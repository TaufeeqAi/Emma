import json
import logging
import re
from typing import Optional, Callable, Awaitable

from livekit.protocol import webhook as webhook_proto
from livekit.protocol import models as lk_models
from livekit.api import WebhookReceiver, TokenVerifier

from backend.config import LIVEKIT_API_KEY, LIVEKIT_API_SECRET
from backend.telephony.call_manager import CallManager

logger = logging.getLogger(__name__)

# LiveKit SIP participant identity prefix
_SIP_IDENTITY_RE = re.compile(r"sip:([^@;]+)@")

# Type alias for hangup callbacks
HangupCallback = Callable[["ActiveCall"], Awaitable[None]]  # type: ignore


class LiveKitWebhookHandler:
    """
    Processes LiveKit webhook events and updates CallManager state.

    Args:
        call_manager:       Process-level call registry.
        on_call_connected:  Optional async callback fired on participant_joined (SIP).
        on_call_ended:      Optional async callback fired on participant_left (SIP).
    """

    def __init__(
        self,
        call_manager: CallManager,
        on_call_connected: Optional[HangupCallback] = None,
        on_call_ended: Optional[HangupCallback] = None,
    ) -> None:
        self._call_manager = call_manager
        self._on_call_connected = on_call_connected
        self._on_call_ended = on_call_ended
        token_verifier = TokenVerifier(
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )
        self._receiver = WebhookReceiver(token_verifier)

    async def process_request(
        self,
        body: bytes,
        auth_header: str,
    ) -> bool:
        """
        Validate and process a LiveKit webhook request.

        Args:
            body:        Raw request body bytes.
            auth_header: Value of the 'Authorization' header.

        Returns:
            True if processed successfully, False if validation failed.
        """
        try:
            event = self._receiver.receive(body.decode("utf-8"), auth_header)
        except Exception as exc:
            logger.warning("LiveKit webhook validation failed: %s", exc)
            return False

        await self._dispatch_event(event)
        return True

    async def _dispatch_event(self, event: webhook_proto.WebhookEvent) -> None:
        event_type = event.WhichOneof("event")

        if event_type == "room_started":
            await self._on_room_started(event.room_started.room)
        elif event_type == "room_finished":
            await self._on_room_finished(event.room_finished.room)
        elif event_type == "participant_joined":
            await self._on_participant_joined(
                event.participant_joined.room,
                event.participant_joined.participant,
            )
        elif event_type == "participant_left":
            await self._on_participant_left(
                event.participant_left.room,
                event.participant_left.participant,
            )
        else:
            logger.debug("Unhandled LiveKit webhook event type: %s", event_type)

    async def _on_room_started(self, room: lk_models.Room) -> None:
        """
        Room created (by SIP dispatch rule). No-op here — the agent entrypoint
        handles call setup when it joins the room.
        """
        logger.info(
            "Room started | name=%s | metadata=%s",
            room.name, room.metadata[:100] if room.metadata else "",
        )

    async def _on_room_finished(self, room: lk_models.Room) -> None:
        """
        Room finished (all participants left). Final cleanup pass.
        The call is usually already ended via participant_left, but this
        covers edge cases (e.g. agent crash without clean disconnect).
        """
        logger.info("Room finished | name=%s", room.name)
        existing = self._call_manager.get_call(room.name)
        if existing:
            ended = self._call_manager.end_call(room.name, reason="room_finished")
            if ended and self._on_call_ended:
                try:
                    await self._on_call_ended(ended)
                except Exception as exc:
                    logger.error("on_call_ended callback error: %s", exc)

    async def _on_participant_joined(
        self,
        room: lk_models.Room,
        participant: lk_models.ParticipantInfo,
    ) -> None:
        """
        Participant joined room. Register call if this is a SIP participant.
        Agent participants are identified by their identity prefix.
        """
        identity = participant.identity
        is_sip = _is_sip_participant(identity)

        logger.info(
            "Participant joined | room=%s | identity=%s | kind=%s | sip=%s",
            room.name, identity, participant.kind.name, is_sip,
        )

        if not is_sip:
            # Agent participant joining — no action needed here
            return

        caller_number = _extract_sip_number(identity) or "anonymous"
        destination_number = _extract_destination_from_metadata(room.metadata)

        call = self._call_manager.register_call(
            room_name=room.name,
            participant_sid=participant.sid,
            caller_identity=identity,
            destination_number=destination_number,
            caller_number=caller_number,
        )

        if self._on_call_connected:
            try:
                await self._on_call_connected(call)
            except Exception as exc:
                logger.error("on_call_connected callback error: %s", exc)

    async def _on_participant_left(
        self,
        room: lk_models.Room,
        participant: lk_models.ParticipantInfo,
    ) -> None:
        """
        Participant left room. End call if this is the SIP caller.
        Agent leaving does not end the call record (agent may reconnect).
        """
        identity = participant.identity
        if not _is_sip_participant(identity):
            logger.debug(
                "Non-SIP participant left | room=%s | identity=%s",
                room.name, identity,
            )
            return

        logger.info(
            "SIP participant left | room=%s | identity=%s",
            room.name, identity,
        )

        ended = self._call_manager.end_call(room.name, reason="sip_participant_left")
        if ended and self._on_call_ended:
            try:
                await self._on_call_ended(ended)
            except Exception as exc:
                logger.error("on_call_ended callback error: %s", exc)


def _is_sip_participant(identity: str) -> bool:
    """SIP participants have identities starting with 'sip:'."""
    return identity.lower().startswith("sip:")


def _extract_sip_number(identity: str) -> Optional[str]:
    """
    Extract the phone number from a SIP URI.

    Examples:
      "sip:+441234567890@provider.com" → "+441234567890"
      "sip:1000@192.168.1.1:5060"     → "1000"
      "sip:anonymous@provider.com"     → "anonymous"
    """
    match = _SIP_IDENTITY_RE.match(identity)
    return match.group(1) if match else None


def _extract_destination_from_metadata(metadata: str) -> str:
    """
    Extract the dialled DDI from room metadata JSON.
    Room metadata is set by the SIP dispatch rule.
    Falls back to "unknown" if not present.
    """
    if not metadata:
        return "unknown"
    try:
        data = json.loads(metadata)
        return data.get("destination", data.get("ddi", "unknown"))
    except (json.JSONDecodeError, TypeError):
        return "unknown"