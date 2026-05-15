import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional

from backend.telephony.ddi_router import get_ddi_router

logger = logging.getLogger(__name__)

MAX_CALL_DURATION_SECONDS: int = 600  # 10 minutes (clinical safety limit)


class CallState(Enum):
    CONNECTING  = auto()  # Agent task dispatched, not yet joined room
    CONNECTED   = auto()  # Agent in room, SIP participant present
    STREAMING   = auto()  # Audio track subscribed, pipeline running
    ENDED       = auto()  # Normal hangup
    TIMED_OUT   = auto()  # Exceeded MAX_CALL_DURATION_SECONDS
    ERROR       = auto()  # Unrecoverable pipeline error


@dataclass
class ActiveCall:
    """State for a single SIP call managed by LiveKit."""
    room_name: str              
    tenant_id: str               
    caller_identity: str        
    caller_number: str           
    destination_number: str      
    participant_sid: str         
    session_id: Optional[str]    
    trace_id: Optional[str]      
    state: CallState
    started_at: str             
    ended_at: Optional[str]
    dtmf_digits: list = field(default_factory=list)
    error: Optional[str] = None

    @property
    def call_uuid(self) -> str:
        """Alias: extract uuid suffix from room_name for backwards-compat logging."""
        parts = self.room_name.split("-", 1)
        return parts[1] if len(parts) == 2 else self.room_name

    @property
    def duration_seconds(self) -> float:
        if not self.started_at:
            return -1.0
        start = datetime.fromisoformat(self.started_at)
        end_time = (
            datetime.fromisoformat(self.ended_at)
            if self.ended_at
            else datetime.now(tz=timezone.utc)
        )
        return (end_time - start).total_seconds()

    @property
    def is_active(self) -> bool:
        return self.state in (CallState.CONNECTED, CallState.STREAMING)

    @property
    def is_over_limit(self) -> bool:
        return self.duration_seconds > MAX_CALL_DURATION_SECONDS

    def to_dict(self) -> dict:
        return {
            "room_name":          self.room_name,
            "call_uuid":          self.call_uuid,
            "tenant_id":          self.tenant_id,
            "caller_identity":    self.caller_identity,
            "caller_number":      self.caller_number,
            "destination_number": self.destination_number,
            "participant_sid":    self.participant_sid,
            "session_id":         self.session_id,
            "trace_id":           self.trace_id,
            "state":              self.state.name,
            "started_at":         self.started_at,
            "ended_at":           self.ended_at,
            "duration_seconds":   round(self.duration_seconds, 1),
            "dtmf_digits":        self.dtmf_digits,
            "error":              self.error,
        }


class CallManager:
    """
    Process-level registry of active SIP calls.

    Keyed by room_name (unique per call, persists for the room lifetime).
    """

    def __init__(self, ddi_router=None) -> None:
        self._calls: dict[str, ActiveCall] = {}
        self._router = ddi_router or get_ddi_router()
        logger.info("CallManager initialised (LiveKit backend).")

    # ── Call lifecycle 

    def register_call(
        self,
        room_name: str,
        participant_sid: str,
        caller_identity: str,
        destination_number: str,
        caller_number: str = "anonymous",
    ) -> ActiveCall:
        """
        Register a new LiveKit SIP call when the agent joins the room.

        Args:
            room_name:           LiveKit room name, format "{tenant_id}-{uuid}".
            participant_sid:     LiveKit SID of the SIP caller participant.
            caller_identity:     LiveKit participant identity (SIP URI).
            destination_number:  DDI the patient called.
            caller_number:       Patient's outbound number (may be "anonymous").

        Returns:
            New ActiveCall in CONNECTED state.
        """
        tenant_id = self._router.tenant_from_room_name(room_name) or self._router.default_tenant
        call = ActiveCall(
            room_name=room_name,
            tenant_id=tenant_id,
            caller_identity=caller_identity,
            caller_number=caller_number,
            destination_number=destination_number,
            participant_sid=participant_sid,
            session_id=None,
            trace_id=None,
            state=CallState.CONNECTED,
            started_at=datetime.now(tz=timezone.utc).isoformat(),
            ended_at=None,
        )
        self._calls[room_name] = call
        logger.info(
            "Call registered | room=%s | tenant=%s | caller=%s | dest=%s",
            room_name, tenant_id, caller_identity, destination_number,
        )
        return call

    def set_streaming(
        self,
        room_name: str,
        session_id: str,
        trace_id: str,
    ) -> None:
        """Mark call as STREAMING once the audio pipeline is active."""
        call = self._calls.get(room_name)
        if call:
            call.state = CallState.STREAMING
            call.session_id = session_id
            call.trace_id = trace_id
            logger.info(
                "Call STREAMING | room=%s | session=%s | trace=%s",
                room_name, session_id, trace_id,
            )
        else:
            logger.warning("set_streaming: unknown room_name=%s", room_name)

    def record_dtmf(self, room_name: str, digit: str) -> None:
        call = self._calls.get(room_name)
        if call:
            call.dtmf_digits.append(digit)
            logger.info(
                "DTMF '%s' | room=%s | sequence=%s",
                digit, room_name, "".join(call.dtmf_digits),
            )

    def end_call(
        self,
        room_name: str,
        reason: str = "participant_left",
        error: Optional[str] = None,
    ) -> Optional[ActiveCall]:
        """
        Deregister a call on hangup, room finish, or error.

        Args:
            room_name: LiveKit room name.
            reason:    Human-readable end reason.
            error:     Set if call ended due to pipeline error.
        """
        call = self._calls.pop(room_name, None)
        if call:
            call.state = CallState.ERROR if error else CallState.ENDED
            call.ended_at = datetime.now(tz=timezone.utc).isoformat()
            call.error = error
            logger.info(
                "Call ended | room=%s | tenant=%s | duration=%.1fs | reason=%s",
                room_name, call.tenant_id, call.duration_seconds, reason,
            )
        else:
            logger.debug("end_call: unknown room_name=%s (already removed)", room_name)
        return call

    # ── Queries 

    def get_call(self, room_name: str) -> Optional[ActiveCall]:
        return self._calls.get(room_name)

    def get_call_by_session(self, session_id: str) -> Optional[ActiveCall]:
        return next(
            (c for c in self._calls.values() if c.session_id == session_id),
            None,
        )

    def get_call_by_participant_sid(self, participant_sid: str) -> Optional[ActiveCall]:
        return next(
            (c for c in self._calls.values() if c.participant_sid == participant_sid),
            None,
        )

    @property
    def active_call_count(self) -> int:
        return len(self._calls)

    @property
    def active_calls_summary(self) -> list[dict]:
        return [c.to_dict() for c in self._calls.values()]

    def get_timed_out_calls(self) -> list[ActiveCall]:
        return [c for c in self._calls.values() if c.is_over_limit]


# ── Module-level singleton 
_call_manager_instance: Optional[CallManager] = None


def get_call_manager() -> CallManager:
    global _call_manager_instance
    if _call_manager_instance is None:
        _call_manager_instance = CallManager()
    return _call_manager_instance