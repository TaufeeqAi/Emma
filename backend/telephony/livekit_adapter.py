import asyncio
import json
import logging
import struct
import uuid
from typing import Optional, AsyncIterator

from livekit import rtc
from livekit.agents import JobContext, AutoSubscribe

from backend.telephony.audio_bridge import AudioBridge
from backend.telephony.call_manager import CallManager, CallState, MAX_CALL_DURATION_SECONDS
from backend.voice.session_manager import SessionManager
from backend.voice.stt import STTHandler
from backend.voice.tts import TTSHandler
from backend.voice.websocket_handler import VoiceSessionHandler

logger = logging.getLogger(__name__)

_SILENCE_FRAME_SAMPLES = 320   # 20ms of silence at 16kHz (for barge-in flush)
_SILENCE_FRAME_BYTES = _SILENCE_FRAME_SAMPLES * 2  # int16
_TIMEOUT_CHECK_INTERVAL = 30   # seconds between timeout checks
_GOODBYE_MESSAGE = "Thank you for calling. Goodbye."


class LiveKitCallAdapter:
    """
    Bridges a LiveKit room's audio to EMMA's VoiceSessionHandler.

    One instance per call. Created by the agent entrypoint (livekit_agent.py).

    Args:
        ctx:             LiveKit JobContext for this room/call.
        call_manager:    Process-level call registry.
        session_manager: Phase 3 voice session registry.
        stt:             Shared STTHandler.
        tts:             Shared TTSHandler.
        room_name:       LiveKit room name (used as call ID).
        tenant_id:       Resolved from room name prefix.
    """

    def __init__(
        self,
        ctx: JobContext,
        call_manager: CallManager,
        session_manager: SessionManager,
        stt: STTHandler,
        tts: TTSHandler,
        room_name: str,
        tenant_id: str,
    ) -> None:
        self._ctx = ctx
        self._call_manager = call_manager
        self._session_manager = session_manager
        self._stt = stt
        self._tts = tts
        self._room_name = room_name
        self._tenant_id = tenant_id
        self._audio_bridge = AudioBridge()
        self._audio_source: Optional[rtc.AudioSource] = None
        self._output_track: Optional[rtc.LocalAudioTrack] = None
        self._session_id: Optional[str] = None
        self._running = False
        self._timeout_task: Optional[asyncio.Task] = None

    async def run(self, sip_participant: rtc.RemoteParticipant) -> None:
        """
        Main entry point. Sets up audio I/O and runs the EMMA pipeline.

        Args:
            sip_participant: The LiveKit participant representing the SIP caller.
        """
        self._running = True

        try:
            # ── Step 1: Publish our output audio track ─────────────────────────
            await self._setup_output_track()

            # ── Step 2: Register the call duration watchdog ────────────────────
            self._timeout_task = asyncio.create_task(
                self._timeout_watchdog(),
                name=f"timeout-{self._room_name}",
            )

            # ── Step 3: Subscribe to caller audio and run pipeline ─────────────
            await self._run_voice_pipeline(sip_participant)

        except asyncio.CancelledError:
            logger.info("CallAdapter cancelled | room=%s", self._room_name)
        except Exception as exc:
            logger.exception(
                "CallAdapter error | room=%s | error=%s", self._room_name, exc
            )
            self._call_manager.end_call(
                self._room_name, reason="pipeline_error", error=str(exc)
            )
        finally:
            await self._cleanup()

    async def _setup_output_track(self) -> None:
        """
        Create and publish our audio output track to the room.

        This track carries EMMA's TTS responses to the SIP caller.
        LiveKit SIP bridge picks up this track and sends it as RTP audio.

        Sample rate: 16kHz (matches EMMA's TTS output; LiveKit handles
        the resampling to 8kHz G.711 for SIP delivery).
        """
        self._audio_source = rtc.AudioSource(
            sample_rate=16000,
            num_channels=1,
            queue_size_ms=1000,  # 1s buffer for jitter; reduce for lower latency
        )
        self._output_track = rtc.LocalAudioTrack.create_audio_track(
            name="emma-voice",
            source=self._audio_source,
        )
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE,
        )
        await self._ctx.room.local_participant.publish_track(
            self._output_track, options
        )
        logger.info(
            "Output audio track published | room=%s | rate=16kHz",
            self._room_name,
        )

    async def _run_voice_pipeline(
        self,
        sip_participant: rtc.RemoteParticipant,
    ) -> None:
        """
        Subscribe to the caller's audio track and run the full EMMA pipeline.

        Uses a custom WebSocket-like adapter to present the LiveKit audio
        stream to VoiceSessionHandler without modifying Phase 3 code.
        """
        # Wait for the caller's audio track to be published
        audio_track = await self._wait_for_audio_track(sip_participant)
        if not audio_track:
            logger.error(
                "No audio track from SIP participant | room=%s", self._room_name
            )
            return

        logger.info(
            "Subscribing to SIP audio | room=%s | participant=%s | track=%s",
            self._room_name, sip_participant.identity, audio_track.sid,
        )

        # Create audio stream at 16kHz (LiveKit resamples from SIP 8kHz internally)
        audio_stream = rtc.AudioStream(
            track=audio_track,
            sample_rate=16000,
            num_channels=1,
        )

        # Create the Phase 3 WebSocket adapter for this LiveKit call
        ws_adapter = _LiveKitWebSocketAdapter(
            audio_stream=audio_stream,
            audio_source=self._audio_source,
            audio_bridge=self._audio_bridge,
            room_name=self._room_name,
        )

        # Create and run the Phase 3 voice session handler (UNCHANGED)
        handler = VoiceSessionHandler(
            tenant_id=self._tenant_id,
            session_manager=self._session_manager,
            stt=self._stt,
            tts=self._tts,
            browser_sample_rate=16000,  # Already at 16kHz from AudioStream
        )

        self._session_id = str(uuid.uuid4())

        # Mark call as STREAMING in CallManager
        self._call_manager.set_streaming(
            room_name=self._room_name,
            session_id=self._session_id,
            trace_id=str(uuid.uuid4()),
        )

        # Subscribe to DTMF data messages
        self._ctx.room.on("data_received", self._handle_data_message)

        # Run the Phase 3 pipeline (blocks until session ends)
        try:
            await handler.handle_session(ws_adapter)  # type: ignore[arg-type]
        finally:
            self._ctx.room.off("data_received", self._handle_data_message)

    async def _wait_for_audio_track(
        self,
        participant: rtc.RemoteParticipant,
        timeout: float = 10.0,
    ) -> Optional[rtc.RemoteAudioTrack]:
        """
        Wait for the SIP participant's audio track to become available.
        SIP participants typically publish their audio track within 1-2s
        of joining the room (after SIP 200 OK and RTP negotiation).

        Failure mode: if track never arrives, we return None and the
        call is cleaned up gracefully.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        track_event = asyncio.Event()
        found_track: list[Optional[rtc.RemoteAudioTrack]] = [None]

        def _on_track(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            remote_participant: rtc.RemoteParticipant,
        ):
            if (
                remote_participant.sid == participant.sid
                and track.kind == rtc.TrackKind.KIND_AUDIO
            ):
                found_track[0] = track  # type: ignore[assignment]
                track_event.set()

        self._ctx.room.on("track_subscribed", _on_track)

        # Check if track was already published before we subscribed
        for pub in participant.track_publications.values():
            if pub.kind == rtc.TrackKind.KIND_AUDIO and pub.track:
                found_track[0] = pub.track  # type: ignore[assignment]
                track_event.set()
                break

        try:
            await asyncio.wait_for(track_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Timeout waiting for audio track | room=%s | participant=%s",
                self._room_name, participant.identity,
            )
        finally:
            self._ctx.room.off("track_subscribed", _on_track)

        return found_track[0]

    def _handle_data_message(
        self,
        data: bytes,
        participant: Optional[rtc.Participant],
        kind: rtc.DataPacketKind,
    ) -> None:
        """
        Handle room data messages — primarily DTMF from SIP bridge.

        LiveKit SIP bridge encodes DTMF as:
          {"type": "dtmf", "digit": "5", "duration": 160}
        """
        try:
            msg = json.loads(data)
            if msg.get("type") == "dtmf":
                digit = str(msg.get("digit", ""))
                if digit:
                    self._call_manager.record_dtmf(self._room_name, digit)
                    logger.info(
                        "DTMF '%s' received | room=%s", digit, self._room_name
                    )
        except (json.JSONDecodeError, Exception) as exc:
            logger.debug("Data message parse error: %s", exc)

    async def _timeout_watchdog(self) -> None:
        """
        Background task that enforces MAX_CALL_DURATION_SECONDS.
        Checks every _TIMEOUT_CHECK_INTERVAL seconds.
        On timeout: plays goodbye message and disconnects.
        """
        while self._running:
            await asyncio.sleep(_TIMEOUT_CHECK_INTERVAL)
            call = self._call_manager.get_call(self._room_name)
            if call and call.is_over_limit:
                logger.warning(
                    "Call exceeded max duration | room=%s | duration=%.0fs",
                    self._room_name, call.duration_seconds,
                )
                call.state = CallState.TIMED_OUT
                # Play goodbye and disconnect
                await self._play_goodbye()
                await self._ctx.room.disconnect()
                return

    async def _play_goodbye(self) -> None:
        """Play a brief goodbye message via TTS before disconnecting."""
        if not self._audio_source:
            return
        try:
            audio_bytes = await self._tts.synthesize(_GOODBYE_MESSAGE)
            chunks = AudioBridge.chunk_tts_for_livekit(audio_bytes)
            for chunk in chunks:
                frame = rtc.AudioFrame(
                    data=chunk,
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=len(chunk) // 2,
                )
                await self._audio_source.capture_frame(frame)
            # Small pause to let audio flush before disconnect
            await asyncio.sleep(0.5)
        except Exception as exc:
            logger.warning("Failed to play goodbye: %s", exc)

    async def _cleanup(self) -> None:
        """Release resources and cancel background tasks."""
        self._running = False
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass

        # Flush remaining audio buffer
        flushed = self._audio_bridge.flush()
        if flushed and self._audio_source:
            try:
                frame = rtc.AudioFrame(
                    data=flushed,
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=len(flushed) // 2,
                )
                await self._audio_source.capture_frame(frame)
            except Exception:
                pass

        logger.info("CallAdapter cleaned up | room=%s", self._room_name)


# ── LiveKit → VoiceSessionHandler adapter ─────────────────────────────────────

class _LiveKitWebSocketAdapter:
    """
    Presents a WebSocket-like interface backed by LiveKit audio streams.

    This allows VoiceSessionHandler (Phase 3, unchanged) to work with
    LiveKit audio tracks as if they were browser WebSocket binary frames.

    Inbound (caller audio):
      VoiceSessionHandler calls receive_bytes() → we pull from AudioStream
      and return VAD-sized 16kHz PCM chunks.

    Outbound (TTS audio):
      VoiceSessionHandler calls send_bytes(data) → we push to AudioSource
      in 20ms chunks for smooth playback.

    Control messages:
      send_json() is used by VoiceSessionHandler for transcript/status events.
      We log these but don't need to forward them (no browser to receive them).
      In Phase 6 (frontend dashboard), we'd forward via a separate SignalR/SSE channel.
    """

    def __init__(
        self,
        audio_stream: rtc.AudioStream,
        audio_source: rtc.AudioSource,
        audio_bridge: AudioBridge,
        room_name: str,
    ) -> None:
        self._audio_stream = audio_stream
        self._audio_source = audio_source
        self._bridge = audio_bridge
        self._room_name = room_name
        self._chunk_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self._ingest_task: Optional[asyncio.Task] = None
        self._start_ingest()

    def _start_ingest(self) -> None:
        """Start background task that pulls AudioStream frames into the chunk queue."""
        self._ingest_task = asyncio.create_task(
            self._ingest_loop(),
            name=f"audio-ingest-{self._room_name}",
        )

    async def _ingest_loop(self) -> None:
        """Pull frames from LiveKit AudioStream and queue as VAD chunks."""
        async for frame_event in self._audio_stream:
            frame: rtc.AudioFrame = frame_event.frame
            chunks = self._bridge.ingest_livekit_frame(
                frame.data.tobytes() if hasattr(frame.data, 'tobytes') else bytes(frame.data),
                frame.sample_rate,
            )
            for chunk in chunks:
                try:
                    self._chunk_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    logger.warning(
                        "Chunk queue full | room=%s — consumer too slow",
                        self._room_name,
                    )

    async def receive_bytes(self) -> bytes:
        """
        VoiceSessionHandler calls this to get the next audio chunk.
        Blocks until a VAD-sized (30ms) chunk is available.
        """
        return await self._chunk_queue.get()

    async def send_bytes(self, data: bytes) -> None:
        """
        VoiceSessionHandler calls this to send TTS audio to the caller.
        Splits into 20ms chunks and pushes to LiveKit AudioSource.
        """
        if not data:
            return
        chunks = AudioBridge.chunk_tts_for_livekit(data)
        for chunk in chunks:
            frame = rtc.AudioFrame(
                data=chunk,
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=len(chunk) // 2,
            )
            try:
                await self._audio_source.capture_frame(frame)
            except Exception as exc:
                logger.warning("AudioSource capture failed: %s", exc)
                break

    async def send_json(self, data: dict) -> None:
        """
        Log control messages from VoiceSessionHandler.
        In Phase 6: forward these via WebSocket to the dashboard frontend.
        """
        msg_type = data.get("type", "unknown")
        logger.debug(
            "VoiceSession event | room=%s | type=%s | data=%s",
            self._room_name, msg_type, str(data)[:200],
        )

    async def accept(self) -> None:
        """No-op: LiveKit connection is already established."""
        pass

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Stop the ingest task and release resources."""
        if self._ingest_task and not self._ingest_task.done():
            self._ingest_task.cancel()
            try:
                await self._ingest_task
            except asyncio.CancelledError:
                pass

    def __getattr__(self, name: str):
        """Proxy unknown attributes to prevent AttributeError in Phase 3 code."""
        logger.debug("_LiveKitWebSocketAdapter: unhandled attr '%s'", name)
        return lambda *a, **kw: None