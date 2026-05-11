import asyncio
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from backend.agents.graph import emma_graph
from backend.agents.state import make_initial_state
from backend.config import (
    AUDIO_SAMPLE_RATE,
    MAX_SILENCE_BEFORE_PROMPT_SECONDS,
)
from backend.voice.audio_utils import resample_audio
from backend.voice.session_manager import (
    SessionManager,
    SessionStatus,
    TurnRecord,
    VoiceSession,
)
from backend.voice.stt import STTHandler
from backend.voice.tts import TTSHandler
from backend.voice.vad import EndOfSpeechDetector, VoiceActivityDetector

logger = logging.getLogger(__name__)

# Silence prompt: sent when patient hasn't spoken for MAX_SILENCE_BEFORE_PROMPT_SECONDS
_SILENCE_PROMPT = (
    "I'm still here if you need help. "
    "You can ask me about appointments, opening hours, or prescriptions."
)

# Turn limit message
_TURN_LIMIT_MESSAGE = (
    "I've reached the limit of what I can help with in this call. "
    "Please hold while I connect you with our reception team."
)

# Session timeout message
_TIMEOUT_MESSAGE = (
    "This call has reached its maximum duration. "
    "Please call back or hold to speak with our team directly."
)


class VoiceSessionHandler:
    """
    Manages the full audio loop for a single patient WebSocket session.

    One instance per WebSocket connection. NOT reusable — create a new
    instance for each incoming WebSocket connection.

    Args:
        tenant_id:       Surgery identifier (validated in main.py before init).
        session_manager: Process-level session registry.
        stt:             Shared STTHandler instance (from app lifespan).
        tts:             Shared TTSHandler instance (from app lifespan).
        browser_sample_rate: Sample rate of incoming browser audio.
                             If different from AUDIO_SAMPLE_RATE, audio is resampled.
    """

    def __init__(
        self,
        tenant_id: str,
        session_manager: SessionManager,
        stt: STTHandler,
        tts: TTSHandler,
        browser_sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> None:
        self._tenant_id = tenant_id
        self._session_manager = session_manager
        self._stt = stt
        self._tts = tts
        self._browser_sample_rate = browser_sample_rate

        # Per-session state
        self._session: Optional[VoiceSession] = None
        self._vad = VoiceActivityDetector()
        self._eos_detector = EndOfSpeechDetector(self._vad)

        # Barge-in signal — set by _listen_for_barge_in() during TTS playback
        self._barge_in_event = asyncio.Event()

        # Currently speaking flag — guards TTS stream against concurrent sends
        self._is_speaking = False

    async def handle_session(self, websocket: WebSocket) -> None:
        """
        Main entry point. Called once per WebSocket connection.

        Manages the full session lifecycle:
          1. Accept WebSocket + create session
          2. Play greeting
          3. Enter listening loop (utterance → pipeline → TTS)
          4. Handle disconnects, timeouts, and turn limits
          5. Clean up session

        Args:
            websocket: FastAPI WebSocket connection (not yet accepted).
        """
        await websocket.accept()

        # Create session
        try:
            self._session = self._session_manager.create_session(self._tenant_id)
        except FileNotFoundError as exc:
            logger.error("Cannot create session — tenant config missing: %s", exc)
            await websocket.close(code=4004, reason="Tenant configuration not found.")
            return

        session = self._session
        logger.info(
            "WebSocket session started | session_id=%s | tenant=%s",
            session.session_id, self._tenant_id,
        )

        try:
            # Play greeting
            session.status = SessionStatus.GREETING
            await self._speak(websocket, session.greeting)

            # Main listening loop
            await self._listening_loop(websocket, session)

        except WebSocketDisconnect:
            logger.info(
                "Patient disconnected | session_id=%s | turns=%d",
                session.session_id, session.turn_count,
            )
        except Exception as exc:
            logger.exception(
                "Unrecoverable session error | session_id=%s | error=%s",
                session.session_id, exc,
            )
            session.end(SessionStatus.ERROR)
            if websocket.client_state == WebSocketState.CONNECTED:
                await self._send_json(websocket, {
                    "type": "error",
                    "message": "An unexpected error occurred. Please call reception directly.",
                    "recoverable": False,
                })
        finally:
            self._session_manager.end_session(session.session_id)
            if websocket.client_state == WebSocketState.CONNECTED:
                await self._send_json(websocket, {
                    "type": "session_end",
                    "reason": session.status.name,
                    "session_summary": session.to_dict(),
                })
                await websocket.close()

    # main listening loop 

    async def _listening_loop(
        self,
        websocket: WebSocket,
        session: VoiceSession,
    ) -> None:
        """
        Core audio loop: receive frames → detect end-of-speech → process → respond.

        Terminates when:
          - Patient disconnects (WebSocketDisconnect propagates up)
          - Session times out (is_timed_out)
          - Turn limit reached (turn_limit_reached)

        Silence tracking uses asyncio.wait_for() on frame receipt with
        MAX_SILENCE_BEFORE_PROMPT_SECONDS timeout.
        """
        session.status = SessionStatus.LISTENING
        last_speech_time = time.monotonic()

        while session.is_active:
            # Session timeout check
            if session.is_timed_out:
                logger.warning(
                    "Session timeout | session_id=%s | turns=%d",
                    session.session_id, session.turn_count,
                )
                await self._speak(websocket, _TIMEOUT_MESSAGE)
                session.end(SessionStatus.ENDED)
                break

            # Turn limit check
            if session.turn_limit_reached:
                logger.warning(
                    "Turn limit reached | session_id=%s", session.session_id
                )
                await self._speak(websocket, _TURN_LIMIT_MESSAGE)
                session.end(SessionStatus.ENDED)
                break

            # Receive next audio chunk (with silence timeout)
            try:
                raw_chunk = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=MAX_SILENCE_BEFORE_PROMPT_SECONDS,
                )
            except asyncio.TimeoutError:
                session.record_silence()
                await self._send_json(websocket, {
                    "type": "silence",
                    "message": _SILENCE_PROMPT,
                })
                await self._speak(websocket, _SILENCE_PROMPT)
                last_speech_time = time.monotonic()
                self._eos_detector.reset()
                continue
            except WebSocketDisconnect:
                raise  # Propagate — handled in handle_session()

            # Resample if browser sends different rate
            chunk = self._maybe_resample(raw_chunk)

            # Barge-in check: if EMMA is speaking and patient starts talking
            if self._is_speaking:
                if self._vad.contains_speech(chunk):
                    logger.info(
                        "Barge-in detected | session_id=%s", session.session_id
                    )
                    session.record_barge_in()
                    self._barge_in_event.set()
                    await self._send_json(websocket, {"type": "barge_in", "message": "Listening..."})
                    self._eos_detector.reset()
                    session.status = SessionStatus.LISTENING
                continue  # Don't process audio while speaking (after barge-in, loop continues)

            # Feed chunk to end-of-speech detector
            self._eos_detector.ingest(chunk)

            # Process complete utterance
            if self._eos_detector.end_of_speech_detected():
                audio_data = self._eos_detector.get_buffered_audio()
                self._eos_detector.reset()
                last_speech_time = time.monotonic()

                await self._process_utterance(websocket, session, audio_data)
                session.status = SessionStatus.LISTENING

    # utterance processing 

    async def _process_utterance(
        self,
        websocket: WebSocket,
        session: VoiceSession,
        audio_bytes: bytes,
    ) -> None:
        """
        Process a complete patient utterance: STT → LangGraph → TTS.
        Records the full turn in session history.
        """
        turn_start = time.perf_counter()
        turn_id = str(uuid.uuid4())
        error_msg: Optional[str] = None

        session.status = SessionStatus.PROCESSING

        # Step 1: STT 
        stt_start = time.perf_counter()
        try:
            stt_result = await self._stt.transcribe(
                audio_bytes,
                sample_rate=AUDIO_SAMPLE_RATE,
                is_wav=False,
            )
        except Exception as exc:
            logger.error(
                "STT failed | session_id=%s | error=%s", session.session_id, exc
            )
            await self._speak(
                websocket,
                "I'm sorry, I couldn't hear that clearly. Could you please repeat?",
            )
            return

        stt_latency_ms = (time.perf_counter() - stt_start) * 1000
        transcript = stt_result["text"]

        # Send transcript to client (for display/debugging)
        await self._send_json(websocket, {
            "type": "transcript",
            "text": transcript,
            "confidence": stt_result["confidence"],
            "session_id": session.session_id,
        })

        # Handle silence detection
        if stt_result["is_silence"] or not transcript:
            logger.debug("STT returned silence | session_id=%s", session.session_id)
            return

        # Handle low confidence — ask patient to repeat
        if stt_result["low_confidence"]:
            logger.info(
                "Low STT confidence %.3f | session_id=%s | transcript='%s'",
                stt_result["confidence"], session.session_id, transcript,
            )
            await self._speak(
                websocket,
                "I'm sorry, I didn't quite catch that. Could you please repeat?",
            )
            return

        logger.info(
            "Utterance | session_id=%s | turn=%d | transcript='%.80s'",
            session.session_id, session.turn_count + 1, transcript,
        )

        # Step 2: LangGraph pipeline 
        pipeline_start = time.perf_counter()
        trace_id = str(uuid.uuid4())

        state = make_initial_state(
            query=transcript,
            tenant_id=self._tenant_id,
            session_id=session.session_id,
            trace_id=trace_id,
        )

        try:
            # Use ainvoke (async) — avoids blocking the event loop during LLM inference
            result = await emma_graph.ainvoke(state)
        except Exception as exc:
            logger.exception(
                "LangGraph pipeline failed | session_id=%s | error=%s",
                session.session_id, exc,
            )
            await self._speak(
                websocket,
                "I'm having trouble accessing that information right now. "
                "Please call reception directly and they'll be happy to help.",
            )
            return

        pipeline_latency_ms = (time.perf_counter() - pipeline_start) * 1000

        final_response = result.get("final_response") or (
            "I don't have that information. Please call our reception team directly."
        )

        # Send agent trace to client
        await self._send_json(websocket, {
            "type": "agent_trace",
            "escalated": result.get("escalate", False),
            "safety_cleared": result.get("safety_cleared", False),
            "verified": result.get("verified", False),
            "latency_ms": pipeline_latency_ms,
            "trace_id": trace_id,
            "session_id": session.session_id,
        })

        # Step 3: TTS 
        tts_start = time.perf_counter()
        session.status = SessionStatus.SPEAKING

        await self._speak(websocket, final_response)

        tts_latency_ms = (time.perf_counter() - tts_start) * 1000
        total_latency_ms = (time.perf_counter() - turn_start) * 1000

        logger.info(
            "Turn complete | session_id=%s | total=%.0fms | "
            "stt=%.0fms | pipeline=%.0fms | tts=%.0fms | escalated=%s",
            session.session_id, total_latency_ms,
            stt_latency_ms, pipeline_latency_ms, tts_latency_ms,
            result.get("escalate"),
        )

        # Record turn
        session.record_turn(TurnRecord(
            turn_id=turn_id,
            transcript=transcript,
            stt_confidence=stt_result["confidence"],
            escalated=result.get("escalate", False),
            final_response=final_response,
            total_latency_ms=total_latency_ms,
            stt_latency_ms=stt_latency_ms,
            pipeline_latency_ms=pipeline_latency_ms,
            tts_latency_ms=tts_latency_ms,
            timestamp=str(time.time()),
            error=result.get("error"),
        ))

    # TTS + barge-in 
    
    async def _speak(self, websocket: WebSocket, text: str) -> None:
        """
        Synthesize and stream TTS audio to the patient with barge-in support.

        Runs two concurrent tasks:
          1. _barge_in_listener:  reads WebSocket frames and sets barge_in_event
             when patient speech is detected.
          2. _audio_streamer:     sends TTS chunks, checks barge_in_event between
             each chunk, stops immediately on barge-in.

        Args:
            websocket: Active FastAPI WebSocket connection.
            text:      Response text to synthesize and speak.
        """
        if not text or not text.strip():
            return

        self._barge_in_event.clear()
        self._is_speaking = True

        if self._session:
            self._session.status = SessionStatus.SPEAKING

        try:
            # Run barge-in listener and audio streamer concurrently
            listener_task = asyncio.create_task(
                self._barge_in_listener(websocket)
            )
            await self._audio_streamer(websocket, text)
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass
        finally:
            self._is_speaking = False
            self._barge_in_event.clear()

    async def _audio_streamer(self, websocket: WebSocket, text: str) -> None:
        """Stream TTS audio chunks to WebSocket, stopping on barge-in."""
        try:
            async for audio_chunk in self._tts.synthesize_streaming_async(text):
                if self._barge_in_event.is_set():
                    logger.debug("_audio_streamer: barge-in event — stopping TTS stream")
                    break
                if websocket.client_state != WebSocketState.CONNECTED:
                    break
                await websocket.send_bytes(audio_chunk)
        except WebSocketDisconnect:
            pass  # Patient disconnected during TTS — handle at session level
        except Exception as exc:
            logger.error("TTS stream error: %s", exc)

    async def _barge_in_listener(self, websocket: WebSocket) -> None:
        """
        Continuously read WebSocket frames during TTS playback.
        Sets _barge_in_event if patient speech is detected.

        This is a background task — cancelled by _speak() after TTS completes.
        """
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_bytes(), timeout=0.1
                    )
                    chunk = self._maybe_resample(raw)
                    if self._vad.contains_speech(chunk):
                        logger.debug("_barge_in_listener: speech detected during TTS")
                        self._barge_in_event.set()
                        break
                except asyncio.TimeoutError:
                    continue  # No audio received — keep listening
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass

    # utilities 

    def _maybe_resample(self, audio_bytes: bytes) -> bytes:
        """Resample audio if browser sends at a different rate than AUDIO_SAMPLE_RATE."""
        if self._browser_sample_rate == AUDIO_SAMPLE_RATE:
            return audio_bytes
        return resample_audio(audio_bytes, self._browser_sample_rate, AUDIO_SAMPLE_RATE)

    @staticmethod
    async def _send_json(websocket: WebSocket, data: dict) -> None:
        """Safe JSON sender — catches errors on disconnected sockets."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(data)
        except Exception as exc:
            logger.debug("_send_json failed (client may have disconnected): %s", exc)