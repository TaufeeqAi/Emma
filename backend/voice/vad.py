"""
backend/voice/vad.py
─────────────────────
Voice Activity Detection (VAD) and end-of-speech detection.

Two distinct problems solved here:
─────────────────────────────────
1. Speech presence detection (is_speech_frame):
   Given a single audio frame, is there speech in it?
   Uses WebRTC VAD — a GMM-based classifier optimised for 300–3400 Hz voice.
   Very fast (<0.1ms per frame), deterministic, no model to load.

2. End-of-speech detection (EndOfSpeechDetector):
   Given a stream of frames, detect when the patient has FINISHED speaking.
   This is the harder problem. We use a state machine:
     SILENCE → SPEECH (when VAD detects speech for N consecutive frames)
     SPEECH → SILENCE (when VAD detects silence for N consecutive frames)
   When the SPEECH→SILENCE transition happens after sufficient audio has
   been buffered, we declare "end of speech" and send to STT.

3. Barge-in detection (is_barge_in):
   Given that EMMA is currently speaking (playing TTS audio), check if
   the patient has started talking. Uses the same VAD but at a lower
   confirmation threshold (1 frame is enough to interrupt — patients
   shouldn't have to shout).

Why not just buffer X seconds of audio and always send?
  Fixed-duration buffering works but has two failure modes:
  a) Short utterances ("yes", "no", "bye") are padded with trailing silence,
     causing Whisper to produce garbage hallucinations at the end.
  b) Long utterances are cut off mid-sentence.
  End-of-speech detection solves both.
"""

import logging
from collections import deque
from enum import Enum, auto
from typing import Optional

import webrtcvad

from backend.voice.audio_utils import (
    compute_rms_energy,
    split_into_vad_frames,
)
from backend.config import (
    AUDIO_SAMPLE_RATE,
    VAD_AGGRESSIVENESS,
    MIN_SPEECH_DURATION_SECONDS,
    END_OF_SPEECH_SILENCE_SECONDS,
)

logger = logging.getLogger(__name__)

# Minimum RMS energy to even attempt VAD — filters dead-silent frames.
# Prevents WebRTC VAD from classifying electrical noise as speech.
_MIN_RMS_FOR_VAD = 30.0


class SpeechState(Enum):
    IDLE = auto()          
    LISTENING = auto()      
    SPEECH = auto()         
    END_OF_SPEECH = auto()  


class VoiceActivityDetector:
    """
    Stateless frame-level speech detector.

    Used for:
      1. Barge-in detection (single-frame check during TTS playback)
      2. Providing raw is_speech signals to EndOfSpeechDetector

    Thread safety: webrtcvad.Vad is not thread-safe. Instantiate one
    VoiceActivityDetector per concurrent WebSocket session.

    Args:
        aggressiveness: VAD sensitivity 0–3. Mode 3 = most aggressive noise
                        filtering. Recommended for phone-quality audio.
        sample_rate:    Audio sample rate. Must be 8000, 16000, 32000, or 48000.
        frame_duration_ms: Frame size for VAD. Must be 10, 20, or 30.
    """

    def __init__(
        self,
        aggressiveness: int = VAD_AGGRESSIVENESS,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        frame_duration_ms: int = 30,
    ) -> None:
        if aggressiveness not in (0, 1, 2, 3):
            raise ValueError(f"VAD aggressiveness must be 0–3. Got: {aggressiveness}")
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"VAD sample_rate must be 8/16/32/48 kHz. Got: {sample_rate}")
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError(f"frame_duration_ms must be 10, 20, or 30. Got: {frame_duration_ms}")

        self._vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        # Bytes per frame: sample_rate * duration_s * 2 bytes/sample (int16)
        self.frame_size_bytes = int(sample_rate * frame_duration_ms / 1000) * 2

        logger.debug(
            "VoiceActivityDetector | aggressiveness=%d sample_rate=%d frame=%dms frame_bytes=%d",
            aggressiveness, sample_rate, frame_duration_ms, self.frame_size_bytes,
        )

    def is_speech_frame(self, frame_bytes: bytes) -> bool:
        """
        Classify a single audio frame as speech or non-speech.

        Args:
            frame_bytes: Exactly frame_size_bytes of PCM 16-bit LE audio.

        Returns:
            True if WebRTC VAD detects speech, False otherwise.
            Returns False (not speech) on any error — safer than crashing.
        """
        if len(frame_bytes) < self.frame_size_bytes:
            return False  # Insufficient data — don't guess

        # Energy gate: skip VAD on silent frames
        if compute_rms_energy(frame_bytes[: self.frame_size_bytes]) < _MIN_RMS_FOR_VAD:
            return False

        try:
            return self._vad.is_speech(
                frame_bytes[: self.frame_size_bytes],
                self.sample_rate,
            )
        except Exception as exc:
            logger.debug("VAD error (returning False): %s", exc)
            return False

    def contains_speech(self, audio_bytes: bytes) -> bool:
        """
        Check if any frame in audio_bytes contains speech.
        Used for barge-in detection: if ANY frame has speech, patient is talking.

        Args:
            audio_bytes: Any length of PCM 16-bit LE audio.

        Returns:
            True if at least one frame contains speech.
        """
        for frame in split_into_vad_frames(
            audio_bytes, self.sample_rate, self.frame_duration_ms
        ):
            if self.is_speech_frame(frame):
                return True
        return False

    def has_sufficient_audio(
        self,
        audio_bytes: bytes,
        min_seconds: float = MIN_SPEECH_DURATION_SECONDS,
    ) -> bool:
        """
        Check if the audio buffer contains at least min_seconds of audio.

        Args:
            audio_bytes: Buffered PCM 16-bit LE audio.
            min_seconds: Minimum duration threshold.

        Returns:
            True if buffer holds at least min_seconds worth of audio.
        """
        min_bytes = int(self.sample_rate * min_seconds) * 2  # 2 bytes/sample
        return len(audio_bytes) >= min_bytes


class EndOfSpeechDetector:
    """
    Stateful end-of-speech detector for streaming audio.

    Maintains a state machine that transitions based on sequences of
    speech/silence frames from VoiceActivityDetector.

    Usage pattern (per WebSocket message loop):
        detector = EndOfSpeechDetector(vad)
        while True:
            chunk = await websocket.receive_bytes()
            detector.ingest(chunk)
            if detector.end_of_speech_detected():
                audio = detector.get_buffered_audio()
                detector.reset()
                # → send audio to STT

    Args:
        vad:              VoiceActivityDetector instance.
        speech_onset_frames:   Number of consecutive speech frames before
                               declaring "patient is speaking". Default: 2.
                               Higher = less false positives. Lower = faster response.
        speech_end_frames:     Number of consecutive silent frames after speech
                               before declaring "patient has finished". Default: 24.
                               At 30ms/frame: 24 frames = 720ms silence.
        min_speech_frames:     Minimum speech frames required to consider an
                               utterance valid (rejects short noise bursts). Default: 5.
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        speech_onset_frames: int = 2,
        speech_end_frames: int = 24,    # 720ms at 30ms/frame
        min_speech_frames: int = 5,     # 150ms minimum utterance
    ) -> None:
        self._vad = vad
        self._speech_onset_frames = speech_onset_frames
        self._speech_end_frames = speech_end_frames
        self._min_speech_frames = min_speech_frames

        # State machine
        self._state = SpeechState.LISTENING
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._total_speech_frames = 0

        # Audio buffer (only accumulates during SPEECH state + padding)
        self._buffer = bytearray()

        # Pre-speech ring buffer: keeps the last N frames before speech onset
        # so we don't clip the start of the utterance
        _ring_size = speech_onset_frames + 2
        self._pre_speech_buffer: deque[bytes] = deque(maxlen=_ring_size)

    def ingest(self, audio_bytes: bytes) -> None:
        """
        Process incoming audio bytes. Updates internal state machine.

        Call this for every chunk received from the WebSocket.
        Chunks can be any size — they are split into frames internally.

        Args:
            audio_bytes: PCM 16-bit LE audio chunk (any length).
        """
        for frame in split_into_vad_frames(
            audio_bytes,
            self._vad.sample_rate,
            self._vad.frame_duration_ms,
        ):
            self._process_frame(frame)

    def end_of_speech_detected(self) -> bool:
        """Returns True if the state machine has reached END_OF_SPEECH."""
        return self._state == SpeechState.END_OF_SPEECH

    def get_buffered_audio(self) -> bytes:
        """
        Return the accumulated speech audio buffer.
        Call only after end_of_speech_detected() returns True.
        """
        return bytes(self._buffer)

    def reset(self) -> None:
        """Reset state machine for the next utterance."""
        self._state = SpeechState.LISTENING
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._total_speech_frames = 0
        self._buffer = bytearray()
        self._pre_speech_buffer.clear()

    @property
    def state(self) -> SpeechState:
        return self._state

    @property
    def is_receiving_speech(self) -> bool:
        return self._state == SpeechState.SPEECH

    def _process_frame(self, frame: bytes) -> None:
        """Apply state machine transitions for a single audio frame."""
        is_speech = self._vad.is_speech_frame(frame)

        if self._state == SpeechState.LISTENING:
            if is_speech:
                self._consecutive_speech += 1
                self._pre_speech_buffer.append(frame)
                if self._consecutive_speech >= self._speech_onset_frames:
                    # Speech onset confirmed — transition to SPEECH
                    self._state = SpeechState.SPEECH
                    # Prepend the pre-speech buffer (captures speech onset)
                    for pre_frame in self._pre_speech_buffer:
                        self._buffer.extend(pre_frame)
                    self._pre_speech_buffer.clear()
                    logger.debug("EndOfSpeechDetector: LISTENING → SPEECH")
            else:
                self._consecutive_speech = 0
                self._pre_speech_buffer.append(frame)

        elif self._state == SpeechState.SPEECH:
            self._buffer.extend(frame)
            if is_speech:
                self._consecutive_silence = 0
                self._total_speech_frames += 1
            else:
                self._consecutive_silence += 1
                if self._consecutive_silence >= self._speech_end_frames:
                    if self._total_speech_frames >= self._min_speech_frames:
                        # Patient has finished speaking — valid utterance
                        self._state = SpeechState.END_OF_SPEECH
                        logger.debug(
                            "EndOfSpeechDetector: SPEECH → END_OF_SPEECH "
                            "(speech_frames=%d, silence_frames=%d)",
                            self._total_speech_frames, self._consecutive_silence,
                        )
                    else:
                        # Too short — likely noise burst, ignore
                        logger.debug(
                            "EndOfSpeechDetector: ignoring short utterance "
                            "(speech_frames=%d < min=%d)",
                            self._total_speech_frames, self._min_speech_frames,
                        )
                        self.reset()