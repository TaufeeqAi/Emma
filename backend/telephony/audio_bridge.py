import logging
import struct
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

# LiveKit AudioFrame constants
_EMMA_SAMPLE_RATE: int = 16000      # Hz — EMMA pipeline native rate
_EMMA_NUM_CHANNELS: int = 1         # Mono
_LIVEKIT_DEFAULT_RATE: int = 48000  # Hz — LiveKit WebRTC internal rate

# Target frame size for VAD (30ms at 16kHz = 480 samples = 960 bytes)
_VAD_FRAME_SAMPLES: int = 480
_VAD_FRAME_BYTES: int = _VAD_FRAME_SAMPLES * 2  # 16-bit = 2 bytes/sample


class AudioBridge:
    """
    Stateful audio buffer + format converter for a single call's audio stream.

    One instance per active call (created by LiveKitCallAdapter).

    Responsibilities:
      1. Accumulate incoming LiveKit AudioFrames into VAD-sized chunks.
      2. Convert between LiveKit AudioFrame dtype and numpy arrays.
      3. Convert numpy TTS output back to LiveKit AudioFrame bytes.
      4. Handle jitter: discard frames if buffer overflows (backpressure).
    """

    def __init__(
        self,
        emma_sample_rate: int = _EMMA_SAMPLE_RATE,
        vad_frame_samples: int = _VAD_FRAME_SAMPLES,
        max_buffer_frames: int = 50,  # ~1.5s of audio; drop frames if exceeded
    ) -> None:
        self._rate = emma_sample_rate
        self._vad_samples = vad_frame_samples
        self._max_buffer = max_buffer_frames * vad_frame_samples * 2  # bytes
        self._buffer = bytearray()
        self._dropped_frames: int = 0

    # ── Inbound: LiveKit → EMMA ─────────────────────────────────────────────────

    def ingest_livekit_frame(self, frame_data: bytes, source_rate: int) -> list[bytes]:
        """
        Accept raw PCM bytes from a LiveKit AudioFrame and return zero or more
        VAD-sized (30ms) chunks ready for the STT pipeline.

        Args:
            frame_data:  Raw bytes from rtc.AudioFrame.data (int16 LE).
            source_rate: frame.sample_rate — may differ from _EMMA_SAMPLE_RATE
                         if AudioStream was not created with sample_rate=16000.

        Returns:
            List of PCM16 byte chunks, each exactly _VAD_FRAME_BYTES long.
        """
        # Resample if source doesn't match our pipeline rate
        if source_rate != self._rate:
            frame_data = _resample_pcm16(frame_data, source_rate, self._rate)

        # Backpressure: drop frame if buffer is saturated (caller talking too fast
        # or pipeline is stalled — clinical safety: don't queue unbounded audio)
        if len(self._buffer) + len(frame_data) > self._max_buffer:
            self._dropped_frames += 1
            if self._dropped_frames % 10 == 0:
                logger.warning(
                    "AudioBridge: buffer saturated, dropped %d frames (check STT latency)",
                    self._dropped_frames,
                )
            return []

        self._buffer.extend(frame_data)

        # Yield complete VAD-sized chunks
        chunks = []
        while len(self._buffer) >= _VAD_FRAME_BYTES:
            chunk = bytes(self._buffer[:_VAD_FRAME_BYTES])
            self._buffer = self._buffer[_VAD_FRAME_BYTES:]
            chunks.append(chunk)

        return chunks

    def flush(self) -> bytes:
        """
        Flush remaining buffered audio at end-of-call.
        Zero-pads to nearest VAD frame boundary.
        Returns PCM16 bytes (may be empty).
        """
        if not self._buffer:
            return b""
        # Zero-pad to VAD frame size
        remainder = len(self._buffer)
        padding_needed = _VAD_FRAME_BYTES - (remainder % _VAD_FRAME_BYTES)
        if padding_needed < _VAD_FRAME_BYTES:
            self._buffer.extend(b"\x00" * padding_needed)
        flushed = bytes(self._buffer)
        self._buffer = bytearray()
        return flushed

    # ── Outbound: EMMA → LiveKit ────────────────────────────────────────────────

    @staticmethod
    def tts_bytes_to_numpy(pcm16_bytes: bytes) -> np.ndarray:
        """
        Convert raw PCM16 bytes from TTSHandler to float32 numpy array.
        Used for optional DSP processing before LiveKit capture.
        """
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0

    @staticmethod
    def numpy_to_livekit_bytes(audio_float32: np.ndarray) -> bytes:
        """
        Convert float32 audio to int16 PCM bytes for LiveKit AudioSource.
        Clips to [-1.0, 1.0] before conversion to prevent clipping distortion.
        """
        clipped = np.clip(audio_float32, -1.0, 1.0)
        int16_samples = (clipped * 32767).astype(np.int16)
        return int16_samples.tobytes()

    @staticmethod
    def chunk_tts_for_livekit(
        pcm16_bytes: bytes,
        chunk_ms: int = 20,
        sample_rate: int = _EMMA_SAMPLE_RATE,
    ) -> list[bytes]:
        """
        Split a TTS audio buffer into LiveKit-friendly chunks.

        LiveKit AudioSource performs best with 20ms chunks (matches
        Opus frame duration and RTP packetisation for SIP delivery).

        Args:
            pcm16_bytes: Full TTS audio as int16 LE PCM.
            chunk_ms:    Chunk duration in milliseconds (default 20ms).
            sample_rate: PCM sample rate (default 16kHz).

        Returns:
            List of byte chunks, each representing chunk_ms of audio.
        """
        samples_per_chunk = int(sample_rate * chunk_ms / 1000)
        bytes_per_chunk = samples_per_chunk * 2  # int16 = 2 bytes

        chunks = []
        for i in range(0, len(pcm16_bytes), bytes_per_chunk):
            chunk = pcm16_bytes[i:i + bytes_per_chunk]
            # Pad last chunk to full size with silence
            if len(chunk) < bytes_per_chunk:
                chunk += b"\x00" * (bytes_per_chunk - len(chunk))
            chunks.append(chunk)

        return chunks


# ── Module-level resampling utility ────────────────────────────────────────────

def _resample_pcm16(
    data: bytes,
    from_rate: int,
    to_rate: int,
) -> bytes:
    """
    Resample PCM16 mono audio using linear interpolation.

    Used as a fallback when AudioStream sample_rate doesn't match our
    pipeline rate (shouldn't happen normally, but defensive).

    For production, use scipy.signal.resample_poly for higher quality.
    This implementation uses numpy interp for zero-dependency operation.
    """
    if from_rate == to_rate:
        return data

    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    n_out = int(len(samples) * to_rate / from_rate)
    x_in = np.linspace(0, len(samples) - 1, len(samples))
    x_out = np.linspace(0, len(samples) - 1, n_out)
    resampled = np.interp(x_out, x_in, samples).astype(np.int16)
    return resampled.tobytes()