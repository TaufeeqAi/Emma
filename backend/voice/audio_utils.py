import io
import logging
import struct
import wave
from typing import Generator, List

import numpy as np

logger = logging.getLogger(__name__)


def pcm16_to_wav(
    pcm_bytes: bytes,
    sample_rate: int,
    channels: int = 1,
) -> bytes:
    """
    Wrap raw PCM 16-bit LE bytes in a WAV container.

    Groq Whisper requires a WAV container — it will reject raw PCM.
    This is cheaper than writing to disk (no disk I/O).

    Args:
        pcm_bytes:   Raw 16-bit little-endian PCM audio bytes.
        sample_rate: Sample rate in Hz (e.g. 16000).
        channels:    Number of audio channels. Default: 1 (mono).

    Returns:
        WAV-formatted bytes (RIFF container with PCM16 payload).
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)          # 16-bit = 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buffer.getvalue()


def wav_to_pcm16(wav_bytes: bytes) -> tuple[bytes, int]:
    """
    Extract raw PCM 16-bit LE bytes from a WAV container.

    Returns:
        Tuple of (pcm_bytes, sample_rate).

    Raises:
        wave.Error: if wav_bytes is not a valid WAV file.
    """
    buffer = io.BytesIO(wav_bytes)
    with wave.open(buffer, "rb") as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    return frames, sample_rate


def float32_audio_to_pcm16(audio_array: np.ndarray) -> bytes:
    """
    Convert float32 numpy audio array (range [-1.0, 1.0]) to PCM 16-bit LE bytes.

    Kokoro TTS outputs float32 arrays. WebRTC VAD and most telephony systems
    expect int16. This conversion is the bridge.

    Args:
        audio_array: numpy float32 array with values in [-1.0, 1.0].

    Returns:
        Raw bytes, 16-bit little-endian, clipped to [-32768, 32767].
    """
    # Clip to prevent overflow artifacts on values slightly outside [-1, 1]
    clipped = np.clip(audio_array, -1.0, 1.0)
    int16_array = (clipped * 32767).astype(np.int16)
    return int16_array.tobytes()


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """
    Convert PCM 16-bit LE bytes to float32 numpy array in [-1.0, 1.0].

    Args:
        pcm_bytes: Raw 16-bit little-endian PCM bytes.

    Returns:
        numpy float32 array.
    """
    int16_array = np.frombuffer(pcm_bytes, dtype=np.int16)
    return int16_array.astype(np.float32) / 32768.0


def split_into_vad_frames(
    pcm_bytes: bytes,
    sample_rate: int,
    frame_duration_ms: int = 30,
) -> Generator[bytes, None, None]:
    """
    Split a PCM byte buffer into fixed-duration frames for WebRTC VAD.

    WebRTC VAD is strict: frames must be exactly 10, 20, or 30ms.
    Any other duration raises a ValueError.

    Args:
        pcm_bytes:        Raw PCM 16-bit LE bytes.
        sample_rate:      Sample rate in Hz.
        frame_duration_ms: Frame size in milliseconds. Must be 10, 20, or 30.

    Yields:
        Fixed-size PCM frame bytes (silent frames at end if audio doesn't divide evenly).

    Raises:
        ValueError: if frame_duration_ms is not 10, 20, or 30.
    """
    if frame_duration_ms not in (10, 20, 30):
        raise ValueError(
            f"WebRTC VAD requires frame_duration_ms in 10, 20, or 30. Got: {frame_duration_ms}"
        )
    # 2 bytes per sample (int16), mono
    frame_size_bytes = int(sample_rate * frame_duration_ms / 1000) * 2

    for offset in range(0, len(pcm_bytes) - frame_size_bytes + 1, frame_size_bytes):
        yield pcm_bytes[offset : offset + frame_size_bytes]


def compute_rms_energy(pcm_bytes: bytes) -> float:
    """
    Compute Root Mean Square energy of a PCM 16-bit audio buffer.

    Used to filter out extremely quiet frames (room noise) before VAD,
    reducing false-positive speech detections in very quiet environments.

    Returns:
        RMS value in range [0.0, 32767.0]. Values below ~50 are silence.
    """
    if not pcm_bytes:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2)))


def resample_audio(
    pcm_bytes: bytes,
    source_rate: int,
    target_rate: int,
) -> bytes:
    """
    Simple linear resampling of PCM 16-bit audio.

    Used when the browser sends audio at a different sample rate than
    WebRTC VAD requires (e.g., browser sends 48kHz, VAD needs 16kHz).

    Note: For production, replace with scipy.signal.resample or librosa
    for higher quality anti-aliased resampling. This linear implementation
    is adequate for voice (300–3400 Hz bandwidth) but not music.

    Args:
        pcm_bytes:   Raw PCM 16-bit LE input.
        source_rate: Original sample rate.
        target_rate: Desired sample rate.

    Returns:
        Resampled PCM 16-bit LE bytes.
    """
    if source_rate == target_rate:
        return pcm_bytes

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    ratio = target_rate / source_rate
    new_length = int(len(samples) * ratio)
    resampled = np.interp(
        np.linspace(0, len(samples) - 1, new_length),
        np.arange(len(samples)),
        samples,
    ).astype(np.int16)
    return resampled.tobytes()