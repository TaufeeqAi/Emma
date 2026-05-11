import asyncio
import logging
from typing import AsyncGenerator, Optional

import numpy as np

from backend.config import TTS_SPEAKING_RATE
from backend.voice.audio_utils import float32_audio_to_pcm16, pcm16_to_wav

logger = logging.getLogger(__name__)

_KOKORO_SAMPLE_RATE = 24000
_PREFERRED_VOICE = "bf_emma"
_FALLBACK_VOICE = "af_sarah"


class TTSHandler:
    """
    Local TTS handler using Kokoro.

    Instantiate once per process — the model is loaded at __init__.
    synthesize() is blocking.
    synthesize_streaming_async() yields chunks from a background worker.

    Args:
        lang_code:    Kokoro language code. 'en-gb' for British English.
        speaking_rate: Speech speed. 1.0 = normal, 0.9 = slightly slower.
        voice:        Kokoro voice name. Default: bf_emma (British female).
    """

    def __init__(
        self,
        lang_code: str = "en-gb",
        speaking_rate: float = TTS_SPEAKING_RATE,
        voice: str = _PREFERRED_VOICE,
    ) -> None:
        logger.info(
            "Loading Kokoro TTS model (lang=%s, runs locally)...",
            lang_code,
        )
        try:
            from kokoro import KPipeline  # noqa: PLC0415

            self._pipeline = KPipeline(lang_code=lang_code)
            self._voice = self._resolve_voice(voice)
        except ImportError as exc:
            raise ImportError(
                "Kokoro TTS not installed. Run: pip install kokoro==0.9.4"
            ) from exc
        except Exception as exc:
            logger.error("Kokoro initialisation failed: %s", exc)
            raise

        self.sample_rate = _KOKORO_SAMPLE_RATE
        self.speaking_rate = speaking_rate
        logger.info(
            "Kokoro TTS ready | voice=%s | rate=%.2f | sample_rate=%d Hz",
            self._voice,
            speaking_rate,
            self.sample_rate,
        )

    def synthesize(self, text: str) -> bytes:
        """
        Synthesize full audio for a text string.

        Blocking call — use for non-interactive paths.
        For streaming voice responses, use synthesize_streaming_async().

        Args:
            text: Response text to synthesize.

        Returns:
            WAV-formatted bytes at 24000 Hz, 16-bit PCM.
            Returns b"" if synthesis fails.
        """
        if not text or not text.strip():
            logger.warning("TTSHandler.synthesize() called with empty text.")
            return b""

        audio_chunks: list[np.ndarray] = []
        try:
            for _, _, audio in self._pipeline(
                text,
                voice=self._voice,
                speed=self.speaking_rate,
            ):
                if audio is not None and len(audio) > 0:
                    audio_chunks.append(self._to_numpy(audio))
        except Exception as exc:
            logger.error("Kokoro synthesis failed: %s", exc)
            return b""

        if not audio_chunks:
            return b""

        combined = np.concatenate(audio_chunks)
        return self._array_to_wav(combined)

    async def synthesize_streaming_async(
        self, text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        True streaming TTS generator.

        WAV chunks are yielded as Kokoro produces them. This avoids collecting
        the full output first and improves perceived latency.

        Args:
            text: Response text to synthesize.

        Yields:
            WAV-formatted bytes for each synthesis chunk.
        """
        if not text or not text.strip():
            logger.warning("synthesize_streaming_async() called with empty text.")
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=1)

        def _produce_sync() -> None:
            """
            Run Kokoro synthesis in a worker thread and push WAV chunks into
            the async queue with backpressure.
            """
            try:
                for _, _, audio in self._pipeline(
                    text,
                    voice=self._voice,
                    speed=self.speaking_rate,
                ):
                    if audio is not None and len(audio) > 0:
                        wav_bytes = self._array_to_wav(audio)
                        future = asyncio.run_coroutine_threadsafe(
                            queue.put(wav_bytes), loop
                        )
                        future.result()
            except Exception as exc:
                logger.error("TTS streaming production error: %s", exc)
            finally:
                try:
                    future = asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                    future.result(timeout=5.0)
                except Exception:
                    pass

        producer_task = loop.run_in_executor(None, _produce_sync)

        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            try:
                await producer_task
            except Exception:
                pass

    # ── Private methods ────────────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(audio_array) -> np.ndarray:
        """
        Convert Kokoro output to a NumPy array safely.
        Handles PyTorch tensors, NumPy arrays, or array-like outputs.
        """
        if hasattr(audio_array, "detach"):
            audio_array = audio_array.detach().cpu().numpy()
        elif hasattr(audio_array, "numpy"):
            audio_array = audio_array.numpy()

        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)

        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        return audio_array

    def _array_to_wav(self, audio_array) -> bytes:
        """Convert float32 audio to WAV bytes at Kokoro's sample rate."""
        audio_np = self._to_numpy(audio_array)
        pcm = float32_audio_to_pcm16(audio_np)
        return pcm16_to_wav(pcm, self.sample_rate)

    def _resolve_voice(self, preferred: str) -> str:
        """
        Resolve the voice name to use.
        Falls back to _FALLBACK_VOICE if the preferred voice is not available.
        """
        try:
            # Attempt a tiny synthesis to verify voice exists
            list(self._pipeline("test", voice=preferred, speed=1.0))
            return preferred
        except Exception:
            logger.warning(
                "Voice '%s' not available — falling back to '%s'",
                preferred,
                _FALLBACK_VOICE,
            )
            return _FALLBACK_VOICE