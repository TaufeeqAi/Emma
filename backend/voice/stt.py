import asyncio
import io
import logging
from typing import Optional

from groq import Groq, APIError

from backend.config import GROQ_API_KEY, STT_CONFIDENCE_THRESHOLD
from backend.voice.audio_utils import pcm16_to_wav

logger = logging.getLogger(__name__)

# Maximum no-speech probability above which we treat the transcript as silence.
# Whisper's no_speech_prob: 1.0 = definitely silence, 0.0 = definitely speech.
_NO_SPEECH_PROB_THRESHOLD = 0.7


class STTHandler:
    """
    Async Speech-to-Text handler using Groq Whisper Large-v3.

    Instantiate once per process (Groq client maintains a connection pool).
    The transcribe() method is async-safe: multiple concurrent sessions can
    call it simultaneously — each call is independent.

    Args:
        api_key:            Groq API key. Defaults to config.GROQ_API_KEY.
        confidence_threshold: Transcriptions below this avg_logprob are flagged
                              as low-confidence. Default: STT_CONFIDENCE_THRESHOLD.
        model:              Groq Whisper model. Default: whisper-large-v3.
    """

    _MODEL = "whisper-large-v3"
    _LANGUAGE = "en"  # Prevent Whisper from switching languages mid-call

    def __init__(
        self,
        api_key: Optional[str] = None,
        confidence_threshold: float = STT_CONFIDENCE_THRESHOLD,
        model: str = _MODEL,
    ) -> None:
        self._client = Groq(api_key=api_key or GROQ_API_KEY)
        self._confidence_threshold = confidence_threshold
        self._model = model
        logger.info(
            "STTHandler initialised | model=%s confidence_threshold=%.2f",
            model, confidence_threshold,
        )

    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        is_wav: bool = False,
    ) -> dict:
        """
        Transcribe audio bytes to text via Groq Whisper.

        Args:
            audio_bytes:  Raw PCM 16-bit LE audio bytes OR WAV-formatted bytes.
            sample_rate:  Audio sample rate in Hz (used only if is_wav=False).
            is_wav:       True if audio_bytes is already WAV-formatted.
                          False (default) if it's raw PCM — we wrap it in WAV.

        Returns:
            Dict:
              {
                "text":           str,   — Cleaned transcript text
                "confidence":     float, — avg_logprob (0=perfect, -inf=garbage)
                "low_confidence": bool,  — True if below threshold
                "is_silence":     bool,  — True if Whisper thinks no speech
                "no_speech_prob": float, — Whisper's silence probability
                "segments":       list,  — Raw Whisper segment objects
                "word_count":     int,   — Approximate word count
              }

        Raises:
            groq.APIError: on Groq API errors (rate limit, auth failure).
                           Caller (websocket_handler) should catch and handle.
        """
        # Convert PCM to WAV if needed
        wav_bytes = audio_bytes if is_wav else pcm16_to_wav(audio_bytes, sample_rate)

        # Run blocking Groq call in thread pool to avoid blocking event loop
        result = await asyncio.to_thread(self._call_groq_api, wav_bytes)
        return result

    def _call_groq_api(self, wav_bytes: bytes) -> dict:
        """
        Synchronous Groq API call. Runs in thread pool via asyncio.to_thread().
        Do not call directly from async context.
        """
        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "audio.wav"  # Groq uses extension to detect format

        try:
            transcription = self._client.audio.transcriptions.create(
                file=audio_file,
                model=self._model,
                response_format="verbose_json",
                language=self._language,
            )
        except APIError as exc:
            logger.error("Groq STT API error: %s", exc)
            raise

        return self._parse_transcription(transcription)

    def _parse_transcription(self, transcription) -> dict:
        """Parse Groq Whisper verbose_json response into our standard format."""
        text = (transcription.text or "").strip()

        # Extract per-segment confidence metrics
        segments = getattr(transcription, "segments", None) or []
        avg_logprob = 0.0
        no_speech_prob = 0.0

        if segments:
            logprobs = [
                s.get("avg_logprob", -1.0) if isinstance(s, dict)
                else getattr(s, "avg_logprob", -1.0)
                for s in segments
            ]
            no_speech_probs = [
                s.get("no_speech_prob", 0.0) if isinstance(s, dict)
                else getattr(s, "no_speech_prob", 0.0)
                for s in segments
            ]
            avg_logprob = sum(logprobs) / len(logprobs)
            no_speech_prob = sum(no_speech_probs) / len(no_speech_probs)

        is_silence = no_speech_prob >= _NO_SPEECH_PROB_THRESHOLD or not text
        low_confidence = avg_logprob < self._confidence_threshold

        logger.info(
            "STT | text='%.80s' | avg_logprob=%.3f | no_speech_prob=%.3f | "
            "low_confidence=%s | is_silence=%s",
            text, avg_logprob, no_speech_prob, low_confidence, is_silence,
        )

        return {
            "text": text,
            "confidence": avg_logprob,
            "low_confidence": low_confidence,
            "is_silence": is_silence,
            "no_speech_prob": no_speech_prob,
            "segments": segments,
            "word_count": len(text.split()) if text else 0,
        }

    @property
    def _language(self) -> str:
        return self._LANGUAGE