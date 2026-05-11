import asyncio
import io
import struct
import uuid
import wave
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient, ASGITransport

from backend.voice.audio_utils import (
    compute_rms_energy,
    float32_audio_to_pcm16,
    pcm16_to_float32,
    pcm16_to_wav,
    resample_audio,
    split_into_vad_frames,
    wav_to_pcm16,
)
from backend.voice.vad import (
    EndOfSpeechDetector,
    SpeechState,
    VoiceActivityDetector,
)
from backend.voice.session_manager import (
    SessionManager,
    SessionStatus,
    TurnRecord,
    VoiceSession,
)
from backend.config import AUDIO_SAMPLE_RATE


# ── Fixtures 

@pytest.fixture(scope="module")
def sample_rate() -> int:
    return AUDIO_SAMPLE_RATE  # 16000 Hz


@pytest.fixture
def silence_pcm(sample_rate) -> bytes:
    """0.5 seconds of silence — 16-bit LE PCM zeros."""
    n_samples = sample_rate // 2
    return b"\x00\x00" * n_samples


@pytest.fixture
def speech_pcm(sample_rate) -> bytes:
    """
    0.5 seconds of synthetic 'speech-like' audio — 440 Hz sine wave at 50% amplitude.
    Not real speech, but has sufficient energy and frequency content to trigger VAD
    at lower aggressiveness settings.
    """
    duration = 0.5
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    # Mix of 300Hz and 600Hz — within voice frequency range
    sine = (np.sin(2 * np.pi * 300 * t) + np.sin(2 * np.pi * 600 * t)) * 0.4
    return (sine * 16384).astype(np.int16).tobytes()


@pytest.fixture
def speech_wav(speech_pcm, sample_rate) -> bytes:
    """WAV-wrapped speech_pcm."""
    return pcm16_to_wav(speech_pcm, sample_rate)


@pytest.fixture
def vad(sample_rate) -> VoiceActivityDetector:
    """VoiceActivityDetector with aggressiveness=1 (less strict — works with synthetic audio)."""
    return VoiceActivityDetector(aggressiveness=1, sample_rate=sample_rate)


# ── AudioUtils tests 

class TestAudioUtils:

    def test_pcm16_to_wav_produces_valid_wav(self, silence_pcm, sample_rate):
        """pcm16_to_wav must produce a parseable WAV file."""
        wav = pcm16_to_wav(silence_pcm, sample_rate)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"

    def test_wav_roundtrip(self, silence_pcm, sample_rate):
        """PCM → WAV → PCM should be lossless."""
        wav = pcm16_to_wav(silence_pcm, sample_rate)
        pcm, extracted_rate = wav_to_pcm16(wav)
        assert extracted_rate == sample_rate
        assert pcm == silence_pcm

    def test_float32_to_pcm16_clipping(self):
        """float32 values outside [-1, 1] must be clipped, not wrapped."""
        audio = np.array([1.5, -1.5, 0.5, -0.5], dtype=np.float32)
        pcm = float32_audio_to_pcm16(audio)
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert samples[0] == 32767    # clipped positive
        assert samples[1] == -32767   # clipped negative
        assert abs(samples[2] - 16383) <= 1
        assert abs(samples[3] + 16384) <= 1

    def test_pcm16_to_float32_range(self):
        """float32 output must be in [-1.0, 1.0]."""
        max_sample = struct.pack("<h", 32767)
        min_sample = struct.pack("<h", -32768)
        pcm = max_sample + min_sample
        floats = pcm16_to_float32(pcm)
        assert floats[0] <= 1.0
        assert floats[1] >= -1.0

    def test_float32_pcm16_roundtrip_fidelity(self):
        """float32 → PCM16 → float32 should have low quantisation error."""
        original = np.array([0.1, 0.5, -0.3, -0.8, 0.0], dtype=np.float32)
        pcm = float32_audio_to_pcm16(original)
        recovered = pcm16_to_float32(pcm)
        np.testing.assert_allclose(original, recovered, atol=1e-4)

    def test_rms_energy_silence(self, silence_pcm):
        """Silence should have near-zero RMS energy."""
        rms = compute_rms_energy(silence_pcm)
        assert rms < 1.0

    def test_rms_energy_speech(self, speech_pcm):
        """Synthesized speech should have measurable RMS energy."""
        rms = compute_rms_energy(speech_pcm)
        assert rms > 100.0

    def test_split_into_vad_frames_count(self, sample_rate):
        """30ms frames at 16kHz: each frame is 960 samples = 1920 bytes."""
        duration_ms = 300  # 10 frames
        n_samples = int(sample_rate * duration_ms / 1000)
        pcm = b"\x00\x00" * n_samples
        frames = list(split_into_vad_frames(pcm, sample_rate, frame_duration_ms=30))
        assert len(frames) == 10
        expected_size = int(sample_rate * 30 / 1000) * 2
        assert all(len(f) == expected_size for f in frames)

    def test_split_into_vad_frames_invalid_duration(self, sample_rate):
        """Invalid frame duration must raise ValueError."""
        with pytest.raises(ValueError, match="10, 20, or 30"):
            list(split_into_vad_frames(b"\x00" * 1000, sample_rate, frame_duration_ms=25))

    def test_resample_same_rate_is_noop(self, silence_pcm, sample_rate):
        """Resampling to the same rate must return identical bytes."""
        result = resample_audio(silence_pcm, sample_rate, sample_rate)
        assert result == silence_pcm

    def test_resample_changes_length(self, silence_pcm, sample_rate):
        """Resampling 16kHz → 8kHz should produce half as many samples."""
        result = resample_audio(silence_pcm, sample_rate, sample_rate // 2)
        # Allow ±2 bytes for rounding
        assert abs(len(result) - len(silence_pcm) // 2) <= 4

    def test_empty_pcm_rms(self):
        """Empty bytes should return 0.0 RMS without error."""
        assert compute_rms_energy(b"") == 0.0


# ── VoiceActivityDetector tests 

class TestVoiceActivityDetector:

    def test_vad_initialises(self, sample_rate):
        vad = VoiceActivityDetector(aggressiveness=3, sample_rate=sample_rate)
        assert vad.sample_rate == sample_rate
        assert vad.frame_duration_ms == 30

    def test_vad_invalid_aggressiveness(self, sample_rate):
        with pytest.raises(ValueError, match="0–3"):
            VoiceActivityDetector(aggressiveness=5, sample_rate=sample_rate)

    def test_vad_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="8/16/32/48"):
            VoiceActivityDetector(sample_rate=22050)

    def test_silence_returns_false(self, vad, silence_pcm):
        """
        WebRTC VAD with mode 3 should NOT detect speech in silence.
        We test contains_speech on silence — should return False.
        """
        result = vad.contains_speech(silence_pcm)
        # Silence should not trigger VAD
        assert result is False

    def test_has_sufficient_audio_short(self, vad, sample_rate):
        """Less than 0.8s of audio should return False."""
        short_pcm = b"\x00\x00" * (sample_rate // 4)  # 0.25s
        assert vad.has_sufficient_audio(short_pcm, min_seconds=0.8) is False

    def test_has_sufficient_audio_long(self, vad, sample_rate):
        """More than 0.8s of audio should return True."""
        long_pcm = b"\x00\x00" * sample_rate  # 1.0s
        assert vad.has_sufficient_audio(long_pcm, min_seconds=0.8) is True

    def test_short_frame_returns_false(self, vad):
        """Frames shorter than frame_size_bytes should return False (not crash)."""
        assert vad.is_speech_frame(b"\x00" * 10) is False


# ── EndOfSpeechDetector tests 

class TestEndOfSpeechDetector:

    def test_initial_state_is_listening(self, vad):
        eos = EndOfSpeechDetector(vad)
        assert eos.state == SpeechState.LISTENING
        assert not eos.end_of_speech_detected()

    def test_reset_clears_state(self, vad, silence_pcm):
        eos = EndOfSpeechDetector(vad)
        eos.ingest(silence_pcm)
        eos.reset()
        assert eos.state == SpeechState.LISTENING
        assert eos.get_buffered_audio() == b""

    def test_silence_only_stays_listening(self, vad, silence_pcm):
        """Silence-only input should never reach END_OF_SPEECH."""
        eos = EndOfSpeechDetector(vad)
        for _ in range(10):
            eos.ingest(silence_pcm)
        assert eos.state in (SpeechState.LISTENING,)
        assert not eos.end_of_speech_detected()

    def test_get_buffered_audio_empty_before_speech(self, vad):
        eos = EndOfSpeechDetector(vad)
        assert eos.get_buffered_audio() == b""

    def test_ingest_does_not_crash_on_empty_bytes(self, vad):
        """Empty byte input must not raise."""
        eos = EndOfSpeechDetector(vad)
        eos.ingest(b"")  # Should not raise

    def test_is_receiving_speech_false_initially(self, vad):
        eos = EndOfSpeechDetector(vad)
        assert eos.is_receiving_speech is False


# ── STTHandler unit tests (no API calls) 

class TestSTTHandlerUnit:
    """
    Unit tests for STTHandler parsing logic.
    All Groq API calls are mocked — these tests don't require a Groq API key.
    """

    def _make_mock_transcription(
        self,
        text: str,
        avg_logprob: float = -0.2,
        no_speech_prob: float = 0.05,
    ):
        """Build a mock Groq transcription object."""
        mock = MagicMock()
        mock.text = text
        mock.segments = [
            {"avg_logprob": avg_logprob, "no_speech_prob": no_speech_prob}
        ]
        return mock

    def test_parse_high_confidence(self):
        from backend.voice.stt import STTHandler
        handler = STTHandler()
        mock = self._make_mock_transcription("What are the opening hours?", -0.1, 0.01)
        result = handler._parse_transcription(mock)
        assert result["text"] == "What are the opening hours?"
        assert result["low_confidence"] is False
        assert result["is_silence"] is False
        assert result["word_count"] == 5

    def test_parse_low_confidence(self):
        from backend.voice.stt import STTHandler
        handler = STTHandler()
        mock = self._make_mock_transcription("muh uh booking", -0.8, 0.1)
        result = handler._parse_transcription(mock)
        assert result["low_confidence"] is True

    def test_parse_silence_detected(self):
        from backend.voice.stt import STTHandler
        handler = STTHandler()
        mock = self._make_mock_transcription("Thank you.", -0.2, 0.9)
        result = handler._parse_transcription(mock)
        assert result["is_silence"] is True

    def test_parse_empty_text(self):
        from backend.voice.stt import STTHandler
        handler = STTHandler()
        mock = self._make_mock_transcription("", -0.2, 0.0)
        result = handler._parse_transcription(mock)
        assert result["is_silence"] is True
        assert result["word_count"] == 0

    def test_parse_no_segments(self):
        from backend.voice.stt import STTHandler
        handler = STTHandler()
        mock = MagicMock()
        mock.text = "Hello"
        mock.segments = []
        result = handler._parse_transcription(mock)
        assert result["text"] == "Hello"
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    @pytest.mark.requires_groq
    async def test_transcribe_real_wav(self, speech_wav, sample_rate):
        """Integration test: real Groq Whisper call with synthetic audio."""
        from backend.voice.stt import STTHandler
        handler = STTHandler()
        result = await handler.transcribe(speech_wav, sample_rate, is_wav=True)
        assert "text" in result
        assert "confidence" in result
        assert isinstance(result["low_confidence"], bool)


# ── TTSHandler tests 

class TestTTSHandler:

    @pytest.mark.requires_kokoro
    def test_tts_loads_successfully(self):
        """Kokoro model should load without error."""
        from backend.voice.tts import TTSHandler
        handler = TTSHandler()
        assert handler.sample_rate == 24000

    @pytest.mark.requires_kokoro
    def test_synthesize_returns_wav(self):
        """synthesize() must return valid WAV bytes."""
        from backend.voice.tts import TTSHandler
        handler = TTSHandler()
        wav = handler.synthesize("Hello, you've reached the surgery.")
        assert isinstance(wav, bytes)
        assert len(wav) > 0
        assert wav[:4] == b"RIFF"

    @pytest.mark.requires_kokoro
    def test_synthesize_empty_text(self):
        """synthesize() with empty text must return empty bytes, not crash."""
        from backend.voice.tts import TTSHandler
        handler = TTSHandler()
        result = handler.synthesize("")
        assert result == b""

    @pytest.mark.requires_kokoro
    @pytest.mark.asyncio
    async def test_synthesize_streaming_yields_chunks(self):
        """synthesize_streaming_async() must yield at least one WAV chunk."""
        from backend.voice.tts import TTSHandler
        handler = TTSHandler()
        chunks = []
        async for chunk in handler.synthesize_streaming_async(
            "Our opening hours are Monday to Friday, eight AM to six PM."
        ):
            chunks.append(chunk)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk[:4] == b"RIFF"


# ── SessionManager tests 

class TestSessionManager:

    def test_create_session(self):
        mgr = SessionManager()
        session = mgr.create_session("surgery_greenfield")
        assert session.session_id is not None
        assert session.tenant_id == "surgery_greenfield"
        assert session.status == SessionStatus.CREATED
        assert mgr.active_session_count == 1

    def test_end_session(self):
        mgr = SessionManager()
        session = mgr.create_session("surgery_greenfield")
        mgr.end_session(session.session_id)
        assert mgr.active_session_count == 0

    def test_end_unknown_session_no_crash(self):
        """Ending a non-existent session should not raise."""
        mgr = SessionManager()
        mgr.end_session("nonexistent-session-id")  # Must not raise

    def test_session_greeting_contains_surgery_name(self):
        mgr = SessionManager()
        session = mgr.create_session("surgery_greenfield")
        assert "Greenfield" in session.greeting or "EMMA" in session.greeting

    def test_session_record_turn(self):
        mgr = SessionManager()
        session = mgr.create_session("surgery_greenfield")
        turn = TurnRecord(
            turn_id=str(uuid.uuid4()),
            transcript="What are the opening hours?",
            stt_confidence=-0.2,
            escalated=False,
            final_response="We're open Monday to Friday, eight AM to six PM.",
            total_latency_ms=800.0,
            stt_latency_ms=200.0,
            pipeline_latency_ms=500.0,
            tts_latency_ms=100.0,
            timestamp="2026-01-01T00:00:00",
        )
        session.record_turn(turn)
        assert session.turn_count == 1
        assert session.turns[0].transcript == "What are the opening hours?"

    def test_session_to_dict(self):
        mgr = SessionManager()
        session = mgr.create_session("surgery_greenfield")
        d = session.to_dict()
        assert "session_id" in d
        assert "tenant_id" in d
        assert d["turn_count"] == 0

    def test_multiple_sessions(self):
        mgr = SessionManager()
        s1 = mgr.create_session("surgery_greenfield")
        s2 = mgr.create_session("surgery_riverside")
        assert mgr.active_session_count == 2
        assert s1.session_id != s2.session_id

    def test_session_is_timed_out_false_new(self):
        mgr = SessionManager()
        session = mgr.create_session("surgery_greenfield")
        assert session.is_timed_out is False

    def test_unknown_tenant_raises(self):
        mgr = SessionManager()
        with pytest.raises(FileNotFoundError):
            mgr.create_session("surgery_does_not_exist_xyz")


# ── FastAPI HTTP endpoint tests 

@pytest.fixture
def mock_app():
    """
    Create a test FastAPI app with mocked STT and TTS singletons.
    Avoids loading the 350MB Kokoro model during HTTP endpoint tests.
    """
    from backend.main import create_app
    from backend.voice.session_manager import SessionManager

    test_app = create_app()

    # Override lifespan: inject mocks directly
    test_app.state.session_manager = SessionManager()
    test_app.state.stt = MagicMock()
    test_app.state.tts = MagicMock()
    return test_app


@pytest.mark.asyncio
async def test_health_endpoint(mock_app):
    """GET /health must return 200 with expected fields."""
    async with AsyncClient(
        transport=ASGITransport(app=mock_app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "tenants" in data
    assert "active_sessions" in data


@pytest.mark.asyncio
async def test_tenants_endpoint(mock_app):
    """GET /tenants must list configured surgeries."""
    async with AsyncClient(
        transport=ASGITransport(app=mock_app), base_url="http://test"
    ) as client:
        response = await client.get("/tenants")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2
    tenant_ids = [t["tenant_id"] for t in data]
    assert "surgery_greenfield" in tenant_ids
    assert "surgery_riverside" in tenant_ids


@pytest.mark.asyncio
async def test_get_tenant_config_valid(mock_app):
    """GET /tenants/{tenant_id}/config returns surgery config."""
    async with AsyncClient(
        transport=ASGITransport(app=mock_app), base_url="http://test"
    ) as client:
        response = await client.get("/tenants/surgery_greenfield/config")
    assert response.status_code == 200
    data = response.json()
    assert data["tenant_id"] == "surgery_greenfield"
    assert "surgery_name" in data


@pytest.mark.asyncio
async def test_get_tenant_config_invalid(mock_app):
    """GET /tenants/{tenant_id}/config for unknown tenant returns 404."""
    async with AsyncClient(
        transport=ASGITransport(app=mock_app), base_url="http://test"
    ) as client:
        response = await client.get("/tenants/surgery_nonexistent/config")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_sessions_endpoint(mock_app):
    """GET /sessions returns session count."""
    async with AsyncClient(
        transport=ASGITransport(app=mock_app), base_url="http://test"
    ) as client:
        response = await client.get("/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "active_count" in data
    assert "sessions" in data


@pytest.mark.asyncio
async def test_transcribe_invalid_tenant(mock_app):
    """POST /transcribe/{tenant_id} for unknown tenant returns 404."""
    async with AsyncClient(
        transport=ASGITransport(app=mock_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/transcribe/surgery_invalid",
            files={"audio_file": ("audio.wav", b"RIFF_fake_wav_bytes", "audio/wav")},
        )
    assert response.status_code == 404
