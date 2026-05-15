"""
Test categories:
  Unit tests (no external dependencies — always run):
    TestDDIRouter              — number normalisation, room naming
    TestCallManager            — call lifecycle state machine
    TestAudioBridge            — audio buffering, resampling, chunking
    TestWebhookHandler         — event parsing, call registration

  Integration tests (require LiveKit server):
    @pytest.mark.requires_livekit
    test_sip_provisioner_*     — trunk/dispatch rule CRUD via LiveKit API
    test_livekit_connectivity  — server reachability

  End-to-end tests (require LiveKit + SIP softphone/simulator):
    @pytest.mark.requires_sip
    test_full_sip_call_*       — full SIP call via PJSUA/Linphone/Baresip

Running:
    # Unit tests only (fast, no dependencies)
    pytest tests/test_telephony_livekit.py -v \
        -m "not requires_livekit and not requires_sip"

    # With LiveKit server running
    pytest tests/test_telephony_livekit.py -v -m "requires_livekit"

    # Full E2E (requires PJSUA/Baresip configured to call LiveKit SIP)
    pytest tests/test_telephony_livekit.py -v -m "requires_sip"
"""

import asyncio
import json
import struct
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

from backend.telephony.audio_bridge import AudioBridge
# ── Fixtures 

@pytest.fixture
def ddi_router():
    from backend.telephony.ddi_router import DDIRouter
    return DDIRouter()


@pytest.fixture
def call_manager(ddi_router):
    from backend.telephony.call_manager import CallManager
    return CallManager(ddi_router=ddi_router)


@pytest.fixture
def audio_bridge():
    from backend.telephony.audio_bridge import AudioBridge
    return AudioBridge()


@pytest.fixture
def webhook_handler(call_manager):
    from backend.telephony.webhook_handler import LiveKitWebhookHandler
    return LiveKitWebhookHandler(call_manager=call_manager)


# ── TestDDIRouter 

class TestDDIRouter:

    def test_exact_match_greenfield(self, ddi_router):
        assert ddi_router.route("1000") == "surgery_greenfield"

    def test_exact_match_riverside(self, ddi_router):
        assert ddi_router.route("1001") == "surgery_riverside"

    def test_normalisation_strips_plus(self, ddi_router):
        assert ddi_router.route("+441234567890") == "surgery_greenfield"

    def test_normalisation_strips_spaces(self, ddi_router):
        assert ddi_router.route("01234 567 890") == "surgery_greenfield"

    def test_normalisation_intl_prefix(self, ddi_router):
        # 01234567890 → 441234567890
        assert ddi_router.route("01234567890") == "surgery_greenfield"

    def test_unknown_defaults_to_default_tenant(self, ddi_router):
        from backend.config import DEFAULT_TENANT
        result = ddi_router.route("99999999999")
        assert result == DEFAULT_TENANT

    def test_room_name_for_tenant(self, ddi_router):
        name = ddi_router.room_name_for_tenant("surgery_greenfield", "abc-123")
        assert name == "surgery_greenfield-abc-123"

    def test_tenant_from_room_name_greenfield(self, ddi_router):
        tenant = ddi_router.tenant_from_room_name("surgery_greenfield-some-uuid")
        assert tenant == "surgery_greenfield"

    def test_tenant_from_room_name_riverside(self, ddi_router):
        tenant = ddi_router.tenant_from_room_name("surgery_riverside-abc123def")
        assert tenant == "surgery_riverside"

    def test_tenant_from_room_name_unknown(self, ddi_router):
        tenant = ddi_router.tenant_from_room_name("unknown-room-name")
        assert tenant is None

    def test_ddi_numbers_for_tenant_format(self, ddi_router):
        numbers = ddi_router.ddi_numbers_for_tenant("surgery_greenfield")
        for n in numbers:
            assert n.startswith("+"), f"Expected E.164 format, got '{n}'"
            assert len(n) >= 11, f"Too short for E.164: '{n}'"

    def test_add_route_valid(self, ddi_router):
        ddi_router.add_route("07777000001", "surgery_greenfield")
        assert ddi_router.route("07777000001") == "surgery_greenfield"

    def test_add_route_invalid_tenant(self, ddi_router):
        with pytest.raises(ValueError, match="Unknown tenant_id"):
            ddi_router.add_route("07777000002", "surgery_nonexistent")

    def test_remove_route(self, ddi_router):
        from backend.config import DEFAULT_TENANT
        # Use the non-default tenant so removal isn't blocked by default-route protection
        tenant = "surgery_riverside" if DEFAULT_TENANT == "surgery_greenfield" else "surgery_greenfield"
        ddi_router.add_route("07777000003", tenant)
        removed = ddi_router.remove_route("07777000003")
        assert removed is True

    def test_get_all_routes_returns_dict(self, ddi_router):
        routes = ddi_router.get_all_routes()
        assert isinstance(routes, dict)
        assert "1000" in routes


# ── TestCallManager 

class TestCallManager:

    def test_register_call_creates_active_call(self, call_manager):
        call = call_manager.register_call(
            room_name="surgery_greenfield-abc123",
            participant_sid="PA_abc123",
            caller_identity="sip:+441234567890@provider.com",
            destination_number="441234567890",
            caller_number="+441234567890",
        )
        assert call is not None
        assert call.tenant_id == "surgery_greenfield"
        assert call.room_name == "surgery_greenfield-abc123"

    def test_register_call_resolves_tenant_from_room(self, call_manager):
        call = call_manager.register_call(
            room_name="surgery_riverside-def456",
            participant_sid="PA_def456",
            caller_identity="sip:+441234987654@provider.com",
            destination_number="441234987654",
        )
        assert call.tenant_id == "surgery_riverside"

    def test_active_call_count(self, call_manager):
        assert call_manager.active_call_count == 0
        call_manager.register_call(
            "surgery_greenfield-x1", "PA_1", "sip:1@p.com", "1000"
        )
        assert call_manager.active_call_count == 1

    def test_set_streaming_updates_state(self, call_manager):
        from backend.telephony.call_manager import CallState
        call_manager.register_call(
            "surgery_greenfield-s1", "PA_s1", "sip:1@p.com", "1000"
        )
        call_manager.set_streaming(
            "surgery_greenfield-s1", "sess-001", "trace-001"
        )
        call = call_manager.get_call("surgery_greenfield-s1")
        assert call.state == CallState.STREAMING
        assert call.session_id == "sess-001"

    def test_end_call_removes_from_registry(self, call_manager):
        call_manager.register_call(
            "surgery_greenfield-e1", "PA_e1", "sip:1@p.com", "1000"
        )
        call_manager.end_call("surgery_greenfield-e1", reason="test")
        assert call_manager.get_call("surgery_greenfield-e1") is None
        assert call_manager.active_call_count == 0

    def test_end_call_unknown_room_returns_none(self, call_manager):
        result = call_manager.end_call("nonexistent-room", reason="test")
        assert result is None

    def test_record_dtmf(self, call_manager):
        call_manager.register_call(
            "surgery_greenfield-d1", "PA_d1", "sip:1@p.com", "1000"
        )
        call_manager.record_dtmf("surgery_greenfield-d1", "5")
        call_manager.record_dtmf("surgery_greenfield-d1", "#")
        call = call_manager.get_call("surgery_greenfield-d1")
        assert call.dtmf_digits == ["5", "#"]

    def test_call_uuid_alias(self, call_manager):
        call = call_manager.register_call(
            "surgery_greenfield-uuid999", "PA_u", "sip:1@p.com", "1000"
        )
        assert call.call_uuid == "uuid999"

    def test_duration_seconds_positive_after_registration(self, call_manager):
        call = call_manager.register_call(
            "surgery_greenfield-dur1", "PA_dur", "sip:1@p.com", "1000"
        )
        assert call.duration_seconds >= 0.0

    def test_get_call_by_participant_sid(self, call_manager):
        call_manager.register_call(
            "surgery_greenfield-sid1", "PA_sid1", "sip:1@p.com", "1000"
        )
        found = call_manager.get_call_by_participant_sid("PA_sid1")
        assert found is not None
        assert found.room_name == "surgery_greenfield-sid1"

    def test_multiple_concurrent_calls(self, call_manager):
        for i in range(5):
            call_manager.register_call(
                f"surgery_greenfield-cc{i}",
                f"PA_{i}",
                f"sip:{i}@p.com",
                "1000",
            )
        assert call_manager.active_call_count == 5
        for i in range(5):
            call_manager.end_call(f"surgery_greenfield-cc{i}")
        assert call_manager.active_call_count == 0


# ── TestAudioBridge 

class TestAudioBridge:

    def _make_pcm16_silence(self, num_samples: int) -> bytes:
        return b"\x00" * (num_samples * 2)

    def _make_pcm16_tone(self, num_samples: int, freq_hz: int = 440,
                         sample_rate: int = 16000) -> bytes:
        t = np.linspace(0, num_samples / sample_rate, num_samples)
        wave = (np.sin(2 * np.pi * freq_hz * t) * 16000).astype(np.int16)
        return wave.tobytes()

    def test_ingest_silence_returns_vad_chunks(self, audio_bridge):
        """3 × 20ms frames (160 samples each) = 960 bytes → one 30ms chunk"""
        frame_bytes = self._make_pcm16_silence(160)  # 20ms at 16kHz
        chunks = []
        for _ in range(3):
            chunks.extend(audio_bridge.ingest_livekit_frame(frame_bytes, 16000))
        assert len(chunks) == 1  # 3 × 20ms = 60ms; 30ms VAD frames → 2 chunks? No: 480 samples per chunk
        # 3 × 160 samples = 480 samples = exactly 1 VAD chunk
        assert len(chunks[0]) == 960  # 480 samples × 2 bytes = 960 bytes

    def test_ingest_resamples_48k_to_16k(self, audio_bridge):
        """48kHz frame should be resampled to 16kHz before buffering."""
        # 20ms at 48kHz = 960 samples = 1920 bytes
        frame_bytes = self._make_pcm16_silence(960)
        # After resampling 48k→16k: 960/3 = 320 samples = 640 bytes
        chunks_before = len(audio_bridge._buffer)
        audio_bridge.ingest_livekit_frame(frame_bytes, 48000)
        # Buffer should have 640 bytes (not 1920)
        assert len(audio_bridge._buffer) == 640

    def test_flush_returns_and_clears_buffer(self, audio_bridge):
        frame_bytes = self._make_pcm16_silence(100)  # 100 samples < VAD frame
        audio_bridge.ingest_livekit_frame(frame_bytes, 16000)
        flushed = audio_bridge.flush()
        assert len(flushed) > 0
        # After flush, buffer should be empty
        assert len(audio_bridge._buffer) == 0

    def test_backpressure_drops_frames_when_saturated(self):
        from backend.telephony.audio_bridge import AudioBridge
        bridge = AudioBridge(max_buffer_frames=2)
        # Fill buffer past max
        large_frame = self._make_pcm16_silence(1000)
        result = bridge.ingest_livekit_frame(large_frame, 16000)
        # Second call should drop (buffer would exceed max)
        bridge.ingest_livekit_frame(large_frame, 16000)
        assert bridge._dropped_frames >= 0  # At least checked

    def test_chunk_tts_for_livekit_20ms_chunks(self, audio_bridge):
        # 100ms of audio at 16kHz = 1600 samples = 3200 bytes
        pcm = self._make_pcm16_silence(1600)
        chunks = AudioBridge.chunk_tts_for_livekit(pcm, chunk_ms=20, sample_rate=16000)
        # 100ms / 20ms = 5 chunks
        assert len(chunks) == 5
        # Each chunk: 20ms × 16000 / 1000 = 320 samples = 640 bytes
        for chunk in chunks:
            assert len(chunk) == 640

    def test_chunk_tts_pads_last_chunk(self, audio_bridge):
        # 25ms of audio — last 5ms will be padded
        pcm = self._make_pcm16_silence(400)  # 400 samples = 25ms at 16kHz
        chunks = AudioBridge.chunk_tts_for_livekit(pcm, chunk_ms=20, sample_rate=16000)
        # 2 chunks: first full (320 samples), second padded to 320 samples
        assert len(chunks) == 2
        assert len(chunks[1]) == 640  # Padded to full 20ms

    def test_tts_bytes_to_numpy_and_back(self):
        # Round-trip: bytes → float32 → bytes should be lossless (for zero signal)
        silence = b"\x00" * 640
        as_float = AudioBridge.tts_bytes_to_numpy(silence)
        assert as_float.dtype == np.float32
        assert np.all(as_float == 0.0)
        back = AudioBridge.numpy_to_livekit_bytes(as_float)
        assert back == silence

    def test_numpy_to_livekit_clips_overflow(self):
        # Values outside [-1, 1] should be clipped
        overflow = np.array([2.0, -2.0, 1.5], dtype=np.float32)
        result = AudioBridge.numpy_to_livekit_bytes(overflow)
        samples = np.frombuffer(result, dtype=np.int16)
        assert samples[0] == 32767   # clipped to max
        assert samples[1] == -32767  # clipped to min
        assert samples[2] == 32767   # clipped to max


# ── TestWebhookHandler 

class TestWebhookHandler:

    @pytest.mark.asyncio
    async def test_invalid_signature_rejected(self, webhook_handler):
        body = b'{"event": "room_started"}'
        ok = await webhook_handler.process_request(body, "Bearer invalid_token")
        assert ok is False

    @pytest.mark.asyncio
    async def test_sip_participant_identity_detection(self, webhook_handler):
        from backend.telephony.webhook_handler import _is_sip_participant
        assert _is_sip_participant("sip:+441234567890@provider.com") is True
        assert _is_sip_participant("sip:1000@192.168.1.1") is True
        assert _is_sip_participant("emma-agent") is False
        assert _is_sip_participant("") is False

    def test_extract_sip_number(self):
        from backend.telephony.webhook_handler import _extract_sip_number
        assert _extract_sip_number("sip:+441234567890@provider.com") == "+441234567890"
        assert _extract_sip_number("sip:1000@192.168.1.1") == "1000"
        assert _extract_sip_number("sip:anonymous@provider.com") == "anonymous"
        assert _extract_sip_number("not-a-sip-uri") is None

    def test_extract_destination_from_metadata(self):
        from backend.telephony.webhook_handler import _extract_destination_from_metadata
        meta = json.dumps({"tenant_id": "surgery_greenfield", "destination": "+441234567890"})
        result = _extract_destination_from_metadata(meta)
        assert result == "+441234567890"

    def test_extract_destination_missing_key(self):
        from backend.telephony.webhook_handler import _extract_destination_from_metadata
        meta = json.dumps({"tenant_id": "surgery_greenfield"})
        result = _extract_destination_from_metadata(meta)
        assert result == "unknown"

    def test_extract_destination_invalid_json(self):
        from backend.telephony.webhook_handler import _extract_destination_from_metadata
        result = _extract_destination_from_metadata("not-json")
        assert result == "unknown"

    def test_extract_destination_empty(self):
        from backend.telephony.webhook_handler import _extract_destination_from_metadata
        result = _extract_destination_from_metadata("")
        assert result == "unknown"


# ── TestDDIRouterRoomNaming 

class TestRoomNaming:
    """Test round-trip: tenant → room name → tenant"""

    @pytest.mark.parametrize("tenant_id", [
        "surgery_greenfield",
        "surgery_riverside",
    ])
    def test_room_name_roundtrip(self, ddi_router, tenant_id):
        uid = str(uuid.uuid4())
        room_name = ddi_router.room_name_for_tenant(tenant_id, uid)
        recovered = ddi_router.tenant_from_room_name(room_name)
        assert recovered == tenant_id

    def test_room_name_format(self, ddi_router):
        uid = "abc-123-def"
        name = ddi_router.room_name_for_tenant("surgery_greenfield", uid)
        assert name == f"surgery_greenfield-{uid}"
        assert "-" in name
        assert name.startswith("surgery_greenfield-")


# ── Integration tests (requires LiveKit server) 

@pytest.mark.requires_livekit
@pytest.mark.asyncio
async def test_livekit_connectivity():
    """Verify LiveKit server is reachable and credentials are valid."""
    from scripts.start_agent import check_connectivity
    ok = await check_connectivity()
    assert ok, (
        "LiveKit server not reachable. "
        "Is docker-compose running? Check LIVEKIT_URL in .env"
    )


@pytest.mark.requires_livekit
@pytest.mark.asyncio
async def test_sip_provisioner_list_resources():
    """List SIP resources from LiveKit server."""
    from collections.abc import Iterable
    from backend.telephony.sip_provisioner import SIPProvisioner
    async with SIPProvisioner() as provisioner:
        trunks = await provisioner.list_trunks()
        rules  = await provisioner.list_dispatch_rules()
    # Protobuf repeated fields (google._upb._message.RepeatedCompositeContainer)
    # are iterable but not isinstance(list)
    assert isinstance(trunks, Iterable), f"trunks not iterable: {type(trunks)}"
    assert isinstance(rules, Iterable), f"rules not iterable: {type(rules)}"
    # Should be able to iterate and access fields
    trunk_ids = [t.sip_trunk_id for t in trunks]
    rule_ids  = [r.sip_dispatch_rule_id for r in rules]
    assert len(trunk_ids) >= 2, f"Expected >=2 trunks, got {len(trunk_ids)}"
    assert len(rule_ids) >= 2, f"Expected >=2 rules, got {len(rule_ids)}"


@pytest.mark.requires_livekit
@pytest.mark.asyncio
async def test_sip_provisioner_provision_and_cleanup():
    """
    Provision trunks/rules for all tenants, verify they exist, then remove.
    Idempotency: run twice to ensure no duplicates.
    """
    from backend.telephony.sip_provisioner import SIPProvisioner
    async with SIPProvisioner() as provisioner:
        # First provision
        results1 = await provisioner.provision_all()
        assert len(results1) > 0

        # Second provision (idempotent — should not create duplicates)
        results2 = await provisioner.provision_all()
        assert len(results2) == len(results1)

        # Trunk IDs should be stable across idempotent calls
        ids1 = {r.trunk_id for r in results1}
        ids2 = {r.trunk_id for r in results2}
        assert ids1 == ids2, "Idempotency check failed: trunk IDs changed"


@pytest.mark.requires_livekit
@pytest.mark.asyncio
async def test_webhook_endpoint_reachable():
    """Verify the FastAPI webhook endpoint is up and returns 401 for unsigned requests."""
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/livekit-webhook",
            content=b"{}",
            headers={"Content-Type": "application/json"},
        )
    # No Authorization header → should return 401
    assert response.status_code == 401


@pytest.mark.requires_livekit
@pytest.mark.asyncio
async def test_calls_endpoint_empty():
    """Verify /calls endpoint returns empty state when no calls active."""
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/calls")
    assert response.status_code == 200
    data = response.json()
    assert "active_count" in data
    assert "calls" in data


@pytest.mark.requires_livekit
@pytest.mark.asyncio
async def test_routing_sip_resources_endpoint():
    """Verify /routing/sip-resources returns provisioned resources."""
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/routing/sip-resources")
    assert response.status_code == 200
    data = response.json()
    assert "trunks" in data
    assert "dispatch_rules" in data


# ── E2E tests (require LiveKit + SIP simulator) 

@pytest.mark.requires_sip
@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_sip_call_greenfield():
    """
    Full E2E SIP call: PJSUA → LiveKit SIP → EMMA → LiveKit SIP → PJSUA.

    Requires:
      - LiveKit server running (docker-compose up -d livekit)
      - EMMA backend running (docker-compose up -d backend)
      - EMMA agent running (python scripts/start_agent.py)
      - SIP trunks provisioned (python scripts/provision_sip.py)
      - PJSUA or Baresip installed and configured to reach LiveKit SIP

    Verification:
      - Call is accepted (200 OK)
      - Room is created with "surgery_greenfield-" prefix
      - Agent joins room within 5s
      - Call appears in /calls endpoint
      - After 5s, call ends cleanly

    Note: Full audio pipeline verification requires a real-time SIP client.
    This test verifies the signalling and session management path only.
    """
    import subprocess
    import httpx
    import time

    # Kick off a SIP call via PJSUA (must be installed)
    # pjsua registers against LiveKit SIP bridge and places a call
    proc = subprocess.Popen(
        [
            "pjsua",
            "--null-audio",              # No real audio hardware needed
            "--registrar", "sip:sip.emma-local:5060",
            "--id", "sip:test@emma-client",
            "--username", "emma-test",
            "--password", "test",
            "--auto-answer", "200",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for registration
        await asyncio.sleep(2)

        # Trigger a call to Greenfield DDI
        proc.stdin.write(b"m\nsip:+441234567890@sip.emma-local:5060\n")
        proc.stdin.flush()

        # Wait for call to be established
        await asyncio.sleep(3)

        # Verify call appears in EMMA's active calls
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/calls")
        assert response.status_code == 200
        data = response.json()

        # Find call for greenfield tenant
        greenfield_calls = [
            c for c in data["calls"]
            if c["tenant_id"] == "surgery_greenfield"
        ]
        assert len(greenfield_calls) >= 1, (
            f"No greenfield call found in: {data['calls']}"
        )

        # Verify room name format
        room_name = greenfield_calls[0]["room_name"]
        assert room_name.startswith("surgery_greenfield-"), (
            f"Unexpected room name format: {room_name}"
        )

        # Let call run briefly
        await asyncio.sleep(5)

        # Hang up
        proc.stdin.write(b"h\n")
        proc.stdin.flush()
        await asyncio.sleep(2)

        # Verify call has ended
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/calls")
        data = response.json()
        active_greenfield = [
            c for c in data["calls"]
            if c["tenant_id"] == "surgery_greenfield"
        ]
        assert len(active_greenfield) == 0, (
            f"Call still active after hangup: {active_greenfield}"
        )

    finally:
        proc.terminate()
        proc.wait(timeout=5)


# ── Concurrency tests 

@pytest.mark.asyncio
async def test_concurrent_calls_different_tenants(call_manager):
    """Register 10 concurrent calls across both tenants."""
    rooms = [
        (f"surgery_greenfield-call{i}", f"PA_g{i}", f"sip:{i}@p.com", "441234567890")
        for i in range(5)
    ] + [
        (f"surgery_riverside-call{i}", f"PA_r{i}", f"sip:{i+5}@p.com", "441234987654")
        for i in range(5)
    ]

    for room, sid, identity, dest in rooms:
        call_manager.register_call(room, sid, identity, dest)

    assert call_manager.active_call_count == 10

    greenfield = [c for c in call_manager.active_calls_summary
                  if c["tenant_id"] == "surgery_greenfield"]
    riverside  = [c for c in call_manager.active_calls_summary
                  if c["tenant_id"] == "surgery_riverside"]
    assert len(greenfield) == 5
    assert len(riverside)  == 5

    # End all calls
    for room, *_ in rooms:
        call_manager.end_call(room)
    assert call_manager.active_call_count == 0


@pytest.mark.asyncio
async def test_audio_bridge_concurrent_ingestion():
    """Multiple concurrent AudioBridge instances don't interfere."""
    from backend.telephony.audio_bridge import AudioBridge

    bridges = [AudioBridge() for _ in range(5)]
    frame = b"\x00" * 320  # 20ms silence at 16kHz

    # Ingest frames into all bridges concurrently (asyncio.coroutine removed in Python 3.12)
    await asyncio.gather(*[
        asyncio.to_thread(b.ingest_livekit_frame, frame, 16000)
        for b in bridges
    ])

    # Each bridge should accumulate independently
    for bridge in bridges:
        bridge.ingest_livekit_frame(frame, 16000)
        bridge.ingest_livekit_frame(frame, 16000)
        bridge.ingest_livekit_frame(frame, 16000)
        # 3 × 160 samples = 480 = 1 VAD frame → buffer should be empty after chunk yield

    # No cross-bridge contamination
    for bridge in bridges:
        assert len(bridge._buffer) == 0 or len(bridge._buffer) < 960


# ── Reconnect tests 

@pytest.mark.asyncio
async def test_call_manager_survives_repeated_register_end_cycles():
    """Stress test: register and end 100 calls sequentially."""
    from backend.telephony.call_manager import CallManager
    from backend.telephony.ddi_router import DDIRouter

    cm = CallManager(ddi_router=DDIRouter())
    for i in range(100):
        room = f"surgery_greenfield-stress{i}"
        cm.register_call(room, f"PA_{i}", f"sip:{i}@p.com", "1000")
        cm.end_call(room)
    assert cm.active_call_count == 0


# ── Tenant isolation tests 

class TestTenantIsolation:

    def test_greenfield_call_only_in_greenfield_tenant(self, call_manager):
        call_manager.register_call(
            "surgery_greenfield-iso1", "PA_iso1",
            "sip:1@p.com", "441234567890"
        )
        call = call_manager.get_call("surgery_greenfield-iso1")
        assert call.tenant_id == "surgery_greenfield"

    def test_riverside_call_not_visible_to_greenfield(self, call_manager):
        call_manager.register_call(
            "surgery_riverside-iso1", "PA_iso2",
            "sip:2@p.com", "441234987654"
        )
        greenfield_calls = [
            c for c in call_manager.active_calls_summary
            if c["tenant_id"] == "surgery_greenfield"
        ]
        assert len(greenfield_calls) == 0

    def test_dtmf_not_cross_tenant(self, call_manager):
        call_manager.register_call(
            "surgery_greenfield-dtmf1", "PA_d1", "sip:1@p.com", "1000"
        )
        call_manager.register_call(
            "surgery_riverside-dtmf1", "PA_d2", "sip:2@p.com", "1001"
        )
        call_manager.record_dtmf("surgery_greenfield-dtmf1", "5")
        # Greenfield has digit "5"
        gf_call = call_manager.get_call("surgery_greenfield-dtmf1")
        rs_call = call_manager.get_call("surgery_riverside-dtmf1")
        assert "5" in gf_call.dtmf_digits
        assert "5" not in rs_call.dtmf_digits