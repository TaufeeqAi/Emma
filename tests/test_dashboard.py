"""
Test categories:
  Unit tests (no server required):
    TestEventBus              — publish, subscribe, history, heartbeat
    TestEvalsAPI              — result caching, mock fallbacks

  Integration tests (require running backend):
    @pytest.mark.requires_backend
    test_sse_events_endpoint  — SSE stream reachable + returns text/event-stream
    test_evals_latest         — /evals/latest returns valid structure
    test_evals_run            — POST /evals/run returns job_id
    test_metrics_summary      — /metrics/summary returns valid structure

Running:
    # Unit tests only
    pytest tests/test_dashboard.py -v -m "not requires_backend"

    # With backend running
    pytest tests/test_dashboard.py -v -m "requires_backend"
"""

import asyncio
import json
import time
import pytest
import pytest_asyncio


# ── TestEventBus 

class TestEventBus:

    @pytest.fixture
    def fresh_bus(self):
        from backend.api.events import EventBus
        return EventBus()

    def test_publish_stores_in_history(self, fresh_bus):
        fresh_bus.publish({"type": "call_started", "room_name": "test-room"})
        assert len(fresh_bus.history) == 1
        assert fresh_bus.history[0]["type"] == "call_started"

    def test_publish_adds_timestamp(self, fresh_bus):
        fresh_bus.publish({"type": "heartbeat"})
        assert "timestamp" in fresh_bus.history[0]

    def test_publish_preserves_existing_timestamp(self, fresh_bus):
        ts = "2025-01-01T00:00:00+00:00"
        fresh_bus.publish({"type": "heartbeat", "timestamp": ts})
        assert fresh_bus.history[0]["timestamp"] == ts

    def test_history_ring_buffer(self, fresh_bus):
        from backend.api.events import MAX_HISTORY
        for i in range(MAX_HISTORY + 10):
            fresh_bus.publish({"type": "heartbeat", "seq": i})
        assert len(fresh_bus.history) == MAX_HISTORY
        # Should keep the most recent events
        assert fresh_bus.history[-1]["seq"] == MAX_HISTORY + 9

    def test_clear_history(self, fresh_bus):
        fresh_bus.publish({"type": "heartbeat"})
        fresh_bus.clear_history()
        assert len(fresh_bus.history) == 0

    @pytest.mark.asyncio
    async def test_subscribe_receives_history(self, fresh_bus):
        fresh_bus.publish({"type": "call_started", "room_name": "r1"})
        fresh_bus.publish({"type": "call_started", "room_name": "r2"})

        received = []
        async def collect():
            async for event in fresh_bus.subscribe():
                received.append(event)
                if len(received) >= 2:
                    break

        await asyncio.wait_for(collect(), timeout=2.0)
        assert len(received) >= 2

    @pytest.mark.asyncio
    async def test_subscribe_receives_live_events(self, fresh_bus):
        received = []

        async def collect():
            async for event in fresh_bus.subscribe():
                if event["type"] != "heartbeat":
                    received.append(event)
                if len(received) >= 1:
                    break

        # Start subscriber before publishing
        task = asyncio.create_task(
            asyncio.wait_for(collect(), timeout=2.0)
        )
        await asyncio.sleep(0.05)
        fresh_bus.publish({"type": "call_started", "room_name": "new-room"})

        await task
        assert any(e.get("room_name") == "new-room" for e in received)

    @pytest.mark.asyncio
    async def test_subscribe_room_filter(self, fresh_bus):
        received = []

        async def collect():
            async for event in fresh_bus.subscribe(room_name="target-room"):
                received.append(event)
                if len(received) >= 1:
                    break

        task = asyncio.create_task(
            asyncio.wait_for(collect(), timeout=2.0)
        )
        await asyncio.sleep(0.05)
        fresh_bus.publish({"type": "call_started", "room_name": "other-room"})
        fresh_bus.publish({"type": "call_started", "room_name": "target-room"})

        await task
        assert all(
            e.get("room_name") == "target-room" or e.get("room_name") is None
            for e in received
        )

    def test_subscriber_count_increases(self, fresh_bus):
        assert fresh_bus.subscriber_count == 0
        q = asyncio.Queue()
        fresh_bus._subscribers.append(q)
        assert fresh_bus.subscriber_count == 1
        fresh_bus._subscribers.remove(q)
        assert fresh_bus.subscriber_count == 0

    def test_full_queue_subscriber_dropped(self, fresh_bus):
        """A slow subscriber with a full queue is removed gracefully."""
        from asyncio import Queue
        full_q = Queue(maxsize=1)
        full_q.put_nowait({"type": "filler"})  # Fill the queue
        fresh_bus._subscribers.append(full_q)

        # Publish — should not raise, should drop the saturated subscriber
        fresh_bus.publish({"type": "heartbeat"})
        assert full_q not in fresh_bus._subscribers


# ── TestEvalsAPI 

class TestEvalsAPI:

    def test_get_latest_returns_no_results_when_empty(self):
        from backend.api import evals_api
        evals_api._latest_results = None  # reset
        result = evals_api.get_latest_results()
        assert result["status"] == "no_results"
        assert result["ragas"] is None

    def test_get_latest_returns_cached(self):
        from backend.api import evals_api
        evals_api._latest_results = {
            "status": "ok",
            "ragas": {"metrics": {"faithfulness": 0.95}},
            "deepeval": {"pass_rate": 1.0},
        }
        result = evals_api.get_latest_results()
        assert result["status"] == "ok"
        assert result["ragas"]["metrics"]["faithfulness"] == 0.95
        evals_api._latest_results = None  # cleanup

    @pytest.mark.asyncio
    async def test_trigger_eval_run_returns_job_id(self):
        from backend.api import evals_api
        evals_api._running_job = None  # ensure clean state
        job_id = await evals_api.trigger_eval_run(tenant_id="surgery_greenfield")
        assert isinstance(job_id, str)
        assert len(job_id) > 0

        # Should not create a second job while one is running
        job_id2 = await evals_api.trigger_eval_run(tenant_id="surgery_greenfield")
        assert job_id2 == job_id

        # Cleanup: wait for job to finish so it doesn't bleed into other tests
        await asyncio.sleep(0.1)
        evals_api._running_job = None

    @pytest.mark.asyncio
    async def test_mock_ragas_results_structure(self):
        from backend.api.evals_api import _mock_ragas_results
        result = _mock_ragas_results("surgery_greenfield")
        assert "metrics" in result
        for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            assert key in result["metrics"]
            assert 0.0 <= result["metrics"][key] <= 1.0

    @pytest.mark.asyncio
    async def test_mock_deepeval_results_structure(self):
        from backend.api.evals_api import _mock_deepeval_results
        result = _mock_deepeval_results()
        assert result["pass_rate"] == 1.0
        assert result["failed"] == 0
        assert isinstance(result["test_cases"], list)
        for tc in result["test_cases"]:
            assert "name" in tc
            assert "passed" in tc


# ── Integration tests (require running backend) 

@pytest.mark.requires_backend
@pytest.mark.asyncio
async def test_sse_events_endpoint_reachable():
    """SSE endpoint returns text/event-stream content type."""
    import httpx
    async with httpx.AsyncClient() as client:
        # Use a short timeout — we just need the headers, not the full stream
        try:
            async with client.stream(
                "GET",
                "http://localhost:8000/events",
                timeout=3.0,
            ) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers.get("content-type", "")
        except httpx.ReadTimeout:
            pass  # Timeout is fine — the stream is infinite


@pytest.mark.requires_backend
@pytest.mark.asyncio
async def test_evals_latest_structure():
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/evals/latest")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    # Either no_results or ok
    assert data["status"] in ("no_results", "ok")


@pytest.mark.requires_backend
@pytest.mark.asyncio
async def test_evals_run_returns_job_id():
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/evals/run?tenant_id=surgery_greenfield"
        )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "status" in data


@pytest.mark.requires_backend
@pytest.mark.asyncio
async def test_metrics_summary_structure():
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/metrics/summary")
    assert response.status_code == 200
    data = response.json()
    required_keys = [
        "active_calls", "calls_in_history", "safety_gate_accuracy",
        "sse_subscribers", "event_history_size",
    ]
    for key in required_keys:
        assert key in data, f"Missing key: {key}"


@pytest.mark.requires_backend
@pytest.mark.asyncio
async def test_health_includes_sse_subscribers():
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/health")
    assert response.status_code == 200
    data = response.json()
    assert "sse_subscribers" in data
    assert data["version"] == "0.6.0"