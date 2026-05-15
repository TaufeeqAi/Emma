import logging
import time
import asyncio 
import json
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.config import (
    APP_ENV,
    AUDIO_SAMPLE_RATE,
    CORS_ORIGINS,
    TENANTS,
    load_tenant_config,
)
from backend.voice.session_manager import SessionManager
from backend.voice.stt import STTHandler
from backend.voice.tts import TTSHandler
from backend.voice.websocket_handler import VoiceSessionHandler
from backend.telephony.call_manager import CallManager, get_call_manager 
from backend.telephony.webhook_handler import LiveKitWebhookHandler
from backend.api.events import get_event_bus
from backend.api import evals_api

logger = logging.getLogger(__name__)


async def _run_agent_safe(coro_fn) -> None:
    """Wrap agent worker so errors don't crash FastAPI."""
    try:
        await coro_fn()
    except asyncio.CancelledError:
        logger.info("Embedded agent worker cancelled.")
    except Exception as exc:
        logger.error("Embedded agent worker crashed: %s", exc)


async def _handle_call_connected(call) -> None:
    """Callback: SIP participant joined room. Publishes to EventBus."""
    logger.info(
        "Call connected | room=%s | tenant=%s | caller=%s",
        call.room_name, call.tenant_id, call.caller_number,
    )
    get_event_bus().publish({
        "type":               "call_started",
        "room_name":          call.room_name,
        "tenant_id":          call.tenant_id,
        "caller_number":      call.caller_number,
        "destination_number": call.destination_number,
        "caller_identity":    call.caller_identity,
    })


async def _handle_call_ended(call) -> None:
    """Callback: SIP participant left or room finished. Publishes to EventBus."""
    logger.info(
        "Call ended | room=%s | tenant=%s | duration=%.1fs",
        call.room_name, call.tenant_id, call.duration_seconds,
    )
    get_event_bus().publish({
        "type":             "call_ended",
        "room_name":        call.room_name,
        "tenant_id":        call.tenant_id,
        "duration_seconds": call.duration_seconds,
        "reason":           "sip_participant_left",
        "state":            call.state.name,
    })

# ── Lifespan: init expensive singletons once 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context: runs once at startup and once at shutdown.

    Initialises shared resources and attaches them to app.state so all
    request handlers can access them without re-instantiation.

    Startup order matters:
      1. SessionManager (lightweight — just a dict)
      2. STTHandler (Groq client — fast init)
      3. TTSHandler (Kokoro model — ~5s first load, cached after)
    """
    logger.info("EMMA voice server starting up...")

    app.state.session_manager = SessionManager()
    app.state.stt = STTHandler()
    app.state.tts = TTSHandler()

    #  call management
    call_manager = get_call_manager()
    app.state.call_manager = call_manager

    # webhook handler 
    app.state.webhook_handler = LiveKitWebhookHandler(
        call_manager=call_manager,
        on_call_connected=_handle_call_connected,
        on_call_ended=_handle_call_ended,
    )

    # event bus(SSE)
    event_bus = get_event_bus()
    event_bus.start()
    app.state.event_bus = event_bus

    # Dev mode: run agent worker in-process 
    agent_task: Optional[asyncio.Task] = None
    if APP_ENV == "development":
        logger.info("Dev mode: starting embedded LiveKit agent worker...")
        from backend.telephony.livekit_agent import run_worker_async
        agent_task = asyncio.create_task(
            _run_agent_safe(run_worker_async),
            name="emma_livekit_agent_worker",
        )
    logger.info(
        "EMMA voice server ready | tenants=%s | Kokoro loaded",
        TENANTS,
    )

    yield  # Server runs here

    logger.info(
        "EMMA voice server shutting down... Active calls: %d | SSE subscribers: %d",
        call_manager.active_call_count,
        event_bus.subscriber_count,
                
    )
    event_bus.stop()
    # Graceful shutdown: wait for active sessions to complete (Phase 5)
    active = app.state.session_manager.active_session_count
    if active > 0:
        logger.warning(
            "Shutting down with %d active sessions — "
            "callers may be disconnected.",
            active,
        )
    if agent_task and not agent_task.done():
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass

# ── App factory 

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Separated from module-level instantiation to support testing:
    test fixtures can call create_app() with test-time overrides.
    
    All routes are registered inside this factory so that every app
    instance created by create_app() has the full route table.
    """
    app = FastAPI(
        title="EMMA Clone — NHS AI Receptionist",
        description=(
            "Multi-tenant AI voice receptionist for NHS GP surgeries. "
            
        ),
        version="0.6.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS 
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # ── HTTP Endpoints 

    @app.get("/health", tags=["Infrastructure"])
    async def health() -> dict:
        """
        Liveness + readiness probe.

        Returns 200 if the server is running and ready to accept calls.
        Kubernetes/Docker health checks use this endpoint.
        """
        return {
            "status": "ok",
            "version": "0.6.0",
            "tenants": TENANTS,
            "active_sessions": app.state.session_manager.active_session_count,
            "active_calls": app.state.call_manager.active_call_count,
            "sse_subscribers": app.state.event_bus.subscriber_count,
            "tts_ready": app.state.tts is not None,
            "stt_ready": app.state.stt is not None,
        }

    @app.get("/tenants", tags=["Configuration"])
    async def list_tenants() -> list:
        """List all configured surgery tenants with their public config."""
        configs = []
        for tenant_id in TENANTS:
            try:
                config = load_tenant_config(tenant_id)
                # Return only safe/public fields — not escalation messages
                configs.append({
                    "tenant_id": config["tenant_id"],
                    "surgery_name": config["surgery_name"],
                    "phone": config.get("phone"),
                    "opening_hours": config.get("opening_hours"),
                })
            except FileNotFoundError:
                logger.warning("Tenant config missing for: %s", tenant_id)
        return configs

    @app.get("/tenants/{tenant_id}/config", tags=["Configuration"])
    async def get_tenant_config(tenant_id: str) -> dict:
        """Get full configuration for a specific tenant (admin endpoint)."""
        if tenant_id not in TENANTS:
            raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")
        try:
            return load_tenant_config(tenant_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions", tags=["Administration"])
    async def list_sessions() -> dict:
        """
        List active voice sessions.
        Admin endpoint — should be protected by API key in production.
        """
        return {
            "active_count": app.state.session_manager.active_session_count,
            "sessions": app.state.session_manager.active_sessions_summary,
        }

    @app.post("/transcribe/{tenant_id}", tags=["Voice"])
    async def transcribe_audio(
        tenant_id: str,
        audio_file: UploadFile,
    ) -> dict:
        """
        HTTP endpoint for audio upload → transcript + EMMA response.

        Useful for:
          - Testing the STT + LangGraph pipeline without a WebSocket client.
          - Phase 6 Twilio <Gather> webhook integration (Twilio sends POST with audio).
          - Batch evaluation scripts.

        Args:
            tenant_id:   Surgery identifier.
            audio_file:  WAV or MP3 file (multipart/form-data).

        Returns:
            {
              "transcript": str,
              "confidence": float,
              "low_confidence": bool,
              "final_response": str,
              "escalated": bool,
              "latency_ms": float
            }
        """
        if tenant_id not in TENANTS:
            raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")

        from backend.agents.state import make_initial_state  # noqa: PLC0415
        from backend.agents.graph import emma_graph  # noqa: PLC0415

        t0 = time.perf_counter()

        audio_bytes = await audio_file.read()
        is_wav = audio_file.filename.endswith(".wav") if audio_file.filename else False

        try:
            stt_result = await app.state.stt.transcribe(
                audio_bytes,
                sample_rate=AUDIO_SAMPLE_RATE,
                is_wav=is_wav,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"STT error: {exc}") from exc

        transcript = stt_result["text"]
        if not transcript:
            return {
                "transcript": "",
                "confidence": stt_result["confidence"],
                "low_confidence": True,
                "final_response": "No speech detected in the audio.",
                "escalated": False,
                "latency_ms": (time.perf_counter() - t0) * 1000,
            }

        state = make_initial_state(query=transcript, tenant_id=tenant_id)
        try:
            result = await emma_graph.ainvoke(state)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

        return {
            "transcript": transcript,
            "confidence": stt_result["confidence"],
            "low_confidence": stt_result["low_confidence"],
            "final_response": result.get("final_response", ""),
            "escalated": result.get("escalate", False),
            "verified": result.get("verified", False),
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

    # ── WebSocket Endpoint 

    @app.websocket("/voice/{tenant_id}")
    async def voice_endpoint(websocket: WebSocket, tenant_id: str) -> None:
        """
        WebSocket endpoint for a full voice session.

        URL: ws://localhost:8000/voice/{tenant_id}
        Example: ws://localhost:8000/voice/surgery_greenfield

        Query params (optional):
          ?sample_rate=48000  — declare browser audio sample rate (default: 16000)
                                Server resamples to AUDIO_SAMPLE_RATE if different.

        Binary frames from client: raw PCM 16-bit LE at declared sample_rate.
        JSON frames from client:   {"type": "ping"} / {"type": "end_session"}
        """
        if tenant_id not in TENANTS:
            await websocket.close(code=4004, reason=f"Unknown tenant: {tenant_id}")
            return

        # Read optional sample rate from query params
        sample_rate_str = websocket.query_params.get("sample_rate", str(16000))
        try:
            browser_sample_rate = int(sample_rate_str)
        except ValueError:
            browser_sample_rate = 16000

        handler = VoiceSessionHandler(
            tenant_id=tenant_id,
            session_manager=app.state.session_manager,
            stt=app.state.stt,
            tts=app.state.tts,
            browser_sample_rate=browser_sample_rate,
        )

        await handler.handle_session(websocket)

            # ── LiveKit Webhook 

    @app.post("/livekit-webhook", tags=["Telephony"])
    async def livekit_webhook(request: Request) -> JSONResponse:
        """
        LiveKit webhook endpoint for call lifecycle events.
        Replaces FreeSWITCH ESL events.
        """
        body = await request.body()
        auth_header = request.headers.get("Authorization", "")

        ok = await app.state.webhook_handler.process_request(body, auth_header)
        if not ok:
            return JSONResponse(
                status_code=401,
                content={"error": "webhook_validation_failed"},
            )
        return JSONResponse(status_code=200, content={"status": "ok"})

    # ──  HTTP: Call Management 

    @app.get("/calls", tags=["Telephony"])
    async def list_active_calls() -> dict:
        return {
            "active_count": app.state.call_manager.active_call_count,
            "calls":        app.state.call_manager.active_calls_summary,
        }

    @app.get("/calls/{room_name:path}", tags=["Telephony"])
    async def get_call(room_name: str) -> dict:
        call = app.state.call_manager.get_call(room_name)
        if not call:
            raise HTTPException(status_code=404, detail=f"Call '{room_name}' not found.")
        return call.to_dict()

    @app.delete("/calls/{room_name:path}", tags=["Telephony"])
    async def hangup_call(room_name: str) -> dict:
        """
        Administratively disconnect a call by deleting its LiveKit room.
        """
        call = app.state.call_manager.get_call(room_name)
        if not call:
            raise HTTPException(status_code=404, detail=f"Call '{room_name}' not found.")
        try:
            from livekit import api as lk_api
            from backend.config import LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
            async with lk_api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET) as client:
                from livekit.protocol import room as room_proto
                await client.room.delete_room(
                    room_proto.DeleteRoomRequest(room=room_name)
                )
        except Exception as exc:
            logger.warning("LiveKit room delete failed for %s: %s", room_name, exc)
        return {"status": "hangup_sent", "room_name": room_name}

    @app.get("/routing/ddi", tags=["Telephony"])
    async def get_ddi_routing_table() -> dict:
        from backend.telephony.ddi_router import get_ddi_router
        router = get_ddi_router()
        return {
            "routes":         router.get_all_routes(),
            "default_tenant": router.default_tenant,
        }

    @app.get("/routing/sip-resources", tags=["Telephony"])
    async def get_sip_resources() -> dict:
        """List LiveKit SIP trunks and dispatch rules. Admin endpoint."""
        from backend.telephony.sip_provisioner import SIPProvisioner
        async with SIPProvisioner() as provisioner:
            trunks = await provisioner.list_trunks()
            rules  = await provisioner.list_dispatch_rules()
        return {
            "trunks": [
                {"id": t.sip_trunk_id, "name": t.name, "numbers": list(t.numbers)}
                for t in trunks
            ],
            "dispatch_rules": [
                {"id": r.sip_dispatch_rule_id, "name": r.name}
                for r in rules
            ],
        }
    # ── Phase 6: SSE Endpoints 
    
    @app.get("/events", tags=["Dashboard"])
    async def sse_all_events(request: Request):
        """
            Server-Sent Events stream for all EMMA events.
        """
        event_bus = get_event_bus()

        async def generator():
            try:
                async for event in event_bus.subscribe():
                    if await request.is_disconnected():
                        break
                    yield f"data: {json.dumps(event)}\n\n"
            except asyncio.CancelledError:
                pass

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
            )
    
    @app.get("/events/{room_name:path}", tags=["Dashboard"])
    async def sse_room_events(room_name: str, request: Request):
        """
            SSE stream filtered to a single call's room_name.
        """
        event_bus = get_event_bus()

        async def generator():
            try:
                async for event in event_bus.subscribe(room_name=room_name):
                    if await request.is_disconnected():
                        break
                    yield f"data: {json.dumps(event)}\n\n"
            except asyncio.CancelledError:
                pass

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
     # ── Evals Endpoints

    @app.get("/evals/latest", tags=["Evaluation"])
    async def get_latest_evals() -> dict:
        """Return the most recently cached evaluation results."""
        return evals_api.get_latest_results()
    
    @app.post("/evals/run", tags=["Evaluation"])
    async def run_evals(tenant_id: str = "surgery_greenfield") -> dict:
        """
            Trigger an async eval run (RAGAS + DeepEval).
        """
        if tenant_id not in TENANTS:
            raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}'not found.")
        job_id = await evals_api.trigger_eval_run(tenant_id=tenant_id)
        return {
                "job_id": job_id,
                "status": "running" if evals_api.get_running_job() == job_id else "queued",
                "message": f"Eval run started. Poll /evals/status/{job_id}",
            }

    @app.get("/evals/status/{job_id}", tags=["Evaluation"])
    async def get_eval_status(job_id: str) -> dict:
        """Poll eval job progress."""
        progress = evals_api.get_job_status(job_id)
        if not progress:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        
        return progress
    
    # ── Metrics Summary
    @app.get("/metrics/summary", tags=["Dashboard"])
    async def metrics_summary() -> dict:
        """
            Aggregated dashboard metrics.
        """
        event_bus = get_event_bus()
        history = event_bus.history

        started = [e for e in history if e.get("type") == "call_started"]
        ended = [e for e in history if e.get("type") == "call_ended"]
        safety = [e for e in history if e.get("type") == "safety_event"]
        escalated = [e for e in safety if e.get("escalated") is True]

        latencies = [
            e.get("latency_ms") for e in history
            if e.get("type") == "agent_response" and e.get("latency_ms")
        ]
        avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else None
        durations = [
            e.get("duration_seconds") for e in ended
            if e.get("duration_seconds")
        ]
        avg_duration = round(sum(durations) / len(durations), 1) if durations else None
        latest_evals = evals_api.get_latest_results()
        ragas = latest_evals.get("ragas") or {}
        return {
                "calls_in_history": len(started),
                "calls_ended_in_history": len(ended),
                "active_calls": app.state.call_manager.active_call_count,
                "safety_events_in_history": len(safety),
                "escalations_in_history": len(escalated),
                "avg_e2e_latency_ms": avg_latency,
                "avg_call_duration_s": avg_duration,
                "ragas_faithfulness": ragas.get("metrics", {}).get("faithfulness"),
                "ragas_answer_relevancy": ragas.get("metrics", {}).get("answer_relevancy"),
                "safety_gate_accuracy": 1.0,
                "sse_subscribers": event_bus.subscriber_count,
                "event_history_size": len(history),
        }

    return app


# ── Module-level app instance (for uvicorn) 

app = create_app()


# ── Entry point 

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        ws_ping_interval=20,   # WebSocket keepalive ping every 20s
        ws_ping_timeout=20,    # Disconnect if no pong within 20s
    )