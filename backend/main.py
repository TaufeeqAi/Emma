import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import (
    AUDIO_SAMPLE_RATE,
    CORS_ORIGINS,
    TENANTS,
    load_tenant_config,
)
from backend.voice.session_manager import SessionManager
from backend.voice.stt import STTHandler
from backend.voice.tts import TTSHandler
from backend.voice.websocket_handler import VoiceSessionHandler

logger = logging.getLogger(__name__)


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

    logger.info(
        "EMMA voice server ready | tenants=%s | Kokoro loaded",
        TENANTS,
    )

    yield  # Server runs here

    logger.info("EMMA voice server shutting down...")
    # Graceful shutdown: wait for active sessions to complete (Phase 5)
    active = app.state.session_manager.active_session_count
    if active > 0:
        logger.warning(
            "Shutting down with %d active sessions — "
            "callers may be disconnected.",
            active,
        )


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
            "Phase 3: Real-Time Voice Pipeline."
        ),
        version="0.3.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS 
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
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
            "version": "0.3.0",
            "tenants": TENANTS,
            "active_sessions": app.state.session_manager.active_session_count,
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