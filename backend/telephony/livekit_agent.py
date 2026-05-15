import asyncio
import json
import logging
import signal
from typing import Optional

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
    metrics,
)

from backend.config import (
    LIVEKIT_URL,
    LIVEKIT_API_KEY,
    LIVEKIT_API_SECRET,
    EMMA_AGENT_NAME,
    EMMA_MAX_CONCURRENT_CALLS,
    TENANTS,
)
from backend.telephony.call_manager import get_call_manager
from backend.telephony.ddi_router import get_ddi_router
from backend.telephony.livekit_adapter import LiveKitCallAdapter
from backend.voice.session_manager import SessionManager
from backend.voice.stt import STTHandler
from backend.voice.tts import TTSHandler

logger = logging.getLogger(__name__)

# Shared singletons — initialised once per worker process
_session_manager: Optional[SessionManager] = None
_stt_handler: Optional[STTHandler] = None
_tts_handler: Optional[TTSHandler] = None


def _get_singletons() -> tuple[SessionManager, STTHandler, TTSHandler]:
    """Lazy-initialise shared Phase 3 handlers (once per worker process)."""
    global _session_manager, _stt_handler, _tts_handler
    if _session_manager is None:
        logger.info("Initialising EMMA voice pipeline singletons (agent worker)...")
        _session_manager = SessionManager()
        _stt_handler = STTHandler()
        _tts_handler = TTSHandler()
        logger.info("EMMA pipeline singletons ready.")
    return _session_manager, _stt_handler, _tts_handler


async def entrypoint(ctx: JobContext) -> None:
    """
    LiveKit agent job entrypoint — called once per incoming SIP call.

    Execution context:
      - Called in a fresh asyncio Task by the LiveKit worker runtime.
      - ctx.room is a Room pre-configured for this job.
      - ctx.job contains job metadata (room name, dispatch info).
      - ctx.connect() must be called before accessing room participants.

    Design notes:
      1. We call ctx.connect() with AutoSubscribe.AUDIO_ONLY — we don't
         need video from SIP calls, and it saves unnecessary track subscriptions.
      2. We wait for the SIP participant with wait_for_participant(). This
         handles both cases: participant already present and future arrival.
         Timeout of 30s: if no SIP participant arrives, we abort cleanly.
      3. Tenant resolution: extracted from room name prefix by DDIRouter.
         This is robust: even if metadata is missing, room name prefix is
         always set by the dispatch rule.
      4. CallManager state: updated by the webhook handler (participant_joined).
         The adapter also updates state (STREAMING) when audio track is active.

    Args:
        ctx: LiveKit JobContext for this room/call.
    """
    room_name = ctx.job.room.name
    logger.info("Agent job started | room=%s", room_name)

    # ── Resolve tenant 
    router = get_ddi_router()
    tenant_id = router.tenant_from_room_name(room_name)

    if not tenant_id:
        # Try to extract from room metadata (fallback)
        try:
            meta = json.loads(ctx.job.room.metadata or "{}")
            tenant_id = meta.get("tenant_id")
        except (json.JSONDecodeError, AttributeError):
            pass

    if not tenant_id or tenant_id not in TENANTS:
        logger.error(
            "Cannot resolve tenant from room '%s' — aborting job.", room_name
        )
        await ctx.disconnect()
        return

    logger.info("Tenant resolved | room=%s | tenant=%s", room_name, tenant_id)

    # ── Connect to room 
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # ── Wait for SIP participant 
    try:
        sip_participant = await asyncio.wait_for(
            ctx.wait_for_participant(),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.error(
            "No SIP participant arrived within 30s | room=%s — aborting", room_name
        )
        await ctx.disconnect()
        return

    logger.info(
        "SIP participant arrived | room=%s | identity=%s | sid=%s",
        room_name, sip_participant.identity, sip_participant.sid,
    )

    # ── Publish agent identity 
    await ctx.room.local_participant.update_metadata(
        json.dumps({
            "agent": EMMA_AGENT_NAME,
            "tenant_id": tenant_id,
            "version": "5.0",
        })
    )

    # ── Initialise pipeline singletons 
    session_manager, stt, tts = _get_singletons()
    call_manager = get_call_manager()

    # ── Run the per-call adapter 
    adapter = LiveKitCallAdapter(
        ctx=ctx,
        call_manager=call_manager,
        session_manager=session_manager,
        stt=stt,
        tts=tts,
        room_name=room_name,
        tenant_id=tenant_id,
    )

    try:
        await adapter.run(sip_participant)
    except Exception as exc:
        logger.exception("Agent job failed | room=%s | error=%s", room_name, exc)
    finally:
        # Ensure call is cleaned up even on unexpected errors
        if call_manager.get_call(room_name):
            call_manager.end_call(room_name, reason="agent_job_complete")
        logger.info("Agent job complete | room=%s", room_name)


async def prewarm(proc) -> None:
    """
    LiveKit prewarm hook: initialise heavy singletons before first call arrives.

    Called by the agent worker runtime when it's ready to accept jobs but
    before any job is assigned. This ensures STT/TTS models are loaded and
    warm, reducing cold-start latency for the first call.

    Args:
        proc: PrewarmedProc (unused; we initialise module-level singletons).
    """
    logger.info("Prewarming EMMA pipeline (loading STT/TTS models)...")
    _get_singletons()
    logger.info("Prewarm complete.")


def make_worker_options() -> WorkerOptions:
    """
    Build LiveKit WorkerOptions for the EMMA agent worker.

    Key decisions:
      worker_type=ROOM:
        One job dispatched per room. Each SIP call gets its own room (via
        SIPDispatchRuleIndividual), so this maps to one job per call.

      agent_name=EMMA_AGENT_NAME:
        Allows dispatch rules to target specific agent names for blue/green
        deploys and canary rollouts.

      prewarm_fnc=prewarm:
        Ensures model weights are loaded before first call arrives.
        Without this, first call incurs Kokoro TTS model load latency (~2s).
    """
    return WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        worker_type=WorkerType.ROOM,
        agent_name=EMMA_AGENT_NAME,
        max_concurrent_jobs=EMMA_MAX_CONCURRENT_CALLS,
    )


async def run_worker_async() -> None:
    """
    Run the agent worker in the current asyncio event loop.

    Used when embedding the worker in FastAPI's lifespan (dev mode only).
    Production: use `python scripts/start_agent.py` (separate process).
    """
    from livekit.agents import Worker

    opts = make_worker_options()
    worker = Worker(
        opts=opts,
        ws_url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    )

    logger.info(
        "Starting EMMA agent worker | url=%s | name=%s | max_jobs=%d",
        LIVEKIT_URL, EMMA_AGENT_NAME, EMMA_MAX_CONCURRENT_CALLS,
    )

    async with worker:
        await worker.run()


if __name__ == "__main__":
    # Standalone entry point — used by scripts/start_agent.py
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            worker_type=WorkerType.ROOM,
            agent_name=EMMA_AGENT_NAME,
            max_concurrent_jobs=EMMA_MAX_CONCURRENT_CALLS,
        )
    )