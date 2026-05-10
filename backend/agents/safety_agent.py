import logging
import time
from datetime import datetime, timezone

from backend.agents.state import AgentState
from backend.rag.embedder import EmbeddingModel
from backend.safety.emergency_keywords import EmergencyDetector

logger = logging.getLogger(__name__)

# Load ONCE at process start. The same model serves both RAG retrieval and emergency detection
_SHARED_EMBEDDER = EmbeddingModel()

_detector = EmergencyDetector(
    semantic_threshold=0.72,
    embedder=_SHARED_EMBEDDER.sentence_transformer,
)


def safety_gate_agent(state: AgentState) -> AgentState:
    """
    LangGraph node: Safety Gate.

    Inspects state["query"] for emergency signals via three-layer detection.
    Sets state["escalate"], state["safety_cleared"], and state["escalation_reason"].

    This node does NOT modify:
      - retrieved_chunks, retrieval_context (retrieval hasn't run)
      - raw_response, final_response (LLM hasn't run)

    Args:
        state: Current AgentState. Must have 'query' and 'tenant_id' populated.

    Returns:
        Updated AgentState with safety fields set.
        Never raises — all exceptions are caught and result in fail-closed escalation.
    """
    t0 = time.perf_counter()
    query = state.get("query", "")
    tenant_id = state.get("tenant_id", "unknown")

    logger.info(
        "safety_gate | session=%s | query='%.80s'",
        state.get("session_id", "?"), query,
    )

    try:
        is_emergency, reason = _detector.is_emergency(query)
    except Exception as exc:  # noqa: BLE001
        # Fail closed: detection error → escalate
        logger.exception(
            "safety_gate EXCEPTION for session=%s — failing closed: %s",
            state.get("session_id", "?"), exc,
        )
        is_emergency = True
        reason = f"Safety gate error (failing closed): {exc}"

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "safety_gate | is_emergency=%s | latency=%.1fms | reason='%s'",
        is_emergency, latency_ms, reason or "none",
    )

    return {
        **state,
        "safety_cleared": not is_emergency,
        "escalate": is_emergency,
        "escalation_reason": reason if is_emergency else None,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }