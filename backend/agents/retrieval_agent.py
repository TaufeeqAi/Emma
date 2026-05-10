import logging
import time

from backend.agents.state import AgentState
from backend.rag.retriever import SurgeryRetriever
from backend.agents.safety_agent import _SHARED_EMBEDDER

logger = logging.getLogger(__name__)

# Module-level singleton — embedding model + cross-encoder loaded once.
_retriever = SurgeryRetriever(embedder=_SHARED_EMBEDDER)


def retrieval_agent(state: AgentState) -> AgentState:
    """
    LangGraph node: Retrieval Agent.

    Retrieves top-K relevant chunks from the tenant's Qdrant collection
    and formats them into a context string for the response_agent.

    Args:
        state: AgentState with safety_cleared=True, query, and tenant_id set.

    Returns:
        Updated AgentState with retrieved_chunks, retrieval_context,
        and retrieval_latency_ms populated.
        On error: sets state["error"] and returns safe empty context.
    """
    query = state["query"]
    tenant_id = state["tenant_id"]
    session_id = state.get("session_id", "?")

    logger.info(
        "retrieval_agent | session=%s | tenant=%s | query='%.80s'",
        session_id, tenant_id, query,
    )

    t0 = time.perf_counter()

    try:
        chunks = _retriever.retrieve(query, tenant_id)
        context = _retriever.format_context(chunks)
        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "retrieval_agent | retrieved %d chunks | latency=%.1fms | session=%s",
            len(chunks), latency_ms, session_id,
        )

        return {
            **state,
            "retrieved_chunks": chunks,
            "retrieval_context": context,
            "retrieval_latency_ms": latency_ms,
        }

    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.exception(
            "retrieval_agent FAILED | session=%s | tenant=%s | error=%s",
            session_id, tenant_id, exc,
        )
        # Return safe empty context — response_agent will acknowledge the gap
        safe_context = (
            "No relevant information found in the surgery guidelines. "
            "Acknowledge this to the patient and direct them to call reception."
        )
        return {
            **state,
            "retrieved_chunks": [],
            "retrieval_context": safe_context,
            "retrieval_latency_ms": latency_ms,
            "error": f"Retrieval error: {exc}",
        }