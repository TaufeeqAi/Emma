import logging

from backend.agents.state import AgentState
from backend.config import load_tenant_config

logger = logging.getLogger(__name__)

# Hardcoded prefix — always prepended, regardless of tenant config.

_EMERGENCY_PREFIX = (
    "If you are in immediate danger or experiencing a life-threatening emergency, "
    "please call 999 now. "
)

# Fallback if tenant config cannot be loaded.
_FALLBACK_ESCALATION = (
    "I'm concerned about what you've described. "
    "Please call 999 immediately for a life-threatening emergency, "
    "or 111 for urgent medical advice. "
    "Do not wait — please seek help now."
)


def escalation_handler(state: AgentState) -> AgentState:
    """
    LangGraph node: Escalation Handler.

    Generates a hardcoded emergency response. Called only when
    safety_gate sets state["escalate"] = True.

    This node makes ZERO LLM calls. It must complete in < 5ms.

    Args:
        state: AgentState with escalate=True and tenant_id set.

    Returns:
        Updated AgentState with final_response set to the emergency message,
        verified=True (hardcoded responses don't need verification),
        and verification_notes documenting the escalation trigger.
    """
    session_id = state.get("session_id", "?")
    tenant_id = state.get("tenant_id", "unknown")
    reason = state.get("escalation_reason", "No reason recorded")

    logger.warning(
        "ESCALATION TRIGGERED | session=%s | tenant=%s | reason='%s'",
        session_id, tenant_id, reason,
    )

    # Load tenant-specific escalation message
    try:
        tenant_config = load_tenant_config(tenant_id)
        tenant_message = tenant_config.get("escalation_message", "")
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "escalation_handler: failed to load tenant config for '%s': %s. "
            "Using fallback escalation message.",
            tenant_id, exc,
        )
        tenant_message = ""

    # Compose final response
    if tenant_message:
        final_response = _EMERGENCY_PREFIX + tenant_message
    else:
        final_response = _FALLBACK_ESCALATION

    logger.info(
        "escalation_handler | response='%.100s...' | session=%s",
        final_response, session_id,
    )

    return {
        **state,
        "final_response": final_response,
        "verified": True,  # Hardcoded — no verification needed
        "verification_notes": f"Emergency escalation — hardcoded response. Trigger: {reason}",
        # Explicitly clear LLM fields to prevent downstream misinterpretation
        "raw_response": None,
        "retrieval_context": None,
        "retrieved_chunks": None,
    }