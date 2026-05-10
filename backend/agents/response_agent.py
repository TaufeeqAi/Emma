import logging
import time

from backend.agents.state import AgentState
from backend.config import GROQ_API_KEY, load_tenant_config
from backend.safety.guardrails_handler import GuardrailsHandler

logger = logging.getLogger(__name__)

# Module-level singletons
_guardrails = GuardrailsHandler()

#  System prompt template 

_BASE_SYSTEM_PROMPT = """\
You are EMMA, an AI receptionist for an NHS GP surgery. You are professional,
calm, empathetic, and concise. Patients calling may be anxious or unwell.

═══════════════════════════════════════════════════════
STRICT RULES — violating any of these is a safety failure:
═══════════════════════════════════════════════════════

1. Answer ONLY using the SURGERY CONTEXT provided below.
2. NEVER suggest a diagnosis, clinical assessment, or medical opinion.
3. NEVER recommend or comment on any medication, dose, or treatment.
4. NEVER invent appointment slots, times, protocols, or phone numbers.
5. If the context does not contain the answer, say:
   "I don't have that information — please call our reception team directly."
6. ALWAYS direct clinical questions to the GP.
7. Keep responses to 2–4 sentences maximum — this is a phone conversation.

═══════════════════════════════════════════════════════
NEGATIVE CONSTRAINTS (what you must NEVER do):
═══════════════════════════════════════════════════════

- Never say "I think" or "I believe" — only state what the context confirms.
- Never mention information not found in the SURGERY CONTEXT below.
- Never diagnose, prescribe, or provide clinical guidance of any kind.
- Never acknowledge that you are an AI unless the patient explicitly asks.
- Never provide the email address, home address, or personal details of staff.

═══════════════════════════════════════════════════════
VOICE FORMAT RULES:
═══════════════════════════════════════════════════════

- Speak naturally as if on a phone call — avoid bullet points or numbered lists.
- Use complete sentences.
- If listing multiple items (e.g., opening hours), use natural connectives:
  "We're open Monday to Friday from eight AM to six PM, and on Saturdays
   from nine AM to one PM. We're closed on Sundays."
"""


def response_agent(state: AgentState) -> AgentState:
    """
    LangGraph node: Response Agent.

    Constructs a surgery-specific system prompt by combining:
      - Base system prompt (rules + constraints)
      - Tenant-specific extras from config.json
      - Retrieved context from retrieval_agent

    Then calls Groq Llama 3.3 via GuardrailsHandler (NeMo output rails active).

    Args:
        state: AgentState with retrieval_context and tenant_id populated.

    Returns:
        Updated AgentState with raw_response and final_response set
        (final_response will be overwritten by verification_agent).
    """
    session_id = state.get("session_id", "?")
    query = state["query"]
    tenant_id = state["tenant_id"]
    context = state.get("retrieval_context") or (
        "No relevant context available. Tell the patient to call reception."
    )

    logger.info(
        "response_agent | session=%s | tenant=%s | query='%.80s'",
        session_id, tenant_id, query,
    )

    # Load tenant-specific prompt additions
    try:
        tenant_config = load_tenant_config(tenant_id)
        tenant_extras = tenant_config.get("system_prompt_extras", "")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "response_agent: failed to load tenant config for '%s': %s",
            tenant_id, exc,
        )
        tenant_extras = ""

    # Compose full system prompt
    system_prompt = (
        f"{_BASE_SYSTEM_PROMPT}\n\n"
        f"{tenant_extras}\n\n"
        f"═══════════════════════════════════════════════════════\n"
        f"SURGERY CONTEXT (use ONLY this information):\n"
        f"═══════════════════════════════════════════════════════\n"
        f"{context}"
    )

    t0 = time.perf_counter()

    try:
        result = _guardrails.generate(
            system_prompt=system_prompt,
            user_message=query,
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=300,
        )
        raw_response = result["content"]
        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "response_agent | latency=%.1fms | nemo_active=%s | nemo_intercepted=%s | session=%s",
            latency_ms,
            result.get("nemo_active"),
            result.get("nemo_intercepted"),
            session_id,
        )

        return {
            **state,
            "raw_response": raw_response,
            "final_response": raw_response,  # overwritten by verification_agent
            "response_latency_ms": latency_ms,
        }

    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.exception(
            "response_agent FAILED | session=%s | error=%s", session_id, exc
        )
        fallback = (
            "I'm sorry, I'm having trouble accessing that information right now. "
            "Please call our reception team directly and they'll be happy to help."
        )
        return {
            **state,
            "raw_response": fallback,
            "final_response": fallback,
            "response_latency_ms": latency_ms,
            "error": f"Response agent error: {exc}",
        }