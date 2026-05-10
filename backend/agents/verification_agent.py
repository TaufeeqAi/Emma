import logging
import time

from backend.agents.state import AgentState
from backend.config import GROQ_API_KEY
from groq import Groq

logger = logging.getLogger(__name__)
_client = Groq(api_key=GROQ_API_KEY)

# Verification prompt
# 1. Return the original response unchanged (all claims supported).
# 2. Return a rewritten response with unsupported claims removed.

_VERIFICATION_PROMPT_TEMPLATE = """\
You are a clinical safety verifier for an NHS GP surgery AI receptionist.

Your ONLY job is to check whether every factual claim in the RESPONSE below
is directly supported by the CONTEXT below.

══════════════════════════════════════════════
RETRIEVED CONTEXT (ground truth):
══════════════════════════════════════════════
{context}

══════════════════════════════════════════════
RESPONSE TO VERIFY:
══════════════════════════════════════════════
{response}

══════════════════════════════════════════════
VERIFICATION RULES:
══════════════════════════════════════════════

1. Read every factual claim in the RESPONSE (times, durations, phone numbers,
   procedures, names).
2. Check each claim against the CONTEXT.
3. If ALL claims are directly supported by the CONTEXT:
   → Return the RESPONSE exactly as written. Do not change a single word.
4. If ANY claim is NOT supported by or contradicts the CONTEXT:
   → Rewrite the RESPONSE, removing or replacing only the unsupported claims.
   → Replace unsupported claims with: "For that information, please speak
     with a member of our team or call reception directly."
   → Keep all supported claims unchanged.

CRITICAL:
- Return ONLY the final response text. No preamble, no explanation,
  no "Verified:" prefix. Just the response the patient will hear.
- Do not add any information not in the RESPONSE or CONTEXT.
- Do not make the response longer than the original.
"""


def verification_agent(state: AgentState) -> AgentState:
    """
    LangGraph node: Verification Agent.

    Checks raw_response faithfulness against retrieval_context using
    a second LLM call (Chain of Verification).

    Args:
        state: AgentState with raw_response and retrieval_context set.

    Returns:
        Updated AgentState with final_response (verified), verified=True,
        verification_notes, and verification_latency_ms.
        On error: passes raw_response through as final_response and logs warning.
    """
    session_id = state.get("session_id", "?")
    raw_response = state.get("raw_response") or ""
    context = state.get("retrieval_context") or ""

    logger.info(
        "verification_agent | session=%s | response_len=%d chars",
        session_id, len(raw_response),
    )

    # Guard: if no raw response exists (e.g., response_agent failed),
    # pass through the existing final_response without modification.
    if not raw_response.strip():
        logger.warning(
            "verification_agent: empty raw_response — skipping verification | session=%s",
            session_id,
        )
        return {
            **state,
            "verified": False,
            "verification_notes": "Skipped — empty raw_response.",
            "verification_latency_ms": 0.0,
        }

    t0 = time.perf_counter()

    try:
        prompt = _VERIFICATION_PROMPT_TEMPLATE.format(
            context=context or "No context available.",
            response=raw_response,
        )

        result = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=350,   
            stream=False,
        )

        verified_response = (result.choices[0].message.content or "").strip()
        latency_ms = (time.perf_counter() - t0) * 1000

        # Detect whether the verifier modified the response
        was_modified = verified_response.strip() != raw_response.strip()
        if was_modified:
            logger.warning(
                "verification_agent MODIFIED response | session=%s | "
                "original='%.100s...' | verified='%.100s...'",
                session_id, raw_response, verified_response,
            )
            notes = "Response modified by Chain of Verification — unsupported claim removed."
        else:
            notes = "All claims verified against context. Response unchanged."

        logger.info(
            "verification_agent | modified=%s | latency=%.1fms | session=%s",
            was_modified, latency_ms, session_id,
        )

        return {
            **state,
            "final_response": verified_response,
            "verified": True,
            "verification_notes": notes,
            "verification_latency_ms": latency_ms,
        }

    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.exception(
            "verification_agent FAILED | session=%s | error=%s — "
            "passing raw_response as final_response",
            session_id, exc,
        )

        return {
            **state,
            "final_response": raw_response,
            "verified": False,
            "verification_notes": f"Verification failed — raw response used: {exc}",
            "verification_latency_ms": latency_ms,
            "error": f"Verification error: {exc}",
        }