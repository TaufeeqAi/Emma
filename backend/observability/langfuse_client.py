import logging
import time
import uuid
from functools import wraps
from typing import Any, Callable, Optional

from backend.config import (
    LANGFUSE_ENABLED,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)
from backend.observability.metrics import EvalMetrics

logger = logging.getLogger(__name__)

# ── Module-level singleton ─────────────────────────────────────────────────────
_langfuse_instance: Optional["LangfuseClient"] = None


def get_langfuse_client() -> Optional["LangfuseClient"]:
    """
    Return the process-level LangfuseClient singleton.

    Returns None if Langfuse is disabled (keys not set) or if the
    langfuse package is not installed. Callers must check for None.

    Usage:
        lf = get_langfuse_client()
        if lf:
            lf.score_turn(trace_id, metrics)
    """
    global _langfuse_instance
    if _langfuse_instance is None and LANGFUSE_ENABLED:
        _langfuse_instance = LangfuseClient()
    return _langfuse_instance


class LangfuseClient:
    """
    Wrapper around the Langfuse Python SDK v2.

    Provides:
      - create_trace():       Start a new trace for a patient utterance.
      - create_span():        Add a span (agent node) to an existing trace.
      - score_turn():         Attach RAGAS scores to a completed trace.
      - trace_pipeline_run(): Context manager for tracing a full pipeline run.
      - flush():              Force-flush all queued traces (use at shutdown).

    All public methods are try/except wrapped — a Langfuse error never
    propagates to the patient-facing pipeline.

    Args:
        public_key:  Langfuse public key. Default: from config.
        secret_key:  Langfuse secret key. Default: from config.
        host:        Langfuse server URL. Default: from config.
        enabled:     Master switch. If False, all methods are no-ops.
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled and LANGFUSE_ENABLED
        self._client = None

        if not self._enabled:
            logger.info("LangfuseClient: disabled (keys not set or enabled=False).")
            return

        try:
            from langfuse import Langfuse  # noqa: PLC0415

            self._client = Langfuse(
                public_key=public_key or LANGFUSE_PUBLIC_KEY,
                secret_key=secret_key or LANGFUSE_SECRET_KEY,
                host=host or LANGFUSE_HOST,
                debug=False,
            )
            logger.info("LangfuseClient connected to %s", host or LANGFUSE_HOST)

        except ImportError:
            logger.warning(
                "langfuse package not installed. Run: pip install langfuse==2.57.4"
            )
            self._enabled = False
        except Exception as exc:
            logger.warning("LangfuseClient init failed: %s — tracing disabled.", exc)
            self._enabled = False

    # ── Trace lifecycle ────────────────────────────────────────────────────────

    def create_trace(
        self,
        trace_id: str,
        name: str,
        query: str,
        tenant_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Any:
        """
        Create a new Langfuse trace for a patient utterance.

        A trace represents the full lifecycle of one patient query:
          STT → safety_gate → retrieval → response → verification → TTS

        Args:
            trace_id:   AgentState["trace_id"] — links trace to agent state.
            name:       Human-readable trace name (e.g. "emma_pipeline_run").
            query:      Patient's utterance (post-STT, pre-safety).
            tenant_id:  Surgery identifier.
            session_id: Call session ID (groups multiple turns).
            metadata:   Additional key-value pairs for filtering in UI.

        Returns:
            Langfuse Trace object, or None if tracing is disabled.
        """
        if not self._enabled or not self._client:
            return None

        try:
            trace = self._client.trace(
                id=trace_id,
                name=name,
                input={"query": query, "tenant_id": tenant_id},
                session_id=session_id,
                metadata={
                    "tenant_id": tenant_id,
                    "session_id": session_id,
                    **(metadata or {}),
                },
                tags=[f"tenant:{tenant_id}"],
            )
            logger.debug("Langfuse trace created: %s", trace_id)
            return trace
        except Exception as exc:
            logger.debug("Langfuse create_trace failed: %s", exc)
            return None

    def create_span(
        self,
        trace_id: str,
        name: str,
        input_data: dict,
        output_data: Optional[dict] = None,
        latency_ms: Optional[float] = None,
        level: str = "DEFAULT",
        status_message: Optional[str] = None,
    ) -> Any:
        """
        Add a span (agent node execution) to an existing trace.

        Args:
            trace_id:      Must match an existing trace's ID.
            name:          Agent node name (e.g. "safety_gate", "retrieval").
            input_data:    Node inputs (query, tenant_id).
            output_data:   Node outputs (escalate, verified, response_preview).
            latency_ms:    Execution time in milliseconds.
            level:         "DEFAULT" | "WARNING" | "ERROR".
            status_message: Error message if level=ERROR.

        Returns:
            Langfuse Span object, or None if tracing is disabled.
        """
        if not self._enabled or not self._client:
            return None

        try:
            # Langfuse spans are created via the trace object reference.
            # Since we pass trace_id as a string (not the trace object),
            # we use the low-level span() method which accepts trace_id directly.
            span = self._client.span(
                trace_id=trace_id,
                name=name,
                input=input_data,
                output=output_data,
                metadata={"latency_ms": latency_ms},
                level=level,
                status_message=status_message,
            )
            logger.debug("Langfuse span created: %s / %s", trace_id, name)
            return span
        except Exception as exc:
            logger.debug("Langfuse create_span failed (name=%s): %s", name, exc)
            return None

    def create_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        prompt: str,
        completion: str,
        latency_ms: Optional[float] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> Any:
        """
        Create a Langfuse Generation span for LLM calls.

        Generations are special spans that Langfuse renders with token counts,
        cost estimates, and model information. Applied to:
          - response_agent  (Groq Llama 3.3 generation)
          - verification_agent (Groq Llama 3.3 verification call)

        Args:
            trace_id:      Parent trace ID.
            name:          Generation name (e.g. "response_agent_llm").
            model:         Model identifier (e.g. "llama-3.3-70b-versatile").
            prompt:        Full prompt sent to LLM (system + user).
            completion:    LLM output text.
            latency_ms:    LLM call latency.
            input_tokens:  Prompt token count (from Groq response headers).
            output_tokens: Completion token count.
        """
        if not self._enabled or not self._client:
            return None

        try:
            generation = self._client.generation(
                trace_id=trace_id,
                name=name,
                model=model,
                input=prompt,
                output=completion,
                metadata={
                    "latency_ms": latency_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )
            logger.debug("Langfuse generation created: %s / %s", trace_id, name)
            return generation
        except Exception as exc:
            logger.debug("Langfuse create_generation failed: %s", exc)
            return None

    def score_turn(
        self,
        trace_id: str,
        metrics: EvalMetrics,
        comment: Optional[str] = None,
    ) -> None:
        """
        Attach RAGAS evaluation scores to a Langfuse trace.

        Scores appear in the Langfuse UI under "Scores" on the trace detail page.
        Use this to tag production traces with their faithfulness score — enables
        filtering traces where faithfulness < threshold for debugging.

        Args:
            trace_id: Must match the trace created in create_trace().
            metrics:  EvalMetrics dataclass with RAGAS scores.
            comment:  Optional annotation (e.g. "automated RAGAS evaluation").
        """
        if not self._enabled or not self._client:
            return

        score_defs = [
            ("ragas_faithfulness",       metrics.faithfulness),
            ("ragas_answer_relevancy",   metrics.answer_relevancy),
            ("ragas_context_precision",  metrics.context_precision),
            ("ragas_context_recall",     metrics.context_recall),
        ]
        for score_name, value in score_defs:
            try:
                self._client.score(
                    trace_id=trace_id,
                    name=score_name,
                    value=value,
                    comment=comment,
                )
            except Exception as exc:
                logger.debug(
                    "Langfuse score failed (name=%s): %s", score_name, exc
                )

    def score_safety_event(
        self,
        trace_id: str,
        escalated: bool,
        reason: Optional[str] = None,
    ) -> None:
        """
        Record a binary safety score on a trace.

        escalated=True  → score=1.0 (emergency correctly identified)
        escalated=False → score=0.0 (routine query, no escalation)

        Filtering traces where safety_escalated=1 in Langfuse UI gives
        a complete audit log of all emergency escalations in production.
        """
        if not self._enabled or not self._client:
            return
        try:
            self._client.score(
                trace_id=trace_id,
                name="safety_escalated",
                value=1.0 if escalated else 0.0,
                comment=reason or ("Emergency escalation" if escalated else "Routine query"),
            )
        except Exception as exc:
            logger.debug("Langfuse score_safety_event failed: %s", exc)

    def update_trace_output(
        self,
        trace_id: str,
        final_response: str,
        verified: bool,
        escalated: bool,
        total_latency_ms: Optional[float] = None,
    ) -> None:
        """
        Update a trace with the final pipeline output.

        Called after the full LangGraph pipeline completes. Records the
        final response and key outcome metrics on the trace root.

        Args:
            trace_id:         Trace to update.
            final_response:   Text sent to the patient.
            verified:         Whether Chain of Verification passed.
            escalated:        Whether emergency escalation triggered.
            total_latency_ms: Full pipeline wall-clock time.
        """
        if not self._enabled or not self._client:
            return
        try:
            self._client.trace(
                id=trace_id,
                output={
                    "final_response": final_response[:200],  # Truncate for UI
                    "verified": verified,
                    "escalated": escalated,
                    "total_latency_ms": total_latency_ms,
                },
            )
        except Exception as exc:
            logger.debug("Langfuse update_trace_output failed: %s", exc)

    def flush(self) -> None:
        """
        Force-flush all queued traces to Langfuse.

        Call at process shutdown or end of eval run to ensure all
        traces are persisted before the process exits.
        """
        if not self._enabled or not self._client:
            return
        try:
            self._client.flush()
            logger.info("Langfuse: all queued traces flushed.")
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", exc)

    @property
    def is_active(self) -> bool:
        """True if Langfuse is configured and the client initialised."""
        return self._enabled and self._client is not None


# ── Decorator: trace a LangGraph agent node ────────────────────────────────────

def trace_agent_call(agent_name: str) -> Callable:
    """
    Decorator: wrap a LangGraph agent node function with Langfuse tracing.

    Creates a Langfuse span for each agent invocation, capturing:
      - Input:   query, tenant_id
      - Output:  relevant state fields (escalate, verified, response_preview)
      - Latency: wall-clock execution time in milliseconds
      - Level:   ERROR if the node sets state["error"]

    Usage:
        @trace_agent_call("safety_gate")
        def safety_gate_agent(state: AgentState) -> AgentState:
            ...

    The decorator is a no-op if Langfuse is not configured:
        - No imports are attempted at decoration time.
        - The original function is returned unchanged if tracing fails.

    Args:
        agent_name: Name of the agent node (used as Langfuse span name).

    Returns:
        Decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: dict, *args, **kwargs) -> dict:
            client = get_langfuse_client()
            trace_id = state.get("trace_id")

            if not client or not trace_id:
                # Langfuse not configured — run agent unchanged
                return func(state, *args, **kwargs)

            # Ensure a trace exists for this trace_id
            # (idempotent: Langfuse deduplicates by trace_id)
            client.create_trace(
                trace_id=trace_id,
                name="emma_pipeline_run",
                query=state.get("query", ""),
                tenant_id=state.get("tenant_id", "unknown"),
                session_id=state.get("session_id"),
            )

            t0 = time.perf_counter()
            error_msg: Optional[str] = None

            try:
                result = func(state, *args, **kwargs)
                latency_ms = (time.perf_counter() - t0) * 1000
                error_msg = result.get("error")

                client.create_span(
                    trace_id=trace_id,
                    name=agent_name,
                    input_data={
                        "query": state.get("query", "")[:200],
                        "tenant_id": state.get("tenant_id"),
                    },
                    output_data={
                        "escalate": result.get("escalate"),
                        "safety_cleared": result.get("safety_cleared"),
                        "verified": result.get("verified"),
                        "response_preview": (
                            (result.get("final_response") or "")[:100]
                        ),
                        "error": error_msg,
                    },
                    latency_ms=latency_ms,
                    level="ERROR" if error_msg else "DEFAULT",
                    status_message=error_msg,
                )

                # After the full pipeline completes (verification node),
                # update the trace root with final output and safety score.
                if agent_name == "verification":
                    client.update_trace_output(
                        trace_id=trace_id,
                        final_response=result.get("final_response") or "",
                        verified=result.get("verified", False),
                        escalated=result.get("escalate", False),
                        total_latency_ms=latency_ms,
                    )
                    client.score_safety_event(
                        trace_id=trace_id,
                        escalated=result.get("escalate", False),
                        reason=result.get("escalation_reason"),
                    )

                elif agent_name == "escalation":
                    # Escalation is terminal — update trace and score immediately
                    client.update_trace_output(
                        trace_id=trace_id,
                        final_response=result.get("final_response") or "",
                        verified=True,  # Hardcoded — no verification needed
                        escalated=True,
                        total_latency_ms=latency_ms,
                    )
                    client.score_safety_event(
                        trace_id=trace_id,
                        escalated=True,
                        reason=state.get("escalation_reason"),
                    )

                return result

            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                client.create_span(
                    trace_id=trace_id,
                    name=agent_name,
                    input_data={"query": state.get("query", "")[:200]},
                    output_data={"error": str(exc)},
                    latency_ms=latency_ms,
                    level="ERROR",
                    status_message=str(exc),
                )
                raise  # Re-raise — don't swallow agent errors

        return wrapper
    return decorator