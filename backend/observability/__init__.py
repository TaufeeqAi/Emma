"""
Import hierarchy:
  metrics         → (no internal deps)
  langfuse_client → metrics, config
  ragas_eval      → langfuse_client, metrics, agents.graph, agents.state
  deepeval_suite  → langfuse_client, metrics, agents.graph, agents.state

Graceful degradation:
  All exports here handle the case where optional dependencies
  (langfuse, ragas, deepeval) are not installed. Import errors are caught
  and replaced with stub objects that log warnings. This allows the core
  pipeline (Phases 1–3) to run without eval dependencies installed.
"""

from backend.observability.metrics import (
    EvalMetrics,
    EvalThresholds,
    PerTenantScores,
    EVAL_TEST_CASES,
    EMERGENCY_SAFETY_CASES,
)

try:
    from backend.observability.langfuse_client import (
        LangfuseClient,
        get_langfuse_client,
        trace_agent_call,
    )
except ImportError:
    LangfuseClient = None  # type: ignore[assignment,misc]
    get_langfuse_client = lambda: None  # type: ignore[assignment]
    trace_agent_call = lambda name: (lambda fn: fn)  # type: ignore[assignment]

__all__ = [
    # Metrics + test cases
    "EvalMetrics",
    "EvalThresholds",
    "PerTenantScores",
    "EVAL_TEST_CASES",
    "EMERGENCY_SAFETY_CASES",
    # Langfuse
    "LangfuseClient",
    "get_langfuse_client",
    "trace_agent_call",
]