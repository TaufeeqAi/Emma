import uuid
from unittest.mock import MagicMock, patch, call
import pytest

from backend.observability.metrics import (
    EMERGENCY_SAFETY_CASES,
    EVAL_TEST_CASES,
    NON_EMERGENCY_CASES,
    EvalMetrics,
    EvalThresholds,
    PerTenantScores,
)


# ── EvalMetrics unit tests 

class TestEvalMetrics:

    def test_passed_when_all_above_threshold(self):
        m = EvalMetrics(
            faithfulness=0.90,
            answer_relevancy=0.85,
            context_precision=0.80,
            context_recall=0.80,
        )
        assert m.passed is True

    def test_fails_when_faithfulness_below_threshold(self):
        m = EvalMetrics(
            faithfulness=0.70,      # Below 0.80 threshold
            answer_relevancy=0.85,
            context_precision=0.80,
            context_recall=0.80,
        )
        assert m.passed is False

    def test_fails_when_answer_relevancy_below_threshold(self):
        m = EvalMetrics(
            faithfulness=0.85,
            answer_relevancy=0.60,  # Below 0.75 threshold
            context_precision=0.80,
            context_recall=0.80,
        )
        assert m.passed is False

    def test_to_dict_structure(self):
        m = EvalMetrics(
            faithfulness=0.88,
            answer_relevancy=0.82,
            context_precision=0.75,
            context_recall=0.78,
            n_cases=10,
            tenant_id="surgery_greenfield",
        )
        d = m.to_dict()
        assert "scores" in d
        assert "thresholds" in d
        assert "passed" in d
        assert d["n_cases"] == 10
        assert d["tenant_id"] == "surgery_greenfield"
        assert round(d["scores"]["faithfulness"], 4) == 0.88

    def test_boundary_exactly_at_threshold(self):
        """Score exactly at threshold should pass (>=, not >)."""
        from backend.config import RAGAS_FAITHFULNESS_THRESHOLD
        m = EvalMetrics(
            faithfulness=RAGAS_FAITHFULNESS_THRESHOLD,
            answer_relevancy=0.75,
            context_precision=0.70,
            context_recall=0.70,
        )
        assert m.passed is True

    def test_error_field_preserved(self):
        m = EvalMetrics(0.0, 0.0, 0.0, 0.0, error="RAGAS API timeout")
        d = m.to_dict()
        assert d["error"] == "RAGAS API timeout"


# ── LangfuseClient unit tests 

class TestLangfuseClient:
    """
    Unit tests for LangfuseClient with mocked Langfuse SDK.
    No real Langfuse server needed.
    """

    def _make_client(self):
        """Create a LangfuseClient with a mocked internal Langfuse instance."""
        from backend.observability.langfuse_client import LangfuseClient
        client = LangfuseClient.__new__(LangfuseClient)
        client._enabled = True
        client._client = MagicMock()
        return client

    def test_create_trace_calls_sdk(self):
        client = self._make_client()
        trace_id = str(uuid.uuid4())
        client.create_trace(
            trace_id=trace_id,
            name="test_trace",
            query="What are the opening hours?",
            tenant_id="surgery_greenfield",
        )
        client._client.trace.assert_called_once()
        call_kwargs = client._client.trace.call_args.kwargs
        assert call_kwargs["id"] == trace_id
        assert call_kwargs["name"] == "test_trace"

    def test_create_span_calls_sdk(self):
        client = self._make_client()
        trace_id = str(uuid.uuid4())
        client.create_span(
            trace_id=trace_id,
            name="safety_gate",
            input_data={"query": "test", "tenant_id": "surgery_greenfield"},
            output_data={"escalate": False},
            latency_ms=5.0,
        )
        client._client.span.assert_called_once()
        call_kwargs = client._client.span.call_args.kwargs
        assert call_kwargs["trace_id"] == trace_id
        assert call_kwargs["name"] == "safety_gate"

    def test_score_turn_creates_four_scores(self):
        client = self._make_client()
        trace_id = str(uuid.uuid4())
        metrics = EvalMetrics(
            faithfulness=0.90,
            answer_relevancy=0.85,
            context_precision=0.80,
            context_recall=0.78,
        )
        client.score_turn(trace_id, metrics)
        assert client._client.score.call_count == 4

    def test_score_safety_event_escalated(self):
        client = self._make_client()
        trace_id = str(uuid.uuid4())
        client.score_safety_event(trace_id, escalated=True, reason="chest pain")
        call_kwargs = client._client.score.call_args.kwargs
        assert call_kwargs["value"] == 1.0
        assert call_kwargs["name"] == "safety_escalated"

    def test_score_safety_event_not_escalated(self):
        client = self._make_client()
        trace_id = str(uuid.uuid4())
        client.score_safety_event(trace_id, escalated=False)
        call_kwargs = client._client.score.call_args.kwargs
        assert call_kwargs["value"] == 0.0

    def test_disabled_client_is_noop(self):
        """All methods must be silent no-ops when disabled."""
        from backend.observability.langfuse_client import LangfuseClient
        client = LangfuseClient.__new__(LangfuseClient)
        client._enabled = False
        client._client = None

        # None of these should raise
        client.create_trace("id", "name", "q", "t")
        client.create_span("id", "name", {})
        client.score_turn("id", EvalMetrics(0.9, 0.8, 0.8, 0.8))
        client.flush()
        assert client.is_active is False

    def test_sdk_exception_is_silenced(self):
        """SDK errors must not propagate to caller."""
        client = self._make_client()
        client._client.trace.side_effect = Exception("Langfuse server down")
        # Must not raise
        result = client.create_trace("id", "name", "q", "t")
        assert result is None

    def test_flush_calls_sdk(self):
        client = self._make_client()
        client.flush()
        client._client.flush.assert_called_once()


# ── trace_agent_call decorator unit tests 

class TestTraceAgentCallDecorator:
    """
    Tests for the @trace_agent_call decorator.
    Verifies that spans are created and agent functions execute correctly.
    """

    def test_decorator_calls_agent_function(self):
        """Decorated function must still be called and return its result."""
        from backend.observability.langfuse_client import trace_agent_call

        mock_agent = MagicMock(return_value={"result": "ok", "escalate": False})
        decorated = trace_agent_call("test_node")(mock_agent)

        state = {
            "query": "test query",
            "tenant_id": "surgery_greenfield",
            "trace_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
        }

        # Patch get_langfuse_client to return None (tracing disabled)
        with patch("backend.observability.langfuse_client.get_langfuse_client", return_value=None):
            result = decorated(state)

        mock_agent.assert_called_once_with(state)
        assert result == {"result": "ok", "escalate": False}

    def test_decorator_noop_without_trace_id(self):
        """If state has no trace_id, decorator runs agent without creating span."""
        from backend.observability.langfuse_client import trace_agent_call

        mock_agent = MagicMock(return_value={"result": "ok"})
        decorated = trace_agent_call("test_node")(mock_agent)

        state = {
            "query": "test query",
            "tenant_id": "surgery_greenfield",
            # No trace_id
        }

        mock_lf = MagicMock()
        with patch("backend.observability.langfuse_client.get_langfuse_client", return_value=mock_lf):
            result = decorated(state)

        # Should still run the agent
        mock_agent.assert_called_once()
        # Should NOT create a span (no trace_id)
        mock_lf.create_span.assert_not_called()

    def test_decorator_creates_span_on_success(self):
        """Successful agent call should create a Langfuse span."""
        from backend.observability.langfuse_client import trace_agent_call

        trace_id = str(uuid.uuid4())
        mock_agent = MagicMock(return_value={
            "escalate": False, "verified": True, "final_response": "test"
        })
        decorated = trace_agent_call("test_node")(mock_agent)

        state = {
            "query": "test", "tenant_id": "t",
            "trace_id": trace_id, "session_id": "s",
        }

        mock_lf = MagicMock()
        mock_lf.create_trace.return_value = None
        mock_lf.create_span.return_value = None

        with patch("backend.observability.langfuse_client.get_langfuse_client", return_value=mock_lf):
            decorated(state)

        mock_lf.create_span.assert_called_once()
        span_kwargs = mock_lf.create_span.call_args.kwargs
        assert span_kwargs["name"] == "test_node"
        assert span_kwargs["trace_id"] == trace_id
        assert span_kwargs["level"] == "DEFAULT"

    def test_decorator_creates_error_span_on_exception(self):
        """Agent exception should create ERROR span and re-raise."""
        from backend.observability.langfuse_client import trace_agent_call

        trace_id = str(uuid.uuid4())
        mock_agent = MagicMock(side_effect=RuntimeError("agent crashed"))
        decorated = trace_agent_call("test_node")(mock_agent)

        state = {
            "query": "test", "tenant_id": "t",
            "trace_id": trace_id, "session_id": "s",
        }

        mock_lf = MagicMock()
        with patch("backend.observability.langfuse_client.get_langfuse_client", return_value=mock_lf):
            with pytest.raises(RuntimeError, match="agent crashed"):
                decorated(state)

        mock_lf.create_span.assert_called_once()
        span_kwargs = mock_lf.create_span.call_args.kwargs
        assert span_kwargs["level"] == "ERROR"
        assert "agent crashed" in span_kwargs["status_message"]

    def test_decorator_preserves_function_metadata(self):
        """@wraps must preserve the original function's __name__ and __doc__."""
        from backend.observability.langfuse_client import trace_agent_call

        def my_agent(state):
            """My agent docstring."""
            return state

        decorated = trace_agent_call("my_agent")(my_agent)
        assert decorated.__name__ == "my_agent"
        assert "docstring" in decorated.__doc__


# ── EVAL_TEST_CASES validation 

class TestEvalTestCases:
    """
    Validate the evaluation dataset configuration.
    These tests don't require any external API — just config validation.
    """

    def test_all_cases_have_required_fields(self):
        for case in EVAL_TEST_CASES:
            assert "question" in case, f"Missing 'question': {case}"
            assert "ground_truth" in case, f"Missing 'ground_truth': {case}"
            assert "tenant_id" in case, f"Missing 'tenant_id': {case}"
            assert "category" in case, f"Missing 'category': {case}"

    def test_tenant_ids_are_valid(self):
        from backend.config import TENANTS
        for case in EVAL_TEST_CASES:
            assert case["tenant_id"] in TENANTS, (
                f"Invalid tenant_id '{case['tenant_id']}' in test case: {case['question']}"
            )

    def test_ground_truths_are_non_empty(self):
        for case in EVAL_TEST_CASES:
            assert len(case["ground_truth"].strip()) > 10, (
                f"Ground truth too short for: {case['question']}"
            )

    def test_emergency_cases_have_required_fields(self):
        for case in EMERGENCY_SAFETY_CASES:
            assert "query" in case, f"Missing 'query': {case}"
            assert "category" in case, f"Missing 'category': {case}"
            assert len(case["query"].strip()) > 5, f"Query too short: {case}"

    def test_both_tenants_represented_in_eval_cases(self):
        tenant_ids = {c["tenant_id"] for c in EVAL_TEST_CASES}
        assert "surgery_greenfield" in tenant_ids
        assert "surgery_riverside" in tenant_ids

    def test_categories_cover_key_clinical_areas(self):
        categories = {c.get("category") for c in EVAL_TEST_CASES}
        required = {"opening_hours", "prescriptions", "appointments"}
        missing = required - categories
        assert not missing, f"Missing required evaluation categories: {missing}"

    def test_no_duplicate_questions_per_tenant(self):
        seen = set()
        for case in EVAL_TEST_CASES:
            key = (case["question"].lower().strip(), case["tenant_id"])
            assert key not in seen, f"Duplicate test case: {case['question']} / {case['tenant_id']}"
            seen.add(key)


# ── Safety audit integration tests 

class TestSafetyAudit:
    """
    Integration tests for emergency escalation safety audit.
    These tests invoke the full pipeline — requires Qdrant + data ingested.
    """

    @pytest.mark.requires_groq
    def test_emergency_recall_is_one(self):
        """
        CRITICAL: 100% of emergency queries must escalate.
        Any recall < 1.0 is a clinical safety failure — block release.
        """
        from backend.observability.deepeval_suite import EMMADeepEvalSuite

        suite = EMMADeepEvalSuite(verbose=False, push_to_langfuse=False)
        result = suite.run_safety_audit(
            emergency_cases=EMERGENCY_SAFETY_CASES,
            non_emergency_cases=NON_EMERGENCY_CASES,
            tenant_id="surgery_greenfield",
        )

        assert result.recall == 1.0, (
            f"SAFETY AUDIT FAILED — {len(result.missed_escalations)} missed escalations:\n"
            + "\n".join(
                f"  - [{m['category']}] {m['query']}"
                for m in result.missed_escalations
            )
        )

    @pytest.mark.requires_groq
    def test_non_emergency_precision_acceptable(self):
        """
        False positive rate should be low (< 20%).
        False positives are inconvenient but not safety-critical.
        """
        from backend.observability.deepeval_suite import EMMADeepEvalSuite

        suite = EMMADeepEvalSuite(verbose=False, push_to_langfuse=False)
        result = suite.run_safety_audit(
            emergency_cases=EMERGENCY_SAFETY_CASES,
            non_emergency_cases=NON_EMERGENCY_CASES,
            tenant_id="surgery_greenfield",
        )

        # Allow up to 20% false positive rate (1 in 5 admin queries triggers escalation)
        # This threshold is deliberately lenient — precision is secondary to recall.
        assert result.precision >= 0.80, (
            f"Too many false positives: precision={result.precision:.3f}. "
            f"False positives: {result.false_positives}"
        )

    @pytest.mark.requires_groq
    def test_safety_audit_result_serialisable(self):
        """SafetyAuditResult.to_dict() must produce JSON-serialisable output."""
        import json
        from backend.observability.deepeval_suite import EMMADeepEvalSuite

        suite = EMMADeepEvalSuite(verbose=False, push_to_langfuse=False)
        # Use a small subset for speed
        result = suite.run_safety_audit(
            emergency_cases=EMERGENCY_SAFETY_CASES[:3],
            non_emergency_cases=NON_EMERGENCY_CASES[:3],
            tenant_id="surgery_greenfield",
        )
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0


# ── RAGAS smoke tests 

class TestRAGASSmoke:
    """
    Smoke tests for the RAGAS evaluation pipeline.
    Full RAGAS run is slow and costs OpenAI credits — mark accordingly.
    """

    @pytest.mark.requires_groq
    @pytest.mark.requires_openai
    @pytest.mark.slow
    def test_ragas_returns_scores(self):
        """RAGAS evaluation should return valid scores in [0, 1]."""
        from backend.observability.ragas_eval import RAGASEvaluator

        # Use 2 cases for speed
        evaluator = RAGASEvaluator(
            test_cases=EVAL_TEST_CASES[:2],
            verbose=False,
        )
        scores = evaluator.run_evaluation(tenant_id="surgery_greenfield")

        assert scores.aggregate is not None
        m = scores.aggregate
        assert 0.0 <= m.faithfulness <= 1.0
        assert 0.0 <= m.answer_relevancy <= 1.0
        assert 0.0 <= m.context_precision <= 1.0
        assert 0.0 <= m.context_recall <= 1.0

    @pytest.mark.requires_groq
    @pytest.mark.requires_openai
    @pytest.mark.slow
    def test_ragas_save_results(self, tmp_path):
        """Results should save to JSON without error."""
        import json
        from backend.observability.ragas_eval import RAGASEvaluator

        evaluator = RAGASEvaluator(test_cases=EVAL_TEST_CASES[:1], verbose=False)
        scores = evaluator.run_evaluation(tenant_id="surgery_greenfield")

        output_path = tmp_path / "test_ragas.json"
        saved_path = evaluator.save_results(scores, output_path)

        assert saved_path.exists()
        with open(saved_path) as f:
            data = json.load(f)
        assert "scores" in data
        assert "passed" in data

    @pytest.mark.requires_groq
    @pytest.mark.requires_openai
    @pytest.mark.slow
    def test_ragas_assert_thresholds_passes(self):
        """assert_thresholds() should not raise when scores are above threshold."""
        from backend.observability.ragas_eval import RAGASEvaluator

        evaluator = RAGASEvaluator(test_cases=EVAL_TEST_CASES[:2], verbose=False)
        scores = evaluator.run_evaluation(tenant_id="surgery_greenfield")

        # Only assert if scores are actually above threshold
        # (we can't guarantee this in all environments)
        if scores.aggregate and scores.aggregate.passed:
            evaluator.assert_thresholds(scores)  # Should not raise

    def test_ragas_assert_thresholds_raises_on_fail(self):
        """assert_thresholds() should raise AssertionError on low scores."""
        from backend.observability.ragas_eval import RAGASEvaluator

        evaluator = RAGASEvaluator(verbose=False)
        # Inject artificially low scores
        scores = PerTenantScores(
            aggregate=EvalMetrics(
                faithfulness=0.40,   # Below threshold
                answer_relevancy=0.80,
                context_precision=0.80,
                context_recall=0.80,
            )
        )
        with pytest.raises(AssertionError, match="faithfulness"):
            evaluator.assert_thresholds(scores)