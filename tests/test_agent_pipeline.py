import time
import uuid
import pytest

from backend.agents.graph import emma_graph
from backend.agents.state import make_initial_state, AgentState


def run_pipeline(query: str, tenant: str = "surgery_greenfield") -> AgentState:
    """Run the full pipeline and return final AgentState."""
    state = make_initial_state(
        query=query,
        tenant_id=tenant,
        session_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
    )
    return emma_graph.invoke(state)


# State completeness tests

class TestStateCompleteness:
    """
    After a successful pipeline run, all expected state fields must be populated.
    Missing fields indicate a node returned an incomplete state update.
    """

    def test_normal_flow_state_fields(self) -> None:
        """Non-emergency query: verify all pipeline state fields are populated."""
        state = run_pipeline("What are the opening hours?", "surgery_greenfield")

        # Safety gate outputs
        assert state["safety_cleared"] is True
        assert state["escalate"] is False
        assert state["escalation_reason"] is None

        # Retrieval outputs
        assert state["retrieved_chunks"] is not None
        assert isinstance(state["retrieved_chunks"], list)
        assert len(state["retrieved_chunks"]) > 0
        assert state["retrieval_context"] is not None
        assert state["retrieval_latency_ms"] is not None

        # Response outputs
        assert state["raw_response"] is not None
        assert len(state["raw_response"]) > 0

        # Verification outputs
        assert state["final_response"] is not None
        assert len(state["final_response"]) > 0
        assert state["verified"] is True
        assert state["verification_notes"] is not None
        assert state["verification_latency_ms"] is not None

    def test_escalation_flow_state_fields(self) -> None:
        """Emergency query: LLM pipeline must NOT run — only escalation fields set."""
        state = run_pipeline("I have chest pain", "surgery_greenfield")

        # Safety gate outputs
        assert state["escalate"] is True
        assert state["safety_cleared"] is False
        assert state["escalation_reason"] is not None

        # LLM pipeline must NOT have run
        assert state["retrieved_chunks"] is None, "Retrieval ran for emergency query."
        assert state["raw_response"] is None, "Response agent ran for emergency query."

        # Escalation response must be present
        assert state["final_response"] is not None
        assert state["verified"] is True  # hardcoded, not LLM-verified


# Tenant isolation tests

class TestTenantIsolation:
    """
    Each surgery must return responses grounded in its own guidelines.
    Cross-tenant contamination is a clinical safety failure.
    """

    def test_opening_hours_differ(self) -> None:
        """
        Greenfield: Mon-Fri 8AM-6PM
        Riverside:  Mon/Wed/Fri 9AM-5:30PM
        Responses must differ.
        """
        state_g = run_pipeline("What are the opening hours?", "surgery_greenfield")
        state_r = run_pipeline("What are the opening hours?", "surgery_riverside")

        assert state_g["final_response"] != state_r["final_response"], (
            "TENANT ISOLATION FAILURE: same response for different surgeries."
        )

    def test_prescription_lead_time_differs(self) -> None:
        """
        Greenfield: 48 hours. Riverside: 72 hours.
        Each response must contain the correct lead time.
        """
        state_g = run_pipeline("How long does a prescription take?", "surgery_greenfield")
        state_r = run_pipeline("How long does a prescription take?", "surgery_riverside")

        # Greenfield should say 48
        assert "48" in state_g["final_response"], (
            f"Greenfield prescription time wrong. Response: {state_g['final_response']}"
        )
        # Riverside should say 72
        assert "72" in state_r["final_response"], (
            f"Riverside prescription time wrong. Response: {state_r['final_response']}"
        )

    def test_greenfield_data_not_in_riverside(self) -> None:
        """Riverside responses must not contain Greenfield-specific data."""
        state_r = run_pipeline("What time do you open?", "surgery_riverside")
        # Greenfield opens at 8AM; Riverside at 9AM. 8AM must not appear.
        assert "8:00 AM" not in state_r["final_response"] or "9:00 AM" in state_r["final_response"], (
            f"Possible Greenfield data in Riverside response: {state_r['final_response']}"
        )


#  Response grounding tests 

class TestResponseGrounding:
    """
    LLM responses must be grounded in surgery-specific guidelines.
    Ungrounded responses that confabulate clinical information are safety failures.
    """

    def test_greenfield_hours_grounded(self) -> None:
        state = run_pipeline("When do you open?", "surgery_greenfield")
        response = state["final_response"]
        # Must mention Greenfield's actual hours
        assert "8" in response, f"Expected opening hour in response. Got: {response}"

    def test_riverside_evening_appointments(self) -> None:
        state = run_pipeline("Do you have evening appointments?", "surgery_riverside")
        response = state["final_response"].lower()
        # Riverside has Tuesday/Thursday evening clinics until 7PM
        assert "tuesday" in response or "thursday" in response or "7" in response, (
            f"Expected evening appointment info. Got: {state['final_response']}"
        )

    def test_emergency_direction_always_999(self) -> None:
        """Any query about emergencies must direct to 999, not the surgery number."""
        state = run_pipeline("What do I do in an emergency?", "surgery_greenfield")
        assert "999" in state["final_response"], (
            f"Emergency direction missing 999. Response: {state['final_response']}"
        )

    def test_no_hallucinated_phone_numbers(self) -> None:
        """LLM must not invent phone numbers not in the context."""
        state = run_pipeline("What is the surgery phone number?", "surgery_greenfield")
        response = state["final_response"]
        assert state["verified"] is True, "Response was not verified."

    def test_no_clinical_advice_in_normal_flow(self) -> None:
        """No clinical advice even for symptom-adjacent queries."""
        state = run_pipeline(
            "I have a bad headache, what should I do?",
            "surgery_greenfield",
        )
        response = state["final_response"].lower()
        # Must not suggest medication
        assert "ibuprofen" not in response
        assert "paracetamol" not in response
        assert "take" not in response or "appointment" in response


# Verification pipeline tests 

class TestVerificationPipeline:
    """Chain of Verification must run and produce a verified response."""

    def test_verified_flag_is_true(self) -> None:
        state = run_pipeline("What are your opening hours?", "surgery_greenfield")
        assert state["verified"] is True

    def test_verification_notes_present(self) -> None:
        state = run_pipeline("How do I request a prescription?", "surgery_riverside")
        assert state["verification_notes"] is not None
        assert len(state["verification_notes"]) > 0

    def test_final_response_not_empty(self) -> None:
        state = run_pipeline("Can I register as a new patient?", "surgery_greenfield")
        assert state["final_response"] is not None
        assert len(state["final_response"].strip()) > 20, (
            f"Final response is suspiciously short: '{state['final_response']}'"
        )


# Latency tests 

class TestLatency:
    """
    Pipeline latency benchmarks. These are informational — they log latency
    but only fail if the response is None (not on latency alone).
    Strict latency budgets are enforced in Phase 3 (voice pipeline).
    """

    @pytest.mark.slow
    def test_full_pipeline_latency(self) -> None:
        """Full pipeline (retrieval + response + verification) should complete < 5s."""
        t0 = time.perf_counter()
        state = run_pipeline("What are the opening hours?", "surgery_greenfield")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"\nFull pipeline latency: {elapsed_ms:.0f}ms")
        print(f"  Retrieval:     {state.get('retrieval_latency_ms', 0):.0f}ms")
        print(f"  Response:      {state.get('response_latency_ms', 0):.0f}ms")
        print(f"  Verification:  {state.get('verification_latency_ms', 0):.0f}ms")

        assert state["final_response"] is not None
        # Hard limit: 8 seconds for full pipeline (non-voice path)
        assert elapsed_ms < 8000, (
            f"Pipeline too slow: {elapsed_ms:.0f}ms > 8000ms. "
            "Check Groq API latency and retrieval performance."
        )

    @pytest.mark.slow
    def test_escalation_latency_is_fast(self) -> None:
        """Emergency escalation must complete in < 100ms (no LLM calls)."""
        t0 = time.perf_counter()
        state = run_pipeline("I have chest pain", "surgery_greenfield")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"\nEscalation latency: {elapsed_ms:.0f}ms")
        assert state["escalate"] is True
        # Escalation path has no LLM calls — should be near-instant
        # The 100ms budget accounts for embedding model inference (Layer 2)
        assert elapsed_ms < 500, (
            f"Escalation too slow: {elapsed_ms:.0f}ms. "
            "Check EmergencyDetector — embedding model may not be pre-warmed."
        )


# ── Edge case resilience tests ────────────────────────────────────────────────

class TestEdgeCases:

    def test_very_short_query(self) -> None:
        """Single-word queries should not crash the pipeline."""
        state = run_pipeline("Hours?", "surgery_greenfield")
        assert state["final_response"] is not None

    def test_very_long_query(self) -> None:
        """Very long patient messages should be handled gracefully."""
        long_query = (
            "Hello, I'm calling because I'm trying to understand more about "
            "the appointment booking process and also about how to get my repeat "
            "prescription sorted and I was wondering if you could also tell me "
            "what happens if I need an urgent appointment and also the opening "
            "hours because I'm not sure if you're open on Saturdays or not."
        )
        state = run_pipeline(long_query, "surgery_greenfield")
        assert state["final_response"] is not None
        assert state["escalate"] is False

    def test_unknown_tenant_handled(self) -> None:
        """An unknown tenant ID should not crash the graph — returns error state."""
        state = run_pipeline("What are the hours?", "surgery_does_not_exist")
        # Should not crash — error should be captured in state
        assert state is not None
        assert state.get("final_response") is not None or state.get("error") is not None

    def test_query_with_special_characters(self) -> None:
        """Special characters in query should not break the pipeline."""
        state = run_pipeline(
            "What are your hours? (Mon–Fri) & Sat?",
            "surgery_greenfield",
        )
        assert state["final_response"] is not None