import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from backend.agents.escalation_handler import escalation_handler
from backend.agents.response_agent import response_agent
from backend.agents.retrieval_agent import retrieval_agent
from backend.agents.safety_agent import safety_gate_agent
from backend.agents.state import AgentState
from backend.agents.verification_agent import verification_agent

logger = logging.getLogger(__name__)


# Routing function 

def route_after_safety(
    state: AgentState,
) -> Literal["escalation", "retrieval"]:
    """
    Conditional edge router: called after safety_gate completes.

    Returns the name of the next node based on the safety decision.
    This function is pure — no side effects, no LLM calls.

    Args:
        state: Current AgentState with escalate field set by safety_gate.

    Returns:
        "escalation" if escalate=True, "retrieval" otherwise.
    """
    if state.get("escalate", False):
        logger.info(
            "route_after_safety → escalation | session=%s",
            state.get("session_id", "?"),
        )
        return "escalation"

    logger.info(
        "route_after_safety → retrieval | session=%s",
        state.get("session_id", "?"),
    )
    return "retrieval"


# Graph factory 

def build_emma_graph():
    """
    Build and compile the EMMA LangGraph StateGraph.

    Returns:
        Compiled LangGraph application (CompiledGraph).
        Call .invoke(state) for synchronous execution.
        Call .ainvoke(state) for async (Phase 3 WebSocket handler).

    Raises:
        ImportError: If langgraph is not installed.
        ValueError: If node registration or edge configuration is invalid.

    Usage:
        graph = build_emma_graph()
        result = graph.invoke(make_initial_state("opening hours?", "surgery_greenfield"))
        print(result["final_response"])
    """
    logger.info("Building EMMA LangGraph pipeline...")

    graph = StateGraph(AgentState)

    # Register nodes 
    graph.add_node("safety_gate", safety_gate_agent)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("response", response_agent)
    graph.add_node("verification", verification_agent)
    graph.add_node("escalation", escalation_handler)

    # Entry point: safety_gate always runs first 
    graph.set_entry_point("safety_gate")

    #  Conditional routing after safety_gate 
    graph.add_conditional_edges(
        source="safety_gate",
        path=route_after_safety,
        path_map={
            "escalation": "escalation",
            "retrieval": "retrieval",
        },
    )

    #  Normal (non-emergency) flow: linear pipeline 
    graph.add_edge("retrieval", "response")
    graph.add_edge("response", "verification")
    graph.add_edge("verification", END)

    # Emergency flow: escalation → END (no further processing) 
    graph.add_edge("escalation", END)

    compiled = graph.compile()
    logger.info("EMMA LangGraph pipeline compiled successfully.")
    return compiled


#  Production singleton 
# Loaded once at process start. Importing this module triggers graph compilation.

emma_graph = build_emma_graph()