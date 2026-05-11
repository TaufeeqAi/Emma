import uuid
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict


class AgentState(TypedDict):
    # Input (caller must provide) 
    query: str         
    tenant_id: str      
    session_id: str     
    trace_id: str       

    # Safety gate outputs 
    safety_cleared: bool            
    escalate: bool                  
    escalation_reason: Optional[str]  

    # Retrieval outputs 
    retrieved_chunks: Optional[list[dict]]  
    retrieval_context: Optional[str]        
    retrieval_latency_ms: Optional[float]    

    #  Response outputs 
    raw_response: Optional[str]   
    response_latency_ms: Optional[float]    

    #  Verification outputs 
    final_response: Optional[str]      
    verified: bool                    
    verification_notes: Optional[str]  
    verification_latency_ms: Optional[float]

    # Error handling 
    error: Optional[str]   
                          

    # Timing and metadata 
    timestamp: str          
    total_latency_ms: Optional[float]  

def make_initial_state(
    query: str,
    tenant_id: str,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> AgentState:
    """
    Factory function: constructs a fully-initialised AgentState.

    Use this instead of manually building the dict — it guarantees all
    required keys are present (LangGraph raises KeyError on missing keys
    even for Optional fields if they're entirely absent from the TypedDict).

    Args:
        query:      Patient utterance to process.
        tenant_id:  Surgery identifier (must match a Qdrant collection).
        session_id: Unique call session ID. Auto-generated if not provided.
        trace_id:   Unique graph invocation ID. Auto-generated if not provided.

    Returns:
        AgentState with all fields initialised to safe defaults.

    Example:
        state = make_initial_state("what are the opening hours?", "surgery_greenfield")
        result = emma_graph.invoke(state)
        print(result["final_response"])
    """
    return AgentState(
        query=query,
        tenant_id=tenant_id,
        session_id=session_id or str(uuid.uuid4()),
        trace_id=trace_id or str(uuid.uuid4()),
        safety_cleared=False,
        escalate=False,
        escalation_reason=None,
        retrieved_chunks=None,
        retrieval_context=None,
        retrieval_latency_ms=None,
        raw_response=None,
        response_latency_ms=None,
        final_response=None,
        verified=False,
        verification_notes=None,
        verification_latency_ms=None,
        error=None,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        total_latency_ms=None,
    )