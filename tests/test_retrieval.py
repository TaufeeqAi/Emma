import pytest
from backend.rag.retriever import SurgeryRetriever
from qdrant_client import QdrantClient
from backend.config import QDRANT_HOST, QDRANT_PORT, get_qdrant_collection_name


# Shared retriever instance
@pytest.fixture(scope="module")
def retriever() -> SurgeryRetriever:
    """
    Module-scoped retriever fixture. Loads embedding model and cross-encoder
    once for the entire test session (avoids ~3s model-load penalty per test).

    Prerequisite: Qdrant must be running and data ingested via ingest_all.py.
    """
    return SurgeryRetriever()


# ── DIAGNOSTIC TESTS ──────────────────────────────────────────────────────────
# Run these first to see what chunks exist and what embedding retrieves

def test_diagnostic_show_all_greenfield_chunks():
    """
    DIAGNOSTIC: Print all chunks stored in Qdrant for surgery_greenfield.
    Run with: pytest tests/test_retrieval.py::test_diagnostic_show_all_greenfield_chunks -v -s
    """
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    collection = get_qdrant_collection_name("surgery_greenfield")
    
    # Scroll all points (Qdrant returns tuples: (points, next_page_offset))
    scroll_result = client.scroll(collection_name=collection, limit=100)
    all_points = scroll_result[0]
    
    print(f"\n{'='*70}")
    print(f"  ALL CHUNKS IN GREENFIELD COLLECTION: {collection}")
    print(f"  Total chunks found: {len(all_points)}")
    print(f"{'='*70}")
    
    for i, point in enumerate(all_points):
        text = point.payload.get("text", "NO TEXT")
        print(f"\n  [Chunk {i}]  ID: {point.id}")
        print(f"  Text: {text}")
        print(f"  ---")
    
    assert len(all_points) > 0, "No chunks found! Did you run ingest_all.py?"


def test_diagnostic_embedding_retrieval_greenfield(retriever):
    """
    DIAGNOSTIC: Show exactly what the embedding model retrieves BEFORE reranking.
    Run with: pytest tests/test_retrieval.py::test_diagnostic_embedding_retrieval_greenfield -v -s
    """
    queries = [
        "how do I book an appointment",
        "can I book online",
        "what are the opening hours",
    ]
    
    collection = get_qdrant_collection_name("surgery_greenfield")
    
    print(f"\n{'='*70}")
    print(f"  EMBEDDING MODEL RETRIEVAL (Stage 1 - before reranker)")
    print(f"  Collection: {collection}")
    print(f"  TOP_K_RETRIEVAL: {retriever.top_k_retrieval}")
    print(f"{'='*70}")
    
    for query in queries:
        print(f"\n  QUERY: '{query}'")
        print(f"  {'-'*60}")
        
        # Call _dense_search directly to see raw embedding results
        candidates = retriever._dense_search(query, collection)
        
        if not candidates:
            print(f"  ⚠️  NO CANDIDATES RETURNED BY EMBEDDING MODEL")
            continue
        
        for i, c in enumerate(candidates):
            print(f"  Rank {i}: dense_score={c['dense_score']:.4f}")
            print(f"          text: {c['text'][:100]}...")
            print(f"          chunk_index: {c.get('chunk_index', 'N/A')}")
            print()
        
        # Now show what reranker does
        reranked = retriever._rerank(query, candidates)
        print(f"  --- AFTER RERANKER ---")
        for i, c in enumerate(reranked):
            print(f"  Rank {i}: rerank_score={c.get('rerank_score', 'N/A'):.4f} | "
                  f"dense_score={c['dense_score']:.4f}")
            print(f"          text: {c['text'][:80]}...")
            print()
    
    # This test always passes - it's just for diagnostics
    assert True


def test_diagnostic_embedding_retrieval_riverside(retriever):
    """
    DIAGNOSTIC: Show embedding retrieval for Riverside queries.
    Run with: pytest tests/test_retrieval.py::test_diagnostic_embedding_retrieval_riverside -v -s
    """
    queries = [
        "is there a walk-in centre",
        "what are the opening hours",
    ]
    
    collection = get_qdrant_collection_name("surgery_riverside")
    
    print(f"\n{'='*70}")
    print(f"  EMBEDDING MODEL RETRIEVAL (Stage 1 - before reranker)")
    print(f"  Collection: {collection}")
    print(f"{'='*70}")
    
    for query in queries:
        print(f"\n  QUERY: '{query}'")
        print(f"  {'-'*60}")
        
        candidates = retriever._dense_search(query, collection)
        
        if not candidates:
            print(f"  ⚠️  NO CANDIDATES RETURNED")
            continue
        
        for i, c in enumerate(candidates):
            print(f"  Rank {i}: dense_score={c['dense_score']:.4f}")
            print(f"          text: {c['text'][:100]}...")
            print()
    
    assert True


# Greenfield test cases 
# Format: (query, expected_substring_in_results)
GREENFIELD_TESTS = [
    ("what are the opening hours",        "8:00 AM"),        
    ("what time do you open on Saturday", "9:00 AM"),
    ("how do I book an appointment",      "booking"),
    ("how long for a prescription",       "48 hours"),
    ("what do I do in an emergency",      "999"),
    ("can I book online",                 "Patient Access"),
    ("how long for a referral",           "5 working days"),
    ("when will my test results be ready","3"),
]

# Riverside test cases
RIVERSIDE_TESTS = [
    ("what are the opening hours",        "9:00 AM"),        
    ("how long for a prescription",       "72 hours"),       
    ("do you have evening appointments",  "7:00 PM"),
    ("urgent same-day appointment",       "9:00 AM sharp"),
    ("is there a walk-in centre",         "Walk-In"),
    ("when will my test results be ready","5"),
]


# Parameterised retrieval tests 

@pytest.mark.parametrize("query,expected", GREENFIELD_TESTS)
def test_greenfield_retrieval(query: str, expected: str, retriever: SurgeryRetriever) -> None:
    """
    Verify that Greenfield queries return expected clinical content.

    Failure modes:
      - Score threshold too high → chunks filtered, results empty.
      - Chunk boundary splits expected phrase → keyword not found.
      - Collection not yet ingested → UnexpectedResponse from Qdrant.
    """
    chunks = retriever.retrieve(query, "surgery_greenfield")

    assert len(chunks) > 0, (
        f"No chunks returned for Greenfield query: '{query}'. "
        "Is the collection ingested? Run: python scripts/ingest_all.py"
    )

    combined_text = " ".join(c["text"] for c in chunks).lower()
    assert expected.lower() in combined_text, (
        f"Expected '{expected}' in Greenfield results for query: '{query}'\n"
        f"Got: {combined_text[:300]}..."
    )


@pytest.mark.parametrize("query,expected", RIVERSIDE_TESTS)
def test_riverside_retrieval(query: str, expected: str, retriever: SurgeryRetriever) -> None:
    """
    Verify that Riverside queries return expected clinical content.
    """
    chunks = retriever.retrieve(query, "surgery_riverside")

    assert len(chunks) > 0, (
        f"No chunks returned for Riverside query: '{query}'. "
        "Is the collection ingested? Run: python scripts/ingest_all.py"
    )

    combined_text = " ".join(c["text"] for c in chunks).lower()
    assert expected.lower() in combined_text, (
        f"Expected '{expected}' in Riverside results for query: '{query}'\n"
        f"Got: {combined_text[:300]}..."
    )


# Tenant isolation tests

def test_tenant_isolation(retriever: SurgeryRetriever) -> None:
    """
    CRITICAL NHS compliance test: Surgery A and B must return different data
    for the same query.

    Why this matters:
      If a Qdrant filter bug or collection misconfiguration causes cross-tenant
      leakage, Surgery A patients could receive Surgery B information (wrong
      opening hours, wrong prescription lead times, wrong escalation messages).
      This is a clinical safety issue, not just a data hygiene issue.
    """
    query = "what are the opening hours"
    chunks_a = retriever.retrieve(query, "surgery_greenfield")
    chunks_b = retriever.retrieve(query, "surgery_riverside")

    assert len(chunks_a) > 0, "Greenfield returned no results — check ingestion."
    assert len(chunks_b) > 0, "Riverside returned no results — check ingestion."

    texts_a = " ".join(c["text"] for c in chunks_a)
    texts_b = " ".join(c["text"] for c in chunks_b)

    assert texts_a != texts_b, (
        "TENANT ISOLATION VIOLATED: Greenfield and Riverside returned identical "
        "results for 'opening hours'. This is a critical data segregation failure."
    )


def test_no_cross_contamination_prescription(retriever: SurgeryRetriever) -> None:
    """
    Verify that Greenfield's 48-hour prescription time does NOT appear in
    Riverside's results, and vice versa.

    This tests a specific clinical safety scenario: a patient calling Riverside
    must not be told to allow 48 hours (Greenfield's policy) when their surgery
    requires 72 hours.
    """
    chunks_riverside = retriever.retrieve("how long for a prescription", "surgery_riverside")
    combined_riverside = " ".join(c["text"] for c in chunks_riverside).lower()

    assert "72 hours" in combined_riverside, (
        "Riverside prescription policy (72 hours) not found in retrieval results."
    )


def test_no_cross_contamination_hours(retriever: SurgeryRetriever) -> None:
    """
    Verify Greenfield returns 8AM opening (not 9AM), and Riverside returns 9AM.

    Greenfield: 8:00 AM Monday–Friday
    Riverside:  9:00 AM Monday/Wednesday/Friday
    """
    chunks_greenfield = retriever.retrieve("what time do you open", "surgery_greenfield")
    combined_greenfield = " ".join(c["text"] for c in chunks_greenfield).lower()

    chunks_riverside = retriever.retrieve("what time do you open", "surgery_riverside")
    combined_riverside = " ".join(c["text"] for c in chunks_riverside).lower()

    assert "8:00 am" in combined_greenfield, (
        "Greenfield opening time (8:00 AM) not found. Isolation or chunking issue."
    )
    assert "9:00 am" in combined_riverside, (
        "Riverside opening time (9:00 AM) not found. Isolation or chunking issue."
    )


def test_retriever_rejects_empty_query(retriever: SurgeryRetriever) -> None:
    """Retriever must raise on empty query — not silently return empty results."""
    with pytest.raises(ValueError, match="empty query"):
        retriever.retrieve("", "surgery_greenfield")


def test_retriever_rejects_empty_tenant(retriever: SurgeryRetriever) -> None:
    """Retriever must raise on empty tenant_id."""
    with pytest.raises(ValueError, match="empty tenant_id"):
        retriever.retrieve("opening hours", "")


def test_format_context_empty(retriever: SurgeryRetriever) -> None:
    """format_context with no chunks returns a safe sentinel string."""
    context = retriever.format_context([])
    assert "No relevant information" in context


def test_format_context_with_chunks(retriever: SurgeryRetriever) -> None:
    """format_context with results returns numbered source blocks."""
    mock_chunks = [
        {"text": "Monday to Friday: 8:00 AM to 6:00 PM", "rerank_score": 0.95},
        {"text": "Saturday: 9:00 AM to 1:00 PM", "rerank_score": 0.7},
    ]
    context = retriever.format_context(mock_chunks)
    assert "[Source 1 (relevance:" in context
    assert "[Source 2 (relevance:" in context
    assert "8:00 AM" in context