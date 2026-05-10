import logging
from typing import List, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import CrossEncoder

from backend.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    RERANK_TOP_K,
    TOP_K_RETRIEVAL,
    get_qdrant_collection_name,
)
from backend.rag.embedder import EmbeddingModel

logger = logging.getLogger(__name__)


class SurgeryRetriever:
    """
    Retrieves relevant GP surgery guidelines for a given query and tenant.
    
    Two-stage retrieval with reranker-gated safety:
      Stage 1: Dense vector search — fetches candidates broadly (no threshold)
      Stage 2: Cross-encoder reranking — precise relevance scoring (safety gate)
    
    Clinical safety rationale:
      The embedding model (bi-encoder) is coarse-grained. It can miss relevant
      chunks when query wording differs from document wording (e.g., "can I book
      online" vs "Online booking: Available..."). The cross-encoder sees both
      texts jointly and catches these semantic relationships. Therefore, the
      cross-encoder — not the embedding score — is the authoritative relevance
      gate. This prevents false negatives (missing correct context) while
      maintaining protection against false positives (irrelevant context).
    """

    _RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    
    _RERANK_SCORE_THRESHOLD = -3.0

    def __init__(
        self,
        top_k_retrieval: int = TOP_K_RETRIEVAL,
        rerank_top_k: int = RERANK_TOP_K,
    ) -> None:
        self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._embedder = EmbeddingModel()
        logger.info("Loading cross-encoder reranker: %s", self._RERANKER_MODEL)
        self._reranker = CrossEncoder(self._RERANKER_MODEL)
        self.top_k_retrieval = top_k_retrieval
        self.rerank_top_k = rerank_top_k
        logger.info(
            "SurgeryRetriever ready | top_k=%d rerank_k=%d rerank_threshold=%.2f",
            top_k_retrieval, rerank_top_k, self._RERANK_SCORE_THRESHOLD,
        )

    def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Full two-stage retrieval with reranker-gated safety filter.
        """
        if not query or not query.strip():
            raise ValueError("retrieve() received an empty query.")
        if not tenant_id or not tenant_id.strip():
            raise ValueError("retrieve() received an empty tenant_id.")

        final_k = top_k or self.rerank_top_k
        collection_name = get_qdrant_collection_name(tenant_id)

        # Stage 1: Dense vector retrieval

        candidates = self._dense_search(query, collection_name)
        if not candidates:
            logger.warning(
                "No candidates at all for query '%s' in '%s'. "
                "Collection may be empty or not ingested.",
                query, tenant_id,
            )
            return []

        # Stage 2: Cross-encoder reranking 
        reranked = self._rerank(query, candidates)
        
        # SAFETY FILTER: Only keep results the cross-encoder approves
        safe_results = [
            c for c in reranked 
            if c.get("rerank_score", float('-inf')) >= self._RERANK_SCORE_THRESHOLD
        ]
        
        if not safe_results:
            logger.warning(
                "Cross-encoder rejected all %d candidates for query '%s' in '%s'. "
                "No safe context available. Reranker scores: %s",
                len(reranked), query, tenant_id,
                [round(c.get("rerank_score", 0), 2) for c in reranked[:5]],
            )
            return []

        result = safe_results[:final_k]
        logger.info(
            "Retrieved %d SAFE chunks for tenant '%s' (query: '%.60s...') | "
            "top rerank score: %.2f",
            len(result), tenant_id, query, 
            result[0].get("rerank_score", 0) if result else 0,
        )
        return result

    def format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into a numbered context block for the LLM.
        """
        if not chunks:
            return (
                "No relevant information found in the surgery guidelines for "
                "this query. Acknowledge the gap honestly to the patient."
            )

        parts = []
        for i, chunk in enumerate(chunks, start=1):
            score_info = ""
            if "rerank_score" in chunk:
                score_info = f" (relevance: {chunk['rerank_score']:.2f})"
            parts.append(f"[Source {i}{score_info}]\n{chunk['text']}")

        return "\n\n".join(parts)

    def _dense_search(
        self, query: str, collection_name: str
    ) -> List[Dict]:
        """
        Stage 1: Embed query and search Qdrant — NO SCORE THRESHOLD.
        
        Fetches top_k candidates by cosine similarity without filtering.
        The cross-encoder in Stage 2 handles quality control.
        """
        query_vector = self._embedder.embed_single(query)

        try:
            results = self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=self.top_k_retrieval,
                with_payload=True,
                with_vectors=False,
            )
        except UnexpectedResponse as e:
            logger.error(
                "Qdrant search failed for collection '%s': %s. "
                "Has the tenant been ingested? Run scripts/ingest_all.py.",
                collection_name, e,
            )
            raise

        return [
            {
                "text": r.payload["text"],
                "dense_score": r.score,
                "tenant_id": r.payload.get("tenant_id", ""),
                "source": r.payload.get("source", ""),
                "chunk_index": r.payload.get("chunk_index", -1),
            }
            for r in results.points
        ]

    def _rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Stage 2: Cross-encoder reranking with safety threshold.
        
        The cross-encoder scores (query, chunk) pairs jointly, capturing
        nuanced relevance that the embedding model misses.
        """
        if len(candidates) == 1:
            # Single candidate: run through reranker anyway for consistent scoring
            score = float(self._reranker.predict([[query, candidates[0]["text"]]])[0])
            candidates[0]["rerank_score"] = score
            return candidates

        pairs = [[query, c["text"]] for c in candidates]
        scores = self._reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.debug(
            "Reranked %d candidates. Top score: %.3f | Bottom score: %.3f",
            len(candidates), 
            candidates[0]["rerank_score"],
            candidates[-1]["rerank_score"],
        )
        return candidates