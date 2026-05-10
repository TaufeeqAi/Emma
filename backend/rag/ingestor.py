import logging
import uuid
from pathlib import Path
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)

from backend.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_DIMENSION,
    QDRANT_HOST,
    QDRANT_PORT,
    get_qdrant_collection_name,
)
from backend.rag.chunker import SemanticChunker
from backend.rag.embedder import EmbeddingModel

logger = logging.getLogger(__name__)


class SurgeryIngestor:

    def __init__(self) -> None:
        logger.info("Initialising SurgeryIngestor...")
        self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
        self._chunker = SemanticChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self._embedder = EmbeddingModel()
        logger.info("SurgeryIngestor ready.")


    def ingest_tenant(self, tenant_id: str) -> dict:
        """
        Full ingestion pipeline for one tenant.

        Args:
            tenant_id: e.g. "surgery_greenfield"

        Returns:
            Summary dict:
              {
                "tenant_id":      str,
                "collection":     str,
                "chunks_created": int,
                "vectors_stored": int,
                "chunk_stats":    dict,
              }

        Raises:
            FileNotFoundError: if guidelines.txt doesn't exist.
            ConnectionRefusedError: if Qdrant is not running.
        """
        logger.info("=== Ingesting tenant: %s ===", tenant_id)
        guidelines_path = DATA_DIR / tenant_id / "guidelines.txt"

        if not guidelines_path.exists():
            raise FileNotFoundError(
                f"guidelines.txt not found for tenant '{tenant_id}': "
                f"{guidelines_path}\n"
                f"Create the file or check that tenant_id is correct."
            )

        collection_name = get_qdrant_collection_name(tenant_id)

        # create collection
        self._create_collection(collection_name)

        # Read raw document
        raw_text = guidelines_path.read_text(encoding="utf-8")
        logger.info("Read guidelines: %d chars", len(raw_text))

        # Chunk
        chunks = self._chunker.chunk(
            raw_text,
            metadata={
                "tenant_id": tenant_id,
                "source": str(guidelines_path),
                "document_type": "surgery_guidelines",
            },
        )
        stats = self._chunker.chunk_stats(chunks)
        logger.info(
            "Chunked into %d chunks | min=%s max=%s avg=%s chars",
            stats["count"], stats["min_chars"], stats["max_chars"], stats["avg_chars"],
        )

        # Embed
        texts = [c["text"] for c in chunks]
        embeddings = self._embedder.embed_batch(texts, show_progress=True)
        logger.info("Generated %d embedding vectors (dim=%d)", len(embeddings), self._embedder.dimension)

        # Upsert to Qdrant
        points = self._build_points(chunks, embeddings, tenant_id)
        self._upsert_points(collection_name, points)

        summary = {
            "tenant_id": tenant_id,
            "collection": collection_name,
            "chunks_created": len(chunks),
            "vectors_stored": len(points),
            "chunk_stats": stats,
        }
        logger.info("✓ Ingestion complete for '%s': %s", tenant_id, summary)
        return summary

    def verify_collection(self, tenant_id: str) -> dict:
        """
        Return collection metadata for post-ingestion verification.

        Returns:
            Dict with collection name, vector count, and config info.
        """
        collection_name = get_qdrant_collection_name(tenant_id)
        try:
            info = self._client.get_collection(collection_name)
            return {
                "collection": collection_name,
                "vectors_count": info.vectors_count,
                "status": info.status,
            }
        except UnexpectedResponse as e:
            logger.error("Collection '%s' not found: %s", collection_name, e)
            return {"collection": collection_name, "error": str(e)}

    def _create_collection(self, collection_name: str) -> None:
        """
        (Re)create a Qdrant collection with COSINE distance.

        """
        logger.info("Creating collection '%s'...", collection_name)
        self._client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
                # on_disk=True 
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=100, 
            ),
        )
        logger.info("Collection '%s' created.", collection_name)

    @staticmethod
    def _build_points(
        chunks: List[dict],
        embeddings,
        tenant_id: str,
    ) -> List[PointStruct]:
        """
        Construct Qdrant PointStruct objects from chunks + embeddings.

        """
        return [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={
                    "text": chunk["text"],
                    "tenant_id": tenant_id,
                    "chunk_index": chunk["chunk_index"],
                    "char_start": chunk["char_start"],
                    "source": chunk["metadata"]["source"],
                    "document_type": chunk["metadata"]["document_type"],
                },
            )
            for chunk, emb in zip(chunks, embeddings)
        ]

    def _upsert_points(
        self,
        collection_name: str,
        points: List[PointStruct],
        batch_size: int = 100,
    ) -> None:
        """
        Upsert points in batches to avoid overwhelming Qdrant's gRPC buffer.

        For Phase 1 with ~20 chunks per surgery, batch_size=100 means a single
        upsert call. For production ingestion of thousands of documents,
        smaller batches with exponential backoff are recommended.
        """
        total = len(points)
        for i in range(0, total, batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,  
            )
            logger.debug(
                "Upserted batch %d-%d / %d",
                i + 1, min(i + batch_size, total), total,
            )
        logger.info("Stored %d vectors in '%s'", total, collection_name)