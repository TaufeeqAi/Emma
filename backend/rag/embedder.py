import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Singleton-friendly wrapper around sentence-transformers.

    Instantiate once per process (not per request) — the model is large
    (~80MB) and loading it is the dominant startup cost.

    Args:
        model_name: HuggingFace model identifier. Default: all-MiniLM-L6-v2.
        device:     'cpu' | 'cuda' | 'mps'. Default: 'cpu' for portability.
        cache_folder: Override HuggingFace cache directory.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_folder: Optional[str] = None,
    ) -> None:
        logger.info("Loading embedding model '%s' on device '%s'", model_name, device)
        self.model_name = model_name
        self.device = device

        kwargs: dict = {"device": device}
        if cache_folder:
            kwargs["cache_folder"] = cache_folder

        self._model = SentenceTransformer(model_name, **kwargs)
        # Dynamic dimension — works with any model, not just MiniLM
        self.dimension = self._model.get_embedding_dimension()
        logger.info("Embedding model loaded. Dimension: %d", self.dimension)

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of texts into L2-normalised vectors.
        """
        if not texts:
            raise ValueError("embed_batch() received an empty text list.")

        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        logger.debug("Embedded %d texts → shape %s", len(texts), vectors.shape)
        return vectors

    def embed_single(self, text: str) -> List[float]:
        """
        Embed a single query string. Optimised for low-latency inference path.
        """
        if not text or not text.strip():
            raise ValueError("embed_single() received an empty string.")

        vector: np.ndarray = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return vector.tolist()

    @property
    def sentence_transformer(self) -> SentenceTransformer:
        """
        Expose the underlying model for sharing with EmergencyDetector.

        This avoids loading the ~80MB model twice into RAM.
        """
        return self._model

    @property
    def info(self) -> dict:
        """Return model metadata dict for logging/observability."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.dimension,
        }