import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_folder: str = None,
    ) -> None:
        logger.info("Loading embedding model '%s' on device '%s'", model_name, device)
        self.model_name = model_name
        self.device = device
        self.dimension = 384  

        kwargs = {"device": device}
        if cache_folder:
            kwargs["cache_folder"] = cache_folder

        self._model = SentenceTransformer(model_name, **kwargs)
        logger.info("Embedding model loaded. Dimension: %d", self.dimension)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of texts into L2-normalised vectors.

        Args:
            texts:         List of strings to embed.
            batch_size:    Texts processed per forward pass. Higher = faster
                           if RAM allows; 32 is safe for 8GB machines.
            show_progress: Show tqdm progress bar (useful for large ingestion).

        Returns:
            np.ndarray of shape (len(texts), 384), float32, L2-normalised.

        Raises:
            ValueError: if texts list is empty.
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

        Returns:
            Flat Python list of 384 floats (Qdrant-compatible format).
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
    def info(self) -> dict:
        """Return model metadata dict for logging/observability."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.dimension,
        }