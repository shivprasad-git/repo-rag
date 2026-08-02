from __future__ import annotations

from app.embeddings.base import EmbeddingProvider
from app.logging_config import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Semantic embedding provider backed by sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers to use the sentence-transformers embedding provider."
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self._dimensions = int(self.model.get_embedding_dimension())
        else:
            self._dimensions = int(self.model.get_sentence_embedding_dimension())
        logger.debug("Loaded embedding model %s (dimensions=%d)", model_name, self._dimensions)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        logger.debug("Embedded %d texts with %s", len(texts), self.model_name)
        return embeddings.astype(float).tolist()
