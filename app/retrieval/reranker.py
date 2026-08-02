from __future__ import annotations

from abc import ABC, abstractmethod

from app.logging_config import get_logger
from app.models import Chunk

logger = get_logger(__name__)


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        question: str,
        matches: list[tuple[Chunk, float]],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        raise NotImplementedError


class CrossEncoderMiniLMReranker(Reranker):
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers to use the MiniLM reranker.") from exc

        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        logger.debug("Loaded cross-encoder reranker %s", model_name)

    def rerank(
        self,
        question: str,
        matches: list[tuple[Chunk, float]],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        if not matches:
            return []

        pairs = [(question, _rerank_text(chunk)) for chunk, _ in matches]
        scores = self.model.predict(pairs, show_progress_bar=False)
        reranked = [
            (chunk, float(score))
            for (chunk, _), score in zip(matches, scores)
        ]
        result = sorted(reranked, key=lambda item: item[1], reverse=True)[:top_k]
        logger.debug("Reranked %d candidates -> %d results", len(matches), len(result))
        return result


def _rerank_text(chunk: Chunk) -> str:
    metadata = chunk.metadata
    header = "\n".join(
        [
            f"file: {metadata.get('file_path', '')}",
            f"symbol: {metadata.get('qualified_symbol') or metadata.get('symbol', '')}",
            f"type: {metadata.get('chunk_type', '')}",
        ]
    )
    return f"{header}\n\n{chunk.content}"
