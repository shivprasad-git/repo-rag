from __future__ import annotations

from app.embeddings import EmbeddingProvider
from app.models import Chunk
from app.retrieval.keyword import BM25KeywordIndex
from app.vectorstore.base import VectorStore


class Retriever:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        vector_weight: float = 0.25,
        keyword_weight: float = 0.75,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    def retrieve(self, question: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        return self.retrieve_hybrid(question, top_k=top_k)

    def retrieve_hybrid(self, question: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        query_embedding = self.embedder.embed(question)
        candidate_count = max(top_k * 4, 20)
        vector_matches = self.vector_store.search(query_embedding, top_k=candidate_count)
        keyword_matches = BM25KeywordIndex(self.vector_store.all_chunks()).search(question, top_k=candidate_count)

        merged: dict[str, tuple[Chunk, float]] = {}
        for chunk, score in _normalize(vector_matches):
            _add_score(merged, chunk, score * self.vector_weight)
        for chunk, score in _normalize(keyword_matches):
            _add_score(merged, chunk, score * self.keyword_weight)

        return sorted(merged.values(), key=lambda item: item[1], reverse=True)[:top_k]


def _normalize(matches: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
    if not matches:
        return []
    max_score = max(score for _, score in matches)
    if max_score <= 0:
        return [(chunk, 0.0) for chunk, _ in matches]
    return [(chunk, score / max_score) for chunk, score in matches]


def _add_score(merged: dict[str, tuple[Chunk, float]], chunk: Chunk, score: float) -> None:
    existing = merged.get(chunk.id)
    if existing is None:
        merged[chunk.id] = (chunk, score)
    else:
        merged[chunk.id] = (existing[0], existing[1] + score)
