from __future__ import annotations

from app.embeddings import EmbeddingProvider
from app.models import Chunk
from app.vectorstore.base import VectorStore


class Retriever:
    def __init__(self, embedder: EmbeddingProvider, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, question: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        query_embedding = self.embedder.embed(question)
        return self.vector_store.search(query_embedding, top_k=top_k)

