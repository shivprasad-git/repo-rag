from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import Chunk


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[Chunk, float]]:
        raise NotImplementedError

    @abstractmethod
    def all_chunks(self) -> list[Chunk]:
        raise NotImplementedError
