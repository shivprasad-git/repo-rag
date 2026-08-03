from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import Chunk
from app.retrieval.filters import MetadataFilters


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: MetadataFilters | None = None,
    ) -> list[tuple[Chunk, float]]:
        raise NotImplementedError

    @abstractmethod
    def all_chunks(self, filters: MetadataFilters | None = None) -> list[Chunk]:
        raise NotImplementedError

    @abstractmethod
    def has_data(self) -> bool:
        """Return True if this store already contains indexed data."""
        raise NotImplementedError


def minimal_vector_metadata(chunk: Chunk) -> dict:
    metadata = chunk.metadata
    keys = (
        "file_path",
        "module",
        "language",
        "commit",
        "chunk_type",
        "symbol",
        "qualified_symbol",
        "start_line",
        "end_line",
        "file_metadata_id",
        "has_parse_errors",
    )
    return {key: metadata.get(key, "") for key in keys}
