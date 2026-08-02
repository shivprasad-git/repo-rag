from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.logging_config import get_logger
from app.models import Chunk
from app.retrieval.filters import MetadataFilters, filter_chunks
from app.vectorstore.base import VectorStore, minimal_vector_metadata

logger = get_logger(__name__)


class SimpleJsonVectorStore(VectorStore):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] | None = None
        self._loaded_mtime: float | None = None

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        records = self._load_records()
        by_id = {record["id"]: record for record in records}
        for chunk, embedding in zip(chunks, embeddings):
            by_id[chunk.id] = {
                "id": chunk.id,
                "metadata": minimal_vector_metadata(chunk),
                "embedding": embedding,
            }
        self.path.write_text(json.dumps(list(by_id.values()), indent=2), encoding="utf-8")
        self._records = list(by_id.values())
        self._loaded_mtime = self.path.stat().st_mtime
        logger.debug("Wrote %d vector records to %s", len(self._records), self.path)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: MetadataFilters | None = None,
    ) -> list[tuple[Chunk, float]]:
        records = self._filtered_records(filters)
        if not records:
            logger.debug("No vector records to search")
            return []

        query = _to_array(query_embedding)
        matrix = _records_to_matrix(records)
        scores = matrix @ query

        scored: list[tuple[Chunk, float]] = []
        for record, score in zip(records, scores.tolist()):
            chunk = Chunk(id=record["id"], content="", metadata=record["metadata"])
            scored.append((chunk, float(score)))
        matches = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
        logger.debug("Vector search over %d records returned %d matches", len(records), len(matches))
        return matches

    def all_chunks(self, filters: MetadataFilters | None = None) -> list[Chunk]:
        return [
            Chunk(id=record["id"], content="", metadata=record["metadata"])
            for record in self._filtered_records(filters)
        ]

    def has_data(self) -> bool:
        return bool(self._load_records())

    def _load_records(self) -> list[dict[str, Any]]:
        """Load (or return cached) vector records.

        The in-memory list is invalidated when the file mtime changes, so
        repeated reads avoid re-parsing the JSON file on every call.
        """
        mtime = self.path.stat().st_mtime if self.path.exists() else None
        if self._records is not None and mtime == self._loaded_mtime:
            return self._records

        if not self.path.exists():
            self._records = []
            self._loaded_mtime = None
            return self._records

        self._records = json.loads(self.path.read_text(encoding="utf-8"))
        self._loaded_mtime = mtime
        logger.debug("Loaded %d vector records from %s", len(self._records), self.path)
        return self._records

    def _filtered_records(self, filters: MetadataFilters | None) -> list[dict[str, Any]]:
        records = self._load_records()
        if filters is None or not filters.has_filters:
            return records
        chunks = [
            Chunk(id=record["id"], content="", metadata=record["metadata"])
            for record in records
        ]
        matching_ids = {chunk.id for chunk in filter_chunks(chunks, filters)}
        return [record for record in records if record["id"] in matching_ids]


def _to_array(embedding: list[float]) -> Any:
    import numpy as np

    return np.asarray(embedding, dtype=np.float32)


def _records_to_matrix(records: list[dict[str, Any]]) -> Any:
    import numpy as np

    embeddings = [record["embedding"] for record in records]
    matrix = np.asarray(embeddings, dtype=np.float32)
    # Embeddings are L2-normalized at creation time in
    # SentenceTransformerEmbeddingProvider, so cosine similarity == dot product.
    return matrix
