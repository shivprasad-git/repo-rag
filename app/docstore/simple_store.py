from __future__ import annotations

import json
from pathlib import Path

from app.models import Chunk
from app.retrieval.filters import MetadataFilters, filter_chunks


class SimpleJsonChunkStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._chunks: dict[str, Chunk] | None = None
        self._loaded_mtime: float | None = None

    def add(self, chunks: list[Chunk]) -> None:
        by_id = {chunk.id: chunk for chunk in self.all_chunks()}
        for chunk in chunks:
            by_id[chunk.id] = chunk

        records = [
            {"id": chunk.id, "content": chunk.content, "metadata": chunk.metadata}
            for chunk in by_id.values()
        ]
        self.path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        self._chunks = by_id
        self._loaded_mtime = self.path.stat().st_mtime

    def get(self, chunk_id: str) -> Chunk | None:
        return self._load_index().get(chunk_id)

    def all_chunks(self, filters: MetadataFilters | None = None) -> list[Chunk]:
        chunks = list(self._load_index().values())
        return filter_chunks(chunks, filters)

    def has_data(self) -> bool:
        return bool(self._load_index())

    def _load_index(self) -> dict[str, Chunk]:
        """Load (or return cached) chunks indexed by id.

        The in-memory dict index is invalidated when the file mtime changes,
        so repeated get() calls are O(1) while external modifications to the
        JSON file are still picked up.
        """
        mtime = self.path.stat().st_mtime if self.path.exists() else None
        if self._chunks is not None and mtime == self._loaded_mtime:
            return self._chunks

        if not self.path.exists():
            self._chunks = {}
            self._loaded_mtime = None
            return self._chunks

        records = json.loads(self.path.read_text(encoding="utf-8"))
        self._chunks = {
            record["id"]: Chunk(id=record["id"], content=record["content"], metadata=record["metadata"])
            for record in records
        }
        self._loaded_mtime = mtime
        return self._chunks


def build_chunk_store(indexes_dir: Path, index_name: str) -> SimpleJsonChunkStore:
    return SimpleJsonChunkStore(indexes_dir / "chunks" / f"{index_name}.json")
