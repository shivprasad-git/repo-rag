from __future__ import annotations

import json
from pathlib import Path

from app.models import Chunk
from app.retrieval.filters import MetadataFilters, filter_chunks


class SimpleJsonChunkStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, chunks: list[Chunk]) -> None:
        by_id = {chunk.id: chunk for chunk in self.all_chunks()}
        for chunk in chunks:
            by_id[chunk.id] = chunk

        records = [
            {"id": chunk.id, "content": chunk.content, "metadata": chunk.metadata}
            for chunk in by_id.values()
        ]
        self.path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def get(self, chunk_id: str) -> Chunk | None:
        for chunk in self.all_chunks():
            if chunk.id == chunk_id:
                return chunk
        return None

    def all_chunks(self, filters: MetadataFilters | None = None) -> list[Chunk]:
        if not self.path.exists():
            return []

        chunks = [
            Chunk(id=record["id"], content=record["content"], metadata=record["metadata"])
            for record in json.loads(self.path.read_text(encoding="utf-8"))
        ]
        return filter_chunks(chunks, filters)

    def has_data(self) -> bool:
        if not self.path.exists():
            return False
        records = json.loads(self.path.read_text(encoding="utf-8"))
        return len(records) > 0


def build_chunk_store(indexes_dir: Path, index_name: str) -> SimpleJsonChunkStore:
    return SimpleJsonChunkStore(indexes_dir / "chunks" / f"{index_name}.json")
