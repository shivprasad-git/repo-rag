from __future__ import annotations

import json
import math
from pathlib import Path

from app.models import Chunk
from app.vectorstore.base import VectorStore


class SimpleJsonVectorStore(VectorStore):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        records = self._load_records()
        by_id = {record["id"]: record for record in records}
        for chunk, embedding in zip(chunks, embeddings):
            by_id[chunk.id] = {
                "id": chunk.id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding": embedding,
            }
        self.path.write_text(json.dumps(list(by_id.values()), indent=2), encoding="utf-8")

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[Chunk, float]]:
        scored: list[tuple[Chunk, float]] = []
        for record in self._load_records():
            score = _cosine_similarity(query_embedding, record["embedding"])
            chunk = Chunk(id=record["id"], content=record["content"], metadata=record["metadata"])
            scored.append((chunk, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    def _load_records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)

