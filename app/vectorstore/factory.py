from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.vectorstore.base import VectorStore
from app.vectorstore.chroma_store import ChromaVectorStore
from app.vectorstore.simple_store import SimpleJsonVectorStore


def build_vector_store(kind: str, settings: Settings, index_name: str | None = None) -> VectorStore:
    name = index_name or settings.collection_name
    if kind == "chroma":
        return ChromaVectorStore(settings.indexes_dir / "chroma", name)
    if kind == "simple":
        return SimpleJsonVectorStore(Path(settings.indexes_dir) / f"{name}.json")
    raise ValueError(f"Unsupported vector store: {kind}")

