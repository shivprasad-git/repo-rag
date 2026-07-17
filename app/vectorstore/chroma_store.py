from __future__ import annotations

from pathlib import Path

from app.models import Chunk
from app.vectorstore.base import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install chromadb or run with --store simple.") from exc

        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[Chunk, float]]:
        result = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        matches: list[tuple[Chunk, float]] = []

        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            score = 1.0 / (1.0 + float(distance))
            matches.append((Chunk(id=chunk_id, content=content, metadata=dict(metadata)), score))

        return matches

    def all_chunks(self) -> list[Chunk]:
        result = self.collection.get(include=["documents", "metadatas"])
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        return [
            Chunk(id=chunk_id, content=content, metadata=dict(metadata))
            for chunk_id, content, metadata in zip(ids, documents, metadatas)
        ]
