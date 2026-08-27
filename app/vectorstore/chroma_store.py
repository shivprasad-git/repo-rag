from __future__ import annotations

from pathlib import Path

from app.logging_config import get_logger
from app.models import Chunk
from app.retrieval.filters import MetadataFilters, filter_chunks
from app.vectorstore.base import VectorStore, minimal_vector_metadata

logger = get_logger(__name__)


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install chromadb or run with --store simple.") from exc

        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(collection_name)
        logger.debug("Connected to Chroma collection %s at %s", collection_name, persist_dir)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=["" for _ in chunks],
            metadatas=[minimal_vector_metadata(chunk) for chunk in chunks],
        )
        logger.debug("Upserted %d vectors into Chroma collection %s", len(chunks), self.collection.name)

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self.collection.delete(ids=chunk_ids)
        logger.debug("Deleted %d vectors from Chroma collection %s", len(chunk_ids), self.collection.name)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: MetadataFilters | None = None,
    ) -> list[tuple[Chunk, float]]:
        where = _chroma_where(filters)
        requested_results = _chroma_requested_count(filters, top_k)
        query_args = {"query_embeddings": [query_embedding], "n_results": requested_results}
        if where:
            query_args["where"] = where
        result = self.collection.query(**query_args)
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        matches: list[tuple[Chunk, float]] = []

        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            score = 1.0 / (1.0 + float(distance))
            matches.append((Chunk(id=chunk_id, content=content, metadata=dict(metadata)), score))

        if filters is None or not filters.path_prefix:
            return matches[:top_k]
        return [(chunk, score) for chunk, score in matches if filters.matches(chunk)][:top_k]

    def all_chunks(self, filters: MetadataFilters | None = None) -> list[Chunk]:
        get_args = {"include": ["documents", "metadatas"]}
        where = _chroma_where(filters)
        if where:
            get_args["where"] = where
        result = self.collection.get(**get_args)
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        chunks = [
            Chunk(id=chunk_id, content=content, metadata=dict(metadata))
            for chunk_id, content, metadata in zip(ids, documents, metadatas)
        ]
        return filter_chunks(chunks, filters)

    def has_data(self) -> bool:
        return self.collection.count() > 0


def _chroma_requested_count(filters: MetadataFilters | None, top_k: int) -> int:
    """Return how many candidate vectors to request from Chroma.

    Chroma can express ``language`` and ``chunk_type`` natively via ``where``,
    but ``path_prefix`` cannot, so it is applied as a post-query filter. When a
    ``path_prefix`` filter is present, request extra candidates so the local
    filter never silently truncates the result to fewer than ``top_k`` items.
    """
    if filters is not None and filters.path_prefix:
        return max(top_k * 4, top_k)
    if _chroma_where(filters) is not None:
        # Chroma applies the where clause server-side; top_k results suffice.
        return top_k
    return max(top_k * 4, top_k)


def _chroma_where(filters: MetadataFilters | None) -> dict | None:
    if filters is None:
        return None

    clauses = []
    if filters.language:
        clauses.append({"language": filters.language})
    if filters.chunk_type:
        clauses.append({"chunk_type": filters.chunk_type_value})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
