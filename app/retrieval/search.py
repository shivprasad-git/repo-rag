from __future__ import annotations

from hashlib import md5

from app.embeddings import EmbeddingProvider
from app.config import Settings
from app.models import Chunk
from app.docstore.simple_store import SimpleJsonChunkStore
from app.ingest.searchable import searchable_chunks
from app.retrieval.filters import MetadataFilters
from app.retrieval.keyword import BM25KeywordIndex
from app.retrieval.reranker import Reranker
from app.vectorstore.base import VectorStore


class Retriever:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        settings: Settings,
        chunk_store: SimpleJsonChunkStore | None = None,
        reranker: Reranker | None = None,
        vector_weight: float = 0.25,
        keyword_weight: float = 0.75,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.settings = settings
        self.chunk_store = chunk_store
        self.reranker = reranker
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self._bm25_index: BM25KeywordIndex | None = None
        self._bm25_chunk_hash: str = ""

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        filters: MetadataFilters | None = None,
    ) -> list[tuple[Chunk, float]]:
        return self.retrieve_hybrid(question, top_k=top_k, filters=filters)

    def retrieve_hybrid(
        self,
        question: str,
        top_k: int = 5,
        filters: MetadataFilters | None = None,
    ) -> list[tuple[Chunk, float]]:
        query_embedding = self.embedder.embed(question)
        candidate_count = max(top_k * 4, 20)
        vector_matches = self.vector_store.search(query_embedding, top_k=candidate_count, filters=filters)
        keyword_matches = self._get_bm25_index(filters).search(
            question,
            top_k=candidate_count,
        )

        merged: dict[str, tuple[Chunk, float]] = {}
        for chunk, score in _normalize(vector_matches):
            _add_score(merged, self._hydrate(chunk), score * self.vector_weight)
        for chunk, score in _normalize(keyword_matches):
            _add_score(merged, chunk, score * self.keyword_weight)

        candidates = sorted(merged.values(), key=lambda item: item[1], reverse=True)
        if self.reranker is not None:
            matches = self.reranker.rerank(question, candidates, top_k=top_k)
        else:
            matches = candidates[:top_k]
        return self._expand_context_parts(matches)

    def _get_bm25_index(self, filters: MetadataFilters | None = None) -> BM25KeywordIndex:
        """Return a cached BM25 index, rebuilding only when chunks change.

        When filters are applied, the BM25 index is always rebuilt from the
        filtered chunk subset because document frequencies change.  For the
        common unfiltered case, the index is cached and reused across queries
        until the underlying chunk store changes.
        """
        if filters is not None and filters.has_filters:
            return BM25KeywordIndex(self._all_full_chunks(filters))

        chunks = self._all_full_chunks(filters=None)
        chunk_hash = _chunk_hash(chunks)
        if self._bm25_index is None or chunk_hash != self._bm25_chunk_hash:
            self._bm25_index = BM25KeywordIndex(chunks)
            self._bm25_chunk_hash = chunk_hash
        return self._bm25_index

    def _all_full_chunks(self, filters: MetadataFilters | None) -> list[Chunk]:
        if self.chunk_store is not None:
            return searchable_chunks(self.chunk_store.all_chunks(filters=filters), self.settings, filters=filters)
        return searchable_chunks(self.vector_store.all_chunks(filters=filters), self.settings, filters=filters)

    def _hydrate(self, chunk: Chunk) -> Chunk:
        if self.chunk_store is None:
            return chunk
        return self.chunk_store.get(chunk.id) or chunk

    def _expand_context_parts(self, matches: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
        if self.chunk_store is None or self.settings.context_window_parts <= 0:
            return matches

        expanded: list[tuple[Chunk, float]] = []
        seen: set[str] = set()
        for chunk, score in matches:
            for context_chunk in self._context_parts(chunk):
                if context_chunk.id in seen:
                    continue
                seen.add(context_chunk.id)
                expanded.append((context_chunk, score))
        return expanded

    def _context_parts(self, chunk: Chunk) -> list[Chunk]:
        metadata = chunk.metadata
        if not metadata.get("is_chunk_part"):
            return [chunk]

        parent_id = str(metadata.get("parent_chunk_id", ""))
        part_index = int(metadata.get("part_index", 0))
        part_count = int(metadata.get("part_count", 0))
        if not parent_id or part_index <= 0 or part_count <= 0:
            return [chunk]

        window = self.settings.context_window_parts
        start = max(1, part_index - window)
        end = min(part_count, part_index + window)
        parts: list[Chunk] = []
        for index in range(start, end + 1):
            part = self.chunk_store.get(f"{parent_id}:part:{index}") if self.chunk_store is not None else None
            if part is None:
                continue
            if part.id == chunk.id:
                parts.append(part)
            else:
                parts.append(_context_expansion(part, chunk.id))
        return parts or [chunk]


def _chunk_hash(chunks: list[Chunk]) -> str:
    """Return a deterministic hash of chunk IDs for change detection."""
    return md5("|".join(chunk.id for chunk in chunks).encode("utf-8")).hexdigest()


def _normalize(matches: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
    if not matches:
        return []
    max_score = max(score for _, score in matches)
    if max_score <= 0:
        return [(chunk, 0.0) for chunk, _ in matches]
    return [(chunk, score / max_score) for chunk, score in matches]


def _add_score(merged: dict[str, tuple[Chunk, float]], chunk: Chunk, score: float) -> None:
    existing = merged.get(chunk.id)
    if existing is None:
        merged[chunk.id] = (chunk, score)
    else:
        merged[chunk.id] = (existing[0], existing[1] + score)


def _context_expansion(chunk: Chunk, source_chunk_id: str) -> Chunk:
    return Chunk(
        id=chunk.id,
        content=chunk.content,
        metadata=chunk.metadata
        | {
            "is_context_expansion": True,
            "context_source_chunk_id": source_chunk_id,
        },
    )
