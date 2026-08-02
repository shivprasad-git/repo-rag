from __future__ import annotations

from pathlib import Path

from app.embeddings import EmbeddingProvider
from app.config import Settings
from app.logging_config import get_logger
from app.models import Chunk
from app.docstore.simple_store import SimpleJsonChunkStore
from app.ingest.searchable import searchable_chunks
from app.retrieval.filters import MetadataFilters
from app.retrieval.keyword import BM25KeywordIndex
from app.retrieval.reranker import Reranker
from app.vectorstore.base import VectorStore

logger = get_logger(__name__)


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
        self._bm25_source_mtime: float | None = None
        self._bm25_source_path: Path | None = None

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
        logger.debug(
            "Hybrid retrieval: %d vector matches, %d keyword matches (candidate_count=%d)",
            len(vector_matches),
            len(keyword_matches),
            candidate_count,
        )

        merged: dict[str, tuple[Chunk, float]] = {}
        for chunk, score in _normalize(vector_matches):
            _add_score(merged, self._hydrate(chunk), score * self.vector_weight)
        for chunk, score in _normalize(keyword_matches):
            _add_score(merged, chunk, score * self.keyword_weight)

        candidates = sorted(merged.values(), key=lambda item: item[1], reverse=True)
        if self.reranker is not None:
            logger.debug("Reranking %d candidates with %s", len(candidates), type(self.reranker).__name__)
            matches = self.reranker.rerank(question, candidates, top_k=top_k)
        else:
            matches = candidates[:top_k]
        expanded = self._expand_context_parts(matches)
        logger.debug("Returning %d matches (top_k=%d)", len(expanded), top_k)
        return expanded

    def _get_bm25_index(self, filters: MetadataFilters | None = None) -> BM25KeywordIndex:
        """Return a cached BM25 index, rebuilding only when chunks change.

        When filters are applied, the BM25 index is always rebuilt from the
        filtered chunk subset because document frequencies change.  For the
        common unfiltered case, the index is cached and reused across queries
        until the underlying chunk store changes.
        """
        if filters is not None and filters.has_filters:
            return BM25KeywordIndex(self._all_full_chunks(filters))

        if self._bm25_index is None or self._bm25_source_changed():
            chunks = self._all_full_chunks(filters=None)
            self._bm25_index = BM25KeywordIndex(chunks)
            self._bm25_source_path = _source_path(self.chunk_store, self.vector_store)
            self._bm25_source_mtime = _source_mtime(self._bm25_source_path)
        return self._bm25_index

    def _bm25_source_changed(self) -> bool:
        """Return True when the underlying chunk source file changed on disk.

        Uses the chunk store file mtime when available, falling back to the
        vector store file.  This avoids re-loading and hashing every chunk on
        each query just to detect a change.

        When no on-disk source is available (e.g. a Chroma-only setup without
        a chunk store), there is no file to watch, so the index is rebuilt on
        every query to stay correct.
        """
        if self._bm25_source_path is None:
            return True
        mtime = _source_mtime(self._bm25_source_path)
        return mtime != self._bm25_source_mtime

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


def _source_path(chunk_store: SimpleJsonChunkStore | None, vector_store: VectorStore) -> Path | None:
    """Return the on-disk source file that backs the BM25 corpus."""
    if chunk_store is not None:
        return chunk_store.path
    path = getattr(vector_store, "path", None)
    return path if isinstance(path, Path) else None


def _source_mtime(path: Path | None) -> float | None:
    return path.stat().st_mtime if path is not None and path.exists() else None


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
