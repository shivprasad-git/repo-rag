from __future__ import annotations

import time
from pathlib import Path

from app.config import Settings
from app.docstore import build_chunk_store
from app.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from app.ingest.chunker import create_chunks
from app.ingest.repository import load_repository
from app.ingest.searchable import searchable_chunks
from app.llm import build_prompt
from app.logging_config import get_logger
from app.retrieval.filters import MetadataFilters
from app.retrieval.reranker import CrossEncoderMiniLMReranker, Reranker
from app.retrieval.search import Retriever
from app.vectorstore import build_vector_store

EMBEDDING_BATCH_SIZE = 64

logger = get_logger(__name__)


def build_embedder(settings: Settings) -> EmbeddingProvider:
    return SentenceTransformerEmbeddingProvider(model_name=settings.embedding_model)


def build_reranker(settings: Settings) -> Reranker | None:
    if not settings.reranker_enabled:
        return None
    return CrossEncoderMiniLMReranker(model_name=settings.reranker_model)


def _embed_in_batches(embedder: EmbeddingProvider, texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
    """Embed texts in fixed-size batches to bound peak memory usage.

    ``embed_many`` on all texts at once can exhaust CPU/GPU memory for large
    repositories; batching keeps the peak memory proportional to the batch
    size rather than the whole corpus.
    """
    if batch_size <= 0:
        return embedder.embed_many(texts)
    embeddings: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        embeddings.extend(embedder.embed_many(batch))
        done = min(start + batch_size, total)
        logger.info("Embedded %d/%d chunks (%.0f%%)", done, total, 100.0 * done / total)
    return embeddings


def index_repository(repo: str, store_kind: str, settings: Settings, index_name: str | None = None) -> tuple[Path, int]:
    start = time.monotonic()
    logger.info("Indexing repository: %s (store=%s, index=%s)", repo, store_kind, index_name or settings.collection_name)

    repo_path = load_repository(repo, settings.repositories_dir)
    logger.info("Repository ready at: %s", repo_path)

    chunks = create_chunks(repo_path, settings)
    logger.info("Created %d chunks", len(chunks))

    embedder = build_embedder(settings)
    vector_chunks = searchable_chunks(chunks, settings)
    logger.info("Embedding %d searchable chunks with %s", len(vector_chunks), settings.embedding_model)
    embeddings = _embed_in_batches(embedder, [chunk.content for chunk in vector_chunks])

    name = index_name or settings.collection_name
    chunk_store = build_chunk_store(settings.indexes_dir, name)
    vector_store = build_vector_store(store_kind, settings, index_name=index_name)
    chunk_store.add(chunks)
    logger.info("Stored %d chunks in document store", len(chunks))
    vector_store.add(vector_chunks, embeddings)
    logger.info("Stored %d embeddings in %s vector store", len(vector_chunks), store_kind)

    elapsed = time.monotonic() - start
    logger.info("Indexing complete in %.2fs (%d chunks)", elapsed, len(chunks))
    return repo_path, len(chunks)


def query_repository(
    question: str,
    store_kind: str,
    settings: Settings,
    top_k: int,
    index_name: str | None = None,
    filters: MetadataFilters | None = None,
):
    start = time.monotonic()
    logger.info("Querying index %s (top_k=%d, store=%s)", index_name or settings.collection_name, top_k, store_kind)
    if filters is not None and filters.has_filters:
        logger.info("Applying filters: language=%s chunk_type=%s path_prefix=%s", filters.language, filters.chunk_type, filters.path_prefix)

    embedder = build_embedder(settings)
    reranker = build_reranker(settings)
    name = index_name or settings.collection_name
    chunk_store = build_chunk_store(settings.indexes_dir, name)
    vector_store = build_vector_store(store_kind, settings, index_name=index_name)
    matches = Retriever(embedder, vector_store, settings=settings, chunk_store=chunk_store, reranker=reranker).retrieve(
        question,
        top_k=top_k,
        filters=filters,
    )
    logger.info("Retrieved %d matches in %.2fs", len(matches), time.monotonic() - start)
    return matches


def ask_repository(
    repo: str,
    question: str,
    store_kind: str,
    settings: Settings,
    top_k: int,
    index_name: str | None = None,
    filters: MetadataFilters | None = None,
) -> str:
    name = index_name or settings.collection_name
    chunk_store = build_chunk_store(settings.indexes_dir, name)
    vector_store = build_vector_store(store_kind, settings, index_name=index_name)
    if not chunk_store.has_data() or not vector_store.has_data():
        logger.info("Index %s is empty; indexing repository before answering", name)
        index_repository(repo, store_kind, settings, index_name=index_name)
    matches = query_repository(question, store_kind, settings, top_k, index_name=index_name, filters=filters)
    return build_prompt(question, matches)
