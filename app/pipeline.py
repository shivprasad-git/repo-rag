from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.docstore import build_chunk_store
from app.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from app.ingest.chunker import create_chunks
from app.ingest.repository import load_repository
from app.ingest.searchable import searchable_chunks
from app.llm import build_prompt
from app.retrieval.filters import MetadataFilters
from app.retrieval.reranker import CrossEncoderMiniLMReranker, Reranker
from app.retrieval.search import Retriever
from app.vectorstore import build_vector_store

EMBEDDING_BATCH_SIZE = 64


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
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings.extend(embedder.embed_many(batch))
    return embeddings


def index_repository(repo: str, store_kind: str, settings: Settings, index_name: str | None = None) -> tuple[Path, int]:
    repo_path = load_repository(repo, settings.repositories_dir)
    chunks = create_chunks(repo_path, settings)
    embedder = build_embedder(settings)
    vector_chunks = searchable_chunks(chunks, settings)
    embeddings = _embed_in_batches(embedder, [chunk.content for chunk in vector_chunks])
    name = index_name or settings.collection_name
    chunk_store = build_chunk_store(settings.indexes_dir, name)
    vector_store = build_vector_store(store_kind, settings, index_name=index_name)
    chunk_store.add(chunks)
    vector_store.add(vector_chunks, embeddings)
    return repo_path, len(chunks)


def query_repository(
    question: str,
    store_kind: str,
    settings: Settings,
    top_k: int,
    index_name: str | None = None,
    filters: MetadataFilters | None = None,
):
    embedder = build_embedder(settings)
    reranker = build_reranker(settings)
    name = index_name or settings.collection_name
    chunk_store = build_chunk_store(settings.indexes_dir, name)
    vector_store = build_vector_store(store_kind, settings, index_name=index_name)
    return Retriever(embedder, vector_store, settings=settings, chunk_store=chunk_store, reranker=reranker).retrieve(
        question,
        top_k=top_k,
        filters=filters,
    )


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
        index_repository(repo, store_kind, settings, index_name=index_name)
    matches = query_repository(question, store_kind, settings, top_k, index_name=index_name, filters=filters)
    return build_prompt(question, matches)
