from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.docstore import build_chunk_store
from app.embeddings import EmbeddingProvider, HashEmbeddingProvider, SentenceTransformerEmbeddingProvider
from app.ingest.chunker import create_chunks
from app.ingest.repository import load_repository
from app.llm import build_prompt
from app.retrieval.filters import MetadataFilters
from app.retrieval.search import Retriever
from app.vectorstore import build_vector_store


def build_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hash":
        return HashEmbeddingProvider(dimensions=settings.embedding_dimensions)
    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(model_name=settings.embedding_model)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def index_repository(repo: str, store_kind: str, settings: Settings, index_name: str | None = None) -> tuple[Path, int]:
    repo_path = load_repository(repo, settings.repositories_dir)
    chunks = create_chunks(repo_path, settings)
    embedder = build_embedder(settings)
    embeddings = embedder.embed_many([chunk.content for chunk in chunks])
    name = index_name or settings.collection_name
    chunk_store = build_chunk_store(settings.indexes_dir, name)
    vector_store = build_vector_store(store_kind, settings, index_name=index_name)
    chunk_store.add(chunks)
    vector_store.add(chunks, embeddings)
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
    name = index_name or settings.collection_name
    chunk_store = build_chunk_store(settings.indexes_dir, name)
    vector_store = build_vector_store(store_kind, settings, index_name=index_name)
    return Retriever(embedder, vector_store, chunk_store=chunk_store).retrieve(question, top_k=top_k, filters=filters)


def ask_repository(
    repo: str,
    question: str,
    store_kind: str,
    settings: Settings,
    top_k: int,
    index_name: str | None = None,
    filters: MetadataFilters | None = None,
) -> str:
    index_repository(repo, store_kind, settings, index_name=index_name)
    matches = query_repository(question, store_kind, settings, top_k, index_name=index_name, filters=filters)
    return build_prompt(question, matches)
