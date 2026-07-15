from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.embeddings import HashEmbeddingProvider
from app.ingest.chunker import create_chunks
from app.ingest.repository import load_repository
from app.llm import build_prompt
from app.retrieval import Retriever
from app.vectorstore import build_vector_store


def build_embedder(settings: Settings) -> HashEmbeddingProvider:
    if settings.embedding_provider != "hash":
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
    return HashEmbeddingProvider(dimensions=settings.embedding_dimensions)


def index_repository(repo: str, store_kind: str, settings: Settings, index_name: str | None = None) -> tuple[Path, int]:
    repo_path = load_repository(repo, settings.repositories_dir)
    chunks = create_chunks(repo_path, settings)
    embedder = build_embedder(settings)
    embeddings = embedder.embed_many([chunk.content for chunk in chunks])
    vector_store = build_vector_store(store_kind, settings, index_name=index_name)
    vector_store.add(chunks, embeddings)
    return repo_path, len(chunks)


def query_repository(
    question: str,
    store_kind: str,
    settings: Settings,
    top_k: int,
    index_name: str | None = None,
):
    embedder = build_embedder(settings)
    vector_store = build_vector_store(store_kind, settings, index_name=index_name)
    return Retriever(embedder, vector_store).retrieve(question, top_k=top_k)


def ask_repository(
    repo: str,
    question: str,
    store_kind: str,
    settings: Settings,
    top_k: int,
    index_name: str | None = None,
) -> str:
    index_repository(repo, store_kind, settings, index_name=index_name)
    matches = query_repository(question, store_kind, settings, top_k, index_name=index_name)
    return build_prompt(question, matches)

