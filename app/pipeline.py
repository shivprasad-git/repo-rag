from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.docstore import build_chunk_store
from app.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from app.ingest.chunker import create_chunks
from app.ingest.repository import load_repository
from app.llm import build_prompt
from app.retrieval.filters import MetadataFilters
from app.retrieval.reranker import CrossEncoderMiniLMReranker, Reranker
from app.retrieval.search import Retriever
from app.vectorstore import build_vector_store


def build_embedder(settings: Settings) -> EmbeddingProvider:
    return SentenceTransformerEmbeddingProvider(model_name=settings.embedding_model)


def build_reranker(settings: Settings) -> Reranker | None:
    if not settings.reranker_enabled:
        return None
    return CrossEncoderMiniLMReranker(model_name=settings.reranker_model)


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
    reranker = build_reranker(settings)
    name = index_name or settings.collection_name
    chunk_store = build_chunk_store(settings.indexes_dir, name)
    vector_store = build_vector_store(store_kind, settings, index_name=index_name)
    return Retriever(embedder, vector_store, chunk_store=chunk_store, reranker=reranker).retrieve(
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
    index_repository(repo, store_kind, settings, index_name=index_name)
    matches = query_repository(question, store_kind, settings, top_k, index_name=index_name, filters=filters)
    return build_prompt(question, matches)
