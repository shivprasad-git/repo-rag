from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urlparse

from app.config import Settings
from app.docstore import SimpleJsonChunkStore, build_chunk_store
from app.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from app.indexing.manifest import IndexManifest, build_manifest_path, file_content_hash, index_config
from app.ingest.chunker import create_chunks_for_files
from app.ingest.discover import discover_files
from app.ingest.repository import load_repository
from app.ingest.searchable import searchable_chunks
from app.llm import build_prompt
from app.logging_config import get_logger
from app.retrieval.filters import MetadataFilters
from app.retrieval.reranker import CrossEncoderMiniLMReranker, Reranker
from app.retrieval.search import Retriever
from app.vectorstore import build_vector_store
from app.vectorstore.base import VectorStore

EMBEDDING_BATCH_SIZE = 64

logger = get_logger(__name__)


def default_index_name(repo: str) -> str:
    """Derive a stable index name from a GitHub URL or local repository path.

    Examples::

        https://github.com/pallets/flask  -> "flask"
        /path/to/local/my-project         -> "my-project"
    """
    source = Path(repo.strip()).expanduser()
    if source.exists():
        name = source.name
    else:
        parsed = urlparse(repo.strip())
        name = Path(parsed.path).name.removesuffix(".git")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return slug or "repo_rag"


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


def _build_stores(
    store_kind: str,
    settings: Settings,
    index_name: str | None,
) -> tuple[SimpleJsonChunkStore, VectorStore]:
    """Build the chunk and vector stores for an index."""
    name = index_name or settings.collection_name
    return build_chunk_store(settings.indexes_dir, name), build_vector_store(store_kind, settings, index_name=index_name)


def _incremental_diff(
    manifest: IndexManifest,
    current_hashes: dict[str, str],
    config: dict,
) -> tuple[list[str], list[str]]:
    """Return ``(deleted_paths, changed_paths)`` for the incremental update plan.

    When the index configuration changed, every indexed file is treated as
    changed so the stores are rebuilt with the new settings.
    """
    manifest_files = manifest.files
    deleted_paths = sorted(set(manifest_files) - set(current_hashes))
    changed_paths = sorted(
        path
        for path, content_hash in current_hashes.items()
        if path not in manifest_files or manifest_files[path].content_hash != content_hash
    )

    if manifest.config and manifest.config != config:
        logger.info("Index config changed; rebuilding all indexed files")
        deleted_paths = sorted(manifest_files)
        changed_paths = sorted(current_hashes)
    return deleted_paths, changed_paths


def _collect_stale_chunk_ids(manifest: IndexManifest, deleted_paths: list[str], changed_paths: list[str]) -> list[str]:
    """Drop changed/deleted files from the manifest and return their chunk ids."""
    stale_chunk_ids: list[str] = []
    for relative_path in deleted_paths:
        stale_chunk_ids.extend(manifest.remove_file(relative_path))
    for relative_path in changed_paths:
        stale_chunk_ids.extend(manifest.remove_file(relative_path))
    return stale_chunk_ids


def index_repository(repo: str, store_kind: str, settings: Settings, index_name: str | None = None) -> tuple[Path, int]:
    name = index_name or default_index_name(repo)
    start = time.monotonic()
    logger.info("Indexing repository: %s (store=%s, index=%s)", repo, store_kind, name)

    repo_path = load_repository(repo, settings.repositories_dir)
    logger.info("Repository ready at: %s", repo_path)

    chunk_store, vector_store = _build_stores(store_kind, settings, name)
    manifest = IndexManifest.load(build_manifest_path(settings.indexes_dir, name))
    config = index_config(settings, store_kind)

    current_hashes = {
        str(file_path.relative_to(repo_path)): file_content_hash(file_path)
        for file_path in discover_files(repo_path, settings)
    }
    deleted_paths, changed_paths = _incremental_diff(manifest, current_hashes, config)
    stale_chunk_ids = _collect_stale_chunk_ids(manifest, deleted_paths, changed_paths)

    if stale_chunk_ids:
        chunk_store.delete(stale_chunk_ids)
        vector_store.delete(stale_chunk_ids)
        logger.info("Removed %d stale chunks from previous index state", len(stale_chunk_ids))

    changed_files = [repo_path / relative_path for relative_path in changed_paths]
    chunks = create_chunks_for_files(repo_path, settings, changed_files)
    logger.info(
        "Incremental index plan: %d changed/new files, %d deleted files, %d unchanged files",
        len(changed_paths),
        len(deleted_paths),
        len(current_hashes) - len(changed_paths),
    )

    if chunks:
        embedder = build_embedder(settings)
        vector_chunks = searchable_chunks(chunks, settings)
        logger.info("Embedding %d searchable chunks with %s", len(vector_chunks), settings.embedding_model)
        embeddings = _embed_in_batches(embedder, [chunk.content for chunk in vector_chunks])

        chunk_store.add(chunks)
        logger.info("Stored %d changed chunks in document store", len(chunks))
        vector_store.add(vector_chunks, embeddings)
        logger.info("Stored %d changed embeddings in %s vector store", len(vector_chunks), store_kind)
    else:
        logger.info("No changed chunks to embed")

    chunks_by_file: dict[str, list] = {}
    for chunk in chunks:
        chunks_by_file.setdefault(str(chunk.metadata.get("file_path", "")), []).append(chunk)
    for relative_path in changed_paths:
        manifest.replace_file(relative_path, current_hashes[relative_path], chunks_by_file.get(relative_path, []))
    manifest.config = config
    manifest.save()

    total_chunks = len(chunk_store.all_chunks())

    elapsed = time.monotonic() - start
    logger.info("Indexing complete in %.2fs (%d total chunks)", elapsed, total_chunks)
    return repo_path, total_chunks


def query_repository(
    question: str,
    store_kind: str,
    settings: Settings,
    top_k: int,
    index_name: str | None = None,
    filters: MetadataFilters | None = None,
    debug_scores: bool = False,
):
    start = time.monotonic()
    logger.info("Querying index %s (top_k=%d, store=%s)", index_name or settings.collection_name, top_k, store_kind)
    if filters is not None and filters.has_filters:
        logger.info("Applying filters: language=%s chunk_type=%s path_prefix=%s", filters.language, filters.chunk_type, filters.path_prefix)

    embedder = build_embedder(settings)
    reranker = build_reranker(settings)
    chunk_store, vector_store = _build_stores(store_kind, settings, index_name)
    matches = Retriever(embedder, vector_store, settings=settings, chunk_store=chunk_store, reranker=reranker).retrieve(
        question,
        top_k=top_k,
        filters=filters,
        debug_scores=debug_scores,
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
    name = index_name or default_index_name(repo)
    chunk_store, vector_store = _build_stores(store_kind, settings, name)
    if not chunk_store.has_data() or not vector_store.has_data():
        logger.info("Index %s is empty; indexing repository before answering", name)
        index_repository(repo, store_kind, settings, index_name=name)
    matches = query_repository(question, store_kind, settings, top_k, index_name=name, filters=filters)
    return build_prompt(question, matches, settings=settings)
