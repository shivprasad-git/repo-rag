from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.docstore import build_chunk_store
from app.indexing.manifest import IndexManifest, build_manifest_path
from app.ingest.searchable import searchable_chunks
from app.vectorstore import build_vector_store


@dataclass(frozen=True)
class IndexInfo:
    index_name: str
    store_kind: str
    embedding_model: str
    embedding_dimensions: int
    reranker_model: str
    reranker_enabled: bool
    max_chunk_tokens: int
    chunk_overlap_tokens: int
    context_window_parts: int
    max_prompt_context_tokens: int
    manifest_path: str
    manifest_exists: bool
    files_indexed: int
    chunks_stored: int
    searchable_chunks: int
    vectors_stored: int


def get_index_info(store_kind: str, settings: Settings, index_name: str | None = None) -> IndexInfo:
    name = index_name or settings.collection_name
    manifest_path = build_manifest_path(settings.indexes_dir, name)
    manifest = IndexManifest.load(manifest_path)
    chunk_store = build_chunk_store(settings.indexes_dir, name)
    vector_store = build_vector_store(store_kind, settings, index_name=index_name)
    chunks = chunk_store.all_chunks()
    vectors = vector_store.all_chunks()

    return IndexInfo(
        index_name=name,
        store_kind=store_kind,
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
        reranker_model=settings.reranker_model,
        reranker_enabled=settings.reranker_enabled,
        max_chunk_tokens=settings.max_chunk_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
        context_window_parts=settings.context_window_parts,
        max_prompt_context_tokens=settings.max_prompt_context_tokens,
        manifest_path=str(manifest_path),
        manifest_exists=manifest_path.exists(),
        files_indexed=len(manifest.files),
        chunks_stored=len(chunks),
        searchable_chunks=len(searchable_chunks(chunks, settings)),
        vectors_stored=len(vectors),
    )


def format_index_info(info: IndexInfo) -> str:
    return "\n".join(
        [
            f"Index: {info.index_name}",
            f"Store: {info.store_kind}",
            f"Manifest: {'present' if info.manifest_exists else 'missing'}",
            f"Manifest path: {info.manifest_path}",
            "",
            f"Embedding model: {info.embedding_model}",
            f"Embedding dimensions: {info.embedding_dimensions}",
            f"Reranker enabled: {'yes' if info.reranker_enabled else 'no'}",
            f"Reranker model: {info.reranker_model}",
            "",
            f"Files indexed: {info.files_indexed}",
            f"Chunks stored: {info.chunks_stored}",
            f"Searchable chunks: {info.searchable_chunks}",
            f"Vectors stored: {info.vectors_stored}",
            "",
            f"Max chunk tokens: {info.max_chunk_tokens}",
            f"Chunk overlap tokens: {info.chunk_overlap_tokens}",
            f"Context window parts: {info.context_window_parts}",
            f"Prompt context budget: {info.max_prompt_context_tokens}",
        ]
    )
