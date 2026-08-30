from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from app.config import Settings
from app.docstore import build_chunk_store
from app.indexing.manifest import IndexManifest, build_manifest_path, file_content_hash, index_config
from app.ingest.repository import load_repository
from app.ingest.searchable import searchable_chunks
from app.vectorstore import build_vector_store
from app.vectorstore.base import VectorStore


@dataclass(frozen=True)
class IndexHealthReport:
    index_name: str
    store_kind: str
    manifest_exists: bool
    config_matches: bool
    manifest_files: int
    doc_chunks: int
    searchable_chunks: int
    vector_chunks: int
    missing_doc_chunks: list[str] = field(default_factory=list)
    doc_chunks_missing_manifest: list[str] = field(default_factory=list)
    missing_vectors: list[str] = field(default_factory=list)
    orphan_vectors: list[str] = field(default_factory=list)
    wrong_embedding_dimensions: list[str] = field(default_factory=list)
    stale_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.manifest_exists
            and self.config_matches
            and not self.missing_doc_chunks
            and not self.doc_chunks_missing_manifest
            and not self.missing_vectors
            and not self.orphan_vectors
            and not self.wrong_embedding_dimensions
            and not self.stale_files
        )


def check_index_health(
    store_kind: str,
    settings: Settings,
    index_name: str | None = None,
    repo: str | None = None,
) -> IndexHealthReport:
    name = index_name or settings.collection_name
    manifest_path = build_manifest_path(settings.indexes_dir, name)
    manifest = IndexManifest.load(manifest_path)
    chunk_store = build_chunk_store(settings.indexes_dir, name)
    vector_store = build_vector_store(store_kind, settings, index_name=index_name)

    doc_chunks = chunk_store.all_chunks()
    vector_chunks = vector_store.all_chunks()
    doc_ids = {chunk.id for chunk in doc_chunks}
    vector_ids = {chunk.id for chunk in vector_chunks}
    manifest_chunk_ids = {
        chunk_id
        for indexed_file in manifest.files.values()
        for chunk_id in indexed_file.chunk_ids
    }
    searchable_ids = {chunk.id for chunk in searchable_chunks(doc_chunks, settings)}

    stale_files = _stale_files(repo, settings, manifest) if repo else []

    return IndexHealthReport(
        index_name=name,
        store_kind=store_kind,
        manifest_exists=manifest_path.exists(),
        config_matches=manifest.config == index_config(settings, store_kind),
        manifest_files=len(manifest.files),
        doc_chunks=len(doc_chunks),
        searchable_chunks=len(searchable_ids),
        vector_chunks=len(vector_chunks),
        missing_doc_chunks=sorted(manifest_chunk_ids - doc_ids),
        doc_chunks_missing_manifest=sorted(doc_ids - manifest_chunk_ids),
        missing_vectors=sorted(searchable_ids - vector_ids),
        orphan_vectors=sorted(vector_ids - doc_ids),
        wrong_embedding_dimensions=_wrong_embedding_dimensions(vector_store, settings),
        stale_files=stale_files,
    )


def format_health_report(report: IndexHealthReport) -> str:
    lines = [
        f"Index health: {'OK' if report.ok else 'FAILED'}",
        "",
        f"Index: {report.index_name}",
        f"Store: {report.store_kind}",
        f"Manifest exists: {'yes' if report.manifest_exists else 'no'}",
        f"Config matches: {'yes' if report.config_matches else 'no'}",
        f"Files in manifest: {report.manifest_files}",
        f"Chunks in doc store: {report.doc_chunks}",
        f"Searchable chunks: {report.searchable_chunks}",
        f"Vectors in vector store: {report.vector_chunks}",
        f"Missing doc chunks: {len(report.missing_doc_chunks)}",
        f"Doc chunks missing manifest: {len(report.doc_chunks_missing_manifest)}",
        f"Missing vectors: {len(report.missing_vectors)}",
        f"Orphan vectors: {len(report.orphan_vectors)}",
        f"Wrong embedding dimensions: {len(report.wrong_embedding_dimensions)}",
        f"Stale files: {len(report.stale_files)}",
    ]
    lines.extend(_section("Missing doc chunks", report.missing_doc_chunks))
    lines.extend(_section("Doc chunks missing manifest", report.doc_chunks_missing_manifest))
    lines.extend(_section("Missing vectors", report.missing_vectors))
    lines.extend(_section("Orphan vectors", report.orphan_vectors))
    lines.extend(_section("Wrong embedding dimensions", report.wrong_embedding_dimensions))
    lines.extend(_section("Stale files", report.stale_files))
    return "\n".join(lines)


def _section(title: str, values: list[str], limit: int = 20) -> list[str]:
    if not values:
        return []
    lines = ["", f"{title}:"]
    lines.extend(f"- {value}" for value in values[:limit])
    if len(values) > limit:
        lines.append(f"- ... {len(values) - limit} more")
    return lines


def _stale_files(repo: str, settings: Settings, manifest: IndexManifest) -> list[str]:
    repo_path = load_repository(repo, settings.repositories_dir)
    stale: list[str] = []
    for relative_path, indexed_file in manifest.files.items():
        file_path = repo_path / relative_path
        if not file_path.exists():
            stale.append(relative_path)
            continue
        if file_content_hash(file_path) != indexed_file.content_hash:
            stale.append(relative_path)
    return sorted(stale)


def _wrong_embedding_dimensions(vector_store: VectorStore, settings: Settings) -> list[str]:
    """Return ids of vectors with a dimension mismatch (only checkable for the simple JSON store)."""
    path = getattr(vector_store, "path", None)
    if not isinstance(path, Path) or not path.exists():
        return []

    records = json.loads(path.read_text(encoding="utf-8"))
    return sorted(
        str(record.get("id", ""))
        for record in records
        if len(record.get("embedding", [])) != settings.embedding_dimensions
    )
