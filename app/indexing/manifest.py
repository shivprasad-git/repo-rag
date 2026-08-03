from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import Chunk


@dataclass(frozen=True)
class IndexedFile:
    content_hash: str
    chunk_ids: list[str]


@dataclass
class IndexManifest:
    path: Path
    config: dict[str, Any] = field(default_factory=dict)
    files: dict[str, IndexedFile] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "IndexManifest":
        if not path.exists():
            return cls(path=path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            config=raw.get("config", {}),
            files={
                file_path: IndexedFile(
                    content_hash=record["content_hash"],
                    chunk_ids=list(record.get("chunk_ids", [])),
                )
                for file_path, record in raw.get("files", {}).items()
            },
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "files": {
                file_path: {
                    "content_hash": record.content_hash,
                    "chunk_ids": record.chunk_ids,
                }
                for file_path, record in sorted(self.files.items())
            },
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def replace_file(self, relative_path: str, content_hash: str, chunks: list[Chunk]) -> None:
        self.files[relative_path] = IndexedFile(
            content_hash=content_hash,
            chunk_ids=[chunk.id for chunk in chunks],
        )

    def remove_file(self, relative_path: str) -> list[str]:
        record = self.files.pop(relative_path, None)
        return record.chunk_ids if record is not None else []


def build_manifest_path(indexes_dir: Path, index_name: str) -> Path:
    return indexes_dir / "manifests" / f"{index_name}.json"


def file_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_config(settings: Settings, store_kind: str) -> dict[str, Any]:
    return {
        "store_kind": store_kind,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "searchable_chunk_types": [chunk_type.value for chunk_type in settings.searchable_chunk_types],
        "splittable_chunk_types": [chunk_type.value for chunk_type in settings.splittable_chunk_types],
        "max_chunk_tokens": settings.max_chunk_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
    }
