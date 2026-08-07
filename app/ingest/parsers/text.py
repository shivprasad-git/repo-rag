"""Plain-text source parser producing a single chunk per file."""

from __future__ import annotations

from pathlib import Path

from app.ingest.parsers.common import _base_metadata, _chunk_id
from app.models import Chunk, ChunkType


def parse_text(path: Path, repo_path: Path, commit: str | None = None) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    metadata = _base_metadata(path, repo_path, "text", commit)
    line_count = max(1, len(text.splitlines()))
    return [
        Chunk(
            id=_chunk_id(metadata, ChunkType.TEXT_FILE, metadata["file_path"], 1, line_count),
            content=text,
            metadata=metadata
            | {
                "chunk_type": ChunkType.TEXT_FILE.value,
                "symbol": metadata["file_path"],
                "start_line": 1,
                "end_line": line_count,
            },
        )
    ] if text.strip() else []