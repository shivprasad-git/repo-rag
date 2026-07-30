from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.ingest.discover import discover_files
from app.ingest.parsers import parse_file
from app.ingest.repository import current_commit
from app.ingest.splitter import split_oversized_chunks
from app.models import Chunk


def create_chunks(repo_path: Path, settings: Settings) -> list[Chunk]:
    commit = current_commit(repo_path)
    chunks: list[Chunk] = []
    for file_path in discover_files(repo_path, settings):
        chunks.extend(parse_file(file_path, repo_path, commit))
    return split_oversized_chunks(chunks, settings)
