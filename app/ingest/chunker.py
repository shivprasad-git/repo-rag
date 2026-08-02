from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.ingest.discover import discover_files
from app.ingest.parsers import parse_file
from app.ingest.repository import current_commit
from app.ingest.splitter import split_oversized_chunks
from app.logging_config import get_logger
from app.models import Chunk

logger = get_logger(__name__)


def create_chunks(repo_path: Path, settings: Settings) -> list[Chunk]:
    commit = current_commit(repo_path)
    logger.info("Parsing repository at %s (commit=%s)", repo_path, commit or "unknown")
    files = discover_files(repo_path, settings)
    logger.info("Discovered %d supported files", len(files))

    chunks: list[Chunk] = []
    for file_path in files:
        file_chunks = parse_file(file_path, repo_path, commit)
        chunks.extend(file_chunks)
        logger.debug("Parsed %s -> %d chunks", file_path.relative_to(repo_path), len(file_chunks))

    logger.info("Created %d chunks before splitting", len(chunks))
    split_chunks = split_oversized_chunks(chunks, settings)
    if len(split_chunks) != len(chunks):
        logger.info("Split oversized chunks: %d -> %d", len(chunks), len(split_chunks))
    return split_chunks
