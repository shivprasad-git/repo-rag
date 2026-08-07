"""File parsers for supported source types."""

from __future__ import annotations

from pathlib import Path

from app.ingest.parsers.markdown import parse_markdown
from app.ingest.parsers.python import parse_python
from app.ingest.parsers.text import parse_text
from app.models import Chunk

__all__ = ["parse_file", "parse_markdown", "parse_python", "parse_text"]


def parse_file(path: Path, repo_path: Path, commit: str | None = None) -> list[Chunk]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return parse_python(path, repo_path, commit)
    if suffix == ".md":
        return parse_markdown(path, repo_path, commit)
    if suffix == ".txt":
        return parse_text(path, repo_path, commit)
    return []