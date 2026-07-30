from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.models import ChunkType


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).resolve().parents[2]
    repositories_dir: Path = base_dir / "data" / "repositories"
    indexes_dir: Path = base_dir / "indexes"
    collection_name: str = "repo_rag"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = True
    supported_extensions: tuple[str, ...] = (".py", ".md", ".txt")
    searchable_chunk_types: tuple[ChunkType, ...] = (
        ChunkType.CLASS,
        ChunkType.FUNCTION,
        ChunkType.METHOD,
        ChunkType.MARKDOWN_SECTION,
        ChunkType.TEXT_FILE,
    )
    splittable_chunk_types: tuple[ChunkType, ...] = (
        ChunkType.FUNCTION,
        ChunkType.METHOD,
        ChunkType.MARKDOWN_SECTION,
        ChunkType.TEXT_FILE,
    )
    max_chunk_tokens: int = 700
    chunk_overlap_tokens: int = 100
    ignored_dirs: tuple[str, ...] = (
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        "coverage",
        "target",
        "__pycache__",
    )
