from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).resolve().parents[2]
    repositories_dir: Path = base_dir / "data" / "repositories"
    indexes_dir: Path = base_dir / "indexes"
    collection_name: str = "repo_rag"
    embedding_model: str = os.getenv("REPO_RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimensions: int = 384
    reranker_model: str = os.getenv("REPO_RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_enabled: bool = os.getenv("REPO_RAG_RERANKER_ENABLED", "true").lower() == "true"
    supported_extensions: tuple[str, ...] = (".py", ".md", ".txt")
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
