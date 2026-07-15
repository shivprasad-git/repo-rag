from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).resolve().parents[2]
    repositories_dir: Path = base_dir / "data" / "repositories"
    indexes_dir: Path = base_dir / "indexes"
    collection_name: str = "repo_rag"
    embedding_provider: str = "hash"
    embedding_dimensions: int = 384
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

