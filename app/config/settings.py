from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

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
    context_window_parts: int = 1
    max_prompt_context_tokens: int = 3000
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


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    return value if value is not None and value != "" else None


def _env_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Maps RAG_* environment variables to Settings fields and their coercion.
_SETTINGS_ENV_OVERRIDES: dict[str, tuple[str, Callable[[str], object]]] = {
    "RAG_COLLECTION_NAME": ("collection_name", str),
    "RAG_EMBEDDING_MODEL": ("embedding_model", str),
    "RAG_EMBEDDING_DIMENSIONS": ("embedding_dimensions", int),
    "RAG_RERANKER_MODEL": ("reranker_model", str),
    "RAG_RERANKER_ENABLED": ("reranker_enabled", _env_bool),
    "RAG_MAX_CHUNK_TOKENS": ("max_chunk_tokens", int),
    "RAG_CHUNK_OVERLAP_TOKENS": ("chunk_overlap_tokens", int),
    "RAG_CONTEXT_WINDOW_PARTS": ("context_window_parts", int),
    "RAG_MAX_PROMPT_CONTEXT_TOKENS": ("max_prompt_context_tokens", int),
}


@dataclass(frozen=True)
class CliDefaults:
    """Command-line defaults configurable via ``.env`` (not :class:`Settings` fields)."""

    store: str = "chroma"
    index_name: str | None = None
    no_reranker: bool = False


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Build :class:`Settings` from built-in defaults plus ``.env``/environment overrides.

    Precedence is: built-in defaults < ``.env`` file < process environment.
    """
    load_dotenv(env_file)
    overrides: dict[str, object] = {}
    for env_var, (field_name, coerce) in _SETTINGS_ENV_OVERRIDES.items():
        raw = _env_value(env_var)
        if raw is not None:
            overrides[field_name] = coerce(raw)
    return Settings(**overrides)


def cli_defaults(env_file: str | Path | None = None) -> CliDefaults:
    """Command-line defaults sourced from the same ``.env``/environment as Settings."""
    load_dotenv(env_file)
    no_reranker = False
    reranker_enabled = _env_value("RAG_RERANKER_ENABLED")
    if reranker_enabled is not None:
        no_reranker = not _env_bool(reranker_enabled)
    return CliDefaults(
        store=_env_value("RAG_STORE") or "chroma",
        index_name=_env_value("RAG_INDEX_NAME"),
        no_reranker=no_reranker,
    )
