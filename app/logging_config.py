from __future__ import annotations

import logging
import os
import sys
from typing import TextIO

DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Third-party loggers that are noisy at INFO level.  They are kept at
# WARNING so the application's own logs remain readable.
QUIET_LOGGERS = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "urllib3",
    "chromadb",
    "uvicorn.access",
)


def setup_logging(
    level: str | int | None = None,
    stream: TextIO | None = None,
) -> None:
    """Configure the root logger for the application.

    The log level is resolved in this order:
      1. Explicit ``level`` argument.
      2. ``REPO_RAG_LOG_LEVEL`` environment variable.
      3. ``DEFAULT_LOG_LEVEL`` (INFO).

    Third-party library loggers are kept at WARNING by default so the
    application's own progress messages remain readable.

    Calling this more than once is safe: the root logger's handlers are
    replaced rather than duplicated.
    """
    resolved_level = _resolve_level(level)
    root = logging.getLogger()
    root.setLevel(resolved_level)

    # Remove existing handlers to avoid duplicate output on re-configuration.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(resolved_level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)

    for logger_name in QUIET_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for *name* using the application's configuration."""
    return logging.getLogger(name)


def _resolve_level(level: str | int | None) -> int:
    if level is not None:
        return _coerce_level(level)
    env_level = os.environ.get("REPO_RAG_LOG_LEVEL")
    if env_level:
        return _coerce_level(env_level)
    return _coerce_level(DEFAULT_LOG_LEVEL)


def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    normalized = level.strip().upper()
    return getattr(logging, normalized, logging.INFO)
