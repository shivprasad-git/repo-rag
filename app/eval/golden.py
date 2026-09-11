"""Golden set model, JSON loader, and chunk selector resolution.

A golden set is a small JSON file describing a repository plus a list of
questions with the chunks judged relevant to each. The runner uses it to
score retrieval quality (see ``app.eval.runner``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.models import Chunk


@dataclass(frozen=True)
class RelevantSelector:
    """A chunk selector used to judge a retrieved result as relevant.

    All provided fields must match the chunk metadata; at least one field must
    be provided. ``file_path`` and ``symbol`` match chunk metadata exactly.
    """

    file_path: str | None = None
    symbol: str | None = None
    chunk_type: str | None = None

    def __post_init__(self) -> None:
        if self.file_path is None and self.symbol is None and self.chunk_type is None:
            raise ValueError(f"Relevant selector needs at least one of file_path, symbol, or chunk_type: {self!r}")

    def matches(self, chunk: Chunk) -> bool:
        metadata = chunk.metadata
        if self.file_path is not None and metadata.get("file_path") != self.file_path:
            return False
        if self.symbol is not None and metadata.get("symbol") != self.symbol:
            return False
        if self.chunk_type is not None and metadata.get("chunk_type") != self.chunk_type:
            return False
        return True


@dataclass(frozen=True)
class GoldenQuestion:
    question: str
    relevant: tuple[RelevantSelector, ...]


@dataclass(frozen=True)
class GoldenSet:
    name: str
    repo: str
    store: str
    questions: tuple[GoldenQuestion, ...]
    top_k: tuple[int, ...] = (5, 10)
    index_name: str | None = None


def load_golden_set(path: str | Path) -> GoldenSet:
    """Load and validate a golden set from a JSON file."""
    golden_path = Path(path)
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden set not found: {golden_path}")
    raw = json.loads(golden_path.read_text(encoding="utf-8"))
    return golden_set_from_dict(raw, source=str(golden_path))


def golden_set_from_dict(raw: dict, source: str = "<golden>") -> GoldenSet:
    """Build a :class:`GoldenSet` from parsed JSON, raising ``ValueError`` on invalid input."""

    def _fail(message: str) -> None:
        raise ValueError(f"Invalid golden set {source}: {message}")

    name = raw.get("name")
    if not name:
        _fail("missing 'name'")
    repo = raw.get("repo")
    if not repo:
        _fail("missing 'repo'")

    top_k = _positive_ints(raw.get("top_k", [5, 10]))
    if not top_k:
        _fail("'top_k' must contain positive integers")

    raw_questions = raw.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        _fail("'questions' must be a non-empty list")

    return GoldenSet(
        name=str(name),
        repo=str(repo),
        store=str(raw.get("store", "simple")),
        top_k=tuple(top_k),
        index_name=raw.get("index_name") or None,
        questions=tuple(_golden_question(record, source=source) for record in raw_questions),
    )


def _golden_question(record: dict, source: str) -> GoldenQuestion:
    if not isinstance(record, dict):
        raise ValueError(f"Invalid golden set {source}: each question must be a JSON object")
    question = record.get("question")
    if not question:
        raise ValueError(f"Invalid golden set {source}: every question needs a non-empty 'question'")
    raw_relevant = record.get("relevant")
    if not isinstance(raw_relevant, list) or not raw_relevant:
        raise ValueError(f"Invalid golden set {source}: question {question!r} needs a non-empty 'relevant' list")
    selectors: list[RelevantSelector] = []
    for selector_record in raw_relevant:
        if not isinstance(selector_record, dict):
            raise ValueError(f"Invalid golden set {source}: every relevant entry must be a JSON object")
        try:
            selectors.append(RelevantSelector(**selector_record))
        except TypeError as exc:
            raise ValueError(f"Invalid golden set {source}: unknown selector field: {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"Invalid golden set {source}: {exc}") from exc
    return GoldenQuestion(question=str(question), relevant=tuple(selectors))


def _positive_ints(values) -> list[int]:
    if isinstance(values, str):
        try:
            return [int(value) for value in values.split(",")]
        except ValueError:
            return []
    if not isinstance(values, list):
        return []
    parsed: list[int] = []
    for value in values:
        try:
            integer = int(value)
        except (TypeError, ValueError):
            return []
        if integer <= 0:
            return []
        parsed.append(integer)
    return sorted(set(parsed))


def resolve_relevant_ids(selectors, chunks: list[Chunk]) -> list[str]:
    """Return the distinct chunk ids matching any of the selectors.

    Selectors are OR-ed together: a chunk that matches any selector is
    relevant. Chunks appear once, in chunk-store order.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for selector in selectors:
        for chunk in chunks:
            if chunk.id in seen:
                continue
            if selector.matches(chunk):
                seen.add(chunk.id)
                ids.append(chunk.id)
    return ids