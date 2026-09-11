"""Markdown source parser producing section chunks."""

from __future__ import annotations

from pathlib import Path

from tree_sitter import Node, Tree

from app.ingest.parsers.common import (
    MARKDOWN_LANGUAGE,
    _base_metadata,
    _decode,
    _first_child_of_type,
    _line_chunk,
    _node_text,
    _parse,
    _parse_error_metadata,
)
from app.models import Chunk, ChunkType


def parse_markdown(path: Path, repo_path: Path, commit: str | None = None) -> list[Chunk]:
    source = path.read_bytes()
    tree = _parse(source, MARKDOWN_LANGUAGE)
    metadata = _base_metadata(path, repo_path, "markdown", commit) | _parse_error_metadata(tree.root_node)
    lines = _decode(source).splitlines()
    headings = _markdown_headings(tree, source)

    if not headings and lines:
        chunk = _line_chunk(lines, 0, len(lines) - 1, metadata, ChunkType.MARKDOWN_SECTION, "Document")
        return [chunk] if chunk.content.strip() else []

    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []
    for index, heading in enumerate(headings):
        start = heading["line"]
        end = headings[index + 1]["line"] - 1 if index + 1 < len(headings) else len(lines) - 1
        level = heading["level"]
        title = heading["title"]
        heading_stack = [(parent_level, parent_title) for parent_level, parent_title in heading_stack if parent_level < level]
        parent_headings = [parent_title for _, parent_title in heading_stack]
        chunks.append(
            _line_chunk(
                lines,
                start,
                end,
                metadata,
                ChunkType.MARKDOWN_SECTION,
                title,
                {
                    "heading": title,
                    "heading_level": level,
                    "parent_headings": " > ".join(parent_headings),
                },
            )
        )
        heading_stack.append((level, title))

    return [chunk for chunk in chunks if chunk.content.strip()]


def _markdown_headings(tree: Tree, source: bytes) -> list[dict]:
    headings: list[dict] = []
    _collect_markdown_headings(tree.root_node, source, headings)
    return sorted(headings, key=lambda heading: heading["line"])


def _collect_markdown_headings(node: Node, source: bytes, headings: list[dict]) -> None:
    if node.type == "atx_heading":
        headings.append(
            {
                "line": node.start_point[0],
                "level": _markdown_heading_level(node),
                "title": _markdown_heading_title(source, node),
            }
        )
    for child in node.children:
        _collect_markdown_headings(child, source, headings)


def _markdown_heading_level(node: Node) -> int:
    for child in node.children:
        if child.type.startswith("atx_h") and child.type.endswith("_marker"):
            return int(child.type.removeprefix("atx_h").removesuffix("_marker"))
    return 1


def _markdown_heading_title(source: bytes, node: Node) -> str:
    inline = _first_child_of_type(node, "inline")
    if inline is None:
        return "Section"
    return _node_text(source, inline).strip() or "Section"