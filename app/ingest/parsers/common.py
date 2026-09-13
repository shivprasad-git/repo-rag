"""Shared helpers used by more than one parser."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tree_sitter import Language, Node, Parser, Tree
import tree_sitter_javascript
import tree_sitter_markdown
import tree_sitter_python
import tree_sitter_typescript

from app.models import Chunk, ChunkType

PYTHON_LANGUAGE = Language(tree_sitter_python.language())
MARKDOWN_LANGUAGE = Language(tree_sitter_markdown.language())
JAVASCRIPT_LANGUAGE = Language(tree_sitter_javascript.language())
TYPESCRIPT_LANGUAGE = Language(tree_sitter_typescript.language_typescript())
TSX_LANGUAGE = Language(tree_sitter_typescript.language_tsx())


def _parse(source: bytes, language: Language) -> Tree:
    parser = Parser()
    parser.language = language
    return parser.parse(source)


def _base_metadata(path: Path, repo_path: Path, language: str, commit: str | None) -> dict:
    relative = path.relative_to(repo_path)
    return {
        "file_path": str(relative),
        "module": _module_name(relative) if language == "python" else "",
        "language": language,
        "commit": commit,
    }


def _parse_error_metadata(root: Node) -> dict:
    error_lines: set[int] = set()
    _collect_error_lines(root, error_lines)
    return {
        "has_parse_errors": root.has_error,
        "parse_error_lines": ",".join(str(line) for line in sorted(error_lines)),
    }


def _line_chunk(
    lines: list[str],
    start: int,
    end: int,
    metadata: dict,
    chunk_type: ChunkType,
    symbol: str,
    extra: dict | None = None,
) -> Chunk:
    return Chunk(
        id=_chunk_id(metadata, chunk_type, symbol, start + 1, end + 1),
        content="\n".join(lines[start : end + 1]),
        metadata=metadata
        | {
            "chunk_type": chunk_type.value,
            "symbol": symbol,
            "qualified_symbol": _qualify_symbol(metadata, symbol),
            "start_line": start + 1,
            "end_line": end + 1,
        }
        | (extra or {}),
    )


def _chunk_id(metadata: dict, chunk_type: ChunkType, symbol: str, start_line: int, end_line: int) -> str:
    raw = "|".join(
        [
            str(metadata.get("commit")),
            str(metadata.get("file_path")),
            chunk_type.value,
            symbol,
            str(start_line),
            str(end_line),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _decode(source: bytes) -> str:
    return source.decode("utf-8", errors="ignore")


def _node_text(source: bytes, node: Node) -> str:
    return _decode(source[node.start_byte : node.end_byte])


def _first_child_of_type(node: Node, node_type: str) -> Node | None:
    for child in node.children:
        if child.type == node_type:
            return child
    return None


def _join_metadata_values(values) -> str:
    return "\n".join(value for value in values if value)


def _collect_error_lines(node: Node, error_lines: set[int]) -> None:
    if node.type == "ERROR" or node.is_missing:
        error_lines.add(node.start_point[0] + 1)
    for child in node.children:
        _collect_error_lines(child, error_lines)


def _light_parse_error_metadata(error_metadata: dict) -> dict:
    return {
        "has_parse_errors": error_metadata["has_parse_errors"],
        "parse_error_lines": "",
    }


def _parse_error_chunks(lines: list[str], root: Node, metadata: dict) -> list[Chunk]:
    error_nodes: list[Node] = []
    _collect_error_nodes(root, error_nodes)
    return [
        _line_chunk(
            lines,
            error_node.start_point[0],
            error_node.end_point[0],
            metadata,
            ChunkType.PARSE_ERROR,
            f"parse_error:{error_node.start_point[0] + 1}",
        )
        for error_node in error_nodes
    ]


def _collect_error_nodes(node: Node, error_nodes: list[Node]) -> None:
    if node.type == "ERROR" or node.is_missing:
        error_nodes.append(node)
    for child in node.children:
        _collect_error_nodes(child, error_nodes)


def _module_name(relative_path: Path) -> str:
    without_suffix = relative_path.with_suffix("")
    parts = list(without_suffix.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _qualify_symbol(metadata: dict, symbol: str) -> str:
    module = metadata.get("module")
    return f"{module}.{symbol}" if module else symbol