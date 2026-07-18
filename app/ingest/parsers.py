from __future__ import annotations

import hashlib
from pathlib import Path

from tree_sitter import Language, Node, Parser, Tree
import tree_sitter_markdown
import tree_sitter_python

from app.models import Chunk


PYTHON_LANGUAGE = Language(tree_sitter_python.language())
MARKDOWN_LANGUAGE = Language(tree_sitter_markdown.language())


def parse_file(path: Path, repo_path: Path, commit: str | None = None) -> list[Chunk]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return parse_python(path, repo_path, commit)
    if suffix == ".md":
        return parse_markdown(path, repo_path, commit)
    if suffix == ".txt":
        return parse_text(path, repo_path, commit)
    return []


def parse_python(path: Path, repo_path: Path, commit: str | None = None) -> list[Chunk]:
    source = path.read_bytes()
    lines = _decode(source).splitlines()
    metadata = _base_metadata(path, repo_path, "python", commit)
    tree = _parse(source, PYTHON_LANGUAGE)
    root = tree.root_node

    chunks: list[Chunk] = []
    imports = _top_level_nodes(root, {"import_statement", "import_from_statement"})
    if imports:
        chunks.append(_import_chunk(source, imports, metadata))

    for node in root.children:
        if node.type == "class_definition":
            chunks.extend(_class_chunks(source, lines, node, metadata))
        elif node.type == "function_definition":
            name = _node_name(source, node)
            chunks.append(_node_chunk(source, lines, node, metadata, "function", name))

    return chunks


def parse_markdown(path: Path, repo_path: Path, commit: str | None = None) -> list[Chunk]:
    source = path.read_bytes()
    tree = _parse(source, MARKDOWN_LANGUAGE)
    metadata = _base_metadata(path, repo_path, "markdown", commit)
    lines = _decode(source).splitlines()
    heading_lines = _markdown_heading_lines(tree)

    if not heading_lines and lines:
        return [_line_chunk(lines, 0, len(lines) - 1, metadata, "markdown_section", "Document")]

    chunks: list[Chunk] = []
    for index, start in enumerate(heading_lines):
        end = heading_lines[index + 1] - 1 if index + 1 < len(heading_lines) else len(lines) - 1
        title = lines[start].lstrip("#").strip() or "Section"
        chunks.append(_line_chunk(lines, start, end, metadata, "markdown_section", title))

    return [chunk for chunk in chunks if chunk.content.strip()]


def parse_text(path: Path, repo_path: Path, commit: str | None = None) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    metadata = _base_metadata(path, repo_path, "text", commit)
    line_count = max(1, len(text.splitlines()))
    return [
        Chunk(
            id=_chunk_id(metadata, "text_file", metadata["file_path"], 1, line_count),
            content=text,
            metadata=metadata
            | {
                "chunk_type": "text_file",
                "symbol": metadata["file_path"],
                "start_line": 1,
                "end_line": line_count,
            },
        )
    ] if text.strip() else []


def _parse(source: bytes, language: Language) -> Tree:
    parser = Parser()
    parser.language = language
    return parser.parse(source)


def _top_level_nodes(root: Node, node_types: set[str]) -> list[Node]:
    return [node for node in root.children if node.type in node_types]


def _class_chunks(source: bytes, lines: list[str], node: Node, metadata: dict) -> list[Chunk]:
    class_name = _node_name(source, node)
    methods = _class_methods(node)
    chunks = [_node_chunk(source, lines, node, metadata, "class", class_name)]

    for method in methods:
        method_name = _node_name(source, method)
        chunks.append(
            _node_chunk(
                source,
                lines,
                method,
                metadata,
                "method",
                f"{class_name}.{method_name}",
                extra={"class_name": class_name},
            )
        )

    return chunks


def _class_methods(class_node: Node) -> list[Node]:
    body = class_node.child_by_field_name("body")
    if body is None:
        return []
    return [node for node in body.children if node.type == "function_definition"]


def _import_chunk(source: bytes, imports: list[Node], metadata: dict) -> Chunk:
    start_line = imports[0].start_point[0] + 1
    end_line = imports[-1].end_point[0] + 1
    content = "\n".join(_decode(source[node.start_byte : node.end_byte]) for node in imports)
    return Chunk(
        id=_chunk_id(metadata, "imports", "imports", start_line, end_line),
        content=content,
        metadata=metadata
        | {
            "chunk_type": "imports",
            "symbol": "imports",
            "start_line": start_line,
            "end_line": end_line,
        },
    )


def _node_chunk(
    source: bytes,
    lines: list[str],
    node: Node,
    metadata: dict,
    chunk_type: str,
    symbol: str,
    extra: dict | None = None,
) -> Chunk:
    start = node.start_point[0]
    end = node.end_point[0]
    return _line_chunk(lines, start, end, metadata, chunk_type, symbol, extra)


def _line_chunk(
    lines: list[str],
    start: int,
    end: int,
    metadata: dict,
    chunk_type: str,
    symbol: str,
    extra: dict | None = None,
) -> Chunk:
    return Chunk(
        id=_chunk_id(metadata, chunk_type, symbol, start + 1, end + 1),
        content="\n".join(lines[start : end + 1]),
        metadata=metadata
        | {
            "chunk_type": chunk_type,
            "symbol": symbol,
            "start_line": start + 1,
            "end_line": end + 1,
        }
        | (extra or {}),
    )


def _node_name(source: bytes, node: Node) -> str:
    name = node.child_by_field_name("name")
    if name is None:
        return "<anonymous>"
    return _decode(source[name.start_byte : name.end_byte])


def _markdown_heading_lines(tree: Tree) -> list[int]:
    heading_lines: set[int] = set()
    _collect_markdown_headings(tree.root_node, heading_lines)
    return sorted(heading_lines)


def _collect_markdown_headings(node: Node, heading_lines: set[int]) -> None:
    if "heading" in node.type:
        heading_lines.add(node.start_point[0])
    for child in node.children:
        _collect_markdown_headings(child, heading_lines)


def _base_metadata(path: Path, repo_path: Path, language: str, commit: str | None) -> dict:
    relative = path.relative_to(repo_path)
    return {
        "repo": repo_path.name,
        "repo_path": str(repo_path),
        "file_path": str(relative),
        "language": language,
        "commit": commit,
    }


def _decode(source: bytes) -> str:
    return source.decode("utf-8", errors="ignore")


def _chunk_id(metadata: dict, chunk_type: str, symbol: str, start_line: int, end_line: int) -> str:
    raw = "|".join(
        [
            str(metadata.get("repo")),
            str(metadata.get("commit")),
            str(metadata.get("file_path")),
            chunk_type,
            symbol,
            str(start_line),
            str(end_line),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()
