from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from app.models import Chunk


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
    source = path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    base_metadata = _base_metadata(path, repo_path, "python", commit)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [_whole_file_chunk(source, base_metadata, "python_file")]

    chunks: list[Chunk] = []
    imports = _collect_imports(tree)
    if imports:
        chunks.append(
            Chunk(
                id=_chunk_id(base_metadata, "imports", "imports", 1, max(1, len(imports))),
                content="\n".join(imports),
                metadata=base_metadata
                | {
                    "chunk_type": "imports",
                    "symbol": "imports",
                    "start_line": 1,
                    "end_line": max(1, len(imports)),
                },
            )
        )

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            chunks.extend(_class_chunks(node, lines, base_metadata))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(_node_chunk(node, lines, base_metadata, "function", node.name))

    if not chunks:
        chunks.append(_whole_file_chunk(source, base_metadata, "python_file"))

    return chunks


def parse_markdown(path: Path, repo_path: Path, commit: str | None = None) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    metadata = _base_metadata(path, repo_path, "markdown", commit)
    chunks: list[Chunk] = []
    section_start = 0
    section_title = "Document"

    for index, line in enumerate(lines):
        if line.startswith("#") and line.lstrip("#").startswith(" "):
            if index > section_start:
                chunks.append(
                    _line_chunk(lines, section_start, index - 1, metadata, "markdown_section", section_title)
                )
            section_start = index
            section_title = line.lstrip("#").strip() or "Section"

    if lines:
        chunks.append(_line_chunk(lines, section_start, len(lines) - 1, metadata, "markdown_section", section_title))

    return [chunk for chunk in chunks if chunk.content.strip()]


def parse_text(path: Path, repo_path: Path, commit: str | None = None) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    metadata = _base_metadata(path, repo_path, "text", commit)
    return [_whole_file_chunk(text, metadata, "text_file")] if text.strip() else []


def _class_chunks(node: ast.ClassDef, lines: list[str], metadata: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    methods = [child for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not methods:
        return [_node_chunk(node, lines, metadata, "class", node.name)]

    class_header_end = min(method.lineno for method in methods) - 2
    if node.lineno - 1 <= class_header_end:
        chunks.append(_line_chunk(lines, node.lineno - 1, class_header_end, metadata, "class", node.name))

    for method in methods:
        chunks.append(
            _node_chunk(method, lines, metadata, "method", f"{node.name}.{method.name}", class_name=node.name)
        )
    return chunks


def _node_chunk(
    node: ast.AST,
    lines: list[str],
    metadata: dict,
    chunk_type: str,
    symbol: str,
    class_name: str | None = None,
) -> Chunk:
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1) - 1
    extra = {"class_name": class_name} if class_name else {}
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


def _whole_file_chunk(content: str, metadata: dict, chunk_type: str) -> Chunk:
    line_count = max(1, len(content.splitlines()))
    return Chunk(
        id=_chunk_id(metadata, chunk_type, metadata["file_path"], 1, line_count),
        content=content,
        metadata=metadata
        | {
            "chunk_type": chunk_type,
            "symbol": metadata["file_path"],
            "start_line": 1,
            "end_line": line_count,
        },
    )


def _base_metadata(path: Path, repo_path: Path, language: str, commit: str | None) -> dict:
    relative = path.relative_to(repo_path)
    return {
        "repo": repo_path.name,
        "repo_path": str(repo_path),
        "file_path": str(relative),
        "language": language,
        "commit": commit,
    }


def _collect_imports(tree: ast.AST) -> list[str]:
    imports: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
    return imports


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

