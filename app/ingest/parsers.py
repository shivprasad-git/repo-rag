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
    import_text = _join_metadata_values(_node_text(source, node) for node in imports)
    python_metadata = metadata | _parse_error_metadata(root) | {"imports": import_text}
    if imports:
        chunks.append(_import_chunk(source, imports, python_metadata))

    for node in root.children:
        definition = _definition_node(node)
        if definition is None:
            continue
        if definition.type == "class_definition":
            chunks.extend(_class_chunks(source, lines, node, definition, python_metadata))
        elif definition.type == "function_definition":
            name = _node_name(source, definition)
            chunks.append(
                _node_chunk(
                    source,
                    lines,
                    node,
                    python_metadata,
                    "function",
                    name,
                    definition_node=definition,
                )
            )

    chunks.extend(_parse_error_chunks(lines, root, python_metadata))
    return chunks


def parse_markdown(path: Path, repo_path: Path, commit: str | None = None) -> list[Chunk]:
    source = path.read_bytes()
    tree = _parse(source, MARKDOWN_LANGUAGE)
    metadata = _base_metadata(path, repo_path, "markdown", commit) | _parse_error_metadata(tree.root_node)
    lines = _decode(source).splitlines()
    headings = _markdown_headings(tree, source)

    if not headings and lines:
        return [_line_chunk(lines, 0, len(lines) - 1, metadata, "markdown_section", "Document")]

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
                "markdown_section",
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


def _class_chunks(source: bytes, lines: list[str], chunk_node: Node, definition_node: Node, metadata: dict) -> list[Chunk]:
    class_name = _node_name(source, definition_node)
    methods = _class_methods(definition_node)
    chunks = [
        _node_chunk(
            source,
            lines,
            chunk_node,
            metadata,
            "class",
            class_name,
            definition_node=definition_node,
        )
    ]

    for method_chunk_node, method_definition in methods:
        method_name = _node_name(source, method_definition)
        chunks.append(
            _node_chunk(
                source,
                lines,
                method_chunk_node,
                metadata,
                "method",
                f"{class_name}.{method_name}",
                extra={"class_name": class_name},
                definition_node=method_definition,
            )
        )

    return chunks


def _class_methods(class_node: Node) -> list[tuple[Node, Node]]:
    body = class_node.child_by_field_name("body")
    if body is None:
        return []
    methods: list[tuple[Node, Node]] = []
    for node in body.children:
        definition = _definition_node(node)
        if definition is not None and definition.type == "function_definition":
            methods.append((node, definition))
    return methods


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
            "qualified_symbol": _qualify_symbol(metadata, "imports"),
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
    definition_node: Node | None = None,
) -> Chunk:
    start = node.start_point[0]
    end = node.end_point[0]
    definition = definition_node or node
    return _line_chunk(
        lines,
        start,
        end,
        metadata,
        chunk_type,
        symbol,
        _python_metadata(source, node, definition, metadata, symbol, extra) | (extra or {}),
    )


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
            "qualified_symbol": _qualify_symbol(metadata, symbol),
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


def _definition_node(node: Node) -> Node | None:
    if node.type in {"class_definition", "function_definition"}:
        return node
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in {"class_definition", "function_definition"}:
                return child
    return None


def _python_metadata(
    source: bytes,
    chunk_node: Node,
    definition_node: Node,
    metadata: dict,
    symbol: str,
    extra: dict | None,
) -> dict:
    parent_symbol = (extra or {}).get("class_name", "")
    return {
        "signature": _signature(source, definition_node),
        "docstring": _docstring(source, definition_node),
        "decorators": _join_metadata_values(_decorators(source, chunk_node)),
        "is_async": _is_async(definition_node),
        "calls": _join_metadata_values(_call_names(source, definition_node)),
        "parent_symbol": parent_symbol,
        "is_test": _is_test(metadata, symbol),
    }


def _signature(source: bytes, node: Node) -> str:
    first_line = _node_text(source, node).splitlines()[0].strip()
    return first_line


def _docstring(source: bytes, node: Node) -> str:
    body = node.child_by_field_name("body")
    if body is None:
        return ""
    for child in body.children:
        if child.type != "expression_statement":
            continue
        string_node = _first_child_of_type(child, "string")
        if string_node is None:
            return ""
        return _strip_string_quotes(_node_text(source, string_node))
    return ""


def _decorators(source: bytes, node: Node) -> list[str]:
    if node.type != "decorated_definition":
        return []
    return [_node_text(source, child).strip() for child in node.children if child.type == "decorator"]


def _is_async(node: Node) -> bool:
    return any(child.type == "async" for child in node.children)


def _call_names(source: bytes, node: Node) -> list[str]:
    calls: set[str] = set()
    _collect_call_names(source, node, calls)
    return sorted(calls)


def _collect_call_names(source: bytes, node: Node, calls: set[str]) -> None:
    if node.type == "call":
        function = node.child_by_field_name("function")
        if function is not None:
            calls.add(_node_text(source, function).strip())
    for child in node.children:
        _collect_call_names(source, child, calls)


def _first_child_of_type(node: Node, node_type: str) -> Node | None:
    for child in node.children:
        if child.type == node_type:
            return child
    return None


def _strip_string_quotes(text: str) -> str:
    stripped = text.strip()
    for quote in ('"""', "'''", '"', "'"):
        if stripped.startswith(quote) and stripped.endswith(quote):
            return stripped[len(quote) : -len(quote)].strip()
    return stripped


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


def _base_metadata(path: Path, repo_path: Path, language: str, commit: str | None) -> dict:
    relative = path.relative_to(repo_path)
    return {
        "repo": repo_path.name,
        "repo_path": str(repo_path),
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


def _parse_error_chunks(lines: list[str], root: Node, metadata: dict) -> list[Chunk]:
    error_nodes: list[Node] = []
    _collect_error_nodes(root, error_nodes)
    return [
        _line_chunk(
            lines,
            error_node.start_point[0],
            error_node.end_point[0],
            metadata,
            "parse_error",
            f"parse_error:{error_node.start_point[0] + 1}",
        )
        for error_node in error_nodes
    ]


def _collect_error_nodes(node: Node, error_nodes: list[Node]) -> None:
    if node.type == "ERROR" or node.is_missing:
        error_nodes.append(node)
    for child in node.children:
        _collect_error_nodes(child, error_nodes)


def _collect_error_lines(node: Node, error_lines: set[int]) -> None:
    if node.type == "ERROR" or node.is_missing:
        error_lines.add(node.start_point[0] + 1)
    for child in node.children:
        _collect_error_lines(child, error_lines)


def _module_name(relative_path: Path) -> str:
    without_suffix = relative_path.with_suffix("")
    parts = list(without_suffix.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _qualify_symbol(metadata: dict, symbol: str) -> str:
    module = metadata.get("module")
    return f"{module}.{symbol}" if module else symbol


def _is_test(metadata: dict, symbol: str) -> bool:
    path = str(metadata.get("file_path", ""))
    name = symbol.rsplit(".", 1)[-1]
    return path.startswith("tests/") or "/tests/" in path or Path(path).name.startswith("test_") or name.startswith("test_")


def _decode(source: bytes) -> str:
    return source.decode("utf-8", errors="ignore")


def _node_text(source: bytes, node: Node) -> str:
    return _decode(source[node.start_byte : node.end_byte])


def _join_metadata_values(values) -> str:
    return "\n".join(value for value in values if value)


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
