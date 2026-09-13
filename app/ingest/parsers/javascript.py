"""JavaScript/TypeScript source parser producing code chunks.

Mirrors the Python parser's chunk model: ``file_metadata``, ``imports``,
``class`` headers, ``method``, ``function`` (including arrow-function
bindings), plus TypeScript-native ``ts_interface``, ``type_alias``, and
``enum`` chunks. Doc comments (``//`` and ``/* */``) directly above a
declaration are captured as the ``docstring`` metadata field.
"""
from __future__ import annotations

from pathlib import Path

from tree_sitter import Node

from app.ingest.parsers.common import (
    JAVASCRIPT_LANGUAGE,
    TYPESCRIPT_LANGUAGE,
    TSX_LANGUAGE,
    _base_metadata,
    _chunk_id,
    _decode,
    _join_metadata_values,
    _light_parse_error_metadata,
    _line_chunk,
    _node_text,
    _parse,
    _parse_error_chunks,
    _parse_error_metadata,
    _qualify_symbol,
)
from app.models import Chunk, ChunkType

_CLASS_ATTRIBUTE_TYPES = {"public_field_definition", "private_field_definition", "field_definition"}
_METHOD_TYPES = {"method_definition", "method_signature"}
_DECLARATION_TYPES = {
    "class_declaration",
    "function_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
}
_VARIABLE_DECLARATION_TYPES = {"lexical_declaration", "variable_declaration"}
def parse_javascript(
    path: Path,
    repo_path: Path,
    commit: str | None = None,
    language: str = "javascript",
    use_tsx: bool = False,
) -> list[Chunk]:
    """Parse a JavaScript/TypeScript source file into chunks."""
    source = path.read_bytes()
    lines = _decode(source).splitlines()
    metadata = _base_metadata(path, repo_path, language, commit)
    metadata["module"] = _module_name(metadata["file_path"])

    tree = _parse(source, _language(use_tsx, language))
    root = tree.root_node

    chunks: list[Chunk] = []
    imports = _top_level_imports(root)
    import_text = _join_metadata_values(_node_text(source, node) for node in imports)
    error_metadata = _parse_error_metadata(root)
    file_metadata_id = _chunk_id(metadata, ChunkType.FILE_METADATA, metadata["file_path"], 1, 1)
    js_metadata = metadata | _light_parse_error_metadata(error_metadata) | {"file_metadata_id": file_metadata_id}
    chunks.append(_file_metadata_chunk(metadata, file_metadata_id, import_text, error_metadata))
    if imports:
        chunks.append(_import_chunk(source, imports, js_metadata))

    for node in _top_level_declarations(root):
        declaration = _declaration(node)
        # Use the statement node (e.g. export_statement) as the anchor for doc
        # comments, since it starts closer to the comment than `declaration`.
        doc_node = node
        if declaration.type == "class_declaration":
            chunks.extend(_class_chunks(source, lines, declaration, js_metadata, doc_node))
        elif declaration.type == "function_declaration":
            name = _node_name(source, declaration)
            chunks.append(_node_chunk(source, lines, declaration, js_metadata, ChunkType.FUNCTION, name, doc_node=doc_node))
        elif declaration.type == "interface_declaration":
            name = _node_name(source, declaration)
            chunks.append(_node_chunk(source, lines, declaration, js_metadata, ChunkType.TS_INTERFACE, name, doc_node=doc_node))
        elif declaration.type == "type_alias_declaration":
            name = _node_name(source, declaration)
            chunks.append(_node_chunk(source, lines, declaration, js_metadata, ChunkType.TYPE_ALIAS, name, doc_node=doc_node))
        elif declaration.type == "enum_declaration":
            name = _node_name(source, declaration)
            chunks.append(_node_chunk(source, lines, declaration, js_metadata, ChunkType.ENUM, name, doc_node=doc_node))
        elif declaration.type in _VARIABLE_DECLARATION_TYPES:
            for binding in _function_bindings(declaration):
                name = _node_name(source, binding)
                chunks.append(_node_chunk(source, lines, binding, js_metadata, ChunkType.FUNCTION, name, doc_node=doc_node))

    chunks.extend(_parse_error_chunks(lines, root, js_metadata))
    return chunks


def _language(use_tsx: bool, language: str):
    if language == "typescript":
        return TSX_LANGUAGE if use_tsx else TYPESCRIPT_LANGUAGE
    return JAVASCRIPT_LANGUAGE


def _module_name(file_path: str) -> str:
    """Dotted module name: path without extension, dropping a trailing ``index``."""
    relative = Path(file_path).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "index":
        parts = parts[:-1]
    return ".".join(parts)


def _top_level_imports(root: Node) -> list[Node]:
    imports: list[Node] = []
    for node in root.children:
        if node.type == "import_statement":
            imports.append(node)
        elif node.type == "export_statement" and node.child_by_field_name("source") is not None:
            imports.append(node)
    return imports


def _top_level_declarations(root: Node) -> list[Node]:
    return [node for node in root.children if _declaration(node) is not None]


def _declaration(node: Node) -> Node | None:
    if node.type == "export_statement":
        return node.child_by_field_name("declaration")
    if node.type in _DECLARATION_TYPES | _VARIABLE_DECLARATION_TYPES:
        return node
    return None


def _function_bindings(declaration: Node) -> list[Node]:
    """Return variable declarators whose value is a function (``const f = () => …``)."""
    bindings: list[Node] = []
    for child in declaration.named_children:
        if child.type != "variable_declarator":
            continue
        value = child.child_by_field_name("value")
        if value is not None and value.type in _FUNCTION_VALUE_TYPES:
            bindings.append(child)
    return bindings
def _class_chunks(source: bytes, lines: list[str], class_node: Node, metadata: dict, doc_node: Node) -> list[Chunk]:
    """Build the class header chunk plus one chunk per class method."""
    class_name = _node_name(source, class_node)
    chunks = [
        _class_header_chunk(source, lines, class_node, metadata, class_name, doc_node),
    ]
    for method_node in _class_methods(class_node):
        method_name = _node_name(source, method_node)
        chunks.append(
            _node_chunk(
                source,
                lines,
                method_node,
                metadata,
                ChunkType.METHOD,
                f"{class_name}.{method_name}",
                extra={"class_name": class_name},
            )
        )
    return chunks


def _class_methods(class_node: Node) -> list[Node]:
    body = class_node.child_by_field_name("body")
    if body is None:
        return []
    return [node for node in body.named_children if node.type in _METHOD_TYPES]


def _class_header_chunk(
    source: bytes,
    lines: list[str],
    class_node: Node,
    metadata: dict,
    class_name: str,
    doc_node: Node,
) -> Chunk:
    signature = _signature(source, class_node)
    content_lines = [signature]
    content_lines.extend(_class_attribute_lines(source, class_node))
    end_line = class_node.start_point[0] + len(content_lines)
    return Chunk(
        id=_chunk_id(metadata, ChunkType.CLASS, class_name, class_node.start_point[0] + 1, end_line),
        content="\n".join(content_lines),
        metadata=metadata
        | {
            "chunk_type": ChunkType.CLASS.value,
            "symbol": class_name,
            "qualified_symbol": _qualify_symbol(metadata, class_name),
            "start_line": class_node.start_point[0] + 1,
            "end_line": end_line,
            "signature": signature,
            "docstring": _leading_doc(source, doc_node),
            "decorators": "",
            "is_async": False,
            "calls": "",
            "parent_symbol": "",
        },
    )


def _class_attribute_lines(source: bytes, class_node: Node) -> list[str]:
    body = class_node.child_by_field_name("body")
    if body is None:
        return []
    return [
        _node_text(source, node).strip()
        for node in body.named_children
        if node.type in _CLASS_ATTRIBUTE_TYPES
    ]


def _import_chunk(source: bytes, imports: list[Node], metadata: dict) -> Chunk:
    start_line = imports[0].start_point[0] + 1
    end_line = imports[-1].end_point[0] + 1
    content = "\n".join(_decode(source[node.start_byte : node.end_byte]) for node in imports)
    return Chunk(
        id=_chunk_id(metadata, ChunkType.IMPORTS, "imports", start_line, end_line),
        content=content,
        metadata=metadata
        | {
            "chunk_type": ChunkType.IMPORTS.value,
            "symbol": "imports",
            "qualified_symbol": _qualify_symbol(metadata, "imports"),
            "start_line": start_line,
            "end_line": end_line,
        },
    )


def _file_metadata_chunk(metadata: dict, file_metadata_id: str, imports: str, error_metadata: dict) -> Chunk:
    content_lines = [
        f"file: {metadata['file_path']}",
        f"module: {metadata.get('module', '')}",
    ]
    if imports:
        content_lines.extend(["imports:", imports])
    if error_metadata["parse_error_lines"]:
        content_lines.append(f"parse_error_lines: {error_metadata['parse_error_lines']}")

    return Chunk(
        id=file_metadata_id,
        content="\n".join(content_lines),
        metadata=metadata
        | error_metadata
        | {
            "chunk_type": ChunkType.FILE_METADATA.value,
            "symbol": metadata["file_path"],
            "qualified_symbol": _qualify_symbol(metadata, metadata["file_path"]),
            "start_line": 1,
            "end_line": 1,
            "imports": imports,
            "file_metadata_id": file_metadata_id,
        },
    )
def _node_chunk(
    source: bytes,
    lines: list[str],
    node: Node,
    metadata: dict,
    chunk_type: ChunkType,
    symbol: str,
    extra: dict | None = None,
    doc_node: Node | None = None,
) -> Chunk:
    return _line_chunk(
        lines,
        node.start_point[0],
        node.end_point[0],
        metadata,
        chunk_type,
        symbol,
        _js_metadata(source, node, metadata, symbol, extra, doc_node=doc_node) | (extra or {}),
    )


def _js_metadata(
    source: bytes,
    node: Node,
    metadata: dict,
    symbol: str,
    extra: dict | None,
    doc_node: Node | None = None,
) -> dict:
    parent_symbol = (extra or {}).get("class_name", "")
    return {
        "signature": _signature(source, node),
        "docstring": _leading_doc(source, doc_node or node),
        "decorators": "",
        "is_async": _has_keyword(node, "async"),
        "is_static": _has_keyword(node, "static"),
        "kind": _method_kind(node),
        "calls": _join_metadata_values(_call_names(source, node)),
        "parent_symbol": parent_symbol,
        "is_test": _is_test(metadata, symbol),
    }


def _node_name(source: bytes, node: Node) -> str:
    name = node.child_by_field_name("name")
    if name is None:
        return "<anonymous>"
    return _decode(source[name.start_byte : name.end_byte])


def _signature(source: bytes, node: Node) -> str:
    """First line of the declaration, stopping at the body ``{`` when present."""
    body = node.child_by_field_name("body")
    end_byte = body.start_byte if body is not None else node.end_byte
    first_line = _decode(source[node.start_byte:end_byte]).splitlines()
    return first_line[0].strip() if first_line else ""


def _has_keyword(node: Node, keyword: str) -> bool:
    if any(child.type == keyword for child in node.children):
        return True
    value = node.child_by_field_name("value")
    return value is not None and any(child.type == keyword for child in value.children)


def _method_kind(node: Node) -> str:
    if node.type not in _METHOD_TYPES:
        return ""
    for child in node.children:
        if child.type == "get":
            return "get"
        if child.type == "set":
            return "set"
    return "method"


def _leading_doc(source: bytes, node: Node) -> str:
    """Return the doc comment (``//`` lines or ``/* block */``) directly above *node*."""
    text = _decode(source)[: node.start_byte]
    lines = text.splitlines()
    doc: list[str] = []
    index = len(lines) - 1
    while index >= 0 and not lines[index].strip():
        index -= 1
    while index >= 0:
        line = lines[index].strip()
        if line.startswith("//"):
            doc.insert(0, line[2:].strip())
            index -= 1
            continue
        if line.startswith("/*") or line.endswith("*/"):
            block_start = index
            while block_start > 0 and "/*" not in lines[block_start]:
                block_start -= 1
            if "/*" not in lines[block_start]:
                break
            doc.insert(0, _strip_block_comment("\n".join(lines[block_start : index + 1])))
            break
        break
    return "\n".join(doc)


def _strip_block_comment(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("/**"):
        stripped = stripped[3:]
    elif stripped.startswith("/*"):
        stripped = stripped[2:]
    if stripped.endswith("*/"):
        stripped = stripped[:-2]
    cleaned = [
        line.strip().lstrip("*").strip() if line.strip().startswith("*") else line.strip()
        for line in stripped.splitlines()
    ]
    return "\n".join(cleaned).strip()


def _call_names(source: bytes, node: Node) -> list[str]:
    calls: set[str] = set()
    _collect_call_names(source, node, calls)
    return sorted(calls)


def _collect_call_names(source: bytes, node: Node, calls: set[str]) -> None:
    if node.type == "call_expression":
        function = node.child_by_field_name("function")
        if function is not None:
            calls.add(_node_text(source, function).strip())
    elif node.type == "new_expression":
        constructor = node.child_by_field_name("constructor")
        if constructor is not None:
            calls.add(f"new {_node_text(source, constructor).strip()}")
    for child in node.children:
        _collect_call_names(source, child, calls)


def _is_test(metadata: dict, symbol: str) -> bool:
    path = str(metadata.get("file_path", ""))
    name = symbol.rsplit(".", 1)[-1]
    if ".test." in path or ".spec." in path or "/__tests__/" in path or path.startswith("tests/"):
        return True
    return name.startswith("test") or name in ("it", "describe")
_FUNCTION_VALUE_TYPES = {"arrow_function", "function_expression", "function"}