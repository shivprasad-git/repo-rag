"""Chunking tests for the JavaScript/TypeScript parser."""
from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.ingest.chunker import create_chunks
from app.ingest.discover import discover_files
from app.models import ChunkType


class TestTokenizer:
    def count(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_many(self, texts):
        return [self.count(text) for text in texts]


def _use_test_tokenizer(monkeypatch) -> None:
    monkeypatch.setattr("app.ingest.splitter.get_tokenizer", lambda model_name: TestTokenizer())


JS_SOURCE = "\n".join(
    [
        'import jwt from "jsonwebtoken";',
        "",
        "/**",
        " * Handles authentication.",
        " */",
        "export class AuthService {",
        "  /** Token ttl in seconds. */",
        "  static TOKEN_TTL = 3600;",
        "",
        "  async login(username, password) {",
        '    return jwt.sign({ sub: username }, "key");',
        "  }",
        "",
        '  get name() {',
        '    return "Auth";',
        "  }",
        "}",
        "",
        "export function helper(x) {",
        "  return x * 2;",
        "}",
        "",
        "const createToken = async (user) => {",
        '  return jwt.sign(user, "key");',
        "};",
        "",
        "function anything(x) {",
        "  return x;",
        "}",
    ]
)


def test_javascript_chunks_classes_methods_functions_and_arrows(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    (tmp_path / "app.js").write_text(JS_SOURCE, encoding="utf-8")

    chunks = create_chunks(tmp_path, Settings())
    by_symbol = {chunk.metadata["symbol"]: chunk for chunk in chunks}

    assert "AuthService" in by_symbol
    assert "AuthService.login" in by_symbol
    assert "AuthService.name" in by_symbol
    assert "helper" in by_symbol
    assert "createToken" in by_symbol

    class_chunk = by_symbol["AuthService"]
    assert class_chunk.metadata["chunk_type"] == "class"
    assert class_chunk.metadata["docstring"] == "Handles authentication."
    assert "class AuthService" in class_chunk.content
    assert "static TOKEN_TTL = 3600" in class_chunk.content
    assert "async login" not in class_chunk.content

    login = by_symbol["AuthService.login"]
    assert login.metadata["chunk_type"] == "method"
    assert login.metadata["is_async"] is True
    assert login.metadata["kind"] == "method"
    assert login.metadata["signature"] == "async login(username, password)"
    assert login.metadata["class_name"] == "AuthService"
    assert login.metadata["parent_symbol"] == "AuthService"
    assert login.metadata["calls"] == "jwt.sign"

    getter = by_symbol["AuthService.name"]
    assert getter.metadata["kind"] == "get"

    create_token = by_symbol["createToken"]
    assert create_token.metadata["chunk_type"] == "function"
    assert create_token.metadata["is_async"] is True
    assert create_token.metadata["parent_symbol"] == ""

    imports = [c for c in chunks if c.metadata["chunk_type"] == "imports"]
    assert len(imports) == 1
    assert "jsonwebtoken" in imports[0].content

    file_metadata = next(c for c in chunks if c.metadata["chunk_type"] == "file_metadata")
    assert file_metadata.metadata["module"] == "app"
    assert file_metadata.metadata["has_parse_errors"] is False
TS_SOURCE = "\n".join(
    [
        "export interface User {",
        "  id: number;",
        "  name: string;",
        "}",
        "",
        "export type ID = string | number;",
        "",
        "export enum Color {",
        "  Red,",
        "  Green,",
        "}",
        "",
        "export class UserRepo implements Repo<User> {",
        "  find(id: ID): User | null {",
        "    return null;",
        "  }",
        "}",
    ]
)


def test_typescript_chunks_interfaces_types_enums_and_classes(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    (tmp_path / "types.ts").write_text(TS_SOURCE, encoding="utf-8")

    chunks = create_chunks(tmp_path, Settings())
    by_symbol = {chunk.metadata["symbol"]: chunk for chunk in chunks}

    assert by_symbol["User"].metadata["chunk_type"] == "ts_interface"
    assert by_symbol["User"].metadata["signature"] == "interface User"
    assert by_symbol["ID"].metadata["chunk_type"] == "type_alias"
    assert by_symbol["Color"].metadata["chunk_type"] == "enum"
    assert by_symbol["Color"].metadata["signature"] == "enum Color"

    repo_class = by_symbol["UserRepo"]
    assert repo_class.metadata["chunk_type"] == "class"
    assert repo_class.metadata["signature"] == "class UserRepo implements Repo<User>"
    assert "find" not in repo_class.content

    find = by_symbol["UserRepo.find"]
    assert find.metadata["chunk_type"] == "method"
    assert find.metadata["signature"] == "find(id: ID): User | null"
    assert find.metadata["class_name"] == "UserRepo"

    assert not any(chunk.metadata["has_parse_errors"] for chunk in chunks)


def test_jsx_and_tsx_components_parse(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    (tmp_path / "App.tsx").write_text(
        'export function App() {\n  return <div className="root">hello</div>;\n}\n',
        encoding="utf-8",
    )
    (tmp_path / "Card.jsx").write_text(
        "function Card({ title }) {\n  return <div>{title}</div>;\n}\n",
        encoding="utf-8",
    )

    chunks = create_chunks(tmp_path, Settings())
    by_symbol = {chunk.metadata["symbol"]: chunk for chunk in chunks}

    assert by_symbol["App"].metadata["language"] == "typescript"
    assert by_symbol["Card"].metadata["language"] == "javascript"
    assert not any(chunk.metadata["has_parse_errors"] for chunk in chunks)


def test_broken_javascript_reports_parse_errors(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    (tmp_path / "broken.js").write_text(
        "function broken( {\n  return 1;\n}\n",
        encoding="utf-8",
    )

    chunks = create_chunks(tmp_path, Settings())

    assert chunks
    assert any(chunk.metadata["has_parse_errors"] for chunk in chunks)
    assert any(chunk.metadata["chunk_type"] == "parse_error" for chunk in chunks)


def test_js_ts_extensions_are_discovered(tmp_path: Path) -> None:
    for name in ("a.js", "b.jsx", "c.ts", "d.tsx", "e.py", "skip.vue"):
        (tmp_path / name).write_text("", encoding="utf-8")

    discovered = sorted(path.name for path in discover_files(tmp_path, Settings()))

    assert discovered == ["a.js", "b.jsx", "c.ts", "d.tsx", "e.py"]


def test_ts_chunk_types_are_searchable_and_splittable() -> None:
    settings = Settings()
    searchable = {chunk_type.value for chunk_type in settings.searchable_chunk_types}
    splittable = set(settings.splittable_chunk_types)

    assert {"ts_interface", "type_alias", "enum"} <= searchable
    assert {ChunkType.TS_INTERFACE, ChunkType.TYPE_ALIAS, ChunkType.ENUM} <= splittable