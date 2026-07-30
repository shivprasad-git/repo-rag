from pathlib import Path

from app.config import Settings
from app.ingest.chunker import create_chunks


class TestTokenizer:
    def count(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_many(self, texts: list[str]) -> list[int]:
        return [self.count(text) for text in texts]


def _use_test_tokenizer(monkeypatch) -> None:
    monkeypatch.setattr("app.ingest.splitter.get_tokenizer", lambda model_name: TestTokenizer())


def test_python_and_markdown_chunking(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    (tmp_path / "auth.py").write_text(
        "\n".join(
            [
                "import jwt",
                "",
                "class AuthService:",
                "    \"\"\"Authentication helpers.\"\"\"",
                "    TOKEN_TTL = 3600",
                "",
                "    @staticmethod",
                "    def validate_token(token):",
                "        \"\"\"Validate a JWT token.\"\"\"",
                "        return jwt.decode(token, 'demo-signing-key')",
                "",
                "@router.post('/login')",
                "async def login_route(username: str) -> str:",
                "    return create_token(username)",
                "",
                "def test_login_route():",
                "    return login_route('shiv')",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# Auth\n\nUse JWT tokens.\n\n## Login\n\nCreate a session.\n\n# Install\n\nRun pip install.\n",
        encoding="utf-8",
    )

    chunks = create_chunks(tmp_path, Settings())
    chunks_by_symbol = {chunk.metadata["symbol"]: chunk for chunk in chunks}
    symbols = {chunk.metadata["symbol"] for chunk in chunks}

    assert "AuthService.validate_token" in symbols
    assert "login_route" in symbols
    assert "test_login_route" in symbols
    assert "auth.py" in symbols
    assert "Auth" in symbols
    assert "Login" in symbols
    assert "Install" in symbols
    class_chunk = chunks_by_symbol["AuthService"]
    assert "class AuthService:" in class_chunk.content
    assert "Authentication helpers." in class_chunk.content
    assert "TOKEN_TTL = 3600" in class_chunk.content
    assert "def validate_token" not in class_chunk.content
    assert "jwt.decode" not in class_chunk.content
    method_metadata = chunks_by_symbol["AuthService.validate_token"].metadata
    assert method_metadata["signature"] == "def validate_token(token):"
    assert method_metadata["docstring"] == "Validate a JWT token."
    assert method_metadata["decorators"] == "@staticmethod"
    assert method_metadata["calls"] == "jwt.decode"
    assert method_metadata["class_name"] == "AuthService"
    assert method_metadata["parent_symbol"] == "AuthService"
    assert method_metadata["qualified_symbol"] == "auth.AuthService.validate_token"
    assert method_metadata["module"] == "auth"
    assert method_metadata["has_parse_errors"] is False
    assert method_metadata["parse_error_lines"] == ""
    assert "imports" not in method_metadata
    file_metadata = chunks_by_symbol["auth.py"].metadata
    assert file_metadata["chunk_type"] == "file_metadata"
    assert file_metadata["imports"] == "import jwt"
    assert method_metadata["file_metadata_id"] == file_metadata["file_metadata_id"]
    route_metadata = chunks_by_symbol["login_route"].metadata
    assert route_metadata["signature"] == "async def login_route(username: str) -> str:"
    assert route_metadata["decorators"] == "@router.post('/login')"
    assert route_metadata["is_async"] is True
    assert route_metadata["calls"] == "create_token"
    assert route_metadata["qualified_symbol"] == "auth.login_route"
    assert chunks_by_symbol["test_login_route"].metadata["is_test"] is True
    assert chunks_by_symbol["Auth"].metadata["heading_level"] == 1
    assert chunks_by_symbol["Login"].metadata["parent_headings"] == "Auth"


def test_tree_sitter_parse_error_metadata(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    (tmp_path / "broken.py").write_text("def broken(:\n    pass\n", encoding="utf-8")

    chunks = create_chunks(tmp_path, Settings())

    assert chunks
    assert any(chunk.metadata["has_parse_errors"] for chunk in chunks)
    file_metadata_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "file_metadata"]
    assert file_metadata_chunks
    assert any(chunk.metadata["parse_error_lines"] for chunk in file_metadata_chunks)


def test_oversized_method_chunks_are_split_into_parts(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    body_lines = [f"        value_{index} = {index}" for index in range(30)]
    (tmp_path / "large.py").write_text(
        "\n".join(["class LargeService:", "    def run(self):", *body_lines, "        return value_29"]),
        encoding="utf-8",
    )

    chunks = create_chunks(
        tmp_path,
        Settings(max_chunk_tokens=40, chunk_overlap_tokens=8),
    )

    method_parts = [
        chunk
        for chunk in chunks
        if chunk.metadata["chunk_type"] == "method"
        and chunk.metadata["symbol"] == "LargeService.run"
    ]

    assert len(method_parts) > 1
    assert all(chunk.metadata["is_chunk_part"] is True for chunk in method_parts)
    assert [chunk.metadata["part_index"] for chunk in method_parts] == list(range(1, len(method_parts) + 1))
    assert {chunk.metadata["part_count"] for chunk in method_parts} == {len(method_parts)}
    assert len({chunk.metadata["parent_chunk_id"] for chunk in method_parts}) == 1
    assert method_parts[0].metadata["start_line"] == 2
    assert method_parts[-1].metadata["end_line"] == 33
