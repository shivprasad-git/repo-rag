from pathlib import Path

from app.config import Settings
from app.ingest.chunker import create_chunks


def test_python_and_markdown_chunking(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text(
        "\n".join(
            [
                "import jwt",
                "",
                "class AuthService:",
                "    \"\"\"Authentication helpers.\"\"\"",
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
    assert "Auth" in symbols
    assert "Login" in symbols
    assert "Install" in symbols
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
    route_metadata = chunks_by_symbol["login_route"].metadata
    assert route_metadata["signature"] == "async def login_route(username: str) -> str:"
    assert route_metadata["decorators"] == "@router.post('/login')"
    assert route_metadata["is_async"] is True
    assert route_metadata["calls"] == "create_token"
    assert route_metadata["qualified_symbol"] == "auth.login_route"
    assert chunks_by_symbol["test_login_route"].metadata["is_test"] is True
    assert chunks_by_symbol["Auth"].metadata["heading_level"] == 1
    assert chunks_by_symbol["Login"].metadata["parent_headings"] == "Auth"


def test_tree_sitter_parse_error_metadata(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def broken(:\n    pass\n", encoding="utf-8")

    chunks = create_chunks(tmp_path, Settings())

    assert chunks
    assert any(chunk.metadata["has_parse_errors"] for chunk in chunks)
    assert any(chunk.metadata["parse_error_lines"] for chunk in chunks)
