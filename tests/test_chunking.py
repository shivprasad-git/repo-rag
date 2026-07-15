from pathlib import Path

from app.config import Settings
from app.ingest.chunker import create_chunks


def test_python_and_markdown_chunking(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text(
        "import jwt\n\nclass AuthService:\n    def validate_token(self, token):\n        return jwt.decode(token, 'demo-signing-key')\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Auth\n\nUse JWT tokens.\n\n# Install\n\nRun pip install.\n", encoding="utf-8")

    chunks = create_chunks(tmp_path, Settings())
    symbols = {chunk.metadata["symbol"] for chunk in chunks}

    assert "AuthService.validate_token" in symbols
    assert "Auth" in symbols
    assert "Install" in symbols
