from pathlib import Path

from app.config import Settings
from app.embeddings import EmbeddingProvider
from app.pipeline import index_repository


class TestEmbeddingProvider(EmbeddingProvider):
    embedded_texts: list[str] = []

    @property
    def dimensions(self) -> int:
        return 2

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]


def test_incremental_indexing_embeds_only_changed_files(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "auth.py").write_text("def login():\n    return 'token'\n", encoding="utf-8")
    (repo_path / "billing.py").write_text("def charge():\n    return 'paid'\n", encoding="utf-8")
    settings = Settings(
        repositories_dir=tmp_path / "repositories",
        indexes_dir=tmp_path / "indexes",
        max_chunk_tokens=0,
    )
    embedder = TestEmbeddingProvider()
    monkeypatch.setattr("app.pipeline.build_embedder", lambda settings: embedder)

    _, first_count = index_repository(str(repo_path), "simple", settings, index_name="test")

    assert first_count > 0
    assert any("def login" in text for text in embedder.embedded_texts)
    assert any("def charge" in text for text in embedder.embedded_texts)

    embedder.embedded_texts.clear()
    (repo_path / "auth.py").write_text("def login():\n    return 'new-token'\n", encoding="utf-8")

    _, second_count = index_repository(str(repo_path), "simple", settings, index_name="test")

    assert second_count == first_count
    assert any("new-token" in text for text in embedder.embedded_texts)
    assert not any("def charge" in text for text in embedder.embedded_texts)


def test_incremental_indexing_removes_deleted_files(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    billing_path = repo_path / "billing.py"
    (repo_path / "auth.py").write_text("def login():\n    return 'token'\n", encoding="utf-8")
    billing_path.write_text("def charge():\n    return 'paid'\n", encoding="utf-8")
    settings = Settings(
        repositories_dir=tmp_path / "repositories",
        indexes_dir=tmp_path / "indexes",
        max_chunk_tokens=0,
    )
    embedder = TestEmbeddingProvider()
    monkeypatch.setattr("app.pipeline.build_embedder", lambda settings: embedder)

    index_repository(str(repo_path), "simple", settings, index_name="test")
    billing_path.unlink()

    _, chunk_count = index_repository(str(repo_path), "simple", settings, index_name="test")
    chunk_records = (settings.indexes_dir / "chunks" / "test.json").read_text(encoding="utf-8")

    assert chunk_count > 0
    assert "billing.py" not in chunk_records
