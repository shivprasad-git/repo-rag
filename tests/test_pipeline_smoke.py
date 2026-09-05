from pathlib import Path

from app.config import Settings
from app.embeddings import EmbeddingProvider
from app.llm import build_prompt
from app.pipeline import index_repository, query_repository


class TestTokenizer:
    def count(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_many(self, texts: list[str]) -> list[int]:
        return [self.count(text) for text in texts]


class KeywordEmbeddingProvider(EmbeddingProvider):
    @property
    def dimensions(self) -> int:
        return 2

    def embed(self, text: str) -> list[float]:
        lower = text.lower()
        if any(term in lower for term in ("login", "password", "session")):
            return [1.0, 0.0]
        return [0.0, 1.0]


def test_index_query_and_prompt_pipeline_with_simple_store(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("app.ingest.splitter.get_tokenizer", lambda model_name: TestTokenizer())
    monkeypatch.setattr("app.llm.prompt.get_tokenizer", lambda model_name: TestTokenizer())
    monkeypatch.setattr("app.pipeline.build_embedder", lambda settings: KeywordEmbeddingProvider())

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text(
        "\n".join(
            [
                "def login_user(username, password):",
                "    if password == 'secret':",
                "        return create_session(username)",
                "    return None",
                "",
                "def render_report():",
                "    return 'monthly report'",
            ]
        ),
        encoding="utf-8",
    )
    (repo / "README.md").write_text("# Auth\n\nUsers log in with a password.\n", encoding="utf-8")

    settings = Settings(
        repositories_dir=tmp_path / "repositories",
        indexes_dir=tmp_path / "indexes",
        reranker_enabled=False,
    )

    repo_path, chunk_count = index_repository(str(repo), "simple", settings, index_name="smoke")
    matches = query_repository("How does password login work?", "simple", settings, top_k=2, index_name="smoke")
    prompt = build_prompt("How does password login work?", matches, settings=settings)

    assert repo_path == repo.resolve()
    assert chunk_count >= 3
    assert matches
    assert matches[0][0].metadata["symbol"] in {"login_user", "Auth"}
    assert "login_user" in prompt
    assert "auth.py" in prompt
    assert "How does password login work?" in prompt
