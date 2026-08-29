from pathlib import Path

from app.config import cli_defaults, load_settings
from app.pipeline import default_index_name


def test_default_index_name_derives_from_github_url() -> None:
    assert default_index_name("https://github.com/pallets/flask") == "flask"
    assert default_index_name("https://github.com/pallets/flask.git") == "flask"


def test_default_index_name_derives_from_local_folder(tmp_path: Path) -> None:
    repo_path = tmp_path / "my-project"
    repo_path.mkdir()
    assert default_index_name(str(repo_path)) == "my-project"


def test_default_index_name_sanitizes_to_a_slug() -> None:
    # Does not exist on disk, so it is treated as a repo reference.
    assert default_index_name("org/My.Repo") == "my-repo"
    assert default_index_name("https://github.com/pallets/flask/") == "flask"


def test_env_overrides_apply_to_settings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("RAG_EMBEDDING_DIMENSIONS", "128")
    monkeypatch.setenv("RAG_RERANKER_ENABLED", "false")
    settings = load_settings(tmp_path / "missing.env")
    assert settings.embedding_model == "custom-model"
    assert settings.embedding_dimensions == 128
    assert settings.reranker_enabled is False


def test_cli_defaults_read_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAG_STORE", "simple")
    monkeypatch.setenv("RAG_INDEX_NAME", "flask")
    monkeypatch.setenv("RAG_RERANKER_ENABLED", "false")
    defaults = cli_defaults(tmp_path / "missing.env")
    assert defaults.store == "simple"
    assert defaults.index_name == "flask"
    assert defaults.no_reranker is True


def test_dotenv_file_is_loaded(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("RAG_STORE=simple\nRAG_INDEX_NAME=my-index\n", encoding="utf-8")
    defaults = cli_defaults(env_file)
    assert defaults.store == "simple"
    assert defaults.index_name == "my-index"