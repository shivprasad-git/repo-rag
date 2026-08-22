from pathlib import Path

from app.config import Settings
from app.docstore import build_chunk_store
from app.indexing.health import check_index_health
from app.indexing.manifest import IndexManifest, build_manifest_path, file_content_hash, index_config
from app.models import Chunk
from app.vectorstore import build_vector_store


def test_index_health_reports_ok_for_consistent_simple_index(tmp_path: Path) -> None:
    settings = Settings(indexes_dir=tmp_path / "indexes", embedding_dimensions=2)
    index_name = "health"
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    file_path = repo_path / "auth.py"
    file_path.write_text("def login():\n    return 'token'\n", encoding="utf-8")
    chunk = Chunk(
        id="auth-login",
        content="def login():\n    return 'token'",
        metadata={"file_path": "auth.py", "chunk_type": "function"},
    )
    chunk_store = build_chunk_store(settings.indexes_dir, index_name)
    vector_store = build_vector_store("simple", settings, index_name=index_name)
    chunk_store.add([chunk])
    vector_store.add([chunk], [[1.0, 0.0]])
    manifest = IndexManifest.load(build_manifest_path(settings.indexes_dir, index_name))
    manifest.config = index_config(settings, "simple")
    manifest.replace_file("auth.py", file_content_hash(file_path), [chunk])
    manifest.save()

    report = check_index_health("simple", settings, index_name=index_name, repo=str(repo_path))

    assert report.ok is True
    assert report.doc_chunks == 1
    assert report.vector_chunks == 1
    assert report.stale_files == []


def test_index_health_reports_missing_vectors(tmp_path: Path) -> None:
    settings = Settings(indexes_dir=tmp_path / "indexes", embedding_dimensions=2)
    index_name = "health"
    chunk = Chunk(
        id="auth-login",
        content="def login():\n    return 'token'",
        metadata={"file_path": "auth.py", "chunk_type": "function"},
    )
    chunk_store = build_chunk_store(settings.indexes_dir, index_name)
    chunk_store.add([chunk])
    manifest = IndexManifest.load(build_manifest_path(settings.indexes_dir, index_name))
    manifest.config = index_config(settings, "simple")
    manifest.replace_file("auth.py", "fake-hash", [chunk])
    manifest.save()

    report = check_index_health("simple", settings, index_name=index_name)

    assert report.ok is False
    assert report.missing_vectors == ["auth-login"]


def test_index_health_reports_wrong_embedding_dimensions(tmp_path: Path) -> None:
    settings = Settings(indexes_dir=tmp_path / "indexes", embedding_dimensions=2)
    index_name = "health"
    chunk = Chunk(
        id="auth-login",
        content="def login():\n    return 'token'",
        metadata={"file_path": "auth.py", "chunk_type": "function"},
    )
    chunk_store = build_chunk_store(settings.indexes_dir, index_name)
    vector_store = build_vector_store("simple", settings, index_name=index_name)
    chunk_store.add([chunk])
    # Stored vectors have 3 dimensions but settings.embedding_dimensions is 2.
    vector_store.add([chunk], [[1.0, 2.0, 3.0]])
    manifest = IndexManifest.load(build_manifest_path(settings.indexes_dir, index_name))
    manifest.config = index_config(settings, "simple")
    manifest.replace_file("auth.py", "fake-hash", [chunk])
    manifest.save()

    report = check_index_health("simple", settings, index_name=index_name)

    assert report.ok is False
    assert report.wrong_embedding_dimensions == ["auth-login"]
