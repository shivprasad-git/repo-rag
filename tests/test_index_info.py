from pathlib import Path

from app.config import Settings
from app.docstore import build_chunk_store
from app.indexing.info import format_index_info, get_index_info
from app.indexing.manifest import IndexManifest, build_manifest_path, index_config
from app.models import Chunk
from app.vectorstore import build_vector_store


def test_index_info_reports_config_and_counts(tmp_path: Path) -> None:
    settings = Settings(indexes_dir=tmp_path / "indexes", embedding_dimensions=2)
    index_name = "sample"
    chunks = [
        Chunk(id="auth-login", content="def login(): pass", metadata={"file_path": "auth.py", "chunk_type": "function"}),
        Chunk(id="auth-meta", content="file: auth.py", metadata={"file_path": "auth.py", "chunk_type": "file_metadata"}),
    ]
    chunk_store = build_chunk_store(settings.indexes_dir, index_name)
    vector_store = build_vector_store("simple", settings, index_name=index_name)
    chunk_store.add(chunks)
    vector_store.add([chunks[0]], [[1.0, 0.0]])
    manifest = IndexManifest.load(build_manifest_path(settings.indexes_dir, index_name))
    manifest.config = index_config(settings, "simple")
    manifest.replace_file("auth.py", "fake-hash", chunks)
    manifest.save()

    info = get_index_info("simple", settings, index_name=index_name)
    output = format_index_info(info)

    assert info.files_indexed == 1
    assert info.chunks_stored == 2
    assert info.searchable_chunks == 1
    assert info.vectors_stored == 1
    assert "Index: sample" in output
    assert "Reranker enabled: yes" in output
