from app.docstore import SimpleJsonChunkStore
from app.config import Settings
from app.embeddings import EmbeddingProvider
from app.models import Chunk, ChunkType
from app.retrieval.filters import MetadataFilters
from app.retrieval.search import Retriever
from app.vectorstore.simple_store import SimpleJsonVectorStore


class TestEmbeddingProvider(EmbeddingProvider):
    @property
    def dimensions(self) -> int:
        return 2

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0] if "login" in text.lower() else [0.0, 1.0]


def test_metadata_filters_match_language_chunk_type_and_path() -> None:
    chunk = Chunk(
        content="def login(): pass",
        metadata={
            "language": "python",
            "chunk_type": "function",
            "file_path": "app/auth.py",
        },
    )

    assert MetadataFilters(language="python").matches(chunk)
    assert MetadataFilters(chunk_type=ChunkType.FUNCTION).matches(chunk)
    assert MetadataFilters(path_prefix="app").matches(chunk)
    assert not MetadataFilters(language="markdown").matches(chunk)
    assert not MetadataFilters(path_prefix="docs").matches(chunk)


def test_path_prefix_matches_on_segment_boundaries() -> None:
    chunk = Chunk(
        content="def login(): pass",
        metadata={
            "language": "python",
            "chunk_type": "function",
            "file_path": "app/auth.py",
        },
    )

    assert MetadataFilters(path_prefix="app").matches(chunk)
    assert MetadataFilters(path_prefix="app/auth.py").matches(chunk)
    assert MetadataFilters(path_prefix="/app/").matches(chunk)
    assert not MetadataFilters(path_prefix="appl").matches(chunk)
    assert not MetadataFilters(path_prefix="app/auth").matches(chunk)


def test_hybrid_retrieval_respects_metadata_filters(tmp_path) -> None:
    store = SimpleJsonVectorStore(tmp_path / "index.json")
    chunk_store = SimpleJsonChunkStore(tmp_path / "chunks.json")
    chunks = [
        Chunk(
            id="auth-doc",
            content="Authentication uses login tokens.",
            metadata={
                "language": "markdown",
                "file_path": "README.md",
                "symbol": "Authentication",
                "chunk_type": "markdown_section",
                "start_line": 1,
                "end_line": 3,
            },
        ),
        Chunk(
            id="auth-method",
            content="def login(self, username, password):\n    return create_token(username)",
            metadata={
                "language": "python",
                "file_path": "auth.py",
                "symbol": "AuthService.login",
                "chunk_type": "method",
                "start_line": 10,
                "end_line": 11,
            },
        ),
    ]
    embedder = TestEmbeddingProvider()
    chunk_store.add(chunks)
    store.add(chunks, embedder.embed_many([chunk.content for chunk in chunks]))

    matches = Retriever(embedder, store, settings=Settings(), chunk_store=chunk_store).retrieve(
        "login",
        top_k=5,
        filters=MetadataFilters(chunk_type=ChunkType.METHOD),
    )

    assert [chunk.id for chunk, _ in matches] == ["auth-method"]


def test_default_retrieval_skips_metadata_chunks(tmp_path) -> None:
    store = SimpleJsonVectorStore(tmp_path / "index.json")
    chunk_store = SimpleJsonChunkStore(tmp_path / "chunks.json")
    settings = Settings()
    indexed_chunks = [
        Chunk(
            id="auth-method",
            content="def login(self):\n    return create_token()",
            metadata={"chunk_type": "method", "file_path": "auth.py"},
        )
    ]
    metadata_chunk = Chunk(
        id="auth-file-metadata",
        content="file: auth.py\nimports:\nimport secret_login_metadata",
        metadata={"chunk_type": "file_metadata", "file_path": "auth.py"},
    )
    embedder = TestEmbeddingProvider()
    chunk_store.add(indexed_chunks + [metadata_chunk])
    store.add(indexed_chunks, embedder.embed_many([chunk.content for chunk in indexed_chunks]))

    matches = Retriever(embedder, store, settings=settings, chunk_store=chunk_store).retrieve(
        "secret_login_metadata",
        top_k=5,
    )

    assert [chunk.id for chunk, _ in matches] == ["auth-method"]


def test_chunk_type_filter_can_search_metadata_chunks(tmp_path) -> None:
    store = SimpleJsonVectorStore(tmp_path / "index.json")
    chunk_store = SimpleJsonChunkStore(tmp_path / "chunks.json")
    settings = Settings()
    indexed_chunk = Chunk(
        id="auth-method",
        content="def login(self):\n    return create_token()",
        metadata={"chunk_type": "method", "file_path": "auth.py"},
    )
    metadata_chunk = Chunk(
        id="auth-file-metadata",
        content="file: auth.py\nimports:\nimport secret_login_metadata",
        metadata={"chunk_type": "file_metadata", "file_path": "auth.py"},
    )
    embedder = TestEmbeddingProvider()
    chunk_store.add([indexed_chunk, metadata_chunk])
    store.add([indexed_chunk], embedder.embed_many([indexed_chunk.content]))

    matches = Retriever(embedder, store, settings=settings, chunk_store=chunk_store).retrieve(
        "secret_login_metadata",
        top_k=5,
        filters=MetadataFilters(chunk_type=ChunkType.FILE_METADATA),
    )

    assert [chunk.id for chunk, _ in matches] == ["auth-file-metadata"]
