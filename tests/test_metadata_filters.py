from app.docstore import SimpleJsonChunkStore
from app.embeddings import EmbeddingProvider
from app.models import Chunk
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
    assert MetadataFilters(chunk_type="function").matches(chunk)
    assert MetadataFilters(path_prefix="app").matches(chunk)
    assert not MetadataFilters(language="markdown").matches(chunk)
    assert not MetadataFilters(path_prefix="docs").matches(chunk)


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

    matches = Retriever(embedder, store, chunk_store=chunk_store).retrieve(
        "login",
        top_k=5,
        filters=MetadataFilters(chunk_type="method"),
    )

    assert [chunk.id for chunk, _ in matches] == ["auth-method"]
