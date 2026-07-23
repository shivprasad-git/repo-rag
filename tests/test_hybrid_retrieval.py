from app.docstore import SimpleJsonChunkStore
from app.embeddings import EmbeddingProvider
from app.models import Chunk
from app.retrieval.reranker import Reranker
from app.retrieval.search import Retriever
from app.vectorstore.simple_store import SimpleJsonVectorStore


class TestEmbeddingProvider(EmbeddingProvider):
    @property
    def dimensions(self) -> int:
        return 2

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0] if "login" in text.lower() else [0.0, 1.0]


class PreferBillingReranker(Reranker):
    def rerank(
        self,
        question: str,
        matches: list[tuple[Chunk, float]],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        scored = [
            (chunk, 10.0 if chunk.id == "billing" else score)
            for chunk, score in matches
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


def test_hybrid_retrieval_uses_keyword_matches(tmp_path) -> None:
    store = SimpleJsonVectorStore(tmp_path / "index.json")
    chunk_store = SimpleJsonChunkStore(tmp_path / "chunks.json")
    chunks = [
        Chunk(
            id="auth-login",
            content="def login_user(username, password):\n    return create_session(username)",
            metadata={
                "file_path": "auth.py",
                "symbol": "login_user",
                "chunk_type": "function",
                "start_line": 1,
                "end_line": 2,
            },
        ),
        Chunk(
            id="billing",
            content="def charge_card(amount):\n    return payment_gateway.charge(amount)",
            metadata={
                "file_path": "billing.py",
                "symbol": "charge_card",
                "chunk_type": "function",
                "start_line": 1,
                "end_line": 2,
            },
        ),
    ]
    embedder = TestEmbeddingProvider()
    chunk_store.add(chunks)
    store.add(chunks, embedder.embed_many([chunk.content for chunk in chunks]))
    vector_record = store._load_records()[0]
    assert "content" not in vector_record
    assert "embedding" in vector_record
    assert set(vector_record["metadata"]) <= {
        "repo",
        "file_path",
        "module",
        "language",
        "commit",
        "chunk_type",
        "symbol",
        "qualified_symbol",
        "start_line",
        "end_line",
        "file_metadata_id",
        "has_parse_errors",
    }

    matches = Retriever(embedder, store, chunk_store=chunk_store).retrieve("login_user", top_k=1)

    assert matches[0][0].id == "auth-login"
    assert matches[0][0].content.startswith("def login_user")


def test_retriever_applies_reranker(tmp_path) -> None:
    store = SimpleJsonVectorStore(tmp_path / "index.json")
    chunk_store = SimpleJsonChunkStore(tmp_path / "chunks.json")
    chunks = [
        Chunk(id="auth-login", content="def login_user(): pass", metadata={"chunk_type": "function"}),
        Chunk(id="billing", content="def charge_card(): pass", metadata={"chunk_type": "function"}),
    ]
    embedder = TestEmbeddingProvider()
    chunk_store.add(chunks)
    store.add(chunks, embedder.embed_many([chunk.content for chunk in chunks]))

    matches = Retriever(
        embedder,
        store,
        chunk_store=chunk_store,
        reranker=PreferBillingReranker(),
    ).retrieve("login_user", top_k=1)

    assert matches[0][0].id == "billing"
    assert matches[0][1] == 10.0
