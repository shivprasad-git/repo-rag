from app.embeddings import HashEmbeddingProvider
from app.models import Chunk
from app.retrieval import Retriever
from app.vectorstore.simple_store import SimpleJsonVectorStore


def test_hybrid_retrieval_uses_keyword_matches(tmp_path) -> None:
    store = SimpleJsonVectorStore(tmp_path / "index.json")
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
    embedder = HashEmbeddingProvider()
    store.add(chunks, embedder.embed_many([chunk.content for chunk in chunks]))

    matches = Retriever(embedder, store).retrieve("login_user", top_k=1)

    assert matches[0][0].id == "auth-login"
