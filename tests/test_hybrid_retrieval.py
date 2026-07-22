from app.docstore import SimpleJsonChunkStore
from app.embeddings import HashEmbeddingProvider
from app.models import Chunk
from app.retrieval.search import Retriever
from app.vectorstore.simple_store import SimpleJsonVectorStore


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
    embedder = HashEmbeddingProvider()
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
