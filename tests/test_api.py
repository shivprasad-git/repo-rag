from pathlib import Path

from fastapi.testclient import TestClient

from app import api
from app.models import Chunk


def test_health_endpoint() -> None:
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "index_repository",
        lambda repo, store, settings, index_name=None: (Path("/tmp/repo"), 42),
    )
    client = TestClient(api.app)
    response = client.post("/index", json={"repo": "https://github.com/pallets/flask"})
    assert response.status_code == 200
    assert response.json() == {"repo_path": "/tmp/repo", "chunks": 42}


def test_index_endpoint_derives_index_name(monkeypatch) -> None:
    captured: dict = {}

    def fake_index(repo, store, settings, index_name=None):
        captured["index_name"] = index_name
        return Path("/tmp/repo"), 1

    monkeypatch.setattr(api, "index_repository", fake_index)
    client = TestClient(api.app)
    client.post("/index", json={"repo": "https://github.com/pallets/flask"})
    assert captured["index_name"] is None


def test_query_endpoint(monkeypatch) -> None:
    chunk = Chunk(
        id="c1",
        content="def login_user(): pass",
        metadata={"file_path": "auth.py", "chunk_type": "function"},
    )
    monkeypatch.setattr(api, "query_repository", lambda *args, **kwargs: [(chunk, 0.9)])
    client = TestClient(api.app)
    response = client.post("/query", json={"question": "how does login work?", "index_name": "flask"})
    assert response.status_code == 200
    body = response.json()
    assert body["matches"][0]["score"] == 0.9
    assert body["matches"][0]["content"] == "def login_user(): pass"
    assert body["matches"][0]["metadata"]["file_path"] == "auth.py"


def test_query_endpoint_forwards_filters(monkeypatch) -> None:
    captured: dict = {}

    def fake_query(question, store, settings, top_k, index_name=None, filters=None):
        captured["filters"] = filters
        return []

    monkeypatch.setattr(api, "query_repository", fake_query)
    client = TestClient(api.app)
    response = client.post(
        "/query",
        json={"question": "login", "language": "python", "path_prefix": "src/"},
    )
    assert response.status_code == 200
    assert response.json() == {"matches": []}
    assert captured["filters"] is not None
    assert captured["filters"].language == "python"
    assert captured["filters"].path_prefix == "src/"


def test_ask_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(api, "ask_repository", lambda *args, **kwargs: "You are answering questions...")
    client = TestClient(api.app)
    response = client.post(
        "/ask",
        json={"repo": "work/sample_repo", "question": "How does auth work?", "top_k": 3},
    )
    assert response.status_code == 200
    assert response.json() == {"prompt": "You are answering questions..."}