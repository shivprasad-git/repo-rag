from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import Settings
from app.pipeline import index_repository, query_repository


app = FastAPI(title="Repo RAG Phase 1")
settings = Settings()


class IndexRequest(BaseModel):
    repo: str
    store: str = "chroma"
    index_name: str | None = None


class QueryRequest(BaseModel):
    question: str
    store: str = "chroma"
    index_name: str | None = None
    top_k: int = 5


@app.post("/index")
def index(request: IndexRequest) -> dict:
    repo_path, chunk_count = index_repository(request.repo, request.store, settings, request.index_name)
    return {"repo_path": str(repo_path), "chunks": chunk_count}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    matches = query_repository(request.question, request.store, settings, request.top_k, request.index_name)
    return {
        "matches": [
            {"score": score, "content": chunk.content, "metadata": chunk.metadata}
            for chunk, score in matches
        ]
    }

