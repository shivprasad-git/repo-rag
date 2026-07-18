from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import Settings
from app.pipeline import index_repository, query_repository
from app.retrieval.filters import MetadataFilters


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
    language: str | None = None
    chunk_type: str | None = None
    path_prefix: str | None = None


@app.post("/index")
def index(request: IndexRequest) -> dict:
    repo_path, chunk_count = index_repository(request.repo, request.store, settings, request.index_name)
    return {"repo_path": str(repo_path), "chunks": chunk_count}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    filters = MetadataFilters(
        language=request.language,
        chunk_type=request.chunk_type,
        path_prefix=request.path_prefix,
    )
    matches = query_repository(
        request.question,
        request.store,
        settings,
        request.top_k,
        request.index_name,
        filters=filters if filters.has_filters else None,
    )
    return {
        "matches": [
            {"score": score, "content": chunk.content, "metadata": chunk.metadata}
            for chunk, score in matches
        ]
    }
