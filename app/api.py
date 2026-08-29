from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import load_settings
from app.logging_config import get_logger, setup_logging
from app.pipeline import index_repository, query_repository
from app.retrieval.filters import MetadataFilters

setup_logging()

app = FastAPI(title="Repo RAG Phase 1")
settings = load_settings()
logger = get_logger(__name__)


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
    logger.info("POST /index repo=%s store=%s index=%s", request.repo, request.store, request.index_name)
    repo_path, chunk_count = index_repository(request.repo, request.store, settings, request.index_name)
    logger.info("POST /index complete: %d chunks from %s", chunk_count, repo_path)
    return {"repo_path": str(repo_path), "chunks": chunk_count}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    logger.info("POST /query question=%r top_k=%d store=%s", request.question, request.top_k, request.store)
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
    logger.info("POST /query returned %d matches", len(matches))
    return {
        "matches": [
            {"score": score, "content": chunk.content, "metadata": chunk.metadata}
            for chunk, score in matches
        ]
    }
