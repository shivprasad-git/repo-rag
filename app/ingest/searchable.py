from __future__ import annotations

from app.config import Settings
from app.models import Chunk
from app.retrieval.filters import MetadataFilters


def is_searchable_chunk(chunk: Chunk, settings: Settings) -> bool:
    searchable_types = {chunk_type.value for chunk_type in settings.searchable_chunk_types}
    return chunk.metadata.get("chunk_type", "") in searchable_types


def searchable_chunks(
    chunks: list[Chunk],
    settings: Settings,
    filters: MetadataFilters | None = None,
) -> list[Chunk]:
    if filters is not None and filters.chunk_type:
        return chunks
    return [chunk for chunk in chunks if is_searchable_chunk(chunk, settings)]
