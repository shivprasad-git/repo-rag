from __future__ import annotations

from app.config import Settings
from app.models import Chunk
from app.retrieval.filters import MetadataFilters


def is_searchable_chunk(chunk: Chunk, settings: Settings) -> bool:
    return chunk.metadata.get("chunk_type", "") in _searchable_type_values(settings)


def searchable_chunks(
    chunks: list[Chunk],
    settings: Settings,
    filters: MetadataFilters | None = None,
) -> list[Chunk]:
    if filters is not None and filters.chunk_type:
        return chunks
    searchable_types = _searchable_type_values(settings)
    return [chunk for chunk in chunks if chunk.metadata.get("chunk_type", "") in searchable_types]


def _searchable_type_values(settings: Settings) -> set[str]:
    return {chunk_type.value for chunk_type in settings.searchable_chunk_types}
