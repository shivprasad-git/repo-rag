from __future__ import annotations

from app.config import Settings
from app.models import Chunk
from app.retrieval.filters import MetadataFilters


def searchable_chunks(
    chunks: list[Chunk],
    settings: Settings,
    filters: MetadataFilters | None = None,
) -> list[Chunk]:
    """Return the subset of *chunks* that the vector store should index.

    Normally only the configured searchable types are indexed. A caller that
    filters by ``chunk_type`` lets non-searchable types (``file_metadata``,
    ``imports``, ``parse_error``) through, since only keyword search can
    surface them.
    """
    if filters is not None and filters.chunk_type:
        return [chunk for chunk in chunks if filters.matches(chunk)]
    searchable_types = _searchable_type_values(settings)
    return [chunk for chunk in chunks if chunk.metadata.get("chunk_type", "") in searchable_types]


def _searchable_type_values(settings: Settings) -> set[str]:
    return {chunk_type.value for chunk_type in settings.searchable_chunk_types}
