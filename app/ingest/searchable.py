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

    Normally only the configured searchable chunk types (by default class,
    function, method, markdown_section, text_file) are indexed.  However, when
    the caller explicitly filters by ``chunk_type``, non-searchable types such
    as ``file_metadata``, ``imports``, and ``parse_error`` are allowed through:
    they live only in the document store, so keyword search is the only path
    that can surface them.

    In the filtered case the requested chunk type is still honored, so this
    function stays correct even for callers passing unfiltered chunks.
    """
    if filters is not None and filters.chunk_type:
        return [chunk for chunk in chunks if filters.matches(chunk)]
    searchable_types = _searchable_type_values(settings)
    return [chunk for chunk in chunks if chunk.metadata.get("chunk_type", "") in searchable_types]


def _searchable_type_values(settings: Settings) -> set[str]:
    return {chunk_type.value for chunk_type in settings.searchable_chunk_types}
