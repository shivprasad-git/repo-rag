from __future__ import annotations

from dataclasses import dataclass

from app.models import Chunk, ChunkType


@dataclass(frozen=True)
class MetadataFilters:
    language: str | None = None
    chunk_type: str | ChunkType | None = None
    path_prefix: str | None = None

    @property
    def has_filters(self) -> bool:
        return bool(self.language or self.chunk_type or self.path_prefix)

    def matches(self, chunk: Chunk) -> bool:
        metadata = chunk.metadata
        if self.language and metadata.get("language") != self.language:
            return False
        if self.chunk_type and metadata.get("chunk_type") != self.chunk_type_value:
            return False
        if self.path_prefix:
            file_path = str(metadata.get("file_path", ""))
            normalized_prefix = self.path_prefix.strip("/")
            prefix_parts = [part for part in normalized_prefix.split("/") if part]
            file_parts = [part for part in file_path.split("/") if part]
            if file_parts[: len(prefix_parts)] != prefix_parts:
                return False
        return True

    @property
    def chunk_type_value(self) -> str | None:
        if isinstance(self.chunk_type, ChunkType):
            return self.chunk_type.value
        return self.chunk_type


def filter_chunks(chunks: list[Chunk], filters: MetadataFilters | None) -> list[Chunk]:
    if filters is None or not filters.has_filters:
        return chunks
    return [chunk for chunk in chunks if filters.matches(chunk)]
