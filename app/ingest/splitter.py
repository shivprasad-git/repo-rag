from __future__ import annotations

from dataclasses import replace

from app.config import Settings
from app.models import Chunk


def split_oversized_chunks(chunks: list[Chunk], settings: Settings) -> list[Chunk]:
    split_chunks: list[Chunk] = []
    for chunk in chunks:
        split_chunks.extend(_split_chunk(chunk, settings))
    return split_chunks


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _split_chunk(chunk: Chunk, settings: Settings) -> list[Chunk]:
    if not _should_split(chunk, settings):
        return [chunk]

    lines = chunk.content.splitlines()
    line_groups = _line_groups(lines, settings.max_chunk_tokens, settings.chunk_overlap_tokens)
    if len(line_groups) <= 1:
        return [chunk]

    return [
        _chunk_part(chunk, part_lines, index + 1, len(line_groups))
        for index, part_lines in enumerate(line_groups)
    ]


def _should_split(chunk: Chunk, settings: Settings) -> bool:
    if settings.max_chunk_tokens <= 0:
        return False
    if chunk.metadata.get("chunk_type", "") not in settings.splittable_chunk_types:
        return False
    return estimate_tokens(chunk.content) > settings.max_chunk_tokens


def _line_groups(lines: list[str], max_tokens: int, overlap_tokens: int) -> list[list[tuple[int, str]]]:
    groups: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_tokens = 0

    for line_number, line in enumerate(lines):
        line_tokens = estimate_tokens(line)
        if current and current_tokens + line_tokens > max_tokens:
            groups.append(current)
            current = _overlap_lines(current, overlap_tokens)
            current_tokens = sum(estimate_tokens(line_text) for _, line_text in current)
        current.append((line_number, line))
        current_tokens += line_tokens

    if current:
        groups.append(current)
    return groups


def _overlap_lines(lines: list[tuple[int, str]], overlap_tokens: int) -> list[tuple[int, str]]:
    if overlap_tokens <= 0:
        return []

    selected: list[tuple[int, str]] = []
    selected_tokens = 0
    for line_number, line in reversed(lines):
        line_tokens = estimate_tokens(line)
        if selected and selected_tokens + line_tokens > overlap_tokens:
            break
        selected.append((line_number, line))
        selected_tokens += line_tokens
    return list(reversed(selected))


def _chunk_part(parent: Chunk, part_lines: list[tuple[int, str]], part_index: int, part_count: int) -> Chunk:
    first_line_offset = part_lines[0][0]
    last_line_offset = part_lines[-1][0]
    parent_start_line = int(parent.metadata.get("start_line", 1))
    start_line = parent_start_line + first_line_offset
    end_line = parent_start_line + last_line_offset
    metadata = parent.metadata | {
        "is_chunk_part": True,
        "part_index": part_index,
        "part_count": part_count,
        "parent_chunk_id": parent.id,
        "start_line": start_line,
        "end_line": end_line,
    }
    return replace(
        parent,
        id=f"{parent.id}:part:{part_index}",
        content="\n".join(line for _, line in part_lines),
        metadata=metadata,
    )
