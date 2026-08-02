from __future__ import annotations

from dataclasses import replace

from app.config import Settings
from app.ingest.tokenizer import get_tokenizer, Tokenizer
from app.logging_config import get_logger
from app.models import Chunk

logger = get_logger(__name__)


def split_oversized_chunks(chunks: list[Chunk], settings: Settings) -> list[Chunk]:
    tokenizer = get_tokenizer(settings.embedding_model)
    split_chunks: list[Chunk] = []
    split_count = 0
    for chunk in chunks:
        parts = _split_chunk(chunk, settings, tokenizer)
        if len(parts) > 1:
            split_count += 1
            logger.debug(
                "Split %s (%s) into %d parts",
                chunk.metadata.get("symbol", chunk.id),
                chunk.metadata.get("chunk_type", "unknown"),
                len(parts),
            )
        split_chunks.extend(parts)
    if split_count:
        logger.info("Split %d oversized chunks into %d total parts", split_count, len(split_chunks))
    return split_chunks


def _split_chunk(chunk: Chunk, settings: Settings, tokenizer: Tokenizer) -> list[Chunk]:
    if not _should_split(chunk, settings, tokenizer):
        return [chunk]

    lines = chunk.content.splitlines()
    line_groups = _line_groups(lines, settings.max_chunk_tokens, settings.chunk_overlap_tokens, tokenizer)
    if len(line_groups) <= 1:
        return [chunk]

    return [
        _chunk_part(chunk, part_lines, index + 1, len(line_groups))
        for index, part_lines in enumerate(line_groups)
    ]


def _should_split(chunk: Chunk, settings: Settings, tokenizer: Tokenizer) -> bool:
    if settings.max_chunk_tokens <= 0:
        return False
    splittable_types = {chunk_type.value for chunk_type in settings.splittable_chunk_types}
    if chunk.metadata.get("chunk_type", "") not in splittable_types:
        return False
    return tokenizer.count(chunk.content) > settings.max_chunk_tokens


def _line_groups(
    lines: list[str],
    max_tokens: int,
    overlap_tokens: int,
    tokenizer: Tokenizer,
) -> list[list[tuple[int, str]]]:
    groups: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_tokens = 0

    # Pre-compute token counts for all lines to avoid repeated encode calls
    line_token_counts = tokenizer.count_many(lines)

    for line_number, (line, line_tokens) in enumerate(zip(lines, line_token_counts)):
        if current and current_tokens + line_tokens > max_tokens:
            groups.append(current)
            current = _overlap_lines(current, overlap_tokens, line_token_counts)
            current_tokens = sum(line_token_counts[ln] for ln, _ in current)
        current.append((line_number, line))
        current_tokens += line_tokens

    if current:
        groups.append(current)
    return groups


def _overlap_lines(
    lines: list[tuple[int, str]],
    overlap_tokens: int,
    line_token_counts: list[int],
) -> list[tuple[int, str]]:
    if overlap_tokens <= 0:
        return []

    selected: list[tuple[int, str]] = []
    selected_tokens = 0
    for line_number, line in reversed(lines):
        line_tokens = line_token_counts[line_number]
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