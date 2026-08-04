from __future__ import annotations

from app.config import Settings
from app.ingest.tokenizer import Tokenizer, get_tokenizer
from app.models import Chunk


def build_prompt(question: str, matches: list[tuple[Chunk, float]], settings: Settings | None = None) -> str:
    settings = settings or Settings()
    selected_blocks, omitted_count = _budgeted_context_blocks(
        matches,
        settings.max_prompt_context_tokens,
        get_tokenizer(settings.embedding_model),
    )

    context = "\n\n---\n\n".join(selected_blocks)
    omitted_note = (
        f"\n\nNote: {omitted_count} retrieved context chunk(s) were omitted due to the prompt token budget."
        if omitted_count
        else ""
    )
    return f"""You are answering questions about a GitHub repository.
Use only the retrieved context below. Cite file paths and line numbers when possible.

Question:
{question}

Retrieved context:
{context}{omitted_note}

Answer:
"""


def _budgeted_context_blocks(
    matches: list[tuple[Chunk, float]],
    max_context_tokens: int,
    tokenizer: Tokenizer,
) -> tuple[list[str], int]:
    ordered_matches = _priority_order(matches)
    blocks: list[str] = []
    used_tokens = 0
    omitted = 0

    for display_index, (chunk, score) in enumerate(ordered_matches, start=1):
        block = _context_block(display_index, chunk, score)
        block_tokens = tokenizer.count(block)
        if max_context_tokens > 0 and used_tokens + block_tokens > max_context_tokens:
            omitted += 1
            continue
        blocks.append(block)
        used_tokens += block_tokens

    return blocks, omitted


def _priority_order(matches: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
    return sorted(
        matches,
        key=lambda item: (
            bool(item[0].metadata.get("is_context_expansion")),
            -item[1],
        ),
    )


def _context_block(index: int, chunk: Chunk, score: float) -> str:
    metadata = chunk.metadata
    location = f"{metadata.get('file_path')}:{metadata.get('start_line')}-{metadata.get('end_line')}"
    return "\n".join(
        [
            f"[Chunk {index}] score={score:.4f}",
            f"Role: {'neighbor context' if metadata.get('is_context_expansion') else 'retrieved match'}",
            f"Location: {location}",
            f"Symbol: {metadata.get('symbol')}",
            f"Type: {metadata.get('chunk_type')}",
            "Content:",
            chunk.content,
        ]
    )
