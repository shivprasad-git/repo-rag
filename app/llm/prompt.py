from __future__ import annotations

from app.models import Chunk


def build_prompt(question: str, matches: list[tuple[Chunk, float]]) -> str:
    context_blocks = []
    for index, (chunk, score) in enumerate(matches, start=1):
        metadata = chunk.metadata
        location = f"{metadata.get('file_path')}:{metadata.get('start_line')}-{metadata.get('end_line')}"
        context_blocks.append(
            "\n".join(
                [
                    f"[Chunk {index}] score={score:.4f}",
                    f"Location: {location}",
                    f"Symbol: {metadata.get('symbol')}",
                    f"Type: {metadata.get('chunk_type')}",
                    "Content:",
                    chunk.content,
                ]
            )
        )

    context = "\n\n---\n\n".join(context_blocks)
    return f"""You are answering questions about a GitHub repository.
Use only the retrieved context below. Cite file paths and line numbers when possible.

Question:
{question}

Retrieved context:
{context}

Answer:
"""

