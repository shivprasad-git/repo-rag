from __future__ import annotations

import json


def format_matches(matches, show_content: bool = False, content_chars: int = 700, debug_scores: bool = False) -> str:
    lines: list[str] = []
    for index, (chunk, score) in enumerate(matches, start=1):
        metadata = chunk.metadata
        lines.append(
            f"{index}. score={score:.4f} {metadata.get('file_path')}:{metadata.get('start_line')}-{metadata.get('end_line')}"
        )
        lines.append(f"   {metadata.get('chunk_type')} {metadata.get('symbol')}")
        if debug_scores:
            lines.append(_debug_score_line(metadata))
        if show_content:
            lines.append(_preview_content(chunk.content, content_chars))
    return "\n".join(lines)


def format_matches_json(matches) -> str:
    records = []
    for chunk, score in matches:
        records.append(
            {
                "id": chunk.id,
                "score": score,
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
        )
    return json.dumps(records, indent=2, sort_keys=True)


def _preview_content(content: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    preview = content.strip()
    if len(preview) > max_chars:
        preview = f"{preview[:max_chars].rstrip()}..."
    indented = "\n".join(f"   | {line}" for line in preview.splitlines())
    return indented or "   |"


def _debug_score_line(metadata: dict) -> str:
    scores = [
        f"vector={float(metadata.get('debug_vector_score', 0.0)):.4f}",
        f"keyword={float(metadata.get('debug_keyword_score', 0.0)):.4f}",
        f"combined={float(metadata.get('debug_combined_score', 0.0)):.4f}",
    ]
    if "debug_reranker_score" in metadata:
        scores.append(f"reranker={float(metadata.get('debug_reranker_score', 0.0)):.4f}")
    return f"   debug: {' '.join(scores)}"
