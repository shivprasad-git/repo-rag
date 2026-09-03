import json

from app.cli_format import format_matches, format_matches_json
from app.models import Chunk


def test_format_matches_can_show_debug_scores_and_content() -> None:
    chunk = Chunk(
        id="auth-login",
        content="def login():\n    return create_token()",
        metadata={
            "file_path": "auth.py",
            "start_line": 10,
            "end_line": 11,
            "chunk_type": "function",
            "symbol": "login",
            "debug_vector_score": 0.8,
            "debug_keyword_score": 0.6,
            "debug_combined_score": 0.65,
            "debug_reranker_score": 1.2,
        },
    )

    output = format_matches([(chunk, 1.2)], show_content=True, debug_scores=True)

    assert "1. score=1.2000 auth.py:10-11" in output
    assert "debug: vector=0.8000 keyword=0.6000 combined=0.6500 reranker=1.2000" in output
    assert "| def login():" in output


def test_format_matches_json_includes_content_and_metadata() -> None:
    chunk = Chunk(
        id="auth-login",
        content="def login(): pass",
        metadata={"file_path": "auth.py", "chunk_type": "function"},
    )

    payload = json.loads(format_matches_json([(chunk, 0.5)]))

    assert payload == [
        {
            "id": "auth-login",
            "score": 0.5,
            "content": "def login(): pass",
            "metadata": {"file_path": "auth.py", "chunk_type": "function"},
        }
    ]
