from app.config import Settings
from app.llm.prompt import build_prompt
from app.models import Chunk


class TestTokenizer:
    def count(self, text: str) -> int:
        return len(text.split())


def test_prompt_budget_prefers_retrieved_matches(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.prompt.get_tokenizer", lambda model_name: TestTokenizer())
    matches = [
        (
            Chunk(
                id="neighbor",
                content="neighbor context " * 50,
                metadata={
                    "is_context_expansion": True,
                    "file_path": "auth.py",
                    "start_line": 1,
                    "end_line": 5,
                    "symbol": "Auth.login",
                    "chunk_type": "method",
                },
            ),
            0.99,
        ),
        (
            Chunk(
                id="match",
                content="direct match",
                metadata={
                    "file_path": "auth.py",
                    "start_line": 6,
                    "end_line": 8,
                    "symbol": "Auth.login",
                    "chunk_type": "method",
                },
            ),
            0.5,
        ),
    ]

    prompt = build_prompt(
        "How does login work?",
        matches,
        settings=Settings(max_prompt_context_tokens=35),
    )

    assert "direct match" in prompt
    assert "neighbor context neighbor context" not in prompt
    assert "omitted due to the prompt token budget" in prompt


def test_prompt_budget_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.prompt.get_tokenizer", lambda model_name: TestTokenizer())
    matches = [
        (
            Chunk(
                id="big",
                content="large content " * 100,
                metadata={
                    "file_path": "auth.py",
                    "start_line": 1,
                    "end_line": 20,
                    "symbol": "Auth.login",
                    "chunk_type": "method",
                },
            ),
            0.9,
        )
    ]

    prompt = build_prompt(
        "How does login work?",
        matches,
        settings=Settings(max_prompt_context_tokens=0),
    )

    assert "large content" in prompt
    assert "omitted due to the prompt token budget" not in prompt
