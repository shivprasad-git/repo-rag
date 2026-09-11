import json

import pytest

from app.eval.golden import GoldenSet, RelevantSelector, golden_set_from_dict, load_golden_set, resolve_relevant_ids
from app.models import Chunk


def test_golden_set_from_dict_parses_valid_set() -> None:
    golden = golden_set_from_dict(
        {
            "name": "sample",
            "repo": "work/sample_repo",
            "store": "simple",
            "index_name": "sample-repo",
            "top_k": [3, 5],
            "questions": [
                {"question": "Q1", "relevant": [{"file_path": "auth.py"}]},
                {"question": "Q2", "relevant": [{"symbol": "AuthService.login", "chunk_type": "method"}]},
            ],
        }
    )
    assert isinstance(golden, GoldenSet)
    assert golden.name == "sample"
    assert golden.repo == "work/sample_repo"
    assert golden.store == "simple"
    assert golden.index_name == "sample-repo"
    assert golden.top_k == (3, 5)
    assert len(golden.questions) == 2
    assert golden.questions[1].relevant[0].symbol == "AuthService.login"


def test_golden_set_defaults() -> None:
    golden = golden_set_from_dict(
        {"name": "x", "repo": "r", "questions": [{"question": "q", "relevant": [{"file_path": "a"}]}]}
    )
    assert golden.store == "simple"
    assert golden.top_k == (5, 10)
    assert golden.index_name is None


@pytest.mark.parametrize(
    "payload",
    [
        {"repo": "r", "questions": [{"question": "q", "relevant": [{"file_path": "a"}]}]},  # missing name
        {"name": "x", "repo": "r", "questions": []},  # empty questions
        {"name": "x", "repo": "r"},  # missing questions
        {"name": "x", "repo": "r", "questions": [{"question": "q", "relevant": []}]},  # empty relevant
        {"name": "x", "repo": "r", "questions": [{"question": "", "relevant": [{"file_path": "a"}]}]},  # blank question
        {"name": "x", "repo": "r", "questions": [{"question": "q", "relevant": [{}]}]},  # selector without fields
        {"name": "x", "repo": "r", "top_k": [0], "questions": [{"question": "q", "relevant": [{"file_path": "a"}]}]},  # bad top-k
        {"name": "x", "repo": "r", "questions": [{"question": "q", "relevant": [{"unknown": "field"}]}]},  # unknown selector field
    ],
)
def test_golden_set_invalid_inputs_raise(payload: dict) -> None:
    with pytest.raises(ValueError):
        golden_set_from_dict(payload)


def test_load_golden_set_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_golden_set("does/not/exist.json")


def test_load_golden_set_round_trips(tmp_path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            {
                "name": "sample",
                "repo": "work/sample_repo",
                "questions": [{"question": "Q", "relevant": [{"symbol": "AuthService.login"}]}],
            }
        ),
        encoding="utf-8",
    )
    golden = load_golden_set(path)
    assert golden.name == "sample"
    assert golden.questions[0].question == "Q"


def _chunk(chunk_id: str, **metadata) -> Chunk:
    return Chunk(id=chunk_id, content="", metadata=metadata)


def test_resolve_relevant_ids_or_combines_selectors() -> None:
    chunks = [
        _chunk("a", file_path="auth.py", symbol="AuthService.login", chunk_type="method"),
        _chunk("b", file_path="auth.py", symbol="AuthService.validate_token", chunk_type="method"),
        _chunk("c", file_path="README.md", symbol="Authentication", chunk_type="markdown_section"),
    ]
    selectors = (RelevantSelector(chunk_type="method"), RelevantSelector(symbol="Authentication"))
    assert resolve_relevant_ids(selectors, chunks) == ["a", "b", "c"]


def test_resolve_relevant_ids_deduplicates_and_matches_nothing() -> None:
    chunks = [
        _chunk("a", file_path="auth.py", symbol="AuthService.login", chunk_type="method"),
        _chunk("b", file_path="README.md", symbol="Authentication", chunk_type="markdown_section"),
    ]
    selectors = (RelevantSelector(file_path="auth.py"), RelevantSelector(symbol="AuthService.login"))
    assert resolve_relevant_ids(selectors, chunks) == ["a"]

    assert resolve_relevant_ids((RelevantSelector(symbol="missing"),), chunks) == []