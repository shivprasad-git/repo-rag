import json
from pathlib import Path

from app.config import Settings
from app.eval import golden_set_from_dict, load_golden_set, load_report, report_to_dict, run_eval, write_report
from app.embeddings import EmbeddingProvider


class UnitTokenizer:
    def count(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_many(self, texts: list[str]) -> list[int]:
        return [self.count(text) for text in texts]


class KeywordEmbeddingProvider(EmbeddingProvider):
    @property
    def dimensions(self) -> int:
        return 2

    def embed(self, text: str) -> list[float]:
        lower = text.lower()
        if any(term in lower for term in ("login", "token", "jwt")):
            return [1.0, 0.0]
        return [0.0, 1.0]


def _write_sample_repo(repo_path: Path) -> None:
    repo_path.mkdir()
    (repo_path / "auth.py").write_text(
        "\n".join(
            [
                "import jwt",
                "",
                "class AuthService:",
                "    def validate_token(self, token):",
                "        payload = jwt.decode(token, 'key', algorithms=['HS256'])",
                "        return payload['sub']",
                "",
                "    def login(self, username, password):",
                "        if username and password:",
                "            return jwt.encode({'sub': username}, 'key', algorithm='HS256')",
                "        return None",
            ]
        ),
        encoding="utf-8",
    )
    (repo_path / "README.md").write_text(
        "\n".join(
            [
                "# Sample",
                "",
                "A small service.",
                "",
                "# Authentication",
                "",
                "Uses JWT tokens to validate and create login sessions.",
            ]
        ),
        encoding="utf-8",
    )


def _use_test_tokenizer(monkeypatch) -> None:
    monkeypatch.setattr("app.ingest.splitter.get_tokenizer", lambda model_name: UnitTokenizer())
    monkeypatch.setattr("app.pipeline.build_embedder", lambda settings: KeywordEmbeddingProvider())


def _eval_settings(tmp_path: Path) -> Settings:
    return Settings(
        repositories_dir=tmp_path / "repositories",
        indexes_dir=tmp_path / "indexes",
        eval_results_dir=tmp_path / "results",
        reranker_enabled=False,
        max_chunk_tokens=0,
    )


def test_run_eval_computes_metrics_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    repo_path = tmp_path / "repo"
    _write_sample_repo(repo_path)
    settings = _eval_settings(tmp_path)

    golden = golden_set_from_dict(
        {
            "name": "eval-test",
            "repo": str(repo_path),
            "store": "simple",
            "index_name": "eval-test",
            "top_k": [2, 5],
            "questions": [
                {
                    "question": "How does login work?",
                    "relevant": [{"symbol": "AuthService.login"}],
                },
                {
                    "question": "How are tokens validated?",
                    "relevant": [{"symbol": "AuthService.validate_token"}],
                },
            ],
        }
    )

    report = run_eval(golden, settings)

    assert report.name == "eval-test"
    assert report.index_name == "eval-test"
    assert report.top_k == [2, 5]
    assert len(report.questions) == 2
    assert all(question.relevant_count >= 1 for question in report.questions)
    assert all(name in report.aggregate for name in ("mrr", "recall@2", "recall@5", "ndcg@5"))

    login_result = next(question for question in report.questions if "login" in question.question)
    assert login_result.metrics.recall_at_k[5] == 1.0
    assert login_result.metrics.mrr > 0.0

    path = write_report(report, settings)
    assert path.parent == settings.eval_results_dir
    assert path.name.startswith("eval-test_simple_")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == report_to_dict(report)
    assert payload["aggregate"]["mrr"] == report.aggregate["mrr"]
    assert load_report(path) == payload


def test_run_eval_reuses_existing_index(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    repo_path = tmp_path / "repo"
    _write_sample_repo(repo_path)
    settings = _eval_settings(tmp_path)
    golden = golden_set_from_dict(
        {
            "name": "eval-test",
            "repo": str(repo_path),
            "store": "simple",
            "index_name": "eval-test",
            "top_k": [2],
            "questions": [{"question": "How does login work?", "relevant": [{"symbol": "AuthService.login"}]}],
        }
    )

    first = run_eval(golden, settings)
    second = run_eval(golden, settings)

    assert first.aggregate == second.aggregate
    # The index still exists and did not need rebuilding.
    assert len(list((settings.indexes_dir / "chunks").glob("*.json"))) == 1


def test_run_eval_with_typescript_golden_set(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    settings = _eval_settings(tmp_path)
    golden = load_golden_set("eval/golden/sample_ts.json")

    report = run_eval(golden, settings)

    assert report.name == "sample_ts"
    assert report.index_name == "sample-ts-repo"
    assert len(report.questions) == 6
    assert all(question.relevant_count >= 1 for question in report.questions)

    login = next(question for question in report.questions if "login" in question.question)
    assert login.metrics.recall_at_k[5] == 1.0
    assert login.metrics.mrr > 0.0

    user_question = next(question for question in report.questions if "shape of a User" in question.question)
    assert user_question.relevant_count == 1
    assert user_question.metrics.recall_at_k[5] == 1.0
    assert user_question.metrics.mrr > 0.0

    assert all(name in report.aggregate for name in ("mrr", "recall@3", "recall@5", "ndcg@5"))