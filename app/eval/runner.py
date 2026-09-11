"""Evaluation runner: index a repository, query it per golden question, and score retrieval.

Produces a JSON report that can be compared over time as retrieval changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.config import Settings
from app.docstore import build_chunk_store
from app.eval.golden import GoldenSet, resolve_relevant_ids
from app.eval.metrics import RetrievalMetrics, aggregate_metrics, evaluate_retrieval
from app.logging_config import get_logger
from app.pipeline import default_index_name, index_repository, query_repository

logger = get_logger(__name__)


@dataclass(frozen=True)
class QuestionResult:
    question: str
    relevant_count: int
    metrics: RetrievalMetrics


@dataclass(frozen=True)
class EvalReport:
    name: str
    store: str
    index_name: str
    top_k: list[int]
    timestamp: str
    questions: list[QuestionResult]
    aggregate: dict[str, float]


def run_eval(golden: GoldenSet, settings: Settings, index_name: str | None = None) -> EvalReport:
    """Run retrieval evaluation for a golden set.

    The repository is indexed lazily when its index is empty, mirroring
    ``ask_repository``. Relevant chunk ids are resolved from the document
    store, then each question is run through the standard retrieval pipeline.
    """
    store = golden.store
    name = index_name or golden.index_name or default_index_name(golden.repo)
    chunk_store = build_chunk_store(settings.indexes_dir, name)

    if not chunk_store.has_data():
        logger.info("Index %s is empty; indexing repository before evaluation", name)
        index_repository(golden.repo, store, settings, index_name=name)

    all_chunks = chunk_store.all_chunks()
    ks = sorted(set(golden.top_k))
    max_k = max(ks)

    results: list[QuestionResult] = []
    for golden_question in golden.questions:
        relevant_ids = set(resolve_relevant_ids(golden_question.relevant, all_chunks))
        matches = query_repository(
            golden_question.question,
            store,
            settings,
            max_k,
            index_name=name,
        )
        retrieved_ids = [chunk.id for chunk, _ in matches]
        results.append(
            QuestionResult(
                question=golden_question.question,
                relevant_count=len(relevant_ids),
                metrics=evaluate_retrieval(relevant_ids, retrieved_ids, ks),
            )
        )

    return EvalReport(
        name=golden.name,
        store=store,
        index_name=name,
        top_k=ks,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        questions=results,
        aggregate=aggregate_metrics([result.metrics for result in results], ks),
    )


def write_report(report: EvalReport, settings: Settings) -> Path:
    """Persist the report to ``eval_results_dir`` and return its path."""
    output_dir = settings.eval_results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.name}_{report.store}_{report.timestamp}.json"
    path.write_text(json.dumps(report_to_dict(report), indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote eval report to %s", path)
    return path


def load_report(path: str | Path) -> dict:
    """Load a previously written eval report JSON as a plain dict."""
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"Eval report not found: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def report_to_dict(report: EvalReport) -> dict:
    """Serialize a report using string metric keys so it round-trips through JSON."""
    return {
        "name": report.name,
        "store": report.store,
        "index_name": report.index_name,
        "top_k": report.top_k,
        "timestamp": report.timestamp,
        "aggregate": report.aggregate,
        "questions": [
            {
                "question": question.question,
                "relevant_count": question.relevant_count,
                "metrics": _metrics_to_dict(question.metrics),
            }
            for question in report.questions
        ],
    }


def _metrics_to_dict(metrics: RetrievalMetrics) -> dict:
    return {
        "relevant_count": metrics.relevant_count,
        "retrieved_count": metrics.retrieved_count,
        "mrr": metrics.mrr,
        "recall_at_k": {str(k): value for k, value in sorted(metrics.recall_at_k.items())},
        "precision_at_k": {str(k): value for k, value in sorted(metrics.precision_at_k.items())},
        "ndcg_at_k": {str(k): value for k, value in sorted(metrics.ndcg_at_k.items())},
    }