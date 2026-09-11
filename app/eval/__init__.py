"""Retrieval evaluation harness: golden sets, metrics, and the runner."""
from app.eval.format import format_eval_report
from app.eval.golden import (
    GoldenQuestion,
    GoldenSet,
    RelevantSelector,
    golden_set_from_dict,
    load_golden_set,
    resolve_relevant_ids,
)
from app.eval.metrics import (
    RetrievalMetrics,
    aggregate_metrics,
    evaluate_retrieval,
    metric_names,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.eval.runner import EvalReport, QuestionResult, load_report, report_to_dict, run_eval, write_report

__all__ = [
    "EvalReport",
    "GoldenQuestion",
    "GoldenSet",
    "QuestionResult",
    "RelevantSelector",
    "aggregate_metrics",
    "evaluate_retrieval",
    "format_eval_report",
    "golden_set_from_dict",
    "load_golden_set",
    "load_report",
    "metric_names",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "report_to_dict",
    "resolve_relevant_ids",
    "run_eval",
    "write_report",
]