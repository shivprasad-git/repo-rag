"""Human-readable formatting of evaluation reports."""
from __future__ import annotations

from app.eval.metrics import metric_names
from app.eval.runner import EvalReport


def format_eval_report(report: EvalReport) -> str:
    lines = [
        f"name: {report.name}",
        f"index: {report.index_name}",
        f"store: {report.store}",
        f"questions: {len(report.questions)}",
        "aggregate:",
    ]
    for name in metric_names(report.top_k):
        lines.append(f"  {name:<12} {report.aggregate.get(name, 0.0):.4f}")

    lines.append("per question:")
    for question in report.questions:
        recall = ", ".join(f"@{k}={value:.3f}" for k, value in sorted(question.metrics.recall_at_k.items()))
        ndcg = ", ".join(f"@{k}={value:.3f}" for k, value in sorted(question.metrics.ndcg_at_k.items()))
        lines.append(f"  {question.question!r} relevant={question.relevant_count}")
        lines.append(f"      recall {recall}  ndcg {ndcg}  mrr={question.metrics.mrr:.3f}")
    return "\n".join(lines)