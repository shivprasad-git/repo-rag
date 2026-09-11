"""Retrieval evaluation metrics.

All metric functions use a binary-relevance model:

    relevant_ids   - set of chunk ids judged relevant to a question
    retrieved_ids  - chunk ids in rank order returned by the retriever
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RetrievalMetrics:
    relevant_count: int
    retrieved_count: int
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]


def recall_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    """Fraction of relevant chunks present in the top ``k`` retrieved ids."""
    if not relevant_ids:
        return 0.0
    returned = set(retrieved_ids[: max(k, 0)])
    return len(returned & relevant_ids) / len(relevant_ids)


def precision_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    """Fraction of the top ``k`` retrieved ids that are relevant."""
    if k <= 0:
        return 0.0
    returned = set(retrieved_ids[:k])
    return len(returned & relevant_ids) / k


def reciprocal_rank(relevant_ids: set[str], retrieved_ids: list[str]) -> float:
    """Returns 1 / rank of the first relevant id, or 0.0 when none is retrieved."""
    if not relevant_ids:
        return 0.0
    for index, chunk_id in enumerate(retrieved_ids):
        if chunk_id in relevant_ids:
            return 1.0 / (index + 1)
    return 0.0


def ndcg_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    """Normalized discounted cumulative gain at ``k`` with binary relevance."""
    if not relevant_ids or k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    dcg = 0.0
    for index, chunk_id in enumerate(retrieved_ids[:k]):
        if chunk_id in relevant:
            dcg += 1.0 / math.log2(index + 2)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    relevant_ids: set[str],
    retrieved_ids: list[str],
    ks: Sequence[int],
) -> RetrievalMetrics:
    """Compute retrieval metrics for one question at every ``k`` value."""
    ordered_ks = sorted(k for k in ks if k > 0)
    return RetrievalMetrics(
        relevant_count=len(relevant_ids),
        retrieved_count=len(retrieved_ids),
        recall_at_k={k: recall_at_k(relevant_ids, retrieved_ids, k) for k in ordered_ks},
        precision_at_k={k: precision_at_k(relevant_ids, retrieved_ids, k) for k in ordered_ks},
        mrr=reciprocal_rank(relevant_ids, retrieved_ids),
        ndcg_at_k={k: ndcg_at_k(relevant_ids, retrieved_ids, k) for k in ordered_ks},
    )


def metric_names(ks: Sequence[int]) -> list[str]:
    """Return the canonical aggregate metric key names for the given ``k`` values."""
    names = ["mrr"]
    for k in sorted(k for k in ks if k > 0):
        names.extend([f"recall@{k}", f"precision@{k}", f"ndcg@{k}"])
    return names


def aggregate_metrics(metrics_list: Sequence[RetrievalMetrics], ks: Sequence[int]) -> dict[str, float]:
    """Average every metric across questions, returning one value per metric name."""
    names = metric_names(ks)
    if not metrics_list:
        return {name: 0.0 for name in names}
    totals = {name: 0.0 for name in names}
    for metrics in metrics_list:
        totals["mrr"] += metrics.mrr
        for k, value in metrics.recall_at_k.items():
            totals[f"recall@{k}"] += value
        for k, value in metrics.precision_at_k.items():
            totals[f"precision@{k}"] += value
        for k, value in metrics.ndcg_at_k.items():
            totals[f"ndcg@{k}"] += value
    count = len(metrics_list)
    return {name: total / count for name, total in totals.items()}