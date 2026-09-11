import pytest

from app.eval.metrics import (
    aggregate_metrics,
    evaluate_retrieval,
    metric_names,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_counts_relevant_hits_in_top_k() -> None:
    relevant = {"a", "b", "c"}
    retrieved = ["a", "x", "b", "y"]
    assert recall_at_k(relevant, retrieved, 1) == 1 / 3
    assert recall_at_k(relevant, retrieved, 3) == 2 / 3
    assert recall_at_k(relevant, retrieved, 10) == 2 / 3


def test_recall_at_k_with_empty_relevant_is_zero() -> None:
    assert recall_at_k(set(), ["a"], 5) == 0.0


def test_precision_at_k() -> None:
    relevant = {"a", "b"}
    retrieved = ["a", "x", "b"]
    assert precision_at_k(relevant, retrieved, 1) == 1.0
    assert precision_at_k(relevant, retrieved, 3) == 2 / 3
    assert precision_at_k(relevant, retrieved, 0) == 0.0


def test_reciprocal_rank() -> None:
    relevant = {"b"}
    assert reciprocal_rank(relevant, ["a", "b", "c"]) == 0.5
    assert reciprocal_rank(relevant, ["a", "c"]) == 0.0
    assert reciprocal_rank(set(), ["a"]) == 0.0


def test_ndcg_at_k_perfect_and_partial() -> None:
    relevant = {"a", "b"}
    assert ndcg_at_k(relevant, ["a", "b"], 2) == pytest.approx(1.0)
    partial = ndcg_at_k(relevant, ["z", "a"], 2)
    assert 0.0 < partial < 1.0
    assert ndcg_at_k(relevant, ["z"], 2) == 0.0
    assert ndcg_at_k(set(), ["a"], 2) == 0.0


def test_evaluate_retrieval_reports_each_k() -> None:
    metrics = evaluate_retrieval({"a", "b"}, ["a", "x", "b"], ks=[1, 2, 5])
    assert metrics.relevant_count == 2
    assert metrics.retrieved_count == 3
    assert metrics.recall_at_k == {1: 0.5, 2: 0.5, 5: 1.0}
    assert metrics.precision_at_k == {1: 1.0, 2: 0.5, 5: 2 / 5}
    assert metrics.mrr == 1.0
    assert metrics.ndcg_at_k[1] == pytest.approx(1.0)


def test_aggregate_metrics_averages_and_defaults_to_zero() -> None:
    one = evaluate_retrieval({"a", "b"}, ["a", "x", "b"], ks=[3])
    two = evaluate_retrieval({"c"}, ["c", "z"], ks=[3])
    aggregate = aggregate_metrics([one, two], ks=[3])
    assert aggregate["recall@3"] == pytest.approx(1.0)
    assert aggregate["mrr"] == pytest.approx(1.0)
    assert aggregate["precision@3"] == pytest.approx((2 / 3 + 1 / 3) / 2)
    # `b` is relevant at rank 3 in `one`, so its ndcg contribution is discounted.
    assert 0.9 < aggregate["ndcg@3"] < 1.0

    defaulted = aggregate_metrics([], ks=[3])
    assert set(defaulted) == set(metric_names([3]))
    assert all(value == 0.0 for value in defaulted.values())


def test_metric_names_are_sorted_and_include_mrr() -> None:
    assert metric_names([5, 3]) == ["mrr", "recall@3", "precision@3", "ndcg@3", "recall@5", "precision@5", "ndcg@5"]