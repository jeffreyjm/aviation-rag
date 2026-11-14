import pytest

from aviation_rag import evaluate as E


def test_rank_metrics():
    assert E.recall_at_k(["a", "b", "c"], {"c"}, 2) == 0.0 and E.recall_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert E.reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert E.reciprocal_rank(["x"], {"y"}) == 0.0



def test_auroc_and_threshold():
    assert E.auroc([3, 4], [1, 2]) == 1.0 and E.auroc([1], [1]) == 0.5
    assert 2 < E.best_threshold([5, 6, 7], [0, 1, 2]) <= 5



def test_sample_queries_deterministic(index):
    a = [q["id"] for q in E.sample_queries(index.reports, 3, seed=1)]
    assert a == [q["id"] for q in E.sample_queries(index.reports, 3, seed=1)]
    assert len(set(a)) == 3
