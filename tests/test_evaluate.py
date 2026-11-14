import pytest

from aviation_rag import evaluate as E


def test_rank_metrics():
    assert E.recall_at_k(["a", "b", "c"], {"c"}, 2) == 0.0 and E.recall_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert E.reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert E.reciprocal_rank(["x"], {"y"}) == 0.0
