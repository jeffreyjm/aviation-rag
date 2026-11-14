import pytest

from aviation_rag import evaluate as E
from aviation_rag.generate import Answer, Source


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



def test_run_modes_and_precedent_metrics(retriever, index):
    queries = [index.reports[i] for i in ("1", "2", "3", "4", "5")]
    runs = E.run_modes(retriever, queries, {"bm25": ("bm25", False), "h": ("hybrid+rerank", False), "h+phase": ("hybrid+rerank", True)})
    res = E.evaluate_retrieval(retriever, queries, runs, k_neighbors=2)
    assert res["n_queries"] == 5 and res["baselines"]["majority_class"] in {"Human Factors", "Aircraft", "Procedure", "Airport"}
    for name in runs:
        m = res["modes"][name]
        assert 0.0 <= m["recall@1"]["mean"] <= 1.0 and m["mrr"]["mean"] >= m["recall@1"]["mean"]
    # synopsis is a different text from the narrative, but tiny fixtures overlap enough to find the report
    assert res["modes"]["h"]["recall@10"]["mean"] == 1.0
    # filtering on the query's own phase can only shrink the candidate pool to same-phase reports
    for q, (ranked, _) in zip(queries, runs["h+phase"]):
        assert all(retriever.index.reports[i]["phase"] == q["phase"] for i in ranked)



def test_abstention_split_is_heldout():
    ood = [{"query": str(i), "group": "near_domain" if i % 2 else "far_domain"} for i in range(8)]
    res = E.evaluate_abstention([10, 11, 12, 13, 14, 15], ood, [0, 1, 2, 3, 4, 5, 6, 7])
    assert res["auroc_all"] == 1.0
    assert res["heldout"]["n_in_domain"] == 3 and res["heldout"]["n_out_of_scope"] == 4
    assert set(res["auroc_by_group"]) == {"near_domain", "far_domain"}



def test_answer_metrics_without_nli():
    a_ok = Answer("x [1]", False, [Source(1, "r1", "t", "s")], [1])
    a_abs = Answer("no", True, [])
    r = E.evaluate_answers([
        (E.GoldItem("q1", {"r1"}, True), a_ok), (E.GoldItem("q2", {"zz"}, True), a_ok),
        (E.GoldItem("q3", set(), False), a_abs), (E.GoldItem("q4", set(), False), a_ok),
    ])
    assert r["abstain_on_out_of_scope"] == 0.5 and r["answered_when_in_domain"] == 1.0
    assert r["citation_recall"] == 0.5 and r["citation_validity"] == 1.0
