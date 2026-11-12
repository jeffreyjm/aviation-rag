import pytest

from aviation_rag.retrieve import Hit, collapse_to_reports, rrf


def test_rrf_math():
    fused = rrf([[10, 20], [20, 30]], k=60)
    assert fused[20] == pytest.approx(1 / 62 + 1 / 61)
    assert max(fused, key=fused.get) == 20


def test_collapse_keeps_best_chunk_per_report():
    out = collapse_to_reports([Hit(0, "a", 3.0), Hit(1, "b", 2.0), Hit(2, "a", 1.0)])
    assert [(h.report_id, h.chunk_idx) for h in out] == [("a", 0), ("b", 1)]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid+rerank"])
def test_search_finds_obvious_report(retriever, mode):
    hits = retriever.search("thrust lever stuck on landing flare engine would not respond", mode=mode, k=3)
    assert hits[0].report_id == "1"


def test_filter_is_applied_before_ranking(retriever):
    # Unfiltered, report 2 is the best match; with phase=Climb it must not appear at all.
    q = "descended below the assigned altitude with 4000 in the altitude window"
    assert retriever.search(q, k=1)[0].report_id == "2"
    hits = retriever.search(q, k=5, filters={"phase": "Climb"})
    assert hits and {h.report_id for h in hits} == {"3"}


def test_filters_numeric_and_substring(retriever):
    assert {h.report_id for h in retriever.search("taxiway", k=5, filters={"year_max": 2015})} == {"4"}
    assert {h.report_id for h in retriever.search("climb traffic", k=5, filters={"aircraft": "a320"})} == {"3"}


def test_filter_with_no_match_returns_empty(retriever):
    assert retriever.search("anything", filters={"phase": "Parked"}) == []


def test_unknown_filter_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("q", filters={"colour": "red"})


def test_exclude_reports(retriever):
    hits = retriever.search("thrust lever stuck landing", k=5, exclude_reports={"1"})
    assert "1" not in {h.report_id for h in hits}


def test_rerank_components_present(retriever):
    h = retriever.search("fuel imbalance crossfeed", mode="hybrid+rerank", k=1)[0]
    assert {"rrf", "bm25", "dense", "rerank"} <= set(h.components) and h.report_id == "5"


def test_unknown_mode(retriever):
    with pytest.raises(ValueError):
        retriever.search("q", mode="magic")


def test_index_roundtrip(index, tmp_path):
    from aviation_rag.index import Index

    index.save(tmp_path)
    loaded = Index.load(tmp_path)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in index.chunks]
    assert loaded.reports["3"]["primary_problem"] == "Procedure"
    assert (loaded.embeddings == index.embeddings).all()
