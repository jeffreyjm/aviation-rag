import pytest

from aviation_rag.retrieve import Hit, collapse_to_reports, rrf


def test_rrf_math():
    fused = rrf([[10, 20], [20, 30]], k=60)
    assert fused[20] == pytest.approx(1 / 62 + 1 / 61)
    assert max(fused, key=fused.get) == 20



def test_collapse_keeps_best_chunk_per_report():
    out = collapse_to_reports([Hit(0, "a", 3.0), Hit(1, "b", 2.0), Hit(2, "a", 1.0)])
    assert [(h.report_id, h.chunk_idx) for h in out] == [("a", 0), ("b", 1)]



def test_index_roundtrip(index, tmp_path):
    from aviation_rag.index import Index

    index.save(tmp_path)
    loaded = Index.load(tmp_path)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in index.chunks]
    assert loaded.reports["3"]["primary_problem"] == "Procedure"
    assert (loaded.embeddings == index.embeddings).all()
