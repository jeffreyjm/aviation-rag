


def test_index_roundtrip(index, tmp_path):
    from aviation_rag.index import Index

    index.save(tmp_path)
    loaded = Index.load(tmp_path)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in index.chunks]
    assert loaded.reports["3"]["primary_problem"] == "Procedure"
    assert (loaded.embeddings == index.embeddings).all()
