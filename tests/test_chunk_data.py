from aviation_rag.chunk import chunk_report, context_header, pack, split_sentences
from aviation_rag.data import normalise


def test_pack_respects_budget_and_overlaps_one_sentence():
    text = " ".join(f"Sentence number {i} is here." for i in range(12))
    chunks = pack(text, budget=100)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) > 1
    last_of_first = split_sentences(chunks[0])[-1]
    assert chunks[1].startswith(last_of_first)  # one-sentence overlap


def test_pack_hard_wraps_runon_text_without_punctuation():
    runon = "we were cleared for the approach " * 60  # no sentence boundaries at all
    chunks = pack(runon, budget=200)
    assert len(chunks) > 5 and all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks).split().count("approach") >= 60  # nothing lost


def test_short_narrative_is_one_chunk(reports):
    assert len(chunk_report(reports[0])) == 1


def test_second_reporter_is_appended():
    r = dict(narrative="First view.", narrative2="Other view.", id="9", year=2020, aircraft="X", phase="Taxi", far_part="")
    texts = [c.text for c in chunk_report(r)]
    assert any(t.startswith("Second reporter:") for t in texts)


def test_no_analyst_labels_leak_into_indexed_text(reports):
    """Precedent-quality eval is only valid if analyst labels are never *added* to indexed text.

    The narrative is raw reporter text and may use any word; what we guard is that the
    chunk header and filterable metadata values (everything we add) carry no labels, and
    that the only metadata keys are the declared context fields."""
    from aviation_rag.chunk import META_KEYS

    for r in reports:
        labels = {x.lower() for x in [r["primary_problem"], *r["anomaly"], *r["contributing"], *r["human_factors"], *r["result"]]}
        for c in chunk_report(r):
            assert set(c.meta) == set(META_KEYS)
            added = [c.header.lower()] + [str(v).lower() for v in c.meta.values()]
            for label in labels:
                assert not any(label in a for a in added), f"label {label!r} leaked into chunk {c.chunk_id}"
            assert c.embed_text == f"[{c.header}] {c.text}"  # nothing else is prepended


def test_header_is_context_only(reports):
    h = context_header(reports[0])
    assert h == "B737-800 | Landing | Part 121 | 2018"


def test_citation_after_period_stays_with_its_sentence():
    assert split_sentences("First. [1] Second. [2][3] Third [2].", keep_citations=True) == [
        "First. [1]", "Second. [2][3]", "Third [2].",
    ]


def test_normalise_parses_lists_and_year():
    raw = {
        "acn_num_ACN": "123", "Time_Date": "201808", "Report 1_Narrative": " We  flew. ",
        "Report 1.2_Synopsis": "Flew.", "Events_Anomaly": "A One; B Two ;", "Aircraft 1.9_Flight Phase": "Cruise",
        "Assessments.1_Primary Problem": "Aircraft",
    }
    r = normalise(raw)
    assert r["year"] == 2018 and r["month"] == 8
    assert r["anomaly"] == ["A One", "B Two"]
    assert r["narrative"] == "We flew." and r["phase"] == "Cruise"


def test_normalise_drops_rows_missing_synopsis_or_narrative():
    base = {"acn_num_ACN": "1", "Report 1_Narrative": "x", "Report 1.2_Synopsis": "y"}
    assert normalise(base) is not None
    assert normalise({**base, "Report 1.2_Synopsis": ""}) is None
    assert normalise({**base, "Report 1_Narrative": ""}) is None
