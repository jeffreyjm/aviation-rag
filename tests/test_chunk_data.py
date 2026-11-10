from aviation_rag.chunk import pack, split_sentences
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
