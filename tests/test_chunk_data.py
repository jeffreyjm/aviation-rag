from aviation_rag.data import normalise


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
