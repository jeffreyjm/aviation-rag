import hashlib

import numpy as np
import pytest

from aviation_rag.index import Index, tokenize


def rec(i, narrative, synopsis, *, aircraft="B737-800", phase="Cruise", far="Part 121", year=2018,
        primary="Human Factors", anomaly=("Deviation - Altitude",), contributing=("Human Factors",),
        human=("Fatigue",), result=("General None Reported",)):
    return {
        "id": str(i), "year": year, "month": 1, "narrative": narrative, "narrative2": "", "synopsis": synopsis,
        "primary_problem": primary, "aircraft": aircraft, "operator": "Air Carrier", "far_part": far,
        "mission": "Passenger", "phase": phase, "conditions": "VMC", "anomaly": list(anomaly),
        "contributing": list(contributing), "human_factors": list(human), "result": list(result),
    }



REPORTS = [
    rec(1, "During the landing flare the thrust lever stuck and the engine would not respond. We went around.",
        "Crew reported a stuck thrust lever on landing.", phase="Landing", primary="Aircraft",
        anomaly=("Aircraft Equipment Problem Critical",), contributing=("Aircraft",), human=()),
    rec(2, "Cleared to descend to 5000 but I set 4000 in the altitude window and descended below the assigned altitude.",
        "Captain descended below assigned altitude after a setting error.", phase="Descent",
        anomaly=("Deviation - Altitude",), human=("Situational Awareness",)),
    rec(3, "A TCAS resolution advisory fired during climb as traffic crossed above us. We followed the RA and climbed.",
        "Flight crew responded to a TCAS RA during climb.", phase="Climb", aircraft="A320",
        primary="Procedure", anomaly=("Conflict Airborne Conflict",), contributing=("Procedure",), human=()),
    rec(4, "The taxiway sign was obscured by snow and we turned onto the wrong taxiway near the ramp.",
        "Pilot taxied onto the wrong taxiway because the sign was obscured by snow.", phase="Taxi", year=2014,
        far="Part 135", primary="Airport", anomaly=("Ground Incursion Taxiway",), contributing=("Airport",), human=("Confusion",)),
    rec(5, "Fuel imbalance warning appeared in cruise and the crew crossfed fuel per the checklist.",
        "Fuel imbalance warning in cruise handled with crossfeed.", phase="Cruise",
        primary="Aircraft", anomaly=("Aircraft Equipment Problem Less Severe",), contributing=("Aircraft",), human=()),
]



class FakeEmbedder:
    model_name = "fake-hash-64"



    def _vec(self, text):
        v = np.zeros(64, dtype=np.float32)
        for t in tokenize(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % 64] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v


    def encode_docs(self, texts, batch_size=64):
        return np.stack([self._vec(t) for t in texts])


    def encode_query(self, q):
        return self._vec(q)



@pytest.fixture
def reports():
    return REPORTS



@pytest.fixture
def index(reports):
    return Index.build(reports, FakeEmbedder())
