"""Investigation assistant: precedents + aggregated factors + cited draft.

Given a free-text description of a new event (and optional filters such as flight
phase or FAR part) this returns:

  1. the most similar past reports (precedents),
  2. what the analysts' structured labels say across those precedents, with *lift*
     versus the corpus base rate so distinctive factors stand out from merely common ones,
  3. a cited draft summary that frames factors as hypotheses to check.

(2) is computed deterministically from the retrieved reports' labels, so it is
grounded by construction. (3) is generated and validated like any other answer.
The tool surfaces precedent; it never asserts a cause.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from .generate import Answer


FACTOR_FIELDS = {
    "primary_problem": "Primary problem",
    "anomaly": "Anomaly",
    "contributing": "Contributing factors",
    "human_factors": "Human factors",
    "result": "Result",
}



@dataclass
class Precedent:
    rank: int
    report_id: str
    score: float
    year: int | None
    aircraft: str
    phase: str
    far_part: str
    synopsis: str



@dataclass
class FactorRow:
    label: str
    count: int  # precedents carrying the label
    share: float  # count / n precedents
    base_rate: float  # share of the whole corpus carrying the label
    lift: float  # share / base_rate (>1: over-represented among precedents)



@dataclass
class Investigation:
    description: str
    filters: dict
    precedents: list[Precedent]
    factors: dict[str, list[FactorRow]]
    draft: Answer
    top_score: float | None
    caveats: list[str] = field(default_factory=list)
