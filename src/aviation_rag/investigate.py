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

from collections import Counter
from dataclasses import dataclass, field

from .generate import ABSTAIN_TEXT, Answer, Generator, finalize, make_sources
from .index import Index
from .retrieve import Hit, Retriever

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


def _labels(rec: dict, name: str) -> list[str]:
    v = rec.get(name)
    if isinstance(v, list):
        return [x for x in dict.fromkeys(v) if x]  # de-dupe, keep order
    return [v] if v else []


def base_rates(index: Index) -> dict[str, dict[str, float]]:
    n = len(index.reports)
    out: dict[str, dict[str, float]] = {}
    for name in FACTOR_FIELDS:
        c: Counter = Counter()
        for r in index.reports.values():
            c.update(_labels(r, name))
        out[name] = {lab: cnt / n for lab, cnt in c.items()}
    return out


def aggregate_factors(
    records: list[dict], rates: dict[str, dict[str, float]], top: int = 4
) -> dict[str, list[FactorRow]]:
    n = len(records)
    out: dict[str, list[FactorRow]] = {}
    for name in FACTOR_FIELDS:
        c: Counter = Counter()
        for r in records:
            c.update(_labels(r, name))
        rows = []
        for lab, cnt in c.most_common():
            br = rates[name].get(lab, 0.0)
            rows.append(FactorRow(lab, cnt, cnt / n, br, (cnt / n) / br if br else float("inf")))
        out[name] = rows[:top]
    return out


def investigate(
    description: str,
    retriever: Retriever,
    generator: Generator,
    k: int = 5,
    filters: dict | None = None,
    abstain_below: float | None = None,
    rates: dict | None = None,
) -> Investigation:
    filters = {k_: v for k_, v in (filters or {}).items() if v not in (None, "", [])}
    hits: list[Hit] = retriever.search(description, mode="hybrid+rerank", k=k, filters=filters)
    top = hits[0].components.get("rerank") if hits else None
    reports = retriever.index.reports
    records = [reports[h.report_id] for h in hits]
    rates = rates or base_rates(retriever.index)

    precedents = [
        Precedent(i, r["id"], h.score, r.get("year"), r.get("aircraft", ""), r.get("phase", ""),
                  r.get("far_part", ""), r["synopsis"])
        for i, (h, r) in enumerate(zip(hits, records), start=1)
    ]
    caveats = [
        "Precedents are voluntary, self-reported and de-identified; absence of a factor is not evidence it was absent.",
        f"Factor counts describe the {len(hits)} retrieved precedents, not incidence rates. Lift compares them with the corpus.",
    ]
    if len(hits) < k:
        caveats.append(f"Only {len(hits)} precedents matched the filters {filters or '{}'}.")

    sources = make_sources(retriever, hits, with_synopsis=True)
    if not hits or (abstain_below is not None and top is not None and top < abstain_below):
        caveats.append("Best match is weak; no draft generated.")
        draft = Answer(ABSTAIN_TEXT, True, sources, top_score=top)
        factors: dict[str, list[FactorRow]] = {name: [] for name in FACTOR_FIELDS}
    else:
        draft = finalize(generator.generate("investigate", description, sources), sources, top)
        factors = aggregate_factors(records, rates)
    return Investigation(description, filters, precedents, factors, draft, top, caveats)


def render(inv: Investigation) -> str:
    """Plain-text report for the CLI."""
    lines = [f"EVENT: {inv.description}"]
    if inv.filters:
        lines.append(f"FILTERS: {inv.filters}")
    lines += ["", "PRECEDENTS"]
    for p in inv.precedents:
        lines.append(f"  [{p.rank}] ASRS {p.report_id} | {p.aircraft} | {p.phase} | {p.far_part} | {p.year}")
        lines.append(f"      {p.synopsis}")
    lines += ["", "DRAFT" + (" (abstained)" if inv.draft.abstained else ""), f"  {inv.draft.text}"]
    if any(inv.factors.values()):
        lines += ["", "FACTORS ACROSS PRECEDENTS (count / share / lift vs corpus)"]
        from .investigate import FACTOR_FIELDS as F

        for name, rows in inv.factors.items():
            if rows:
                body = "; ".join(f"{r.label} {r.count}/{r.share:.0%}/x{r.lift:.1f}" for r in rows)
                lines.append(f"  {F[name]}: {body}")
    lines += [""] + [f"NOTE: {c}" for c in inv.caveats]
    return "\n".join(lines)
