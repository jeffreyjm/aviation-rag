"""Retrieval: BM25, dense, hybrid (RRF), cross-encoder rerank, with metadata filters.

Filters apply *before* ranking (masked candidates can never be returned), so a
filtered search is a real constrained search, not a post-hoc trim of top-k.
"""

from __future__ import annotations

from dataclasses import dataclass, field


MODES = ("bm25", "dense", "hybrid", "hybrid+rerank")


FILTER_KEYS = {
    "phase": ("phase", "eq"),
    "far_part": ("far_part", "eq"),
    "operator": ("operator", "eq"),
    "conditions": ("conditions", "eq"),
    "aircraft": ("aircraft", "contains"),
    "year_min": ("year", "min"),
    "year_max": ("year", "max"),
}



@dataclass
class Hit:
    chunk_idx: int
    report_id: str
    score: float
    components: dict[str, float] = field(default_factory=dict)



def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum_r w_r / (k + rank_r(d)), ranks 1-based."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + w / (k + rank)
    return fused



def collapse_to_reports(hits: list[Hit]) -> list[Hit]:
    """Keep each report's best-ranked chunk, preserving order."""
    seen: set[str] = set()
    out: list[Hit] = []
    for h in hits:
        if h.report_id not in seen:
            seen.add(h.report_id)
            out.append(h)
    return out



def matches(meta: dict, filters: dict) -> bool:
    for name, want in filters.items():
        if want in (None, "", []):
            continue
        key, kind = FILTER_KEYS[name]
        have = meta.get(key)
        if kind == "eq" and str(have).lower() != str(want).lower():
            return False
        if kind == "contains" and str(want).lower() not in str(have).lower():
            return False
        if kind == "min" and (have in (None, "") or int(have) < int(want)):
            return False
        if kind == "max" and (have in (None, "") or int(have) > int(want)):
            return False
    return True
