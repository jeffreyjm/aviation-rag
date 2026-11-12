"""Retrieval: BM25, dense, hybrid (RRF), cross-encoder rerank, with metadata filters.

Filters apply *before* ranking (masked candidates can never be returned), so a
filtered search is a real constrained search, not a post-hoc trim of top-k.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .index import Index, tokenize
from .models import Embedder, Reranker

MODES = ("bm25", "dense", "hybrid", "hybrid+rerank")

# filter name -> (chunk.meta key, kind)
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


class Retriever:
    def __init__(self, index: Index, embedder: Embedder | None = None, reranker: Reranker | None = None):
        self.index = index
        self.embedder = embedder or Embedder()
        self.reranker = reranker or Reranker()
        self._mask_cache: dict[tuple, np.ndarray] = {}

    def mask(self, filters: dict | None) -> np.ndarray | None:
        filters = {k: v for k, v in (filters or {}).items() if v not in (None, "", [])}
        if not filters:
            return None
        bad = set(filters) - set(FILTER_KEYS)
        if bad:
            raise ValueError(f"unknown filters {sorted(bad)}; valid: {sorted(FILTER_KEYS)}")
        key = tuple(sorted(filters.items()))
        if key not in self._mask_cache:
            self._mask_cache[key] = np.array([matches(c.meta, filters) for c in self.index.chunks])
        return self._mask_cache[key]

    def _topn(self, scores: np.ndarray, n: int, mask: np.ndarray | None) -> list[int]:
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:n]
        return [int(i) for i in order if np.isfinite(scores[i])]

    def search(
        self,
        query: str,
        mode: str = "hybrid+rerank",
        k: int = 10,
        filters: dict | None = None,
        candidates: int = 80,
        rerank_top: int = 40,
        collapse: bool = True,
        exclude_reports: set[str] | None = None,
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        chunks = self.index.chunks
        mask = self.mask(filters)
        if exclude_reports:
            ex = np.array([c.report_id not in exclude_reports for c in chunks])
            mask = ex if mask is None else (mask & ex)

        bm = self.index.bm25.get_scores(tokenize(query)) if mode != "dense" else None
        de = self.index.embeddings @ self.embedder.encode_query(query) if mode != "bm25" else None

        if mode == "bm25":
            order = self._topn(bm, candidates, mask)
            hits = [Hit(i, chunks[i].report_id, float(bm[i]), {"bm25": float(bm[i])}) for i in order]
        elif mode == "dense":
            order = self._topn(de, candidates, mask)
            hits = [Hit(i, chunks[i].report_id, float(de[i]), {"dense": float(de[i])}) for i in order]
        else:
            fused = rrf([self._topn(de, candidates, mask), self._topn(bm, candidates, mask)])
            ranked = sorted(fused, key=fused.get, reverse=True)
            hits = [
                Hit(i, chunks[i].report_id, fused[i],
                    {"rrf": fused[i], "bm25": float(bm[i]), "dense": float(de[i])})
                for i in ranked
            ]
            if mode == "hybrid+rerank" and hits:
                top = hits[:rerank_top]
                ce = self.reranker.score(query, [chunks[h.chunk_idx].embed_text for h in top])
                for h, s in zip(top, ce):
                    h.components["rerank"] = float(s)
                    h.score = float(s)
                top.sort(key=lambda h: h.score, reverse=True)
                hits = top

        if collapse:
            hits = collapse_to_reports(hits)
        return hits[:k]
