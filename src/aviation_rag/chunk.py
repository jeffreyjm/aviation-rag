"""Character-budget chunking of incident narratives.

Pilot narratives are messy: shorthand, ALL-CAPS, missing punctuation, run-on text.
So chunking packs sentences up to a character budget (with one sentence of overlap)
and hard-wraps any single "sentence" that is itself too long. Each chunk is
prefixed with neutral context (aircraft / phase / FAR part / year), never with
analyst labels, so chunks are self-describing without leaking evaluation targets.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

_ABBREV = re.compile(r"\b(approx|etc|vs|e\.g|i\.e|no|rwy|twy|alt|hdg|dept|apch|cpt|fo|capt)\.", re.I)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")
_MASK = "․"  # one-dot leader: stands in for protected periods



def split_sentences(text: str) -> list[str]:
    masked = _ABBREV.sub(lambda m: m.group(0)[:-1] + _MASK, text)
    parts = _SPLIT.split(masked.strip())
    return [p.replace(_MASK, ".").strip() for p in parts if p.strip()]


def _hard_wrap(s: str, limit: int) -> list[str]:
    """Break an over-long run-on sentence at word boundaries."""
    out: list[str] = []
    cur: list[str] = []
    n = 0
    for w in s.split():
        if n + len(w) + 1 > limit and cur:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(w)
        n += len(w) + 1
    if cur:
        out.append(" ".join(cur))
    return out


def pack(text: str, budget: int = 700) -> list[str]:
    """Pack sentences into chunks of at most ~`budget` chars, overlapping by one sentence."""
    sents: list[str] = []
    for s in split_sentences(text):
        sents.extend(_hard_wrap(s, budget) if len(s) > budget else [s])
    chunks: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for s in sents:
        if cur and size + len(s) + 1 > budget:
            chunks.append(cur)
            keep = cur[-1] if len(cur[-1]) + len(s) + 1 <= budget else None
            cur = [keep] if keep else []
            size = sum(len(x) + 1 for x in cur)
        cur.append(s)
        size += len(s) + 1
    if cur:
        chunks.append(cur)
    return [" ".join(c) for c in chunks]


def context_header(rec: dict) -> str:
    parts = [rec.get("aircraft"), rec.get("phase"), rec.get("far_part"), str(rec["year"]) if rec.get("year") else ""]
    return " | ".join(p for p in parts if p)


META_KEYS = ("aircraft", "operator", "far_part", "phase", "conditions", "year")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<report_id>#<n>"
    report_id: str
    text: str
    header: str
    position: int
    meta: dict = field(default_factory=dict)  # filterable context only, never labels

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def embed_text(self) -> str:
        return f"[{self.header}] {self.text}" if self.header else self.text


def chunk_report(rec: dict, budget: int = 700) -> list[Chunk]:
    header = context_header(rec)
    meta = {k: rec.get(k, "") for k in META_KEYS}
    pieces = pack(rec["narrative"], budget)
    if rec.get("narrative2"):
        pieces += pack("Second reporter: " + rec["narrative2"], budget)
    return [Chunk(f"{rec['id']}#{n}", rec["id"], t, header, n, meta) for n, t in enumerate(pieces)]


def chunk_reports(recs: list[dict], **kw) -> list[Chunk]:
    out: list[Chunk] = []
    for r in recs:
        out.extend(chunk_report(r, **kw))
    return out
