"""Character-budget chunking of incident narratives.

Pilot narratives are messy: shorthand, ALL-CAPS, missing punctuation, run-on text.
So chunking packs sentences up to a character budget (with one sentence of overlap)
and hard-wraps any single "sentence" that is itself too long. Each chunk is
prefixed with neutral context (aircraft / phase / FAR part / year), never with
analyst labels, so chunks are self-describing without leaking evaluation targets.
"""

from __future__ import annotations

import re


_ABBREV = re.compile(r"\b(approx|etc|vs|e\.g|i\.e|no|rwy|twy|alt|hdg|dept|apch|cpt|fo|capt)\.", re.I)


_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")


_MASK = "․"  # one-dot leader: stands in for protected periods



def split_sentences(text: str) -> list[str]:
    masked = _ABBREV.sub(lambda m: m.group(0)[:-1] + _MASK, text)
    parts = _SPLIT.split(masked.strip())
    return [p.replace(_MASK, ".").strip() for p in parts if p.strip()]
