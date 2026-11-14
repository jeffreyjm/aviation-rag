"""Evaluation harness (no hand-written gold set required).

The gold standard comes from the data itself. Every ASRS report has an
analyst-written *synopsis* that the index never sees. So:

1. Known-item retrieval - query = a report's synopsis, target = that report.
   Synopses paraphrase the narrative, so there is a genuine vocabulary gap, and
   hundreds of queries come for free with no author bias.

2. Precedent quality - for the same queries, remove the query's own report and ask
   whether the nearest *other* reports share its analyst labels (primary problem,
   anomaly). Labels are never in the index text, so this is not leakage. Each result
   is reported next to its random-neighbour baseline.

3. Abstention - can the top reranker score separate in-domain synopses from
   out-of-scope event descriptions written in the same voice? Threshold is
   calibrated on one half and scored on the other.

4. Answers - citation validity/recall and NLI groundedness of generated drafts.
"""

from __future__ import annotations
import re

import numpy as np


_CITE = re.compile(r"\[(\d+)\]")



def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0



def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, rid in enumerate(ranked, start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0



def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))



def _summ(values) -> dict:
    return {"mean": float(np.mean(values)), "ci95": bootstrap_ci(values)}



def auroc(pos, neg) -> float:
    p, n = np.asarray(pos), np.asarray(neg)
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))



def best_threshold(pos, neg) -> float:
    """Threshold maximising balanced accuracy (in-domain >= t, out-of-scope < t)."""
    best_t, best = None, -1.0
    for t in sorted(set(pos) | set(neg)):
        bacc = (np.mean([s >= t for s in pos]) + np.mean([s < t for s in neg])) / 2
        if bacc > best:
            best_t, best = t, float(bacc)
    return float(best_t)
