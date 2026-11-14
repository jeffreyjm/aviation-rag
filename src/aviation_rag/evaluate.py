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

import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from .index import tokenize
from .retrieve import Retriever


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



def sample_queries(reports: dict[str, dict], n: int, seed: int = 0) -> list[dict]:
    ids = sorted(reports)
    random.Random(seed).shuffle(ids)
    return [reports[i] for i in ids[:n]]



def load_ood(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]



def synopsis_overlap(queries: list[dict]) -> float:
    """Mean share of synopsis content words that also occur in the report's own narrative."""
    vals = []
    for r in queries:
        q = set(tokenize(r["synopsis"]))
        if q:
            vals.append(len(q & set(tokenize(r["narrative"] + " " + r["narrative2"]))) / len(q))
    return float(np.mean(vals))



def run_modes(retriever: Retriever, queries: list[dict], variants: dict[str, tuple[str, bool]], k: int = 11) -> dict:
    """variants: name -> (mode, use_phase_filter). Returns name -> [(ranked_ids, top_rerank_score)]."""
    out: dict[str, list] = {}
    for name, (mode, use_phase) in variants.items():
        rows = []
        for r in queries:
            filt = {"phase": r["phase"]} if use_phase and r.get("phase") else None
            hits = retriever.search(r["synopsis"], mode=mode, k=k, filters=filt)
            rows.append(([h.report_id for h in hits], hits[0].components.get("rerank") if hits else None))
        out[name] = rows
    return out



def _neighbors(ranked: list[str], own_id: str, k: int) -> list[str]:
    return [i for i in ranked if i != own_id][:k]



def _share_anomaly(a: dict, b: dict) -> bool:
    return bool(set(a["anomaly"]) & set(b["anomaly"]))



def random_pair_anomaly_rate(reports: dict[str, dict], n_pairs: int = 50000, seed: int = 0) -> float:
    rng = random.Random(seed)
    recs = list(reports.values())
    hits = sum(_share_anomaly(*rng.sample(recs, 2)) for _ in range(n_pairs))
    return hits / n_pairs



def evaluate_retrieval(retriever: Retriever, queries: list[dict], runs: dict, k_neighbors: int = 5) -> dict:
    reports = retriever.index.reports
    pp_counts = Counter(r["primary_problem"] for r in reports.values() if r["primary_problem"])
    total_pp = sum(pp_counts.values())
    pp_rate = {c: v / total_pp for c, v in pp_counts.items()}
    majority = max(pp_rate, key=pp_rate.get)
    anomaly_base = random_pair_anomaly_rate(reports)

    labelled = [i for i, r in enumerate(queries) if r["primary_problem"] not in ("", "Ambiguous")]
    report: dict = {
        "n_queries": len(queries),
        "n_labelled": len(labelled),
        "synopsis_overlap": synopsis_overlap(queries),
        "baselines": {
            "random_neighbour_primary_problem_precision": float(np.mean([pp_rate[queries[i]["primary_problem"]] for i in labelled])),
            "majority_class": majority,
            "majority_class_accuracy": float(np.mean([queries[i]["primary_problem"] == majority for i in labelled])),
            "random_pair_anomaly_overlap": anomaly_base,
        },
        "modes": {},
    }
    for name, rows in runs.items():
        per = {"recall@1": [], "recall@5": [], "recall@10": [], "mrr": []}
        misses = []
        for q, (ranked, _) in zip(queries, rows):
            rel = {q["id"]}
            per["recall@1"].append(recall_at_k(ranked, rel, 1))
            per["recall@5"].append(recall_at_k(ranked, rel, 5))
            per["recall@10"].append(recall_at_k(ranked, rel, 10))
            per["mrr"].append(reciprocal_rank(ranked, rel))
            if q["id"] not in ranked[:10] and len(misses) < 5:
                misses.append({"id": q["id"], "synopsis": q["synopsis"], "top3": ranked[:3]})

        prec, anom, correct = [], [], []
        for i in labelled:
            q, (ranked, _) = queries[i], runs[name][i]
            nb = _neighbors(ranked, q["id"], k_neighbors)
            if not nb:
                prec.append(0.0)
                correct.append(0.0)
                continue
            labs = [reports[j]["primary_problem"] for j in nb]
            prec.append(np.mean([l == q["primary_problem"] for l in labs]))
            vote = Counter(l for l in labs if l)
            correct.append(float(bool(vote) and vote.most_common(1)[0][0] == q["primary_problem"]))
        for q, (ranked, _) in zip(queries, rows):
            nb = _neighbors(ranked, q["id"], k_neighbors)
            anom.append(np.mean([_share_anomaly(q, reports[j]) for j in nb]) if nb else 0.0)

        report["modes"][name] = {
            **{m: _summ(v) for m, v in per.items()},
            "primary_problem_precision@5": _summ(prec),
            "primary_problem_vote_accuracy": _summ(correct),
            "anomaly_overlap@5": _summ(anom),
            "misses@10": misses,
        }
    return report
