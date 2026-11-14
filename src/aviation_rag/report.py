"""Render evaluation results as Markdown."""

from __future__ import annotations


KI_COLS = ["recall@1", "recall@5", "recall@10", "mrr"]


PREC_COLS = [
    ("primary_problem_precision@5", "primary-problem precision@5"),
    ("primary_problem_vote_accuracy", "primary-problem vote accuracy"),
    ("anomaly_overlap@5", "anomaly overlap@5"),
]



def _fmt(m: dict) -> str:
    lo, hi = m["ci95"]
    return f"{m['mean']:.3f} [{lo:.2f}, {hi:.2f}]"



def _known_item(res: dict) -> list[str]:
    r = res["retrieval"]
    lines = [
        f"### Known-item retrieval (query = analyst synopsis, n={r['n_queries']}; "
        f"synopsis/narrative lexical overlap {r['synopsis_overlap']:.2f})",
        "",
        "| mode | " + " | ".join(KI_COLS) + " |",
        "|---|" + "---|" * len(KI_COLS),
    ]
    for name, m in r["modes"].items():
        lines.append(f"| {name} | " + " | ".join(_fmt(m[c]) for c in KI_COLS) + " |")
    lines += ["", "Cells are mean [95% bootstrap CI over queries]. The target is the single report the synopsis was written for.", ""]
    return lines



def _precedent(res: dict) -> list[str]:
    r = res["retrieval"]
    b = r["baselines"]
    lines = [
        f"### Precedent quality (top-5 neighbours excluding the query's own report; n={r['n_labelled']} labelled queries)",
        "",
        "| mode | " + " | ".join(t for _, t in PREC_COLS) + " |",
        "|---|" + "---|" * len(PREC_COLS),
    ]
    for name, m in r["modes"].items():
        lines.append(f"| {name} | " + " | ".join(_fmt(m[k]) for k, _ in PREC_COLS) + " |")
    lines += [
        "",
        "Baselines: random neighbours share the primary problem "
        f"{b['random_neighbour_primary_problem_precision']:.3f} of the time; always predicting "
        f"'{b['majority_class']}' is right {b['majority_class_accuracy']:.3f} of the time; "
        f"two random reports share an anomaly category {b['random_pair_anomaly_overlap']:.3f} of the time.",
        "",
    ]
    return lines
