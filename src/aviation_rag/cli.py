"""Command line: aviation-rag {data,index,ask,investigate,eval}."""

from __future__ import annotations
import json
import os
import sys

from . import config


FILTER_ARGS = ("phase", "far_part", "operator", "conditions", "aircraft", "year_min", "year_max")



def _retriever():
    from .index import Index
    from .retrieve import Retriever

    if not (config.INDEX_DIR / "meta.json").exists():
        sys.exit("No index found. Run: aviation-rag data && aviation-rag index")
    return Retriever(Index.load(config.INDEX_DIR))



def _filters(args) -> dict:
    return {k: getattr(args, k) for k in FILTER_ARGS if getattr(args, k, None) not in (None, "")}



def _generator(name: str, retriever):
    from .generate import AnthropicGenerator, ExtractiveGenerator

    if name == "auto":
        name = "llm" if os.environ.get("ANTHROPIC_API_KEY") else "extractive"
    return AnthropicGenerator() if name == "llm" else ExtractiveGenerator(retriever.reranker)



def cmd_data(args):
    from .data import build_reports, download

    recs = build_reports(download())
    print(f"wrote {len(recs)} reports -> {config.REPORTS_PATH}")



def cmd_index(args):
    from .data import load_reports
    from .index import Index
    from .models import Embedder

    idx = Index.build(load_reports(), Embedder())
    idx.save(config.INDEX_DIR)
    print(f"indexed {idx.meta['n_reports']} reports / {idx.meta['n_chunks']} chunks -> {config.INDEX_DIR}")



def cmd_ask(args):
    from .generate import answer

    r = _retriever()
    gen = _generator(args.generator, r)
    res = answer(args.question, r, gen, k=args.k, abstain_below=args.abstain_below, filters=_filters(args))
    print(res.text, "\n")
    if res.top_score is not None:
        print(f"(top rerank score {res.top_score:.2f}; generator {type(gen).__name__})\n")
    for s in res.sources:
        print(f" {'*' if s.n in res.cited else ' '}[{s.n}] ASRS {s.report_id}  {s.title}")



def cmd_investigate(args):
    from .investigate import investigate, render

    r = _retriever()
    gen = _generator(args.generator, r)
    inv = investigate(args.description, r, gen, k=args.k, filters=_filters(args), abstain_below=args.abstain_below)
    print(render(inv))



def cmd_eval(args):
    from . import evaluate as E
    from .data import load_reports  # noqa: F401  (ensures data layer importable)
    from .generate import answer  # noqa: F401
    from .investigate import investigate
    from .models import NLI
    from .report import write_report

    r = _retriever()
    queries = E.sample_queries(r.index.reports, args.n, seed=0)
    ood = E.load_ood(config.OOD_PATH)
    results: dict = {"index": r.index.meta}

    print(f"retrieval: {len(queries)} queries x {len(E.DEFAULT_VARIANTS)} variants ...", flush=True)
    runs = E.run_modes(r, queries, E.DEFAULT_VARIANTS)
    results["retrieval"] = E.evaluate_retrieval(r, queries, runs)

    print("abstention ...", flush=True)
    in_scores = [s for _, s in runs["hybrid+rerank"] if s is not None]
    ood_scores = [(r.search(o["query"], mode="hybrid+rerank", k=1) or [None])[0].components["rerank"] for o in ood]
    results["abstention"] = E.evaluate_abstention(in_scores, ood, ood_scores)
    threshold = results["abstention"]["threshold_calibrated_on_half"]

    print("investigation drafts (extractive) ...", flush=True)
    gen = _generator("extractive", r)
    from .investigate import base_rates

    rates = base_rates(r.index)
    pairs = []
    for q in queries[: args.n_answers]:
        inv = investigate(q["synopsis"], r, gen, k=5, abstain_below=threshold, rates=rates)
        pairs.append((E.GoldItem(q["synopsis"], {q["id"]}, True), inv.draft))
    for o in ood:
        inv = investigate(o["query"], r, gen, k=5, abstain_below=threshold, rates=rates)
        pairs.append((E.GoldItem(o["query"], set(), False), inv.draft))
    results["answers_extractive"] = E.evaluate_answers(pairs, NLI())

    if args.llm:
        print("investigation drafts (LLM) ...", flush=True)
        llm = _generator("llm", r)
        pairs = []
        for q in queries[: args.n_answers]:
            inv = investigate(q["synopsis"], r, llm, k=5, abstain_below=threshold, rates=rates)
            pairs.append((E.GoldItem(q["synopsis"], {q["id"]}, True), inv.draft))
        for o in ood:
            inv = investigate(o["query"], r, llm, k=5, abstain_below=threshold, rates=rates)
            pairs.append((E.GoldItem(o["query"], set(), False), inv.draft))
        results["answers_llm"] = E.evaluate_answers(pairs, NLI())

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    md = write_report(results)
    (config.RESULTS_DIR / "results.md").write_text(md)
    print(md)
