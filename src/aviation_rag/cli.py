"""Command line: aviation-rag {data,index,ask,investigate,eval}."""

from __future__ import annotations
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
