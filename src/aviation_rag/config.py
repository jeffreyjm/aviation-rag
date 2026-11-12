from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_PATH = DATA_DIR / "reports.jsonl"
INDEX_DIR = DATA_DIR / "index"
RESULTS_DIR = ROOT / "eval" / "results"
OOD_PATH = ROOT / "eval" / "ood_queries.jsonl"

HF_BASE = "https://huggingface.co/datasets/elihoole/asrs-aviation-reports/resolve/main/asrs-aviation-reports-{split}.jsonl"
SPLITS = ("test", "validation")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
LLM_MODEL = "claude-sonnet-5-5"
