from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


DATA_DIR = ROOT / "data"


RAW_DIR = DATA_DIR / "raw"


REPORTS_PATH = DATA_DIR / "reports.jsonl"


INDEX_DIR = DATA_DIR / "index"


RESULTS_DIR = ROOT / "eval" / "results"


OOD_PATH = ROOT / "eval" / "ood_queries.jsonl"
