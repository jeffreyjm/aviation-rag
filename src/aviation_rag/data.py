"""Load and normalise NASA ASRS reports.

Source: the `elihoole/asrs-aviation-reports` mirror of public NASA Aviation Safety
Reporting System records (de-identified voluntary reports). We keep the fields an
investigator would use, and split them into two groups on purpose:

  * context (aircraft, phase, FAR part, year ...) -> may appear in the index text
  * labels  (anomaly, primary problem, factors)   -> NEVER indexed; used only to
                evaluate whether retrieved precedents are meaningful.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from . import config

COL = {
    "id": "acn_num_ACN",
    "date": "Time_Date",
    "aircraft": "Aircraft 1.2_Make Model Name",
    "operator": "Aircraft 1.1_Aircraft Operator",
    "far_part": "Aircraft 1.5_Operating Under FAR Part",
    "mission": "Aircraft 1.7_Mission",
    "phase": "Aircraft 1.9_Flight Phase",
    "conditions": "Environment_Flight Conditions",
    "anomaly": "Events_Anomaly",
    "contributing": "Assessments_Contributing Factors / Situations",
    "primary_problem": "Assessments.1_Primary Problem",
    "human_factors": "Person 1.7_Human Factors",
    "result": "Events.5_Result",
    "detector": "Events.3_Detector",
    "synopsis": "Report 1.2_Synopsis",
    "narrative": "Report 1_Narrative",
    "narrative2": "Report 2_Narrative",
}

CONTEXT_FIELDS = ("aircraft", "operator", "far_part", "mission", "phase", "conditions")
LIST_FIELDS = ("anomaly", "contributing", "human_factors", "result")


def _s(v) -> str:
    return " ".join(str(v or "").split())


def _split_list(v) -> list[str]:
    return [x.strip() for x in _s(v).split(";") if x.strip()]


def normalise(raw: dict) -> dict | None:
    """Raw ASRS row -> compact record, or None if it lacks an id, narrative or synopsis."""
    narrative = _s(raw.get(COL["narrative"]))
    synopsis = _s(raw.get(COL["synopsis"]))
    acn = _s(raw.get(COL["id"]))
    if not (acn and narrative and synopsis):
        return None
    date = _s(raw.get(COL["date"]))
    rec = {
        "id": acn,
        "year": int(date[:4]) if date[:4].isdigit() else None,
        "month": int(date[4:6]) if date[4:6].isdigit() else None,
        "narrative": narrative,
        "narrative2": _s(raw.get(COL["narrative2"])),
        "synopsis": synopsis,
        "primary_problem": _s(raw.get(COL["primary_problem"])),
    }
    for f in CONTEXT_FIELDS:
        rec[f] = _s(raw.get(COL[f]))
    for f in LIST_FIELDS:
        rec[f] = _split_list(raw.get(COL[f]))
    return rec


def download(raw_dir: Path = config.RAW_DIR, log=print) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for split in config.SPLITS:
        dest = raw_dir / f"asrs-{split}.jsonl"
        if not dest.exists():
            log(f"downloading {split} ...")
            urllib.request.urlretrieve(config.HF_BASE.format(split=split), dest)
        paths.append(dest)
    return paths


def build_reports(raw_paths: list[Path], out_path: Path = config.REPORTS_PATH) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for p in raw_paths:
        with p.open() as f:
            for line in f:
                rec = normalise(json.loads(line))
                if rec and rec["id"] not in seen:
                    seen.add(rec["id"])
                    out.append(rec)
    out.sort(key=lambda r: r["id"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def load_reports(path: Path = config.REPORTS_PATH) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
