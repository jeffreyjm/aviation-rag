"""Chunk store + dense matrix + BM25 + report store, persisted to disk."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .chunk import Chunk, chunk_reports

_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were which with we i our can do did "
    "had my me not but so if then there their they them he she his her you your".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class Index:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, reports: dict[str, dict], meta: dict):
        assert len(chunks) == len(embeddings)
        self.chunks = chunks
        self.embeddings = embeddings
        self.reports = reports  # id -> full record (labels live here, never in chunk text)
        self.meta = meta
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks])

    @classmethod
    def build(cls, reports: list[dict], embedder, **chunk_kw) -> "Index":
        chunks = chunk_reports(reports, **chunk_kw)
        emb = embedder.encode_docs([c.embed_text for c in chunks])
        meta = {
            "embed_model": getattr(embedder, "model_name", "unknown"),
            "n_reports": len(reports),
            "n_chunks": len(chunks),
        }
        return cls(chunks, emb, {r["id"]: r for r in reports}, meta)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        with (directory / "chunks.jsonl").open("w") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        with (directory / "reports.jsonl").open("w") as f:
            for r in self.reports.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "Index":
        emb = np.load(directory / "embeddings.npy")
        with (directory / "chunks.jsonl").open() as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        with (directory / "reports.jsonl").open() as f:
            reports = {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}
        return cls(chunks, emb, reports, json.loads((directory / "meta.json").read_text()))
