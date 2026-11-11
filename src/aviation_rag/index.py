"""Chunk store + dense matrix + BM25 + report store, persisted to disk."""

from __future__ import annotations
import re

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
