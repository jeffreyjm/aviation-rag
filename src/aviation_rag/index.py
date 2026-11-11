"""Chunk store + dense matrix + BM25 + report store, persisted to disk."""

from __future__ import annotations
import re


_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were which with we i our can do did "
    "had my me not but so if then there their they them he she his her you your".split()
)


_TOKEN = re.compile(r"[a-z0-9]+")



def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]
