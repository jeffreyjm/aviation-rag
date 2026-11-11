"""Thin, lazily-loaded wrappers around the embedding / reranking / NLI models.

Keeping them behind small classes lets tests inject fakes and keeps import
time low (torch is only touched when a model is actually used).
"""

from __future__ import annotations

import numpy as np

from . import config


class Embedder:



    def __init__(self, model_name: str = config.EMBED_MODEL, query_prefix: str = config.QUERY_PREFIX):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self._model = None


    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model


    def encode_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=len(texts) > 256, convert_to_numpy=True,
        ).astype(np.float32)


    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            [self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)[0]



class Reranker:
    """Cross-encoder: scores (query, passage) pairs jointly. Slower, more precise."""



    def __init__(self, model_name: str = config.RERANK_MODEL):
        self.model_name = model_name
        self._model = None


    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model


    def score(self, query: str, passages: list[str], batch_size: int = 32) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        pairs = [(query, p) for p in passages]
        return np.asarray(self.model.predict(pairs, batch_size=batch_size), dtype=np.float32)
