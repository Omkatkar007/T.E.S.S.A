"""
Local cross-encoder reranking (no external API calls).
Cross-encoders jointly encode (query, passage) pairs, which is far more
accurate than comparing independently-encoded embeddings — but too slow
to run over the whole corpus, so it only re-scores the small fused
candidate set that survives RRF.
"""
from __future__ import annotations
from sentence_transformers import CrossEncoder

from .config import config


class Reranker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = CrossEncoder(config.RERANKER_MODEL)
        return cls._instance

    def rerank(self, query: str, candidates: list[dict], top_n: int = config.RERANK_TOP_N) -> list[dict]:
        """
        candidates: list of dicts each containing at least {"id", "text"}.
        Returns candidates sorted by cross-encoder score, with a
        "rerank_score" field added, truncated to top_n.
        """
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_n]


def get_reranker() -> Reranker:
    return Reranker()
