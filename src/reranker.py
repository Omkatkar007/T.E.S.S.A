"""
Pass-through "reranker" — the cross-encoder has been intentionally removed.

Why: the cross-encoder was a full separate transformer model loaded
alongside the embedder, and was one of the biggest contributors to the
out-of-memory crash on Render's free tier (512MB). Dropping it removes
an entire model's worth of RAM.

Trade-off: candidates are now used in their RRF-fused order directly,
rather than being re-scored by a cross-encoder for finer-grained
relevance. RRF fusion (BM25 + dense) is still a solid ranking on its
own — you lose some precision at the margins, not the ability to
retrieve relevant results.

IMPORTANT: "rerank_score" here is the candidate's real retrieval
confidence (best of dense cosine similarity / normalized BM25 score,
computed in pipeline.py), NOT a rank-position placeholder. A rank-based
placeholder would always give the top candidate a perfect score
regardless of whether it's actually relevant — which would silently
disable the sufficiency guardrail. Using real retrieval confidence keeps
that guardrail meaningful.

If you later move to a host with more memory, restoring the cross-encoder
is just swapping this file back — nothing else in the pipeline needs to
change, since it exposes the same rerank(query, candidates, top_n)
interface and still returns candidates with a "rerank_score" field.
"""
from __future__ import annotations

from .config import config


class PassthroughReranker:
    def rerank(self, query: str, candidates: list[dict], top_n: int = config.RERANK_TOP_N) -> list[dict]:
        """
        candidates: list of dicts, already sorted best-first by RRF fusion,
        each carrying a "retrieval_confidence" field set in pipeline.py.
        Returns the top_n candidates with "rerank_score" set from that
        real confidence value, so the sufficiency guardrail still gets a
        meaningful signal instead of a rank-based placeholder.
        """
        if not candidates:
            return []
        top = candidates[:top_n]
        for c in top:
            c["rerank_score"] = c.get("retrieval_confidence", 0.0)
        return top


def get_reranker() -> PassthroughReranker:
    return PassthroughReranker()
