"""
Reciprocal Rank Fusion (RRF).

RRF avoids needing to normalize dense cosine scores against BM25 scores
(different scales/distributions) by fusing on RANK instead of raw score:

    RRF(d) = sum over retrievers r of 1 / (k + rank_r(d))

k=60 is the standard damping constant from the original RRF paper
(Cormack et al., 2009) — it flattens the influence of any single
retriever's top pick so fusion isn't dominated by one system.
"""
from __future__ import annotations
from collections import defaultdict

from .config import config


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    lexical_results: list[tuple[str, float]],
    k: int = config.RRF_K,
    top_n: int = config.FUSED_TOP_N,
) -> list[tuple[str, float]]:
    """
    dense_results / lexical_results: list of (doc_id, score), already
    sorted best-first by each retriever.
    Returns: list of (doc_id, fused_score) sorted best-first.
    """
    fused_scores: dict[str, float] = defaultdict(float)

    for rank, (doc_id, _score) in enumerate(dense_results):
        fused_scores[doc_id] += 1.0 / (k + rank + 1)

    for rank, (doc_id, _score) in enumerate(lexical_results):
        fused_scores[doc_id] += 1.0 / (k + rank + 1)

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


if __name__ == "__main__":
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    lexical = [("b", 5.2), ("d", 4.1), ("a", 3.9)]
    print(reciprocal_rank_fusion(dense, lexical, top_n=4))
