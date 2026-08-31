import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fusion import reciprocal_rank_fusion


def test_doc_ranked_first_by_both_retrievers_wins():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    lexical = [("a", 5.0), ("c", 4.0), ("b", 3.0)]
    fused = reciprocal_rank_fusion(dense, lexical, k=60, top_n=3)
    assert fused[0][0] == "a"


def test_doc_only_in_one_list_still_included():
    dense = [("a", 0.9)]
    lexical = [("b", 5.0)]
    fused = reciprocal_rank_fusion(dense, lexical, k=60, top_n=5)
    ids = {doc_id for doc_id, _ in fused}
    assert ids == {"a", "b"}


def test_top_n_truncates_results():
    dense = [(str(i), 1.0 - i * 0.01) for i in range(20)]
    lexical = []
    fused = reciprocal_rank_fusion(dense, lexical, top_n=5)
    assert len(fused) == 5
