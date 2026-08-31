import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.bm25 import BM25Index, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("TCS's Bench-Time!") == ["tcs", "s", "bench", "time"]


def test_bm25_ranks_relevant_doc_higher():
    idx = BM25Index()
    idx.build([
        ("1", "TCS bench period is long and pay during bench is low"),
        ("2", "Infosys WFH policy is flexible and hikes are decent"),
        ("3", "Wipro appraisal cycle is slow"),
    ])
    results = idx.search("TCS bench pay", top_k=3)
    assert results[0][0] == "1"
    assert results[0][1] > 0


def test_bm25_empty_query_returns_no_matches():
    idx = BM25Index()
    idx.build([("1", "TCS bench period"), ("2", "Infosys WFH")])
    results = idx.search("xyzabc", top_k=5)
    assert results == []


def test_bm25_requires_build_before_search():
    idx = BM25Index()
    try:
        idx.search("test")
        assert False, "should have raised"
    except RuntimeError:
        pass
