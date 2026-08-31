import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.context_builder import build_context, count_tokens
from src import config as config_module


def make_chunk(text, company="tcs", source="ambitionbox", rerank_score=0.5):
    return {"text": text, "payload": {"company": company, "source": source}, "rerank_score": rerank_score}


def test_respects_max_chunks():
    chunks = [make_chunk(f"Review number {i} about bench and pay.") for i in range(10)]
    context, used = build_context(chunks)
    assert len(used) <= config_module.config.MAX_CHUNKS


def test_respects_token_budget():
    huge_text = "bench pay culture " * 2000  # way over budget
    chunks = [make_chunk(huge_text) for _ in range(5)]
    context, used = build_context(chunks)
    assert count_tokens(context) <= config_module.config.MAX_CONTEXT_TOKENS + 10  # small tag overhead


def test_includes_company_tag():
    chunks = [make_chunk("bench pay is low", company="infosys", source="glassdoor_cons")]
    context, used = build_context(chunks)
    assert "infosys" in context.lower()
    assert "glassdoor_cons" in context.lower()
