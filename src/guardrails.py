"""
4-layer guardrail pipeline, applied in order — each layer can short-circuit
and return a refusal before the next layer (or the LLM call) runs:

1. Off-topic filter   — is this even a company/placement question?
2. Safety filter       — block prompt-injection / jailbreak attempts in the query
3. Sufficiency check   — did retrieval actually find relevant evidence?
4. Grounding check     — does the generated answer overlap with the retrieved
                          context, or did the LLM drift into ungrounded claims?
"""
from __future__ import annotations
import re
from dataclasses import dataclass

from .config import config
from .bm25 import tokenize

_INJECTION_PATTERNS = [
    r"ignore (all|previous|the) instructions",
    r"disregard (your|the) (system|prior) prompt",
    r"you are now",
    r"act as (?!a placement)",
    r"reveal (your|the) (system prompt|instructions)",
]


@dataclass
class GuardrailResult:
    passed: bool
    layer: str | None = None
    reason: str | None = None


def check_off_topic(query: str) -> GuardrailResult:
    # Off-topic checks are disabled for the HR pivot since HR teams 
    # query their own specific internal datasets.
    return GuardrailResult(passed=True)


def check_safety(query: str) -> GuardrailResult:
    q = query.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, q):
            return GuardrailResult(
                passed=False, layer="safety",
                reason="Query matched a prompt-injection / instruction-override pattern.",
            )
    return GuardrailResult(passed=True)


def check_sufficiency(reranked_chunks: list[dict]) -> GuardrailResult:
    if not reranked_chunks:
        return GuardrailResult(passed=False, layer="sufficiency", reason="No candidates retrieved.")
    top_score = reranked_chunks[0].get("rerank_score", 0.0)
    if top_score < config.SUFFICIENCY_MIN_RERANK_SCORE:
        return GuardrailResult(
            passed=False, layer="sufficiency",
            reason=f"Top rerank score {top_score:.3f} below threshold "
                   f"{config.SUFFICIENCY_MIN_RERANK_SCORE} — evidence too weak.",
        )
    return GuardrailResult(passed=True)


def check_grounding(answer: str, context: str) -> GuardrailResult:
    # Stopwords that inflate the denominator without adding grounding signal
    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "because", "but", "and",
        "or", "if", "while", "about", "up", "it", "its", "this", "that",
        "these", "those", "i", "you", "he", "she", "we", "they", "me", "him",
        "her", "us", "them", "my", "your", "his", "our", "their", "what",
        "which", "who", "whom", "also", "however", "based", "according",
    }
    answer_tokens = set(tokenize(answer)) - _STOPWORDS
    context_tokens = set(tokenize(context)) - _STOPWORDS
    if not answer_tokens:
        return GuardrailResult(passed=False, layer="grounding", reason="Empty answer.")
    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    if overlap < config.GROUNDING_MIN_OVERLAP:
        return GuardrailResult(
            passed=False, layer="grounding",
            reason=f"Word-overlap {overlap:.2f} between answer and context is below "
                   f"threshold {config.GROUNDING_MIN_OVERLAP} — likely hallucination.",
        )
    return GuardrailResult(passed=True)


REFUSAL_MESSAGES = {
    "off_topic": "I can only answer questions about company placement experiences "
                 "(pay, WFH, bench time, culture, interviews) for tracked companies.",
    "safety": "I can't process that request.",
    "sufficiency": "I don't have enough reliable review data to answer that confidently. "
                   "Try asking about a specific tracked company or a narrower topic.",
    "grounding": "I couldn't produce an answer that's fully backed by the retrieved reviews, "
                 "so I'm not going to guess. Try rephrasing the question.",
}
