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
    q = query.lower()
    has_company = any(c in q for c in config.COMPANIES)
    has_topic_word = any(kw in q for kw in config.OFF_TOPIC_KEYWORDS_ALLOW)
    if not (has_company or has_topic_word):
        return GuardrailResult(
            passed=False, layer="off_topic",
            reason="Query doesn't reference a tracked company or a placement/work-life topic.",
        )
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
    answer_tokens = set(tokenize(answer))
    context_tokens = set(tokenize(context))
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
