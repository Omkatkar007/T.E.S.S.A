"""
Builds the final context block handed to the LLM: picks 3-5 reranked
chunks that fit inside a hard token budget, tagged with source metadata
so guardrails/citations can point back to a specific company + review.
"""
from __future__ import annotations
import re

from .config import config

# tiktoken's cl100k_base encoding requires downloading its BPE merge file from
# openaipublic.blob.core.windows.net at first use, which isn't reachable from
# every deployment environment (this sandbox included). We use a fast,
# dependency-free approximation instead: ~4 chars/token for English prose is
# tiktoken's own documented rule of thumb, and it's accurate enough for a
# token *budget* (we're not counting exact billing tokens, just making sure
# we don't blow past the context window).
#
# If you have real network access to Hugging Face / OpenAI's blob storage,
# swap this back to `tiktoken.get_encoding("cl100k_base")` for exact counts —
# the rest of the pipeline doesn't care which one you use.
_WORD_RE = re.compile(r"\S+")


def count_tokens(text: str) -> int:
    """Approximate token count: max(word-count, char-count/4)."""
    if not text:
        return 0
    words = len(_WORD_RE.findall(text))
    chars = len(text)
    return max(words, chars // 4)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens tokens (char-based cut)."""
    approx_chars = max_tokens * 4
    if len(text) <= approx_chars:
        return text
    return text[:approx_chars].rsplit(" ", 1)[0]


def build_context(reranked_chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    reranked_chunks: output of Reranker.rerank(), sorted best-first, each
    with {"id", "text", "payload": {"company": ..., "source": ...}, "rerank_score"}.

    Returns (context_string, chunks_used) — chunks_used is capped to
    MIN..MAX_CHUNKS and respects MAX_CONTEXT_TOKENS.
    """
    chunks_used = []
    total_tokens = 0
    blocks = []

    for chunk in reranked_chunks:
        if len(chunks_used) >= config.MAX_CHUNKS:
            break
        company = chunk.get("payload", {}).get("company", "Unknown")
        source = chunk.get("payload", {}).get("source", "review")
        block = f"[{company} | {source}] {chunk['text']}"
        block_tokens = count_tokens(block)

        if total_tokens + block_tokens > config.MAX_CONTEXT_TOKENS:
            if len(chunks_used) >= config.MIN_CHUNKS:
                break
            # still under MIN_CHUNKS — truncate this block to fit rather than skip it
            remaining = config.MAX_CONTEXT_TOKENS - total_tokens
            if remaining <= 50:
                break
            block = _truncate_to_tokens(block, remaining)
            block_tokens = count_tokens(block)

        blocks.append(block)
        chunks_used.append(chunk)
        total_tokens += block_tokens

    context_string = "\n\n".join(blocks)
    return context_string, chunks_used
