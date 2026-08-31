"""
Groq generation client. Kept deliberately dumb (no retries/streaming
logic beyond basics) so the guardrails module stays the source of
truth for answer quality, not the LLM call itself.
"""
from __future__ import annotations
from groq import Groq

from .config import config

_SYSTEM_PROMPT = """You are Placement Truth Check, an assistant that answers questions about \
Indian IT companies (TCS, Infosys, Wipro, Cognizant, Capgemini, etc.) using ONLY the employee \
review excerpts given to you in the context below.

Rules:
- Answer ONLY using facts present in the provided context.
- If the context does not contain enough information to answer, say so explicitly \
  instead of guessing or using general knowledge.
- Do not invent numbers (salary, bench duration, ratings) that are not in the context.
- Keep the answer concise and cite which company each claim is about.
"""


class GroqGenerator:
    def __init__(self):
        self.client = Groq(api_key=config.GROQ_API_KEY)

    def generate(self, query: str, context: str) -> str:
        user_prompt = f"Context (real employee reviews):\n{context}\n\nQuestion: {query}"
        response = self.client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=config.GROQ_MAX_OUTPUT_TOKENS,
            temperature=config.GROQ_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
