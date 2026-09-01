"""
Central configuration for the Placement Truth Check RAG pipeline.
All tunables live here so the rest of the codebase never hardcodes constants.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # --- Embeddings (Cohere API) ---
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    COHERE_EMBED_MODEL: str = "embed-english-light-v3.0"   # 384-dim, free tier
    COHERE_RERANK_MODEL: str = "rerank-english-v3.0"        # free tier
    EMBEDDING_DIM: int = 384

    # --- Qdrant (dense retrieval) ---
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = "company_reviews"

    # --- BM25 (lexical retrieval) ---
    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    # --- Retrieval / fusion ---
    DENSE_TOP_K: int = 25
    LEXICAL_TOP_K: int = 25
    RRF_K: int = 60          # standard RRF damping constant
    FUSED_TOP_N: int = 15    # candidates handed to reranker

    # --- Reranking ---
    RERANK_TOP_N: int = 5    # candidates kept after reranking

    # --- Context builder ---
    MAX_CONTEXT_TOKENS: int = 1500
    MIN_CHUNKS: int = 3
    MAX_CHUNKS: int = 5

    # --- Generation (Groq) ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_MAX_OUTPUT_TOKENS: int = 256
    GROQ_TEMPERATURE: float = 0.1

    # --- STT (Sarvam) ---
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
    SARVAM_MODEL: str = "saaras:v3"

    # --- Guardrails ---
    GROUNDING_MIN_OVERLAP: float = 0.15       # min word-overlap ratio answer<->context
    SUFFICIENCY_MIN_RERANK_SCORE: float = 0.2  # below this, treat retrieval as "insufficient evidence"
    OFF_TOPIC_KEYWORDS_ALLOW: tuple = (
        "salary", "pay", "ctc", "wfh", "remote", "bench", "notice period",
        "work life", "culture", "interview", "onboarding", "appraisal",
        "hike", "job security", "layoff", "review", "rating", "manager",
        "growth", "promotion", "hr", "policy",
    )
    COMPANIES: tuple = ( "tcs", "infosys", "wipro", "cognizant", "capgemini", "accenture", "hcl", "tech mahindra", "ibm", "amazon", "deloitte", "capgemini employee", "mahindra", "tata motors", "samsung india electronics", "samsung",)

    DATA_DIR: str = "data"
    LOG_LEVEL: str = "INFO"


config = Config()
