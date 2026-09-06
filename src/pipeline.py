"""
End-to-end pipeline:
Voice/Text -> STT -> Query Processor -> Hybrid Retrieval -> Fusion ->
Reranking -> Context Builder -> LLM -> Guardrails -> Response
"""
from __future__ import annotations
from dataclasses import dataclass, field
import uuid

import pandas as pd

from . import guardrails as gr
from .bm25 import BM25Index
from .config import config
from .context_builder import build_context
from .embeddings import get_embedder
from .fusion import reciprocal_rank_fusion
from .llm import GroqGenerator
from .qdrant_store import QdrantStore
from .reranker import get_reranker
from .stt import transcribe


@dataclass
class PipelineResponse:
    answer: str
    grounded: bool
    refusal_layer: str | None = None
    sources: list[dict] = field(default_factory=list)


class TessaPipeline:
    def __init__(self):
        """
        Initializes an empty pipeline. Data must be loaded via ingest_dataframe().
        """
        self.bm25 = None
        self.doc_lookup = {}
        self.embedder = get_embedder()
        self.qdrant = QdrantStore()
        self.reranker = get_reranker()
        self.llm = GroqGenerator()

    def ingest_dataframe(self, df: pd.DataFrame, progress_callback=None):
        """
        Reads a dataframe, extracts text, embeds it, and populates the in-memory stores.
        """
        docs = []
        for _, row in df.iterrows():
            text_parts = []
            payload = {}
            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    text_parts.append(f"{col}: {val}")
                    payload[str(col)] = str(val)
            
            text = " | ".join(text_parts)
            if len(text) > 10:
                docs.append({
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "payload": payload
                })
        
        if not docs:
            return

        self.qdrant.ensure_collection(recreate=True)

        batch_size = 64
        total_batches = (len(docs) + batch_size - 1) // batch_size
        
        for batch_idx, i in enumerate(range(0, len(docs), batch_size)):
            batch = docs[i:i + batch_size]
            texts = [d["text"] for d in batch]
            vecs = self.embedder.encode(texts)
            ids = [d["id"] for d in batch]
            payloads = [d["payload"] | {"text": d["text"]} for d in batch]
            self.qdrant.upsert(ids, vecs, payloads)
            
            if progress_callback:
                progress_callback(min(1.0, (batch_idx + 1) / total_batches))

        self.bm25 = BM25Index()
        self.bm25.build([(d["id"], d["text"]) for d in docs])

        self.doc_lookup = {d["id"]: {"text": d["text"], "payload": d["payload"]} for d in docs}

    def answer(self, query: str | None = None, audio_path: str | None = None) -> PipelineResponse:
        if not self.bm25:
            return PipelineResponse(answer="Please upload a CSV file first.", grounded=False, refusal_layer="system")

        # --- Stage 0: STT (optional) ---
        if audio_path is not None:
            query = transcribe(audio_path)
        if not query:
            return PipelineResponse(answer="No query provided.", grounded=False, refusal_layer="input")

        # --- Layer 1 & 2: off-topic + safety guardrails (pre-retrieval, cheap) ---
        for check in (gr.check_off_topic, gr.check_safety):
            result = check(query)
            if not result.passed:
                return PipelineResponse(
                    answer=gr.REFUSAL_MESSAGES.get(result.layer, "Blocked by guardrail."),
                    grounded=False,
                    refusal_layer=result.layer,
                )

        # --- Stage 1: Hybrid retrieval ---
        query_vec = self.embedder.encode_one(query)
        dense_hits = self.qdrant.search(query_vec, top_k=config.DENSE_TOP_K)
        dense_results = [(h["id"], h["score"]) for h in dense_hits]
        dense_score_map = {h["id"]: h["score"] for h in dense_hits}

        lexical_results = self.bm25.search(query, top_k=config.LEXICAL_TOP_K)
        lexical_score_map = dict(lexical_results)
        max_lexical_score = max((s for _, s in lexical_results), default=0.0) or 1.0

        # --- Stage 2: RRF fusion ---
        fused = reciprocal_rank_fusion(dense_results, lexical_results, top_n=config.FUSED_TOP_N)

        candidates = []
        for doc_id, _fused_score in fused:
            doc = self.doc_lookup.get(doc_id)
            if doc:
                dense_conf = dense_score_map.get(doc_id, 0.0)
                lexical_conf = lexical_score_map.get(doc_id, 0.0) / max_lexical_score
                retrieval_confidence = max(dense_conf, lexical_conf)
                candidates.append({
                    "id": doc_id,
                    "text": doc["text"],
                    "payload": doc["payload"],
                    "retrieval_confidence": retrieval_confidence,
                })

        # --- Stage 3: Reranking ---
        reranked = self.reranker.rerank(query, candidates, top_n=config.RERANK_TOP_N)

        # --- Layer 3: sufficiency guardrail ---
        result = gr.check_sufficiency(reranked)
        if not result.passed:
            return PipelineResponse(
                answer=gr.REFUSAL_MESSAGES.get(result.layer, "Insufficient evidence."), grounded=False, refusal_layer=result.layer,
            )

        # --- Stage 4: Context building ---
        context, chunks_used = build_context(reranked)

        # --- Stage 5: Generation ---
        raw_answer = self.llm.generate(query, context)

        # --- Layer 4: grounding guardrail (post-generation) ---
        result = gr.check_grounding(raw_answer, context)
        if not result.passed:
            return PipelineResponse(
                answer=gr.REFUSAL_MESSAGES.get(result.layer, "Ungrounded answer."), grounded=False, refusal_layer=result.layer,
                sources=chunks_used,
            )

        return PipelineResponse(answer=raw_answer, grounded=True, sources=chunks_used)
