"""
End-to-end pipeline:
Voice/Text -> STT -> Query Processor -> Hybrid Retrieval -> Fusion ->
Reranking -> Context Builder -> LLM -> Guardrails -> Response
"""
from __future__ import annotations
from dataclasses import dataclass, field

from . import guardrails as gr
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


class PlacementTruthCheckPipeline:
    def __init__(self, bm25_index, doc_lookup: dict[str, dict]):
        """
        bm25_index: a built BM25Index (see ingest.py)
        doc_lookup: {doc_id: {"text": ..., "payload": {...}}} for fetching
                    text back after BM25/Qdrant return only ids.
        """
        self.bm25 = bm25_index
        self.doc_lookup = doc_lookup
        self.embedder = get_embedder()
        self.qdrant = QdrantStore()
        self.reranker = get_reranker()
        self.llm = GroqGenerator()

    def answer(self, query: str | None = None, audio_path: str | None = None) -> PipelineResponse:
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
                    answer=gr.REFUSAL_MESSAGES[result.layer],
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
                # Without a cross-encoder, we need a genuinely meaningful
                # 0-1-ish confidence score for the sufficiency guardrail —
                # NOT just a rank position (which would always give the top
                # result a perfect score regardless of actual relevance).
                # Dense cosine similarity is naturally ~0-1 for normalized
                # vectors; BM25's raw score is normalized against the best
                # score in this result set to put it on a comparable scale.
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
                answer=gr.REFUSAL_MESSAGES[result.layer], grounded=False, refusal_layer=result.layer,
            )

        # --- Stage 4: Context building ---
        context, chunks_used = build_context(reranked)

        # --- Stage 5: Generation ---
        raw_answer = self.llm.generate(query, context)

        # --- Layer 4: grounding guardrail (post-generation) ---
        result = gr.check_grounding(raw_answer, context)
        if not result.passed:
            return PipelineResponse(
                answer=gr.REFUSAL_MESSAGES[result.layer], grounded=False, refusal_layer=result.layer,
                sources=chunks_used,
            )

        return PipelineResponse(answer=raw_answer, grounded=True, sources=chunks_used)
