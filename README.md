# T.E.S.S.A. (Truth Extraction & Statement Scrutiny Assistant)

A RAG system that answers company-comparison questions ("TCS vs Infosys — real pay,
WFH, bench time") grounded in **real employee reviews** (AmbitionBox + Glassdoor),
not marketing claims — with hallucination guardrails that refuse to answer when
the evidence is weak instead of making things up.

## Architecture

```
Voice/Text → STT → Query Processor → Hybrid Retrieval → Fusion → Reranking
           → Context Builder → LLM → Guardrails → Response
```

| Stage | Implementation | File |
|---|---|---|
| STT | Sarvam `saaras:v3` (optional, voice input only) | `src/stt.py` |
| Embedding | Local MiniLM, singleton, 384-dim | `src/embeddings.py` |
| Dense retrieval | Qdrant ANN, cosine similarity | `src/qdrant_store.py` |
| Lexical retrieval | In-memory Okapi BM25 (from scratch) | `src/bm25.py` |
| Fusion | Reciprocal Rank Fusion, k=60 | `src/fusion.py` |
| Reranking | Local cross-encoder (`ms-marco-MiniLM-L-6-v2`) | `src/reranker.py` |
| Context builder | Token-budgeted, ≤1500 tokens, 3–5 chunks | `src/context_builder.py` |
| Generation | Groq `llama-3.1-70b-versatile`, ≤256 output tokens | `src/llm.py` |
| Guardrails | 4-layer: off-topic → safety → sufficiency → grounding | `src/guardrails.py` |
| Orchestration | Wires all stages together | `src/pipeline.py` |

## Setup

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # fill in GROQ_API_KEY, SARVAM_API_KEY (optional)
```

Start Qdrant locally (Docker):
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

## Data

Download from Kaggle and drop the CSVs into `data/raw/`:
- **AmbitionBox Indian companies dataset** → `data/raw/ambitionbox_reviews.csv`
- **Glassdoor reviews dataset** → `data/raw/glassdoor_reviews.csv`

Column names vary between Kaggle uploads — check `scripts/ingest.py`'s
`AMBITIONBOX_COLUMNS` / `GLASSDOOR_COLUMNS` dicts and adjust them to match
whatever headers your specific CSV actually has (`df.columns` will tell you).

## Run

```bash
# Phase 1+2: clean, embed, index into Qdrant + BM25
python scripts/ingest.py

# Phase 3: ask questions
python scripts/query.py
```

## Tests

The retrieval-logic modules (BM25, RRF fusion, guardrails, context builder) are
unit-tested without needing live services:

```bash
pytest tests/ -v
```

`qdrant_store.py`, `llm.py`, and `stt.py` need live services (Qdrant, Groq API key,
Sarvam API key) and aren't unit-tested here — test those end-to-end via
`scripts/query.py` once `ingest.py` has run.

`tests/test_pipeline_smoke.py` covers the full orchestration
(`PlacementTruthCheckPipeline.answer()`) with those three network-dependent
pieces mocked out, so you get one test asserting a grounded answer flows
through every stage, plus one per refusal path (off-topic, safety, grounding)
— all runnable with no external services.

**Note on `count_tokens`:** `context_builder.py` uses a char/word-based token
approximation rather than `tiktoken`, since `tiktoken`'s `cl100k_base` encoding
downloads its BPE file from `openaipublic.blob.core.windows.net` on first use,
which isn't reachable in every environment. Swap back to real `tiktoken` if
you want exact counts and have that domain reachable — nothing else in the
pipeline depends on which one you use.

## Why this is more than a toy RAG demo

- **Hybrid retrieval, not just cosine similarity.** BM25 catches exact terms
  (company names, numbers) that dense embeddings can blur; RRF fusion combines
  both without needing to normalize incompatible score scales.
- **Local cross-encoder reranking** — no extra API cost/latency, and materially
  more accurate than bi-encoder similarity alone since it jointly encodes
  (query, passage) pairs.
- **Grounding guardrails are the actual hard part.** Layer 4 checks word-overlap
  between the generated answer and the retrieved context, and refuses to answer
  if the LLM appears to have drifted into unsupported claims (e.g. inventing a
  specific salary figure that isn't in any review).
- **Sufficiency guardrail** stops the system from confidently answering when
  retrieval genuinely found nothing relevant, instead of forcing the LLM to
  hallucinate an answer from thin context.

## Suggested resume line

> Built a hybrid-retrieval RAG system (BM25 + dense embeddings + RRF fusion,
> local cross-encoder reranking) over real employee reviews (AmbitionBox/Glassdoor)
> to answer company-comparison queries, with a 4-layer guardrail pipeline preventing
> hallucinated claims; served via Groq Llama-3.1-70B with token-budgeted context
> construction.

## Possible extensions (9-10/10 stretch)

- Hand-label 20–30 Q&A pairs and report retrieval precision/recall as a metric.
- Wire up the STT layer end-to-end for voice queries.
- Add a small eval harness that flags grounding-guardrail trigger rate over time.
