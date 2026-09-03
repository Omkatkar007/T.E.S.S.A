# The T.E.S.S.A. Development Journey

This is the honest, unfiltered story of how this project actually got built — including the wrong turns, the crashes, and the debugging. It's here because the process taught as much as the final architecture diagram shows, and because a resume line ("built a RAG system") doesn't capture what it actually took to get a hybrid-retrieval system with real guardrails onto a live URL.

If you're a recruiter or interviewer skimming this: the short version is that this project survived a full-stack failure at almost every layer — bad data, a deprecated LLM model, a naive Docker mental model, three different cloud memory-limit crashes, and a broken deploy — and each one got diagnosed and fixed rather than worked around. That process is arguably the more interesting engineering story than the architecture itself.

---

## Phase 1 — Picking the project

The starting point was a simple, honest question: what's an 8/10-difficulty portfolio project for a Data Analyst-track resume that isn't a generic "chat with a PDF" RAG demo? The answer that stuck: **Placement Truth Check** — a system comparing what companies *say* about themselves against what real employees *actually said*, grounded in AmbitionBox and Glassdoor review data, built for placement-drive prep against mass IT-services recruiters (TCS, Infosys, Wipro, Cognizant, Capgemini).

The architecture was specified upfront:
```
Voice/Text → STT → Query Processor → Hybrid Retrieval → Reranking →
Context Builder → LLM → Guardrails → Response
```
with hybrid retrieval (BM25 + dense embeddings), RRF fusion, cross-encoder reranking, and a 4-layer guardrail system (off-topic, safety, sufficiency, grounding) as the pieces that would separate this from a toy project.

## Phase 2 — Building the skeleton

The initial pipeline (BM25 from scratch, RRF fusion, context builder, guardrails, pipeline orchestration) was built and unit-tested in a sandboxed environment first — 24 tests covering every stage, including a full mocked end-to-end smoke test proving the wiring was correct before ever touching a real dataset. Network-dependent pieces (the embedding model, Qdrant, the LLM) were stubbed for testing since the sandbox itself couldn't reach Hugging Face, Groq, or Qdrant Cloud — a limitation that turned out to foreshadow a lot of what came next.

## Phase 3 — Moving to a real machine, and the first real wall: Docker and Qdrant

Once the code moved to a real Windows laptop, the first genuine confusion wasn't about RAG at all — it was "why do I need Docker?" The honest answer took a few tries to land: Qdrant (the vector database) is a separate program that needs to stay running in the background, and Docker is just the easiest way to switch it on without a manual install. That distinction — Python does the embedding math, Docker just keeps the storage box switched on — became the mental model for the rest of the project.

## Phase 4 — The AmbitionBox data was never actually reviews

This was the single biggest, most repeated lesson of the whole project. `ingest.py` reported **0 AmbitionBox documents loaded** — every single row got filtered out. Debugging revealed the real problem: the downloaded "AmbitionBox dataset" wasn't individual reviews at all. It was a company *directory* — one row per company, with columns like `rating`, `review count` (a string like `"16.1k Reviews"`, not review text), `HQ`, `company age`, `employee count`. There was no actual review text anywhere in the file.

This happened **twice** — a second "AmbitionBox" download (`company_data.csv`) had the exact same problem, just with slightly different column names (`Salary Data`, `Interview Data`, `Review Count` — still all stats, no text). It took a systematic Kaggle search to find datasets that actually had real per-review text: `Likes`/`Dislikes` columns with genuine sentences, one row per individual review — starting with a Capgemini-specific dataset, then extending to Mahindra, Tata Motors, and Samsung India Electronics.

**Lesson:** "the file is called reviews.csv" doesn't mean it contains reviews. Always inspect actual columns and sample values before trusting a dataset's name.

## Phase 5 — Adding data without redoing the slow part

Once real per-company review data was found, a naive re-run of `ingest.py` would have wiped and re-embedded the entire existing Glassdoor corpus from scratch — a multi-hour cost already paid once. Instead, a separate `add_ambitionbox_companies.py` script was built: it merges new documents into the *existing* BM25 index and doc lookup, and upserts into Qdrant with `recreate=False`, so adding a new company never means re-processing everything already indexed.

This surfaced a subtle guardrail bug: newly added companies (Mahindra, Tata Motors) were being wrongly refused as "off-topic," because the off-topic guardrail checked queries against a hardcoded `COMPANIES` list in config — which nobody had updated after adding new data. Fixed by keeping that list in sync with what's actually indexed.

## Phase 6 — The Groq model got deprecated mid-project

A working query suddenly started failing with `model_decommissioned` — Groq had retired `llama-3.1-70b-versatile` (and even its successor, `llama-3.3-70b-versatile`) between when the architecture was specified and when it was actually run. Fixed by switching to `openai/gpt-oss-120b`, Groq's then-current recommended general-purpose model. A reminder that "pin a model name and forget about it" doesn't hold for fast-moving LLM APIs.

## Phase 7 — Moving to the cloud, and the real cost of scale

The local Glassdoor file turned out to be a **1.67 million-row global dump**, not a focused Indian-IT-services dataset. Pushing all of it to a free-tier Qdrant Cloud cluster over a home internet connection was projected to take **13+ hours** — and then crashed partway through with a network timeout anyway.

Two fixes, done in sequence:
1. **Filtering**: keep every row for actually-tracked companies, but cap "everything else" at a fixed sample size — cutting the dataset from 1.67M rows to ~245K.
2. **Resumable ingestion**: `qdrant_store.py` gained retry logic with exponential backoff (covering `ReadTimeout`, `ReadError`, connection resets — the retry coverage had to be widened twice, after a second crash type slipped through the first attempt), and `ingest.py` gained a JSON checkpoint file that records the last successfully completed batch, so a crash at 40% never means restarting from 0% again.

This combination turned an unreliable 13-hour, no-recovery process into a resumable one that reliably finished in under an hour.

## Phase 8 — Choosing a frontend and backend shape

An initial Streamlit app (`app.py`) was built and confirmed working end-to-end as a quick, low-effort UI. The project later pivoted to a custom HTML/Tailwind frontend (built with Antigravity + Stitch) talking to a dedicated FastAPI backend (`api_server.py`), exposing a clean `POST /api/ask` REST endpoint — a more "real" architecture, and better suited to a standalone polished UI.

## Phase 9 — Render deployment: five distinct failures, five distinct fixes

Getting from "working locally" to "live on the internet" surfaced a full tour of real deployment pitfalls, each diagnosed from its actual error rather than guessed at:

1. **`pywin32` in requirements.txt** — a Windows-only package that had been frozen into `requirements.txt` directly from a local Windows `.venv`. Render's Linux build servers have no wheel for it at all; the build failed immediately. Fixed by removing it.

2. **A hardcoded Windows file path** — `BASE_DIR` in `api_server.py` was literally `r"D:\Btech 24-28\...\placement-truth-check"`, which obviously doesn't exist on Render's servers. Fixed with `os.path.dirname(os.path.abspath(__file__))` — and this fix had to be reapplied a second time after an initial attempt didn't actually make it into the deployed file.

3. **Wrong port binding** — Render assigns a port via a `$PORT` environment variable, not a fixed 8000. The `uvicorn` start command and the `__main__` fallback both needed to read `$PORT` dynamically.

4. **A numpy/Python version mismatch** — `numpy==2.5.2` had no pre-built wheel for whatever Python version Render defaulted to. Fixed by loosening the pin (`numpy>=1.26,<2.0`) and explicitly pinning Render's Python version via a `.python-version` file, so the build stops depending on whatever Render happens to default to.

5. **Git LFS pointer files with no actual content** — an attempt to use Git LFS for the two large `.pkl` index files (61MB and 55MB) resulted in Render's clone step failing with `Object does not exist on the server (404)` — the real binary content never made it to GitHub's LFS storage, only the small pointer file did. Since both files were comfortably under GitHub's actual 100MB hard limit anyway, the fix was to abandon LFS entirely and commit them as plain files.

## Phase 10 — Three rounds of out-of-memory crashes

Even after a clean deploy, the app was killed by Render's 512MB free-tier memory limit — repeatedly, in three distinct rounds, each requiring a real architectural change rather than a config tweak:

**Round 1**: The original embedding model (`sentence-transformers`, backed by PyTorch) and a local cross-encoder reranker were both loaded into memory simultaneously. PyTorch's runtime alone typically costs 200-300MB before any model weights are even loaded. First fix: switch to ONNX Runtime for a quantized version of the same MiniLM model (no PyTorch dependency at all), and drop the cross-encoder reranker in favor of using the RRF-fused ranking directly.

This fix also surfaced a subtle correctness bug before it shipped: a naive rank-based placeholder score for the dropped reranker would have given the top-ranked candidate a perfect `1.0` score *regardless of actual relevance* — silently disabling the sufficiency guardrail. Caught in review and fixed by carrying the real dense/BM25 retrieval confidence through instead of a synthetic rank number.

**Round 2**: Still over the limit. The real remaining cost turned out to be the ~300,000-document BM25 index and document-text lookup dictionary, held fully in memory — Python's object overhead means a 60MB pickle file can easily balloon to 300MB+ once actually loaded. Fixed with a `cap_per_company.py` script capping every company (including a few disproportionately large global ones — IBM, Amazon, Deloitte — that had swallowed a huge share of the dataset in the original "keep all target company rows" filter) at a fixed number of reviews each.

**Round 3**: Rather than keep squeezing a fully local model stack into 512MB, the project switched to using **Cohere's API** for both embeddings (`embed-english-light-v3.0`) and reranking (`rerank-english-v3.0`) — eliminating locally-loaded ML models from the deployed process entirely. This is the architecture that finally deployed successfully.

## Phase 11 — The last bug: a frontend pointed at nobody's localhost

Even after a successful deploy, the frontend's JavaScript had `BACKEND_URL` hardcoded to `http://localhost:8000/api/ask` — meaning the *live* deployed site was fully broken for any real visitor, since their browser would try (and fail) to reach `localhost` on their own machine. Caught during a pre-launch review and fixed by switching to a relative path (`/api/ask`), which works correctly whether the frontend is being served locally or from Render — since the frontend and API are served from the same origin either way.

---

## What this journey actually demonstrates

Beyond the finished architecture, the parts of this process worth highlighting in an interview:
- **Data validation discipline** — never trusting a filename, and inspecting actual columns/values before building on top of a dataset (twice).
- **Designing for interruption** — a checkpoint-based resumable ingestion pipeline, built after directly experiencing what a 13-hour uninterruptible job costs when it fails at 40%.
- **Real memory-constrained systems design** — diagnosing *which* component was actually responsible for a memory ceiling (PyTorch runtime vs. cross-encoder weights vs. in-memory corpus size) rather than guessing, across three separate rounds.
- **Deployment debugging from first principles** — five distinct, unrelated Render failures, each traced to its actual root cause (a Windows-only package, a hardcoded path, a port assumption, a wheel-availability mismatch, a broken LFS upload) rather than trial-and-error.
- **Catching a correctness bug before it shipped** — recognizing that a "quick fix" (rank-based placeholder scores) would have silently disabled a safety guardrail, and fixing the underlying signal instead of the symptom.
- **A final pre-launch review catching a bug that would have made the live demo appear completely broken** to anyone but the developer testing locally.
