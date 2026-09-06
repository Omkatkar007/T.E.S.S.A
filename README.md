<div align="center">

# 🔍 T.E.S.S.A.
### **Truth Extraction & Statement Scrutiny Assistant**

*An AI-powered RAG chatbot for internal HR teams — powered by real employee data.*

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLM-orange?style=for-the-badge)
![Qdrant](https://img.shields.io/badge/Qdrant-In--Memory-purple?style=for-the-badge)

</div>

---

## 📖 What is T.E.S.S.A.?

**T.E.S.S.A.** is a **B2B Internal HR Intelligence Tool** — an intelligent Retrieval-Augmented Generation (RAG) system that lets HR teams upload their own internal data (exit interviews, employee engagement surveys, Slack feedback forms, etc.) and instantly query it using natural language.

Instead of relying on scraping public data or maintaining a static company database, T.E.S.S.A. is **data-agnostic** — it works with *your* data. HR departments simply upload a CSV, and T.E.S.S.A. embeds it in-memory within seconds. No databases to configure. No servers to provision.

> *"What is the main reason junior developers in the Bangalore office are quitting?"*  
> T.E.S.S.A. scans your internal exit interviews and tells you the truth — grounded in what your employees **actually said**.

---

## 🧩 Name Breakdown

| Letter | Word | Meaning |
|--------|------|---------|
| **T** | Truth | Retrieves verified answers from real internal employee data |
| **E** | Extraction | Extracts relevant information using hybrid search (dense + lexical) |
| **S** | Statement | Generates clear, structured answers grounded in your data |
| **S** | Scrutiny | Scrutinizes every response through 4 guardrail layers |
| **A** | Assistant | Provides a conversational, professional HR chat interface |

---

## 💡 The Problem We Solve

HR teams sit on a goldmine of unstructured data — thousands of exit interview responses, engagement survey free-text fields, and one-on-one feedback transcripts — but have **no easy way to query it**.

Traditional approaches require:
- ❌ Manual reading and tagging of hundreds of responses
- ❌ Building custom BI dashboards that only show pre-defined metrics
- ❌ Hiring data analysts for one-off questions
- ❌ Months of database setup and maintenance

**T.E.S.S.A. solves this in under 30 seconds:**
1. Upload your CSV
2. T.E.S.S.A. embeds it in-memory instantly
3. Ask any question in plain English
4. Get a grounded, cited answer — immediately

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📂 **Zero-Config Ingestion** | Upload any CSV — T.E.S.S.A. automatically sniffs text columns and embeds them in-memory in seconds |
| 🔀 **Hybrid Retrieval** | Combines dense vector search (ONNX MiniLM) with lexical BM25 search for superior recall |
| 🔗 **Reciprocal Rank Fusion** | Merges dense + lexical results intelligently without score normalization headaches |
| 🛡️ **4-Layer Guardrails** | Prompt injection blocking, retrieval sufficiency checks, post-generation grounding verification |
| ⚡ **Fully In-Memory** | No Docker, no cloud databases, no ingestion scripts — everything runs inside Streamlit's process |
| 🎨 **Premium Dark UI** | Animated glassmorphism interface with source pills, verified response badges, and suggestion cards |
| 🔊 **Voice Input** | Optional STT support via Sarvam AI for hands-free HR queries |

---

## 🏗️ Architecture

```
                    ┌──────────────────────────────────────┐
                    │          CSV Upload (HR Data)         │
                    └─────────────────┬────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────┐
                    │      In-Memory Embedding Engine       │
                    │   ONNX MiniLM (384-dim, no PyTorch)  │
                    └────────────┬─────────────┬───────────┘
                                 │             │
                    ┌────────────▼──────┐  ┌───▼────────────┐
                    │  Qdrant In-Memory │  │  BM25 Index    │
                    │  (Dense Search)   │  │ (Lexical Search)│
                    └────────────┬──────┘  └───┬────────────┘
                                 │             │
                    ┌────────────▼─────────────▼───────────┐
                    │     Reciprocal Rank Fusion (RRF)      │
                    └─────────────────┬────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────┐
                    │      Cohere Cross-Encoder Reranker    │
                    └─────────────────┬────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────┐
                    │           4-Layer Guardrails          │
                    │  L1: Safety  |  L2: Sufficiency       │
                    │  L3: Grounding  |  L4: Prompt Guard   │
                    └─────────────────┬────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────┐
                    │         Groq LLM Generation           │
                    │    (Fast, grounded, cited answer)     │
                    └──────────────────────────────────────┘
```

---

## 🛡️ 4-Layer Guardrail System

Every query passes through four guardrail layers before and after generation:

| Layer | Name | When Applied | Purpose |
|-------|------|-------------|---------|
| **1** | Safety Guard | Pre-retrieval | Blocks prompt injection and jailbreak attempts |
| **2** | Sufficiency Guard | Post-retrieval | Rejects answers if retrieved evidence is too weak |
| **3** | Grounding Guard | Post-generation | Ensures the answer overlaps sufficiently with context |
| **4** | Refusal Guard | Always | Returns a calibrated refusal message instead of hallucinating |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A [Groq API Key](https://console.groq.com) (free tier)
- Optionally: A [Cohere API Key](https://dashboard.cohere.com) for reranking (free tier)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Omkatkar007/T.E.S.S.A.git
cd T.E.S.S.A/placement-truth-check

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your environment
cp .env.example .env
# Edit .env and add your API keys
```

### Environment Variables

Create a `.env` file in the project root with the following:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional — enables smarter reranking
COHERE_API_KEY=your_cohere_api_key_here

# Optional — enables voice input
SARVAM_API_KEY=your_sarvam_api_key_here
```

### Run the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser. You'll see the T.E.S.S.A. upload screen, ready to accept your HR data.

---

## 📊 Usage

### Step 1 — Prepare Your CSV

T.E.S.S.A. accepts **any CSV file**. There's no required schema. The more descriptive your column names, the better the retrieval quality.

Example format:

```csv
Department,Role,Feedback,Exit_Reason,Date
Engineering,Senior Developer,"Compensation wasn't competitive...",Compensation,2026-07-15
Marketing,Content Strategist,"Culture is great but growth is slow...",Growth,2026-07-20
```

You can also use: exit interview exports, survey monkey CSVs, Glassdoor private exports, engagement survey responses — anything you can export to CSV.

### Step 2 — Upload and Index

- Click **"Browse files"** in the Streamlit sidebar
- Select your CSV
- T.E.S.S.A. embeds all rows in-memory within seconds (progress bar shown)

### Step 3 — Ask Natural Language Questions

Use the chat input or click a suggestion card:

| Example Query | What T.E.S.S.A. does |
|---------------|---------------------|
| *"Why are junior developers leaving?"* | Scans exit interviews, surfaces top reasons |
| *"What is feedback on the remote work policy?"* | Retrieves WFH sentiment across all departments |
| *"Summarize compensation concerns from Q3 2026"* | Focuses on salary/hike feedback with date filtering |
| *"Which department has the most disengaged employees?"* | Cross-department engagement analysis |

---

## 🗂️ Project Structure

```
placement-truth-check/
├── app.py                    # Streamlit UI — upload, embed, chat
├── hr_feedback.csv           # Sample HR dataset for testing
├── requirements.txt
├── .env                      # API keys (not committed)
│
└── src/
    ├── pipeline.py           # Core RAG pipeline with TessaPipeline class
    ├── embeddings.py         # ONNX MiniLM embedding engine
    ├── qdrant_store.py       # In-memory Qdrant vector store
    ├── bm25.py               # BM25 lexical index
    ├── fusion.py             # Reciprocal Rank Fusion
    ├── reranker.py           # Cohere cross-encoder reranker
    ├── context_builder.py    # Context window assembly
    ├── guardrails.py         # 4-layer guardrail pipeline
    ├── llm.py                # Groq LLM generation
    ├── stt.py                # Sarvam AI voice-to-text
    └── config.py             # Central configuration
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **UI** | Streamlit with custom glassmorphism CSS |
| **Embeddings** | ONNX MiniLM-L6-v2 (384-dim, no PyTorch needed) |
| **Vector Store** | Qdrant (in-memory, no Docker required) |
| **Lexical Search** | BM25 (custom implementation) |
| **Fusion** | Reciprocal Rank Fusion (RRF) |
| **Reranker** | Cohere `rerank-english-v3.0` |
| **LLM** | Groq (fast inference, grounded generation) |
| **Voice STT** | Sarvam AI `saaras:v3` |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📋 Roadmap

- [ ] Multi-file upload support (upload multiple CSVs, query across all)
- [ ] Department filter (e.g., "only show Engineering feedback")
- [ ] Date range filtering in queries
- [ ] Export answers as PDF reports
- [ ] Streamlit Cloud / Hugging Face Spaces one-click deploy

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

---

<div align="center">

**Built with ❤️ for HR teams who deserve the truth.**

*T.E.S.S.A. — because your employees already told you why they're leaving. You just couldn't read all 10,000 responses.*

</div>
