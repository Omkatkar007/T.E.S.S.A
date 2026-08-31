"""
Streamlit web app for Placement Truth Check.

Local run:
    streamlit run app.py

Deployed on Streamlit Community Cloud:
    Set these as "Secrets" in the app's settings (NOT in this file, NOT
    committed to git):
        GROQ_API_KEY = "gsk_..."
        QDRANT_URL = "https://xxxx.cloud.qdrant.io"
        QDRANT_API_KEY = "..."
"""
import os
import pickle
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

# --- Wire up Streamlit's secrets to the same env vars src/config.py reads ---
# (config.py reads from os.environ; Streamlit Cloud only gives us st.secrets,
# so we copy them across before importing anything that touches config.)
# Locally, there's no secrets.toml file at all (you're using .env instead),
# and st.secrets raises an error just from being touched in that case — so
# we skip this entirely if no secrets file exists.
try:
    for key in ("GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "SARVAM_API_KEY"):
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = st.secrets[key]
except st.errors.StreamlitSecretNotFoundError:
    pass  # no secrets.toml (expected locally) — config.py will read .env instead

from src.config import config
from src.pipeline import PlacementTruthCheckPipeline

PROCESSED_DIR = os.path.join(config.DATA_DIR, "processed")

st.set_page_config(page_title="Placement Truth Check", page_icon="🔍", layout="centered")


@st.cache_resource(show_spinner=False)
def load_pipeline() -> PlacementTruthCheckPipeline | None:
    bm25_path = os.path.join(PROCESSED_DIR, "bm25_index.pkl")
    lookup_path = os.path.join(PROCESSED_DIR, "doc_lookup.pkl")

    if not (os.path.exists(bm25_path) and os.path.exists(lookup_path)):
        return None

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(lookup_path, "rb") as f:
        doc_lookup = pickle.load(f)

    return PlacementTruthCheckPipeline(bm25_index=bm25, doc_lookup=doc_lookup)


st.title("🔍 Placement Truth Check")
st.caption(
    "Ask what real employees actually said — not the marketing pitch. "
    "Answers are grounded in AmbitionBox/Glassdoor reviews, or refused if "
    "there isn't enough evidence."
)

missing_secrets = [k for k in ("GROQ_API_KEY", "QDRANT_URL") if not os.environ.get(k)]
if missing_secrets:
    st.error(
        f"Missing required configuration: {', '.join(missing_secrets)}. "
        "If you're the developer, set these in .env locally or in "
        "Streamlit Cloud's 'Secrets' settings."
    )
    st.stop()

with st.spinner("Loading models and indexes (only happens once)..."):
    pipeline = load_pipeline()

if pipeline is None:
    st.error(
        "No index found. Run `python scripts/ingest.py` (and optionally "
        "`python scripts/add_ambitionbox_companies.py`) first, and make "
        "sure data/processed/bm25_index.pkl and doc_lookup.pkl are "
        "committed to your repo."
    )
    st.stop()

query = st.text_input(
    "Your question",
    placeholder="e.g. Is Capgemini good for freshers?",
)
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and query.strip():
    with st.spinner("Searching reviews and generating an answer..."):
        response = pipeline.answer(query=query.strip())

    st.markdown("### Answer")
    st.write(response.answer)

    if response.grounded and response.sources:
        with st.expander(f"Sources ({len(response.sources)})"):
            for s in response.sources:
                payload = s.get("payload", {})
                company = payload.get("company", "unknown")
                source = payload.get("source", "unknown")
                score = s.get("rerank_score", 0)
                st.markdown(f"- **{company}** · _{source}_ · score={score:.3f}")
    elif response.refusal_layer:
        st.caption(f"(No grounded answer — refused at layer: {response.refusal_layer})")

elif ask_clicked:
    st.warning("Type a question first.")

st.divider()
st.caption(
    "Built with hybrid retrieval (BM25 + dense embeddings + RRF fusion), "
    "cross-encoder reranking, and 4-layer grounding guardrails. "
    "Data: AmbitionBox + Glassdoor employee reviews."
)
