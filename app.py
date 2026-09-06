"""
Streamlit web app for T.E.S.S.A. HR Insights.
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

try:
    for key in ("GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "SARVAM_API_KEY"):
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = st.secrets[key]
except st.errors.StreamlitSecretNotFoundError:
    pass

from src.pipeline import TessaPipeline

st.set_page_config(page_title="T.E.S.S.A. HR", page_icon="🔍", layout="centered")

st.markdown("""
<style>
/* Background Animation */
.stApp {
    background-color: #050510;
    color: #e4e1ed;
}
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: -1;
    background-image:
        radial-gradient(at 40% 20%, hsla(243,100%,74%,0.15) 0px, transparent 50%),
        radial-gradient(at 80% 0%, hsla(271,81%,65%,0.15) 0px, transparent 50%),
        radial-gradient(at 0% 50%, hsla(189,94%,43%,0.15) 0px, transparent 50%);
    animation: mesh-morph 20s infinite alternate linear;
    filter: blur(80px);
}
@keyframes mesh-morph {
    0% { background-position: 0% 0%; }
    100% { background-position: 100% 100%; }
}

/* Glassmorphism Panels */
.glass-panel {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(32px);
    -webkit-backdrop-filter: blur(32px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Typography & Colors */
.gradient-text-animate {
    background: linear-gradient(90deg, #6366f1, #a855f7, #06b6d4, #6366f1);
    background-size: 300% auto;
    color: transparent;
    -webkit-background-clip: text;
    background-clip: text;
    animation: gradient-shimmer 4s linear infinite;
    font-weight: 800;
    font-size: 3rem;
    text-align: center;
}
@keyframes gradient-shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 300% center; }
}

header[data-testid="stHeader"] {
    background: transparent;
}

div[data-testid="stHorizontalBlock"] button {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    color: #e4e1ed;
    transition: all 0.3s ease;
    height: 100px;
    white-space: normal;
}
div[data-testid="stHorizontalBlock"] button:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 10px 30px -10px rgba(168, 85, 247, 0.3);
    border-color: rgba(168, 85, 247, 0.5);
    color: white;
}

div[data-testid="stChatInput"] {
    background: transparent !important;
}
div[data-testid="stChatInput"] textarea {
    background: #0d0d15 !important;
    border-radius: 20px;
    color: white !important;
    border: 2px solid transparent !important;
}

.verified-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(16, 185, 129, 0.1); color: #34d399;
    padding: 4px 12px; border-radius: 9999px;
    border: 1px solid rgba(16, 185, 129, 0.2);
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    margin-top: 10px;
}
.verified-dot { width: 8px; height: 8px; border-radius: 50%; background: #34d399; }

.guardrail-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(245, 158, 11, 0.1); color: #fbbf24;
    padding: 4px 12px; border-radius: 9999px;
    border: 1px solid rgba(245, 158, 11, 0.2);
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    margin-top: 10px;
}
.guardrail-dot { width: 8px; height: 8px; border-radius: 50%; background: #fbbf24; }

.source-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255, 255, 255, 0.05); color: #e4e1ed;
    padding: 4px 12px; border-radius: 9999px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 11px; margin-right: 6px; margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)


if "pipeline" not in st.session_state:
    st.session_state.pipeline = TessaPipeline()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


if not st.session_state.data_loaded:
    st.markdown("<div class='gradient-text-animate'>T.E.S.S.A. HR</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #c7c4d7; font-size: 1.1rem;'>"
                "Truth Extraction & Statement Scrutiny Assistant for HR<br>"
                "<span style='font-size: 0.8rem; opacity: 0.8;'>Upload your internal feedback or exit interview CSV to begin.</span></p>", 
                unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Feedback CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded CSV with {len(df)} rows.")
            
            progress_bar = st.progress(0, text="Embedding data into memory...")
            def update_progress(pct):
                progress_bar.progress(pct, text=f"Embedding data into memory... {int(pct*100)}%")
                
            with st.spinner("Processing..."):
                st.session_state.pipeline.ingest_dataframe(df, progress_callback=update_progress)
            
            progress_bar.empty()
            st.session_state.data_loaded = True
            st.rerun()
            
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            
else:
    # --- HERO & SUGGESTIONS (Only show if no messages) ---
    if not st.session_state.messages:
        st.markdown("<div class='gradient-text-animate'>T.E.S.S.A. HR</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #c7c4d7; font-size: 1.1rem;'>"
                    "Dataset Loaded Successfully.<br>"
                    "<span style='font-size: 0.8rem; opacity: 0.8;'>Ask questions about your internal feedback data.</span></p>", 
                    unsafe_allow_html=True)
        
        st.write("")
        col1, col2 = st.columns(2)
        
        def set_query(q):
            st.session_state.query_to_run = q
            
        with col1:
            if st.button("🚪 Exit Interviews\n\nWhy are junior developers leaving?", use_container_width=True):
                set_query("Why are junior developers leaving?")
            if st.button("🏢 Work Culture\n\nWhat is the general sentiment around work-life balance?", use_container_width=True):
                set_query("What is the general sentiment around work-life balance?")
        with col2:
            if st.button("💵 Compensation\n\nAre employees satisfied with their salary and hikes?", use_container_width=True):
                set_query("Are employees satisfied with their salary and hikes?")
            if st.button("⏰ Remote Policy\n\nWhat is the feedback on remote work and WFH?", use_container_width=True):
                set_query("What is the feedback on remote work and WFH?")

    # --- DISPLAY CHAT MESSAGES ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # --- INPUT HANDLING ---
    query = st.chat_input("Ask about employee feedback, exit reasons, or culture...")

    if getattr(st.session_state, 'query_to_run', None):
        query = st.session_state.query_to_run
        st.session_state.query_to_run = None

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching internal data..."):
                response = st.session_state.pipeline.answer(query=query)
                
            content_html = f"<div style='margin-bottom: 10px;'>{response.answer}</div>"
            
            if not response.grounded or response.refusal_layer:
                content_html += f"""
<div class="guardrail-pill">
    <div class="guardrail-dot"></div>
    Guardrail: {response.refusal_layer or 'blocked'}
</div>
"""
            else:
                if response.sources:
                    content_html += f"""
<div style="margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px;">
<span style="font-size: 10px; color: #908fa0; text-transform: uppercase; margin-right: 8px;">Sources</span><br/>
"""
                    for s in response.sources:
                        score = s.get("retrieval_confidence", 0) * 100
                        content_html += f"""
<span class="source-pill">
    <span style="width:8px; height:8px; border-radius:50%; background:#a855f7; display:inline-block;"></span>
    Review Match <span style="color:#908fa0; margin-left:4px;">{score:.0f}%</span>
</span>
"""
                    content_html += "</div>"
                
                content_html += """
<div class="verified-pill">
    <div class="verified-dot"></div>
    Verified Response
</div>
"""
                
            st.markdown(content_html, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": content_html})
