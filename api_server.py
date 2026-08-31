"""
FastAPI server that exposes the Placement Truth Check pipeline as a REST API.

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8000
Or:
    python api_server.py
"""
import os
import pickle
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------- Path setup ----------
# Everything lives in this one folder
BASE_DIR = r"D:\Btech 24-28\3 rd\ml\placement-truth-check\placement-truth-check"
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

sys.path.insert(0, BASE_DIR)

from src.config import config
from src.pipeline import PlacementTruthCheckPipeline

# ---------- App setup ----------
app = FastAPI(title="Placement Truth Check API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pipeline singleton ----------
PROCESSED_DIR = os.path.join(config.DATA_DIR, "processed")
pipeline: PlacementTruthCheckPipeline | None = None


@app.on_event("startup")
def load_pipeline():
    global pipeline
    bm25_path = os.path.join(PROCESSED_DIR, "bm25_index.pkl")
    lookup_path = os.path.join(PROCESSED_DIR, "doc_lookup.pkl")

    if not (os.path.exists(bm25_path) and os.path.exists(lookup_path)):
        print(
            "⚠️  No index found. Run `python scripts/ingest.py` first.\n"
            f"   Expected: {bm25_path} and {lookup_path}"
        )
        return

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(lookup_path, "rb") as f:
        doc_lookup = pickle.load(f)

    pipeline = PlacementTruthCheckPipeline(bm25_index=bm25, doc_lookup=doc_lookup)
    print("✅ Pipeline loaded successfully.")


# ---------- Request / Response models ----------
class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    refusal_layer: str | None = None
    sources: list[dict] = []


# ---------- API endpoint ----------
@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not loaded. Run ingest.py to build the index."
        )

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = pipeline.answer(query=req.query.strip())

    return AskResponse(
        answer=result.answer,
        grounded=result.grounded,
        refusal_layer=result.refusal_layer,
        sources=result.sources,
    )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
    }


# ---------- Serve frontend at / ----------
@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend")


# ---------- Run directly ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
