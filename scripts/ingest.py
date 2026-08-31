"""
Phase 1+2: load AmbitionBox/Glassdoor CSVs, clean/normalize, embed, and
index into Qdrant (dense) + build a BM25 index (lexical), then pickle
the BM25 index + doc lookup so pipeline.py can load them without
re-embedding every run.

Expected input CSVs in data/raw/:
    ambitionbox_reviews.csv  — columns like: company, title, review_text, rating, ...
    glassdoor_reviews.csv    — columns like: firm, headline, pros, cons, ...

Column names vary a lot between Kaggle dumps — adjust COLUMN_MAP below
to match whatever you actually downloaded.
"""
import json
import os
import pickle
import sys
import uuid

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.bm25 import BM25Index
from src.config import config
from src.embeddings import get_embedder
from src.qdrant_store import QdrantStore

RAW_DIR = os.path.join(config.DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(config.DATA_DIR, "processed")

# --- adjust to match your actual downloaded column names ---
AMBITIONBOX_COLUMNS = {"company": "Company_Name", "text": "Title", "extra": "likes"}
GLASSDOOR_COLUMNS = {"company": "firm", "pros": "pros", "cons": "cons"}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = " ".join(text.split())  # collapse whitespace
    return text


def load_ambitionbox(path: str) -> list[dict]:
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        company = clean_text(str(row.get(AMBITIONBOX_COLUMNS["company"], "")))
        text = clean_text(str(row.get(AMBITIONBOX_COLUMNS["text"], "")))
        if not company or not text or len(text) < 15:
            continue
        docs.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "payload": {"company": company.lower(), "source": "ambitionbox"},
        })
    return docs


def load_glassdoor(path: str) -> list[dict]:
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        company = clean_text(str(row.get(GLASSDOOR_COLUMNS["company"], "")))
        pros = clean_text(str(row.get(GLASSDOOR_COLUMNS["pros"], "")))
        cons = clean_text(str(row.get(GLASSDOOR_COLUMNS["cons"], "")))
        for label, text in (("pros", pros), ("cons", cons)):
            if not company or not text or len(text) < 15:
                continue
            docs.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "payload": {"company": company.lower(), "source": f"glassdoor_{label}"},
            })
    return docs


def dedupe(docs: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for d in docs:
        key = (d["payload"]["company"], d["text"].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    docs = []

    ab_path = os.path.join(RAW_DIR, "ambitionbox_reviews.csv")
    gd_path = os.path.join(RAW_DIR, "glassdoor_reviews_filtered.csv")

    if os.path.exists(ab_path):
        docs += load_ambitionbox(ab_path)
        print(f"Loaded {len(docs)} AmbitionBox docs so far")
    if os.path.exists(gd_path):
        before = len(docs)
        docs += load_glassdoor(gd_path)
        print(f"Loaded {len(docs) - before} Glassdoor docs")

    if not docs:
        print(f"No raw CSVs found in {RAW_DIR}/. Place your Kaggle downloads there and re-run.")
        return

    docs = dedupe(docs)
    print(f"Total unique docs after dedup: {len(docs)}")

    # --- Resume support: remember which batch we last finished, so a crash
    # partway through a long cloud upload doesn't force starting over. ---
    checkpoint_path = os.path.join(PROCESSED_DIR, "ingest_checkpoint.json")
    batch_size = 128
    total_batches = (len(docs) + batch_size - 1) // batch_size

    start_batch = 0
    is_resuming = False
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        # Only trust the checkpoint if it's for this exact same dataset size —
        # otherwise the batch numbers would point at the wrong rows.
        if checkpoint.get("total_docs") == len(docs):
            start_batch = checkpoint.get("last_completed_batch", -1) + 1
            is_resuming = start_batch > 0
            if is_resuming:
                print(f"Resuming from batch {start_batch}/{total_batches} "
                      f"(checkpoint found from a previous run).")
        else:
            print("Checkpoint found but doesn't match current dataset size — starting fresh.")

    # --- Embed + push to Qdrant ---
    embedder = get_embedder()
    store = QdrantStore()
    # Only wipe the collection on a genuinely fresh run, never when resuming —
    # otherwise every retry after a crash would throw away prior progress.
    store.ensure_collection(recreate=not is_resuming)

    for batch_idx in tqdm(range(start_batch, total_batches), desc="Embedding + upserting",
                           initial=start_batch, total=total_batches):
        i = batch_idx * batch_size
        batch = docs[i:i + batch_size]
        texts = [d["text"] for d in batch]
        vecs = embedder.encode(texts)
        ids = [d["id"] for d in batch]
        payloads = [d["payload"] | {"text": d["text"]} for d in batch]
        store.upsert(ids, vecs, payloads)

        # Save progress after every successful batch.
        with open(checkpoint_path, "w") as f:
            json.dump({"total_docs": len(docs), "last_completed_batch": batch_idx}, f)

    # All batches done — checkpoint no longer needed.
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # --- Build BM25 index ---
    bm25 = BM25Index()
    bm25.build([(d["id"], d["text"]) for d in docs])

    # --- Persist BM25 + doc lookup for pipeline.py ---
    doc_lookup = {d["id"]: {"text": d["text"], "payload": d["payload"]} for d in docs}
    with open(os.path.join(PROCESSED_DIR, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    with open(os.path.join(PROCESSED_DIR, "doc_lookup.pkl"), "wb") as f:
        pickle.dump(doc_lookup, f)

    print(f"Done. Indexed {len(docs)} docs into Qdrant + BM25.")
    print(f"Artifacts saved to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
