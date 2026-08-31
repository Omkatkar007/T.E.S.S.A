"""
Phase 1b (add-on): add real per-review AmbitionBox company CSVs on top of
whatever's already indexed (e.g. your existing Glassdoor data), WITHOUT
re-embedding everything from scratch.

Use this instead of re-running ingest.py whenever you add a new
"<Company>_Employee_Reviews_from_AmbitionBox.csv" style file — the kind
with real per-review columns like Likes/Dislikes, Title, Department, etc.
(NOT a company-directory file with just rating/review-count/HQ columns).

How company name is determined:
    The company name is NOT a column in these files — it's derived from
    the filename. "Capgemini_Employee_Reviews_from_AmbitionBox.csv"
    becomes company = "capgemini". Rename files if you want a cleaner
    company name to show up in answers/sources.

Usage:
    1. Drop your new CSV(s) into data/raw/ambitionbox_companies/
    2. python scripts/add_ambitionbox_companies.py
"""
import os
import pickle
import re
import sys
import uuid

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.bm25 import BM25Index
from src.config import config
from src.embeddings import get_embedder
from src.qdrant_store import QdrantStore

AMBITIONBOX_COMPANIES_DIR = os.path.join(config.DATA_DIR, "raw", "ambitionbox_companies")
PROCESSED_DIR = os.path.join(config.DATA_DIR, "processed")

# Real per-review text columns in this file format (NOT a stats/directory file)
TEXT_COLUMNS = ["Likes", "Dislikes"]

# Matches e.g. "Capgemini_Employee_Reviews_from_AmbitionBox.csv" -> "capgemini"
_FILENAME_SUFFIX_RE = re.compile(
    r"_Employee_Reviews?_from_AmbitionBox", re.IGNORECASE
)


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = " ".join(text.split())
    return text


def company_name_from_filename(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = _FILENAME_SUFFIX_RE.sub("", stem)
    stem = stem.replace("_", " ").strip()
    return stem.lower()


def load_company_reviews(path: str) -> list[dict]:
    company = company_name_from_filename(path)
    df = pd.read_csv(path)

    available_text_cols = [c for c in TEXT_COLUMNS if c in df.columns]
    if not available_text_cols:
        print(f"  WARNING: no Likes/Dislikes-style columns found in {path}, skipping.")
        print(f"  Columns found: {df.columns.tolist()}")
        return []

    docs = []
    for _, row in df.iterrows():
        for col in available_text_cols:
            text = clean_text(row.get(col, ""))
            if not text or len(text) < 15:
                continue
            docs.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "payload": {
                    "company": company,
                    "source": f"ambitionbox_{col.lower()}",
                },
            })
    return docs


def dedupe_against_existing(new_docs: list[dict], existing_lookup: dict) -> list[dict]:
    seen = {(d["payload"]["company"], d["text"].lower()) for d in existing_lookup.values()}
    unique = []
    for d in new_docs:
        key = (d["payload"]["company"], d["text"].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique


def main():
    os.makedirs(AMBITIONBOX_COMPANIES_DIR, exist_ok=True)

    csv_paths = [
        os.path.join(AMBITIONBOX_COMPANIES_DIR, f)
        for f in os.listdir(AMBITIONBOX_COMPANIES_DIR)
        if f.lower().endswith(".csv")
    ]
    if not csv_paths:
        print(f"No CSVs found in {AMBITIONBOX_COMPANIES_DIR}/.")
        print("Drop your '<Company>_Employee_Reviews_from_AmbitionBox.csv' files there and re-run.")
        return

    new_docs = []
    for path in csv_paths:
        company = company_name_from_filename(path)
        docs = load_company_reviews(path)
        print(f"{os.path.basename(path)} -> company='{company}', {len(docs)} usable review rows")
        new_docs += docs

    if not new_docs:
        print("No usable review rows found across all CSVs. Nothing to add.")
        return

    # --- Load existing BM25 + doc_lookup (built by ingest.py) so we can merge, not overwrite ---
    bm25_path = os.path.join(PROCESSED_DIR, "bm25_index.pkl")
    lookup_path = os.path.join(PROCESSED_DIR, "doc_lookup.pkl")

    if os.path.exists(lookup_path):
        with open(lookup_path, "rb") as f:
            doc_lookup = pickle.load(f)
    else:
        doc_lookup = {}

    new_docs = dedupe_against_existing(new_docs, doc_lookup)
    print(f"{len(new_docs)} genuinely new docs after de-duping against existing index.")

    if not new_docs:
        print("Everything was already indexed. Nothing to add.")
        return

    # --- Embed + upsert into the SAME Qdrant collection (recreate=False!) ---
    embedder = get_embedder()
    store = QdrantStore()
    store.ensure_collection(recreate=False)  # <-- critical: do NOT wipe existing Glassdoor data

    batch_size = 128
    for i in tqdm(range(0, len(new_docs), batch_size), desc="Embedding + upserting new companies"):
        batch = new_docs[i:i + batch_size]
        texts = [d["text"] for d in batch]
        vecs = embedder.encode(texts)
        ids = [d["id"] for d in batch]
        payloads = [d["payload"] | {"text": d["text"]} for d in batch]
        store.upsert(ids, vecs, payloads)

    # --- Merge into doc_lookup and rebuild BM25 over the COMBINED doc set ---
    for d in new_docs:
        doc_lookup[d["id"]] = {"text": d["text"], "payload": d["payload"]}

    bm25 = BM25Index()
    bm25.build([(doc_id, entry["text"]) for doc_id, entry in doc_lookup.items()])

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    with open(lookup_path, "wb") as f:
        pickle.dump(doc_lookup, f)

    print(f"Done. Added {len(new_docs)} new docs. Total docs now: {len(doc_lookup)}.")
    print(f"Companies now indexed: {sorted(set(v['payload']['company'] for v in doc_lookup.values()))}")


if __name__ == "__main__":
    main()
