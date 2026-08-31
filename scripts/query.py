"""
Interactive CLI: loads the persisted BM25 index + doc lookup from
scripts/ingest.py's output, then answers questions in a loop.

Usage:
    python scripts/query.py
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import config
from src.pipeline import PlacementTruthCheckPipeline

PROCESSED_DIR = os.path.join(config.DATA_DIR, "processed")


def main():
    bm25_path = os.path.join(PROCESSED_DIR, "bm25_index.pkl")
    lookup_path = os.path.join(PROCESSED_DIR, "doc_lookup.pkl")

    if not (os.path.exists(bm25_path) and os.path.exists(lookup_path)):
        print("Run scripts/ingest.py first to build the indexes.")
        return

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(lookup_path, "rb") as f:
        doc_lookup = pickle.load(f)

    pipeline = PlacementTruthCheckPipeline(bm25_index=bm25, doc_lookup=doc_lookup)

    print("Placement Truth Check — ask about TCS, Infosys, Wipro, etc. (Ctrl+C to quit)\n")
    while True:
        try:
            query = input("Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not query:
            continue

        response = pipeline.answer(query=query)
        print(f"\nA: {response.answer}\n")
        if response.grounded and response.sources:
            print("Sources:")
            for s in response.sources:
                p = s.get("payload", {})
                print(f"  - [{p.get('company')} | {p.get('source')}] score={s.get('rerank_score', 0):.3f}")
        elif response.refusal_layer:
            print(f"(refused at layer: {response.refusal_layer})")
        print()


if __name__ == "__main__":
    main()
