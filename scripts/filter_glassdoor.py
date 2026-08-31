"""
Your glassdoor_reviews.csv has 1.67 million rows — global companies, not
just your target list. Uploading all of that to a free-tier cloud Qdrant
cluster over the internet is impractically slow (estimated 13+ hours, and
prone to timing out partway through, which is exactly what just happened).

This script creates a much smaller, focused version of that file:
data/raw/glassdoor_reviews_filtered.csv

It keeps:
  1. ALL rows for your actual target/tracked companies (cheap to check,
     keeps everything relevant no matter how rare)
  2. A random sample of everything else, capped at MAX_OTHER_ROWS, just
     to preserve some variety/comparison companies without keeping millions
     of irrelevant rows

Usage:
    python scripts/filter_glassdoor.py
Then point ingest.py at the filtered file instead of the original by
renaming/swapping it in, OR just update GD_FILENAME below and re-run.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import config

RAW_DIR = os.path.join(config.DATA_DIR, "raw")
INPUT_PATH = os.path.join(RAW_DIR, "glassdoor_reviews.csv")
OUTPUT_PATH = os.path.join(RAW_DIR, "glassdoor_reviews_filtered.csv")

COMPANY_COLUMN = "firm"  # matches GLASSDOOR_COLUMNS["company"] in ingest.py
TARGET_COMPANIES = list(config.COMPANIES)  # your tracked companies list
MAX_OTHER_ROWS = 30000  # cap on everything else, keeps file small + fast


def main():
    print(f"Reading {INPUT_PATH} (this may take a minute for a large file)...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Total rows: {len(df)}")

    company_col = df[COMPANY_COLUMN].astype(str).str.lower()
    is_target = company_col.apply(lambda c: any(t in c for t in TARGET_COMPANIES))

    target_rows = df[is_target]
    other_rows = df[~is_target]

    print(f"Rows matching target companies: {len(target_rows)}")
    print(f"Other rows available: {len(other_rows)}")

    if len(other_rows) > MAX_OTHER_ROWS:
        other_rows = other_rows.sample(n=MAX_OTHER_ROWS, random_state=42)

    result = pd.concat([target_rows, other_rows], ignore_index=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Done. Wrote {len(result)} rows to {OUTPUT_PATH}")
    print(f"(down from {len(df)} — a {len(df) / max(len(result), 1):.0f}x reduction)")
    print()
    print("Next step: in scripts/ingest.py, change this line:")
    print('    gd_path = os.path.join(RAW_DIR, "glassdoor_reviews.csv")')
    print("to:")
    print('    gd_path = os.path.join(RAW_DIR, "glassdoor_reviews_filtered.csv")')


if __name__ == "__main__":
    main()
