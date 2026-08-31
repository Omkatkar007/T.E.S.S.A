"""
In-memory Okapi BM25 — implemented directly from the formula (not a
full-text-search engine wrapper) so scoring behavior is transparent and
tunable (k1, b) for this specific corpus of short review snippets.

BM25(q, d) = sum over query terms t of:
    IDF(t) * ( f(t,d) * (k1+1) ) / ( f(t,d) + k1 * (1 - b + b * |d|/avgdl) )
"""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict

from .config import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = config.BM25_K1, b: float = config.BM25_B):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_freqs: list[Counter] = []     # term freq per doc
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.df: dict[str, int] = defaultdict(int)   # doc frequency per term
        self.idf: dict[str, float] = {}
        self.N: int = 0
        self._built = False

    def build(self, documents: list[tuple[str, str]]):
        """documents: list of (doc_id, text)"""
        self.doc_ids = []
        self.doc_freqs = []
        self.doc_lens = []
        self.df = defaultdict(int)

        for doc_id, text in documents:
            tokens = tokenize(text)
            tf = Counter(tokens)
            self.doc_ids.append(doc_id)
            self.doc_freqs.append(tf)
            self.doc_lens.append(len(tokens))
            for term in tf.keys():
                self.df[term] += 1

        self.N = len(self.doc_ids)
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 0.0

        # Robertson-Sparck Jones IDF with +1 smoothing (never negative)
        self.idf = {
            term: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }
        self._built = True

    def score(self, query: str, doc_index: int) -> float:
        q_tokens = tokenize(query)
        tf = self.doc_freqs[doc_index]
        dl = self.doc_lens[doc_index]
        score = 0.0
        for term in q_tokens:
            if term not in tf:
                continue
            idf = self.idf.get(term, 0.0)
            f = tf[term]
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = config.LEXICAL_TOP_K) -> list[tuple[str, float]]:
        if not self._built:
            raise RuntimeError("BM25Index.build() must be called before search().")
        scores = [(self.doc_ids[i], self.score(query, i)) for i in range(self.N)]
        scores = [s for s in scores if s[1] > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


if __name__ == "__main__":
    idx = BM25Index()
    idx.build([
        ("1", "TCS has a long bench period with low pay during bench"),
        ("2", "Infosys offers good WFH flexibility and decent hikes"),
        ("3", "Wipro appraisal cycle is slow and hikes are minimal"),
    ])
    print(idx.search("bench pay TCS"))
