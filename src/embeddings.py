"""
Local, quantized MiniLM embedding model.
Loaded once as a process-wide singleton so ingestion and query time reuse
the same in-memory model instead of reloading weights per call.
"""
from __future__ import annotations
import threading
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import config


class _EmbeddingSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._load()
        return cls._instance

    def _load(self):
        # sentence-transformers pulls the equivalent PyTorch weights for
        # the same MiniLM checkpoint used by the Xenova ONNX build.
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # cosine similarity == dot product
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype("float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def get_embedder() -> _EmbeddingSingleton:
    return _EmbeddingSingleton()


if __name__ == "__main__":
    emb = get_embedder()
    v = emb.encode_one("TCS bench time and WFH policy")
    print("dim:", v.shape, "norm:", float(np.linalg.norm(v)))
