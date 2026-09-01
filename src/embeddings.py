"""
Local, quantized MiniLM embedding model — ONNX Runtime version.

Why ONNX instead of sentence-transformers/PyTorch:
    PyTorch's runtime alone typically costs 200-300MB+ of RAM just by being
    imported, before any model weights are even loaded. On a memory-capped
    host (e.g. Render's free 512MB tier), that overhead alone can be enough
    to crash the process. ONNX Runtime is a much lighter, purpose-built
    inference engine with no PyTorch dependency at all.

This produces embeddings from the SAME underlying model checkpoint
(all-MiniLM-L6-v2) using the same mean-pooling + L2-normalization that
sentence-transformers used, so vectors are compatible with whatever you
already have stored in Qdrant from the original ingest run — no
re-embedding needed. Quantization introduces tiny numeric differences,
which do not meaningfully affect cosine-similarity search quality.

The ONNX model + tokenizer files are downloaded once (cached locally by
huggingface_hub) from the community ONNX conversion at
huggingface.co/Xenova/all-MiniLM-L6-v2 — the same conversion referenced
in this project's original architecture spec.
"""
from __future__ import annotations
import threading

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

_REPO_ID = "Xenova/all-MiniLM-L6-v2"
_ONNX_SUBPATH = "onnx/model_quantized.onnx"
_TOKENIZER_SUBPATH = "tokenizer.json"
_MAX_SEQ_LEN = 256


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
        onnx_path = hf_hub_download(repo_id=_REPO_ID, filename=_ONNX_SUBPATH)
        tokenizer_path = hf_hub_download(repo_id=_REPO_ID, filename=_TOKENIZER_SUBPATH)

        # Single-threaded intra-op is intentional: keeps peak memory and CPU
        # contention low on small/free-tier hosts. Fine for the request
        # volumes a resume-demo project actually sees.
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(length=None)  # pad per-batch to longest in batch
        self.tokenizer.enable_truncation(max_length=_MAX_SEQ_LEN)

    def _mean_pool(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        # Same mean-pooling sentence-transformers uses: average token vectors,
        # weighted by attention_mask so padding tokens don't contribute.
        mask = attention_mask[..., None].astype("float32")
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return summed / counts

    def _l2_normalize(self, vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return vecs / norms

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encodings = self.tokenizer.encode_batch(batch)

            input_ids = np.array([e.ids for e in encodings], dtype="int64")
            attention_mask = np.array([e.attention_mask for e in encodings], dtype="int64")
            token_type_ids = np.zeros_like(input_ids)

            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )
            token_embeddings = outputs[0]  # (batch, seq_len, 384)
            pooled = self._mean_pool(token_embeddings, attention_mask)
            normalized = self._l2_normalize(pooled)
            all_vecs.append(normalized.astype("float32"))

        return np.concatenate(all_vecs, axis=0)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def get_embedder() -> _EmbeddingSingleton:
    return _EmbeddingSingleton()


if __name__ == "__main__":
    emb = get_embedder()
    v = emb.encode_one("TCS bench time and WFH policy")
    print("dim:", v.shape, "norm:", float(np.linalg.norm(v)))
