"""
Thin wrapper around the Qdrant client for dense vector storage + cosine ANN search.
Run a local Qdrant instance with:
    docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
"""
from __future__ import annotations
import time

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import httpx

from .config import config


class QdrantStore:
    def __init__(self):
        # Cloud upserts over the internet are much slower and flakier than a
        # local Docker instance, so we need a generous timeout (default is
        # only 5s) plus our own retry loop below.
        self.client = self._new_client()
        self.collection = config.QDRANT_COLLECTION

    def _new_client(self) -> QdrantClient:
        return QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY or None,
            timeout=120,  # seconds
        )

    def ensure_collection(self, recreate: bool = False):
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
            )

    def upsert(self, ids: list[str], vectors, payloads: list[dict], max_retries: int = 6):
        points = [
            PointStruct(id=i, vector=vec.tolist(), payload=payload)
            for i, vec, payload in zip(ids, vectors, payloads)
        ]
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                self.client.upsert(collection_name=self.collection, points=points)
                return
            except (
                httpx.TransportError,      # covers ReadTimeout, ConnectTimeout,
                                            # ConnectError, ReadError, WriteError, etc.
                httpx.HTTPError,
                ConnectionError,
                OSError,                   # covers WinError 10054-style resets
            ) as e:
                last_error = e
                wait = min(2 ** attempt, 30)  # exponential backoff, capped at 30s
                print(f"  [retry {attempt}/{max_retries}] Qdrant upsert failed "
                      f"({type(e).__name__}), retrying in {wait}s...")
                time.sleep(wait)
                # A stale/half-broken connection sometimes needs a fresh client,
                # not just a retry on the same one.
                self.client = self._new_client()
        raise RuntimeError(
            f"Qdrant upsert failed after {max_retries} retries. Last error: {last_error}"
        )

    def search(self, query_vector, top_k: int = config.DENSE_TOP_K):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        ).points
        return [
            {"id": str(r.id), "score": float(r.score), "payload": r.payload}
            for r in results
        ]
