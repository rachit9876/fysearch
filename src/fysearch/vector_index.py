from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float


class VectorIndex(Protocol):
    dim: int

    def add(self, doc_ids: list[str], vectors: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, top_k: int) -> list[SearchHit]: ...


class BruteForceIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._doc_ids: list[str] = []
        self._matrix: np.ndarray | None = None  # Pre-stacked for fast search

    def add(self, doc_ids: list[str], vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors shape (n,{self.dim}), got {vectors.shape}")
        # Pre-stack the matrix so we don't re-stack on every search call
        mat = vectors.astype(np.float32, copy=False)
        if self._matrix is not None:
            self._matrix = np.vstack([self._matrix, mat])
        else:
            self._matrix = mat.copy()
        self._doc_ids.extend(doc_ids)

    def search(self, query: np.ndarray, top_k: int) -> list[SearchHit]:
        if query.shape != (self.dim,):
            raise ValueError(f"Expected query shape ({self.dim},), got {query.shape}")
        if self._matrix is None or len(self._doc_ids) == 0:
            return []
        # Cosine similarity via dot product (assumes vectors are already L2-normalized)
        scores = self._matrix @ query.astype(np.float32, copy=False)
        # Use argpartition for faster top-k selection on large arrays
        k = min(top_k, len(scores))
        if k >= len(scores):
            idx = np.argsort(-scores)[:k]
        else:
            # argpartition is O(n) vs argsort O(n log n)
            top_indices = np.argpartition(-scores, k)[:k]
            idx = top_indices[np.argsort(-scores[top_indices])]
        return [SearchHit(doc_id=self._doc_ids[i], score=float(scores[i])) for i in idx]


class FaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        try:
            import faiss  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("faiss not installed; install extras: pip install -e .[faiss]") from e

        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._doc_ids: list[str] = []

    def add(self, doc_ids: list[str], vectors: np.ndarray) -> None:
        vectors = vectors.astype(np.float32, copy=False)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors shape (n,{self.dim}), got {vectors.shape}")
        self._index.add(vectors)
        self._doc_ids.extend(doc_ids)

    def search(self, query: np.ndarray, top_k: int) -> list[SearchHit]:
        query = query.astype(np.float32, copy=False).reshape(1, -1)
        scores, indices = self._index.search(query, top_k)
        hits: list[SearchHit] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0 or idx >= len(self._doc_ids):
                continue
            hits.append(SearchHit(doc_id=self._doc_ids[idx], score=float(score)))
        return hits
