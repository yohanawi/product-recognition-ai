"""Similarity utilities for normalized jewelry image embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency at runtime
    faiss = None


class FaissUnavailableError(RuntimeError):
    """Raised when FAISS-backed search is requested without faiss-cpu installed."""


def ensure_2d_float32(vectors: Iterable[Sequence[float]]) -> np.ndarray:
    array = np.asarray(list(vectors), dtype=np.float32)
    if array.size == 0:
        raise ValueError("At least one vector is required")
    if array.ndim == 1:
        array = np.expand_dims(array, axis=0)
    if array.ndim != 2:
        raise ValueError("Vectors must be a 1D or 2D numeric array")
    return array


def normalize_rows(vectors: Iterable[Sequence[float]]) -> np.ndarray:
    array = ensure_2d_float32(vectors)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Vectors must have non-zero length")
    return array / norms


def cosine_similarity(query_vector: Sequence[float], candidate_vectors: Iterable[Sequence[float]]) -> np.ndarray:
    normalized_query = normalize_rows([query_vector])[0]
    normalized_candidates = normalize_rows(candidate_vectors)
    return normalized_candidates @ normalized_query


def build_faiss_index(vectors: Iterable[Sequence[float]]):
    if faiss is None:
        raise FaissUnavailableError(
            "faiss-cpu is not installed. Add it to requirements and install dependencies first."
        )

    normalized_vectors = normalize_rows(vectors)
    dimension = normalized_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_vectors)
    return index, normalized_vectors


def search_faiss_index(index, query_vector: Sequence[float], top_k: int = 10):
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    normalized_query = normalize_rows([query_vector])
    scores, indices = index.search(normalized_query, top_k)
    results = []
    for rank, candidate_index in enumerate(indices[0].tolist()):
        if candidate_index < 0:
            continue
        results.append(
            {
                "rank": rank + 1,
                "index": int(candidate_index),
                "score": float(scores[0][rank]),
            }
        )
    return results


def save_faiss_artifacts(index, metadata: Sequence[dict], index_path: str | Path, metadata_path: str | Path) -> None:
    if faiss is None:
        raise FaissUnavailableError(
            "faiss-cpu is not installed. Add it to requirements and install dependencies first."
        )

    index_target = Path(index_path)
    metadata_target = Path(metadata_path)
    index_target.parent.mkdir(parents=True, exist_ok=True)
    metadata_target.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_target))
    metadata_target.write_text(json.dumps(list(metadata), indent=2), encoding="utf-8")


def load_faiss_artifacts(index_path: str | Path, metadata_path: str | Path):
    if faiss is None:
        raise FaissUnavailableError(
            "faiss-cpu is not installed. Add it to requirements and install dependencies first."
        )

    index = faiss.read_index(str(index_path))
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return index, metadata
