from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_glove_txt(
    glove_path: str | Path,
    max_words: int | None = None,
    expected_dim: int | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    path = Path(glove_path)
    if not path.exists():
        raise FileNotFoundError(f"GloVe file not found: {path}")

    embeddings: dict[str, np.ndarray] = {}
    dim: int | None = None

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_words is not None and i >= max_words:
                break

            parts = line.rstrip("\n").split(" ")
            if len(parts) < 2:
                continue

            token = parts[0]
            values = np.asarray(parts[1:], dtype=np.float32)

            if dim is None:
                dim = int(values.shape[0])
                if expected_dim is not None and dim != expected_dim:
                    raise ValueError(
                        f"Expected {expected_dim} dimensions but found {dim} in {path}"
                    )

            if values.shape[0] != dim:
                continue

            embeddings[token] = values

    if dim is None:
        raise ValueError(f"No embeddings loaded from {path}")

    return embeddings, dim


def _pool_vectors(vectors: list[np.ndarray], dim: int) -> tuple[np.ndarray, np.ndarray]:
    if not vectors:
        return np.zeros(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32)

    stacked = np.vstack(vectors)
    mean_vec = stacked.mean(axis=0).astype(np.float32)
    max_vec = stacked.max(axis=0).astype(np.float32)

    return mean_vec, max_vec


def build_glove_feature_frame(
    tokenized_texts: Iterable[list[str]],
    embeddings: dict[str, np.ndarray],
    dim: int,
    prefix: str = "glove",
) -> pd.DataFrame:
    rows: list[np.ndarray] = []

    for tokens in tokenized_texts:
        vectors = [embeddings[t] for t in tokens if t in embeddings]
        mean_vec, max_vec = _pool_vectors(vectors, dim)
        rows.append(np.concatenate([mean_vec, max_vec]))

    columns = [f"{prefix}_mean_{i:03d}" for i in range(dim)]
    columns.extend([f"{prefix}_max_{i:03d}" for i in range(dim)])

    return pd.DataFrame(rows, columns=columns)
