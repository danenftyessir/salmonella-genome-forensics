"""Clustering quality metrics: Silhouette Score."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def compute_silhouette(
    features,
    labels,
    metric: str = "euclidean",
    sample_size: int | None = None,
    random_state: int = 42,
) -> float:
    """
    Silhouette score in [-1, 1].  Higher = tighter, better-separated clusters.

    Parameters
    ----------
    features : DataFrame or array, shape (n_samples, n_features)
        If metric='precomputed', pass the square distance matrix.
    labels   : Series or array, shape (n_samples,)
        Cluster assignment per sample.
    metric   : sklearn-compatible distance metric or 'precomputed'

    Returns NaN when score cannot be computed (< 2 clusters, < 2 samples
    in any cluster, or other sklearn error).
    """
    X = features.values if isinstance(features, pd.DataFrame) else np.asarray(features, dtype=float)
    y = labels.values   if isinstance(labels, pd.Series)    else np.asarray(labels)

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        print(f"[WARN] Silhouette: hanya {len(unique_labels)} cluster — skip.")
        return float("nan")

    if any(np.sum(y == lbl) < 2 for lbl in unique_labels):
        print("[WARN] Silhouette: ada cluster dengan < 2 sampel — skip.")
        return float("nan")

    try:
        kw: dict = {"metric": metric, "random_state": random_state}
        if sample_size is not None:
            kw["sample_size"] = min(sample_size, len(y))
        return float(silhouette_score(X, y, **kw))
    except Exception as exc:
        print(f"[WARN] Silhouette gagal: {exc}")
        return float("nan")
