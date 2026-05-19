"""Hierarchical clustering using SNP distance matrix."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def hierarchical_clustering(dist_df: pd.DataFrame, method: str = "ward") -> np.ndarray:
    """Return scipy linkage matrix from a square distance DataFrame."""
    condensed = squareform(dist_df.values.astype(float))
    return linkage(condensed, method=method)


def assign_clusters(
    linkage_matrix: np.ndarray,
    labels: list,
    n_clusters: int = None,
    distance_threshold: float = None,
) -> pd.Series:
    """
    Cut the dendrogram and return cluster assignments as a Series.
    Provide either n_clusters or distance_threshold.
    """
    if n_clusters is not None:
        cluster_ids = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    elif distance_threshold is not None:
        cluster_ids = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")
    else:
        raise ValueError("Provide either n_clusters or distance_threshold.")
    return pd.Series(cluster_ids, index=labels, name="hclust_label")
