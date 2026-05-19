"""Compute N×N pairwise SNP Hamming distance matrix."""

import numpy as np
import pandas as pd


def compute_distance_matrix(snp_df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of pairwise SNP Hamming distances."""
    accessions = snp_df.index.tolist()
    arr = snp_df.values
    n = len(accessions)
    dist = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            d = int(np.sum(arr[i] != arr[j]))
            dist[i, j] = d
            dist[j, i] = d

    return pd.DataFrame(dist, index=accessions, columns=accessions)


def nearest_neighbors(dist_df: pd.DataFrame, accession: str, top_k: int = 5) -> pd.Series:
    """Return top-k closest isolates to a given accession."""
    row = dist_df.loc[accession].drop(labels=[accession])
    return row.nsmallest(top_k)
