"""Compute N×N pairwise SNP Hamming distance matrix."""

import numpy as np
import pandas as pd


def compute_distance_matrix(snp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise SNP Hamming distance matrix.

    Only positions where BOTH isolates have a called base (A/C/G/T) are
    counted.  Positions where either isolate has 'N' (uncalled/missing)
    are excluded from the comparison — treating them as missing data, not
    as a difference.

    Uses a semi-vectorised loop (O(n) iterations, each fully vectorised
    over all SNP positions and all j simultaneously) for speed.
    """
    accessions = snp_df.index.tolist()
    n = len(accessions)

    if snp_df.shape[1] == 0:
        print("[WARN] SNP matrix kosong — distance matrix akan berisi semua nol.")
        return pd.DataFrame(
            np.zeros((n, n), dtype=np.int32),
            index=accessions, columns=accessions,
        )

    # Encode characters → uint8: A=1 C=2 G=3 T=4; N / anything else → 0 (missing)
    enc = (
        snp_df.replace({"A": 1, "C": 2, "G": 3, "T": 4, "N": 0})
        .fillna(0)
        .values.astype(np.uint8)
    )
    # enc shape: (n_isolates, n_snp_positions)

    dist = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        # Broadcast enc[i] (1, n_snp) against enc (n, n_snp) in one numpy call.
        # both_called: True only where BOTH isolates have a valid base (non-zero).
        # differ: True where both called AND bases are different.
        both_called = (enc[i] > 0) & (enc > 0)     # (n, n_snp) bool
        differ      = both_called & (enc[i] != enc)  # (n, n_snp) bool
        dist[i]     = differ.sum(axis=1)             # (n,) int

    np.fill_diagonal(dist, 0)

    result = pd.DataFrame(dist, index=accessions, columns=accessions)
    nonzero_pairs = int((dist > 0).sum()) // 2
    print(f"[DIST] Distance matrix: {n}×{n}, {nonzero_pairs} pasang isolat dengan jarak > 0  "
          f"(min={dist[dist > 0].min() if nonzero_pairs else 0}, "
          f"max={dist.max()}, mean={dist[dist > 0].mean():.1f} jika nonzero)")
    return result


def nearest_neighbors(dist_df: pd.DataFrame, accession: str, top_k: int = 5) -> pd.Series:
    """Return top-k closest isolates to a given accession."""
    row = dist_df.loc[accession].drop(labels=[accession])
    return row.nsmallest(top_k)
