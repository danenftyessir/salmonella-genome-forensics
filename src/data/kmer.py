"""E5 baseline — alignment-free k-mer frequency features.

Tetranucleotide (k=4) frequency is a standard alignment-free genomic representation.
It counts every k-mer in the concatenated genome sequence and normalises by the total
number of valid (non-N) k-mers.

Why k=4?
  • 4^4 = 256 features — compact enough for small datasets
  • Captures local sequence composition without alignment
  • Widely used in bacterial genomics as a simple baseline

Compared to DNABERT embeddings (768-dim, context-aware via transformer attention),
k-mer frequency is a "bag-of-words" approach: fast, interpretable, but blind to
the positional context of each k-mer within the genome.
"""

from __future__ import annotations

from collections import Counter
from itertools import product

import numpy as np
import pandas as pd

_BASES = "ATGC"


def _all_kmers(k: int) -> list[str]:
    return ["".join(p) for p in product(_BASES, repeat=k)]


def extract_kmer_features(
    genomes: dict[str, str],
    k: int = 4,
) -> pd.DataFrame:
    """
    Compute normalised k-mer frequency for each isolate.

    Parameters
    ----------
    genomes : {assembly_accession: concatenated_sequence}
    k       : k-mer length (default 4 → 256 features)

    Returns
    -------
    DataFrame shape (n_isolates, 4^k), index = assembly_accession.
    Values are frequencies in [0, 1] (count / total_valid_kmers).
    """
    kmers = _all_kmers(k)
    rows: dict[str, dict] = {}

    for acc, seq in genomes.items():
        seq_upper = seq.upper()
        counts: Counter = Counter()
        for i in range(len(seq_upper) - k + 1):
            window = seq_upper[i: i + k]
            if all(c in _BASES for c in window):
                counts[window] += 1
        total = sum(counts.values()) or 1
        rows[acc] = {km: counts.get(km, 0) / total for km in kmers}

    df = pd.DataFrame(rows).T
    df.index.name = "assembly_accession"
    print(f"K-mer features ({k}-mer): {df.shape}  ({len(kmers)} features per isolat)")
    return df
