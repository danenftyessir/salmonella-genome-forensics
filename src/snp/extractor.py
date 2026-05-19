"""Build SNP matrix by comparing all genomes position-by-position."""

import numpy as np
import pandas as pd


BASE_TO_INT: dict[str, int] = {"A": 0, "T": 1, "G": 2, "C": 3, "N": -1}


def build_snp_matrix(genomes: dict) -> pd.DataFrame:
    """
    Compare each genome against all others at every position.
    Rows = accessions, Columns = SNP positions (variable sites, excluding N-only columns).

    Note: uses reference-free pairwise comparison after truncating to the
    shortest genome.  A true reference-based approach (minimap2/MUMmer)
    would be more accurate but requires external binaries.
    """
    accessions = list(genomes.keys())
    seqs = list(genomes.values())
    min_len = min(len(s) for s in seqs)
    seqs_clipped = [s[:min_len].upper() for s in seqs]

    snp_positions = [
        pos for pos in range(min_len)
        if len({s[pos] for s in seqs_clipped} - {"N"}) > 1
    ]

    matrix = {
        acc: [seq[pos] for pos in snp_positions]
        for acc, seq in zip(accessions, seqs_clipped)
    }

    df = pd.DataFrame(matrix, index=snp_positions).T
    df.index.name = "assembly_accession"
    print(f"SNP ditemukan: {len(snp_positions)} posisi dari {len(accessions)} isolat")
    return df


def encode_snp_matrix(snp_df: pd.DataFrame, method: str = "integer") -> pd.DataFrame:
    """
    Encode SNP bases to numeric values for ML/distance computation.

    method='integer': A=0, T=1, G=2, C=3, N=-1
    Returns a DataFrame of the same shape with integer dtype.
    """
    if method != "integer":
        raise ValueError(f"Encoding method tidak didukung: '{method}'. Gunakan 'integer'.")

    encoded = snp_df.replace(BASE_TO_INT)
    # Any base not in BASE_TO_INT (e.g. IUPAC ambiguity codes) maps to -1
    encoded = encoded.apply(pd.to_numeric, errors="coerce").fillna(-1).astype(int)
    encoded.index.name = "assembly_accession"
    print(f"SNP encoded ({method}): shape={encoded.shape}")
    return encoded
