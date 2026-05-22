"""Filter SNP positions by quality criteria (MAF, N-density, parsimony)."""

import pandas as pd


def filter_snp_positions(snp_df: pd.DataFrame, min_maf: float = 0.05) -> pd.DataFrame:
    """
    Remove SNP columns where the minor allele frequency (MAF) is below threshold.
    MAF is computed only among valid bases (A/C/G/T); 'N' positions are excluded
    so missing data doesn't distort allele frequencies.
    """
    keep = []
    for col in snp_df.columns:
        valid = snp_df[col][snp_df[col] != "N"]
        if len(valid) == 0:
            continue
        counts = valid.value_counts(normalize=True)
        maf = counts.iloc[-1] if len(counts) > 1 else 0.0
        if maf >= min_maf:
            keep.append(col)
    print(f"SNP setelah filter MAF ≥ {min_maf}: {len(keep)} / {len(snp_df.columns)}")
    return snp_df[keep]


def remove_invariant(snp_df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns where all isolates have the same base."""
    variable = snp_df.nunique(axis=0) > 1
    return snp_df.loc[:, variable]


def remove_high_n_columns(snp_df: pd.DataFrame, max_n_frac: float = 0.2) -> pd.DataFrame:
    """Drop SNP positions where more than max_n_frac isolates have 'N'."""
    n_frac = (snp_df == "N").mean(axis=0)
    return snp_df.loc[:, n_frac <= max_n_frac]
