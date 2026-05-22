"""AMR gene profile feature extraction.

Parses the `amr_genes` column in NCBI metadata into a binary one-hot feature
matrix. Each column represents the presence/absence of one AMR gene.
"""
from __future__ import annotations

import pandas as pd

_MISSING = {"", "nan", "not provided", "not available", "na", "none"}


def extract_amr_features(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Parse AMR gene profile → binary feature matrix.

    Parameters
    ----------
    metadata_df : cleaned metadata with 'amr_genes' column

    Returns
    -------
    DataFrame shape (n_isolates, n_unique_genes), index = assembly_accession.
    Values are 0/1 (gene present/absent). Empty DataFrame if no AMR data.
    """
    if "amr_genes" not in metadata_df.columns:
        print("[AMR] Kolom 'amr_genes' tidak ditemukan, skip.")
        return pd.DataFrame()

    def _parse(val) -> list[str]:
        if pd.isna(val) or str(val).strip().lower() in _MISSING:
            return []
        return [g.strip() for g in str(val).split(",") if g.strip()]

    indexed = metadata_df.set_index("assembly_accession")
    gene_lists = indexed["amr_genes"].apply(_parse)

    all_genes = sorted({g for genes in gene_lists for g in genes})
    if not all_genes:
        print("[AMR] Tidak ada AMR gene ditemukan.")
        return pd.DataFrame()

    rows = {
        acc: {f"amr_{g}": 1 for g in genes}
        for acc, genes in gene_lists.items()
    }
    df = (
        pd.DataFrame(rows)
        .T
        .reindex(columns=[f"amr_{g}" for g in all_genes])
        .fillna(0)
        .astype(int)
    )
    df.index.name = "assembly_accession"
    print(f"[AMR] Features: {df.shape}  ({len(all_genes)} gen unik: {all_genes})")
    return df
