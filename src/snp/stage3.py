"""
Stage 3 orchestrator: load SNP matrix from the best available source.

Priority order:
  1. core_clean.aln  (snippy-core output after post-QC filter)
  2. core.aln        (raw snippy-core output, all isolates)
  3. k-mer fallback  (exact 21-mer, less accurate, no RC support)

Called from the notebook cell so that the notebook contains minimal code
and is not affected by VS Code autosave overwriting our changes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .snippy_parser import load_snippy_core_snps
from .extractor import build_core_snp_matrix
from .filter import filter_snp_positions, remove_high_n_columns


def build_snp_stage3(
    cfg: dict,
    genomes_contigs: dict,
    reference_seq: str,
    project_root: str,
) -> pd.DataFrame:
    """
    Load the SNP matrix for Stage 3.

    Parameters
    ----------
    cfg             : full config dict (from load_config)
    genomes_contigs : {accession: [contig, ...]}  — needed for k-mer fallback only
    reference_seq   : reference genome string      — needed for k-mer fallback only
    project_root    : absolute path to project root (used to resolve relative paths)

    Returns
    -------
    DataFrame (isolates × variable SNP positions), character-encoded (A/T/G/C).
    """
    snp_cfg = cfg["snp"]

    def _abs(rel: str) -> str:
        return str(Path(project_root) / rel)

    core_clean = _abs(snp_cfg.get("snippy_core_clean_aln", "data/processed/snippy/core_clean.aln"))
    core_raw   = _abs(snp_cfg.get("snippy_core_aln",       "data/processed/snippy/core.aln"))

    # ── Choose alignment source ───────────────────────────────────────────────
    if os.path.exists(core_clean):
        print(f"[SNP] Menggunakan core_clean.aln (post-QC): {core_clean}")
        aln_path = core_clean
    elif os.path.exists(core_raw):
        print(f"[WARN] core_clean.aln tidak ditemukan — pakai core.aln (semua isolat, mungkin ada outlier).")
        print(f"       Jalankan: python scripts/snippy_qc_filter.py  untuk core_clean.aln")
        aln_path = core_raw
    else:
        print(f"[WARN] Tidak ada Snippy alignment ditemukan.")
        print(f"       Jalankan: bash scripts/run_snippy.sh")
        print(f"       Lalu    : python scripts/snippy_qc_filter.py")
        print(f"[WARN] Fallback ke exact k-mer caller (tidak handle reverse-complement)...")
        df = build_core_snp_matrix(
            genomes_contigs,
            reference_seq,
            k                       = snp_cfg["k"],
            min_core_fraction       = snp_cfg["min_core_fraction"],
            min_isolate_callability = snp_cfg["min_isolate_callability"],
        )
        return _apply_filters(df)

    df = load_snippy_core_snps(aln_path, drop_n=True, drop_gaps=True)
    print(f"[SNP] Setelah load alignment : {df.shape}")
    return _apply_filters(df)


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df.shape[1] == 0:
        return df
    df = remove_high_n_columns(df)
    print(f"[SNP] Setelah remove_high_n  : {df.shape}")
    df_maf = filter_snp_positions(df)
    print(f"[SNP] Setelah filter MAF      : {df_maf.shape}")
    if df_maf.shape[1] == 0 and df.shape[1] > 0:
        print("[WARN] MAF filter menghapus semua posisi — gunakan hasil sebelum MAF.")
        return df
    return df_maf
