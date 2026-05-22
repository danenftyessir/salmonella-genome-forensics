"""
Parse Snippy core-genome SNP alignment output into the same
(isolates × variable-positions) DataFrame format produced by
build_core_snp_matrix().

Typical snippy-core outputs consumed here:
  core.aln      — multi-FASTA, variable SNP positions only  (preferred)
  core.full.aln — multi-FASTA, full core-genome alignment   (larger, slower)

Usage
-----
from snp.snippy_parser import load_snippy_core_snps

snp_df = load_snippy_core_snps("data/processed/snippy/core.aln")
# Returns DataFrame: index = accession, columns = 0-based integer SNP indices,
#                    values = 'A'/'T'/'G'/'C'/'N'
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from Bio import AlignIO as _AlignIO
    _BIOPYTHON = True
except ImportError:
    _BIOPYTHON = False


# Characters that mark an uncallable or gapped position in snippy output
_SKIP_CHARS: frozenset[str] = frozenset({"N", "-", "n"})


def load_snippy_core_snps(
    core_aln_path: str | Path,
    ref_id: str = "Reference",
    drop_n: bool = True,
    drop_gaps: bool = True,
    min_isolate_presence: float = 0.0,
) -> pd.DataFrame:
    """
    Parse a snippy-core alignment file into a character SNP DataFrame.

    Parameters
    ----------
    core_aln_path        : path to core.aln (SNP-only) or core.full.aln
    ref_id               : row label snippy uses for the reference sequence;
                           it is excluded from the output DataFrame
    drop_n               : drop columns where any isolate has 'N' (uncallable)
    drop_gaps            : drop columns where any isolate has '-' (insertion gap)
    min_isolate_presence : drop isolates present in < this fraction of columns
                           (0.0 = keep all; raise to ~0.5 to filter divergent)

    Returns
    -------
    DataFrame  (index = assembly_accession,
                columns = integer 1..n_snps,
                values  = 'A'/'T'/'G'/'C')

    Raises
    ------
    ImportError    if biopython is not installed
    FileNotFoundError if the alignment file does not exist
    """
    if not _BIOPYTHON:
        raise ImportError(
            "BioPython is required to parse Snippy alignments.\n"
            "Install: pip install biopython   OR   conda install biopython"
        )

    path = Path(core_aln_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Snippy core alignment not found: {path}\n"
            "Run scripts/run_snippy.sh first to generate the alignment.\n"
            "See README / Stage 3 notes for installation instructions."
        )

    print(f"[SNIPPY] Membaca alignment: {path}")
    alignment = _AlignIO.read(str(path), "fasta")

    records: dict[str, list[str]] = {}
    for rec in alignment:
        if rec.id == ref_id:
            continue
        records[rec.id] = list(str(rec.seq).upper())

    if not records:
        print(f"[WARN] Alignment tidak mengandung isolat (hanya '{ref_id}'?)")
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "assembly_accession"
    print(f"[SNIPPY] Raw alignment  : {df.shape[0]} isolat × {df.shape[1]:,} posisi")

    # ── Remove positions where any isolate is N or gap ────────────────────────
    if drop_n:
        n_mask = (df == "N").any(axis=0)
        df = df.loc[:, ~n_mask]
        print(f"[SNIPPY] Setelah drop-N : {df.shape[1]:,} posisi")

    if drop_gaps:
        gap_mask = (df == "-").any(axis=0)
        df = df.loc[:, ~gap_mask]
        print(f"[SNIPPY] Setelah drop-gap: {df.shape[1]:,} posisi")

    # ── Filter low-presence isolates ──────────────────────────────────────────
    if min_isolate_presence > 0.0:
        bad_chars = {"N", "-"}
        presence = df.apply(lambda row: (~row.isin(bad_chars)).mean(), axis=1)
        keep = presence >= min_isolate_presence
        n_dropped = int((~keep).sum())
        if n_dropped:
            print(
                f"[SNIPPY] Dropping {n_dropped} isolat dengan kehadiran "
                f"< {min_isolate_presence:.0%} di core positions"
            )
        df = df.loc[keep]

    # ── Keep only variable sites ──────────────────────────────────────────────
    # core.aln from snippy-core should already be SNP-only, but re-check
    variable_mask = df.nunique(axis=0) > 1
    n_invariant = int((~variable_mask).sum())
    if n_invariant:
        print(f"[SNIPPY] Dropping {n_invariant} invariant columns")
    df = df.loc[:, variable_mask]

    # ── Relabel columns as 1-based integers ──────────────────────────────────
    df.columns = range(1, len(df.columns) + 1)

    print(
        f"[SNIPPY] Final core-SNP matrix: "
        f"{df.shape[0]} isolat × {df.shape[1]:,} posisi variabel"
    )

    if df.shape[1] == 0:
        print(
            "[WARN] SNP matrix masih kosong setelah parsing.\n"
            "  Kemungkinan: snippy-core tidak menemukan core SNP di dataset ini.\n"
            "  Coba periksa core.txt untuk diagnosis jumlah isolat / core positions."
        )

    return df
