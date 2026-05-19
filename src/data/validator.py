"""Validate accessions, genome quality, and filter by configurable thresholds."""

import pandas as pd
from utils.seq import gc_content


def filter_metadata(
    df: pd.DataFrame,
    serovar: str = None,
    source: str = None,
    min_isolates: int = None,
    max_isolates: int = None,
) -> pd.DataFrame:
    if serovar:
        df = df[df["serovar"].str.contains(serovar, case=False, na=False)]
    if source:
        df = df[df["isolation_source"].str.contains(source, case=False, na=False)]
    if max_isolates and len(df) > max_isolates:
        df = df.sample(n=max_isolates, random_state=42)
    if min_isolates and len(df) < min_isolates:
        raise ValueError(f"Isolat terlalu sedikit setelah filter: {len(df)} < {min_isolates}")
    return df.reset_index(drop=True)


def validate_accessions(df: pd.DataFrame, genomes: dict) -> pd.DataFrame:
    valid = df["assembly_accession"].isin(genomes.keys())
    n_missing = (~valid).sum()
    if n_missing > 0:
        print(f"[WARN] {n_missing} accession tidak punya FASTA, dilewati.")
    return df[valid].reset_index(drop=True)


def check_genome_size(genomes: dict, min_bp: int = 4_000_000, max_bp: int = 6_000_000) -> dict:
    """Return only genomes within expected Salmonella size range (~4.8 Mb)."""
    filtered = {}
    for acc, seq in genomes.items():
        if min_bp <= len(seq) <= max_bp:
            filtered[acc] = seq
        else:
            print(f"[WARN] {acc} ukuran tidak wajar: {len(seq):,} bp — dilewati")
    return filtered


def genome_qc_report(
    genomes: dict,
    contig_counts: dict,
    min_bp: int = 4_000_000,
    max_bp: int = 6_000_000,
    max_n_frac: float = 0.05,
    max_contigs: int = 500,
) -> pd.DataFrame:
    """
    Compute per-isolate QC statistics and assign pass/fail.

    Columns: assembly_accession, genome_length_bp, gc_content,
             n_fraction, contig_count, qc_pass.
    """
    rows = []
    for acc, seq in genomes.items():
        length = len(seq)
        seq_upper = seq.upper()
        gc = gc_content(seq_upper)
        n_frac = seq_upper.count("N") / length if length > 0 else 1.0
        n_contigs = contig_counts.get(acc, 0)
        qc_pass = (
            min_bp <= length <= max_bp
            and n_frac <= max_n_frac
            and n_contigs <= max_contigs
        )
        rows.append({
            "assembly_accession": acc,
            "genome_length_bp": length,
            "gc_content": round(gc, 4),
            "n_fraction": round(n_frac, 6),
            "contig_count": n_contigs,
            "qc_pass": qc_pass,
        })

    df = pd.DataFrame(rows)
    n_pass = int(df["qc_pass"].sum())
    n_fail = len(df) - n_pass
    print(f"Genome QC: {n_pass} lulus / {n_fail} gagal dari {len(df)} isolat")

    if n_fail:
        failed = df.loc[~df["qc_pass"], ["assembly_accession", "genome_length_bp",
                                          "n_fraction", "contig_count"]]
        for _, row in failed.iterrows():
            print(
                f"  [QC FAIL] {row['assembly_accession']}: "
                f"len={row['genome_length_bp']:,}bp  "
                f"N={row['n_fraction']:.2%}  "
                f"contigs={row['contig_count']}"
            )
    return df


def filter_by_qc(genomes: dict, qc_df: pd.DataFrame) -> dict:
    """Remove isolates that did not pass QC."""
    passed = set(qc_df.loc[qc_df["qc_pass"], "assembly_accession"])
    removed = [acc for acc in genomes if acc not in passed]
    for acc in removed:
        print(f"[QC REMOVE] {acc}")
    return {acc: seq for acc, seq in genomes.items() if acc in passed}
