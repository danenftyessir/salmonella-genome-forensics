"""Load raw metadata CSV and FASTA genome files from NCBI."""

import os
import pandas as pd
from utils.io import read_fasta


REQUIRED_COLUMNS = [
    "assembly_accession", "isolate_name", "serovar",
    "isolation_source", "geo_loc_name", "collection_date",
    "snp_cluster", "amr_genes",
]


def load_metadata(metadata_path: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan di metadata: {missing}")
    print(f"Metadata dimuat: {len(df)} baris dari {metadata_path}")
    return df


def load_genomes(genomes_dir: str, accessions: list) -> tuple[dict, dict]:
    """
    Return (genomes, contig_counts) where:
      genomes       = {accession: concatenated_sequence_str}
      contig_counts = {accession: int}
    Used by: DNABERT sliding-window extraction (needs single string per isolate).
    """
    genomes: dict[str, str] = {}
    contig_counts: dict[str, int] = {}
    for acc in accessions:
        fasta_path = os.path.join(genomes_dir, f"{acc}.fna")
        if os.path.exists(fasta_path):
            records = read_fasta(fasta_path)
            genomes[acc] = "".join(records.values())
            contig_counts[acc] = len(records)
        else:
            print(f"[WARN] FASTA tidak ditemukan: {fasta_path}")
    print(f"Genome dimuat: {len(genomes)} dari {len(accessions)} accession")
    return genomes, contig_counts


def load_genomes_contigs(genomes_dir: str, accessions: list) -> dict[str, list[str]]:
    """
    Return {accession: [contig_seq, ...]} keeping contigs separate.

    Used by: build_core_snp_matrix — the k-mer alignment aligns each contig
    individually to the reference, which is necessary for multi-contig
    assemblies (contigs have arbitrary ordering relative to the reference).
    """
    genomes: dict[str, list[str]] = {}
    for acc in accessions:
        fasta_path = os.path.join(genomes_dir, f"{acc}.fna")
        if os.path.exists(fasta_path):
            records = read_fasta(fasta_path)
            genomes[acc] = [seq.upper() for seq in records.values()]
        else:
            print(f"[WARN] FASTA tidak ditemukan: {fasta_path}")
    print(f"Genome contigs dimuat: {len(genomes)} dari {len(accessions)} accession")
    return genomes


def load_reference_genome(genomes_dir: str, ref_accession: str) -> str:
    """
    Load the reference genome and return as a single concatenated string.

    Parameters
    ----------
    genomes_dir    : Directory containing *.fna files.
    ref_accession  : NCBI accession of the reference (e.g. 'GCF_000006945.2').

    Returns full sequence (chromosome + plasmids concatenated, upper-cased).
    """
    path = os.path.join(genomes_dir, f"{ref_accession}.fna")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Reference genome tidak ditemukan: {path}\n"
            f"Pastikan {ref_accession}.fna ada di {genomes_dir}"
        )
    records = read_fasta(path)
    seq = "".join(records.values()).upper()
    print(f"Reference genome dimuat: {ref_accession} ({len(seq):,} bp, "
          f"{len(records)} contig(s))")
    return seq
