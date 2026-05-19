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
