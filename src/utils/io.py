"""File I/O helpers: FASTA reading, CSV writing, directory creation."""

import os
import pandas as pd
from Bio import SeqIO


def read_fasta(fasta_path: str) -> dict:
    """Return {record_id: sequence_str} from a FASTA file."""
    records = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        records[record.id] = str(record.seq)
    return records


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
