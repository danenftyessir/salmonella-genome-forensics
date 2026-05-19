"""Clean sequences, sliding-window extraction, SNP-context windows, GC stats."""

import os
import numpy as np
from utils.seq import validate_sequence, gc_content, has_ambiguous
from utils.io import ensure_dir


# ---------------------------------------------------------------------------
# Sequence cleaning
# ---------------------------------------------------------------------------

def clean_sequence(seq: str) -> str:
    return validate_sequence(seq)


# ---------------------------------------------------------------------------
# Window filtering
# ---------------------------------------------------------------------------

def filter_windows_by_n(windows: list[str], max_n_frac: float = 0.10) -> list[str]:
    """Drop windows where the fraction of N bases exceeds max_n_frac."""
    kept = []
    for w in windows:
        n_frac = w.upper().count("N") / max(len(w), 1)
        if n_frac <= max_n_frac:
            kept.append(w)
    return kept


# ---------------------------------------------------------------------------
# Sliding windows (fallback when no SNP positions available)
# ---------------------------------------------------------------------------

def sliding_windows(seq: str, window_size: int = 512, step: int = 256) -> list[tuple[int, str]]:
    """Return list of (start_pos, window_str) tuples."""
    return [
        (i, seq[i:i + window_size])
        for i in range(0, len(seq) - window_size + 1, step)
    ]


def sample_windows(windows: list, max_windows: int) -> list:
    """Evenly subsample windows if count exceeds cap."""
    if len(windows) <= max_windows:
        return windows
    idx = np.linspace(0, len(windows) - 1, max_windows, dtype=int)
    return [windows[i] for i in idx]


def extract_windows(
    genomes: dict,
    window_size: int = 512,
    max_windows: int = 50,
    max_n_frac: float = 0.10,
) -> dict[str, list[str]]:
    """
    Sliding-window extraction.
    Returns {accession: [window_str, ...]} cleaned, N-filtered, and capped.
    """
    result = {}
    for acc, seq in genomes.items():
        seq = clean_sequence(seq)
        wins = sliding_windows(seq, window_size)
        wins = [w for _, w in wins]
        wins = filter_windows_by_n(wins, max_n_frac)
        wins = sample_windows(wins, max_windows)
        result[acc] = wins
    return result


# ---------------------------------------------------------------------------
# SNP-context windows (preferred for DNABERT input)
# ---------------------------------------------------------------------------

def extract_snp_context_windows(
    genomes: dict,
    snp_positions: list[int],
    flank: int = 100,
    max_windows: int = 50,
    max_n_frac: float = 0.10,
) -> dict[str, list[str]]:
    """
    For each isolate extract flanking sequences around SNP positions.

    Each window spans [pos - flank, pos + flank + 1], clipped to genome
    boundaries.  Windows with too many N bases are dropped, then the list
    is subsampled evenly to at most max_windows entries.

    Rationale: SNP-flanking regions carry phylogenetic signal and are
    directly relevant to source attribution, giving DNABERT more
    discriminative input than arbitrary sliding windows.
    """
    result = {}
    for acc, seq in genomes.items():
        seq = clean_sequence(seq)
        contexts: list[str] = []
        for pos in snp_positions:
            start = max(0, pos - flank)
            end = min(len(seq), pos + flank + 1)
            w = seq[start:end]
            if w:
                contexts.append(w)
        contexts = filter_windows_by_n(contexts, max_n_frac)
        contexts = sample_windows(contexts, max_windows)
        result[acc] = contexts
    return result


# ---------------------------------------------------------------------------
# Backward-compat alias used in older code paths
# ---------------------------------------------------------------------------

def extract_snp_context(seq: str, snp_positions: list[int], flank: int = 100) -> list[str]:
    """Return flanking sequences around each SNP position for a single isolate."""
    seq = clean_sequence(seq)
    contexts = []
    for pos in snp_positions:
        start = max(0, pos - flank)
        end = min(len(seq), pos + flank + 1)
        contexts.append(seq[start:end])
    return contexts


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_gc_stats(genomes: dict) -> dict[str, float]:
    """Return {accession: gc_content_float}."""
    return {acc: gc_content(seq) for acc, seq in genomes.items()}


def flag_low_quality(genomes: dict, ambiguous_threshold: float = 0.05) -> list[str]:
    """Return list of accessions with too many N bases."""
    return [acc for acc, seq in genomes.items() if has_ambiguous(seq, ambiguous_threshold)]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_windows(windows: dict, out_dir: str) -> None:
    ensure_dir(out_dir)
    for acc, wins in windows.items():
        np.save(os.path.join(out_dir, f"{acc}.npy"), np.array(wins, dtype=object))
    print(f"Windows disimpan ke: {out_dir}")
