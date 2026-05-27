"""Orchestrate DNABERT-2 encoding with NPZ checkpoint cache."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils.checkpoint import save_embeddings, load_embeddings
from .model import load_model
from .encoder import embed_isolate, embed_isolate_stat, embed_isolate_windows


def _resolve_window_cache_path(cfg: dict) -> Path:
    """Return the absolute path for the per-window embeddings NPZ cache."""
    artifacts = cfg.get("artifacts", {})
    rel = artifacts.get("window_embeddings_path", "artifacts/embeddings/dnabert_window_embeddings.npz")
    return Path(rel)


def _resolve_cache_path(cfg: dict) -> Path:
    """Return the absolute path for the embeddings NPZ cache."""
    artifacts = cfg.get("artifacts", {})
    rel = artifacts.get("embeddings_path", "artifacts/embeddings/dnabert_embeddings.npz")
    return Path(rel)


def _resolve_stat_cache_path(cfg: dict) -> Path:
    """Return the absolute path for the stat (mean+max+std) embeddings NPZ cache."""
    artifacts = cfg.get("artifacts", {})
    rel = artifacts.get("stat_embeddings_path", "artifacts/embeddings/dnabert_stat_embeddings.npz")
    return Path(rel)


def generate_embeddings(windows_dict: dict, cfg: dict) -> pd.DataFrame:
    """
    Generate mean-pooled DNABERT-2 embeddings for each isolate.

    Cache behaviour
    ---------------
    Embeddings are stored as a compressed .npz file at
    artifacts/embeddings/dnabert_embeddings.npz (or the path in config).
    On the next run the cache is returned immediately if it covers all
    isolates in windows_dict — skipping the BERT forward pass entirely.

    To force regeneration set  cfg['pipeline']['force_recompute'] = True
    or delete the .npz file manually.
    """
    force = cfg.get("pipeline", {}).get("force_recompute", False)
    cache_path = _resolve_cache_path(cfg)

    if cache_path.exists() and not force:
        df = load_embeddings(cache_path)
        missing = set(windows_dict.keys()) - set(df.index)
        if not missing:
            print(f"[LOAD]    dnabert_embeddings.npz  ← {cache_path}  shape={df.shape}")
            return df
        print(f"[CACHE]   Cache tidak lengkap ({len(missing)} isolat hilang), regenerate...")

    # ── Run BERT inference ──────────────────────────────────────────────────
    device    = cfg["dnabert"]["device"]
    tokenizer, model = load_model(cfg["dnabert"]["model_id"], cfg["dnabert"]["cache_dir"])
    model = model.to(device)

    rows: dict[str, object] = {}
    total   = len(windows_dict)
    skipped = 0
    for i, (acc, windows) in enumerate(windows_dict.items(), 1):
        if not windows:
            print(f"  [{i}/{total}] {acc}: tidak ada window valid — dilewati")
            skipped += 1
            continue
        print(f"  [{i}/{total}] Embedding {acc}  ({len(windows)} windows)...")
        rows[acc] = embed_isolate(windows, tokenizer, model, device)

    if not rows:
        raise RuntimeError("Tidak ada isolat yang berhasil di-embed. Periksa windows_dict.")

    df = pd.DataFrame(rows).T
    df.index.name = "assembly_accession"
    df.columns = [f"dim_{i}" for i in range(df.shape[1])]
    print(f"Embedding selesai: {df.shape}  (dilewati: {skipped})")

    # ── Persist NPZ cache ───────────────────────────────────────────────────
    save_embeddings(df, cache_path)

    return df


def generate_stat_embeddings(windows_dict: dict, cfg: dict) -> pd.DataFrame:
    """
    Generate mean+max+std-pooled DNABERT-2 embeddings for each isolate (2304-dim).

    Uses a separate NPZ cache at stat_embeddings_path so it coexists with the
    standard mean-pooled embeddings.  Cache behaviour mirrors generate_embeddings.
    """
    force = cfg.get("pipeline", {}).get("force_recompute", False)
    cache_path = _resolve_stat_cache_path(cfg)

    if cache_path.exists() and not force:
        df = load_embeddings(cache_path)
        missing = set(windows_dict.keys()) - set(df.index)
        if not missing:
            print(f"[LOAD]    dnabert_stat_embeddings.npz  ← {cache_path}  shape={df.shape}")
            return df
        print(f"[CACHE]   Stat-cache tidak lengkap ({len(missing)} isolat hilang), regenerate...")

    device    = cfg["dnabert"]["device"]
    tokenizer, model = load_model(cfg["dnabert"]["model_id"], cfg["dnabert"]["cache_dir"])
    model = model.to(device)

    rows: dict[str, object] = {}
    total   = len(windows_dict)
    skipped = 0
    for i, (acc, windows) in enumerate(windows_dict.items(), 1):
        if not windows:
            print(f"  [{i}/{total}] {acc}: tidak ada window valid — dilewati")
            skipped += 1
            continue
        print(f"  [{i}/{total}] Stat-embedding {acc}  ({len(windows)} windows)...")
        rows[acc] = embed_isolate_stat(windows, tokenizer, model, device)

    if not rows:
        raise RuntimeError("Tidak ada isolat yang berhasil di-embed (stat). Periksa windows_dict.")

    df = pd.DataFrame(rows).T
    df.index.name = "assembly_accession"
    df.columns = [f"stat_dim_{i}" for i in range(df.shape[1])]
    print(f"Stat-embedding selesai: {df.shape}  (dilewati: {skipped})")

    save_embeddings(df, cache_path)
    return df


def generate_window_embeddings(windows_dict: dict, cfg: dict) -> dict:
    """
    Generate per-window DNABERT-2 embeddings for each isolate.

    Returns {accession: np.ndarray of shape (n_windows, 768)}.
    This ragged dict is the input for the Attention MIL classifier.

    Cache format: NPZ with accession strings as keys and 2D float32 arrays
    as values.  Accession names with slashes are replaced with underscores
    to be valid NPZ keys.

    Cache behaviour mirrors generate_embeddings — respects force_recompute.
    """
    force = cfg.get("pipeline", {}).get("force_recompute", False)
    cache_path = _resolve_window_cache_path(cfg)

    # ── Try loading from cache ──────────────────────────────────────────────
    def _safe_key(acc: str) -> str:
        return acc.replace("/", "_").replace("\\", "_")

    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=False)
        cached_accs = set(data.files)
        expected    = {_safe_key(a) for a in windows_dict}
        if expected <= cached_accs:
            result = {acc: data[_safe_key(acc)] for acc in windows_dict}
            print(f"[LOAD]    dnabert_window_embeddings.npz  ← {cache_path}  ({len(result)} isolat)")
            return result
        print(f"[CACHE]   Window-cache tidak lengkap, regenerate...")

    # ── Run BERT inference ──────────────────────────────────────────────────
    device    = cfg["dnabert"]["device"]
    tokenizer, model = load_model(cfg["dnabert"]["model_id"], cfg["dnabert"]["cache_dir"])
    model = model.to(device)

    rows: dict[str, object] = {}
    total   = len(windows_dict)
    skipped = 0
    for i, (acc, windows) in enumerate(windows_dict.items(), 1):
        if not windows:
            print(f"  [{i}/{total}] {acc}: tidak ada window valid — dilewati")
            skipped += 1
            continue
        print(f"  [{i}/{total}] Window-embedding {acc}  ({len(windows)} windows)...")
        rows[acc] = embed_isolate_windows(windows, tokenizer, model, device)

    if not rows:
        raise RuntimeError("Tidak ada isolat yang berhasil di-embed (window). Periksa windows_dict.")

    print(f"Window-embedding selesai: {len(rows)} isolat  (dilewati: {skipped})")

    # ── Persist NPZ cache ───────────────────────────────────────────────────
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), **{_safe_key(acc): arr for acc, arr in rows.items()})
    print(f"[SAVE]    {cache_path}")

    return rows
