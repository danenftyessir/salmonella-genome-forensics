"""Orchestrate DNABERT-2 encoding with NPZ checkpoint cache."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from utils.checkpoint import save_embeddings, load_embeddings
from .model import load_model
from .encoder import embed_isolate


def _resolve_cache_path(cfg: dict) -> Path:
    """Return the absolute path for the embeddings NPZ cache."""
    artifacts = cfg.get("artifacts", {})
    rel = artifacts.get("embeddings_path", "artifacts/embeddings/dnabert_embeddings.npz")
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
