"""
Stage-level checkpoint utilities.

Pattern everywhere:
    result = load_or_compute(path, compute_fn, save_fn, load_fn, force=FORCE)

Supported artifact formats:
  • Parquet  — metadata, SNP matrix (tabular, fast, typed)
  • CSV      — sequence QC report, distance matrix (human-readable)
  • NPZ      — DNABERT embeddings (compressed float array + index)
  • Joblib   — scikit-learn models
  • JSON     — metrics, manifest, run config
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core pattern
# ---------------------------------------------------------------------------

def load_or_compute(path, compute_fn, save_fn, load_fn, force: bool = False):
    """
    Checkpoint wrapper for a single pipeline stage.

    Args:
        path:       artifact file path (str or Path)
        compute_fn: zero-argument callable that returns the artifact
        save_fn:    callable(artifact, path) that persists the artifact
        load_fn:    callable(path) that restores the artifact
        force:      if True, always recompute even if file exists

    Returns:
        The artifact (loaded or freshly computed).
    """
    path = Path(path)
    if path.exists() and not force:
        print(f"[LOAD]    {path.name}  ← {path}")
        return load_fn(path)

    print(f"[COMPUTE] {path.name}")
    result = compute_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_fn(result, path)
    print(f"[SAVE]    {path}")
    return result


# ---------------------------------------------------------------------------
# Parquet  (metadata, SNP matrix — tabular)
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True, compression="snappy")


def load_parquet(path) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# CSV  (sequence QC, distance matrix — human-readable)
# ---------------------------------------------------------------------------

def save_csv_ckpt(df: pd.DataFrame, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def load_csv_ckpt(path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


# ---------------------------------------------------------------------------
# NPZ  (DNABERT embeddings — compressed float32 + accession index)
# ---------------------------------------------------------------------------

def save_embeddings(df: pd.DataFrame, path) -> None:
    """
    Save embedding DataFrame to a compressed .npz file.

    Stored arrays:
        embeddings  — float32, shape (n_isolates, dim)
        isolate_ids — str array of assembly_accession values
        columns     — str array of column names (dim_0..dim_767)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        embeddings=df.values.astype(np.float32),
        isolate_ids=np.array(df.index.tolist(), dtype=object),
        columns=np.array(df.columns.tolist(), dtype=object),
    )


def load_embeddings(path) -> pd.DataFrame:
    """Restore embedding DataFrame from .npz, preserving index and column names."""
    data = np.load(str(path), allow_pickle=True)
    df = pd.DataFrame(
        data["embeddings"].astype(np.float64),
        index=data["isolate_ids"].tolist(),
        columns=data["columns"].tolist(),
    )
    df.index.name = "assembly_accession"
    return df


# ---------------------------------------------------------------------------
# Joblib  (scikit-learn models)
# ---------------------------------------------------------------------------

def save_model(model, path) -> None:
    """
    Persist a scikit-learn estimator with joblib.

    Security note: only load models from trusted sources — joblib/pickle
    can execute arbitrary code during deserialization.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(path))
    print(f"[MODEL]   {path}")


def load_model(path):
    return joblib.load(str(path))


# ---------------------------------------------------------------------------
# JSON  (metrics, manifest)
# ---------------------------------------------------------------------------

def save_json(data: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def save_manifest(
    cfg: dict,
    metadata_df: pd.DataFrame,
    snp_df: pd.DataFrame,
    all_metrics: dict,
    path,
) -> None:
    """
    Write a manifest.json that captures the full provenance of a run.
    Useful as an audit trail and for the final report.

    Best mode is selected by balanced_accuracy (handles class imbalance better
    than plain accuracy).  Each mode's train_ids / test_ids are preserved for
    reproducibility verification.
    """
    valid = {k: v for k, v in all_metrics.items() if v.get("balanced_accuracy", 0) > 0}
    best_mode = max(valid, key=lambda k: valid[k]["balanced_accuracy"], default=None)
    best_m = all_metrics.get(best_mode, {}) if best_mode else {}

    manifest = {
        "project": cfg.get("project", {}).get("name", "SalmoTrace-BERT"),
        "organism": cfg.get("metadata", {}).get("organism", "Salmonella enterica"),
        "dataset_source": "NCBI Pathogen Detection",
        "num_isolates": int(len(metadata_df)),
        "reference_genome": cfg["data"]["reference_genome"],
        "serovar": (
            metadata_df["serovar"].value_counts().index[0]
            if "serovar" in metadata_df.columns and len(metadata_df) > 0
            else "N/A"
        ),
        "snp_positions": int(snp_df.shape[1]),
        "dnabert_model": cfg["dnabert"]["model_id"],
        "features": ["SNP", "DNABERT-2"],
        "model": cfg["ml"]["model"],
        "n_estimators": cfg["ml"].get("n_estimators", 100),
        "target_col": cfg["ml"]["target_col"],
        "group_col": "snp_cluster",
        "split_method": "StratifiedGroupKFold → GroupShuffleSplit → Stratified → Random",
        "test_size": cfg["ml"].get("test_size", 0.2),
        "random_state": int(cfg["ml"]["random_state"]),
        "best_feature_mode": best_mode,
        "best_balanced_accuracy": round(best_m.get("balanced_accuracy", 0), 4),
        "best_f1_macro": round(best_m.get("f1_macro", 0), 4),
        "best_f1_weighted": round(best_m.get("f1_weighted", 0), 4),
        "all_modes": {
            k: {
                "f1_macro":          round(v.get("f1_macro", 0), 4),
                "balanced_accuracy": round(v.get("balanced_accuracy", 0), 4),
                "f1_weighted":       round(v.get("f1_weighted", 0), 4),
                "split_type":        v.get("split_type", "-"),
                "n_train":           len(v.get("train_ids", [])),
                "n_test":            len(v.get("test_ids", [])),
            }
            for k, v in all_metrics.items()
        },
        "created_at": datetime.now().isoformat(),
    }
    save_json(manifest, path)
    print(f"[MANIFEST] {path}")
