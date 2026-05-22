"""Experiment tracking utilities for SalmoTrace-BERT.

Two-layer tracking:
  Layer 1 (always on) — local CSV + enriched metrics.json with bio_interpretation.
  Layer 2 (opt-in)    — MLflow local tracking (set tracking.mlflow_enabled: true).

Usage
-----
from utils.tracking import enrich_metrics, save_model_comparison, log_mlflow_all_modes

# After run_pipeline() returns all_metrics:
bio_ctx = {"w_m": 450.2, "b_m": 3200.1, "ari_e1": 0.38}
all_metrics = enrich_metrics(all_metrics, bio_ctx)
save_model_comparison(all_metrics, cfg, path="artifacts/reports/model_comparison.csv")
log_mlflow_all_modes(all_metrics, cfg, artifact_dir="outputs/figures")
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Biological interpretation (concise, 1-2 sentences per mode)
# ---------------------------------------------------------------------------

_MODE_LABELS = {
    "snp_only":    "E2  SNP-only",
    "dnabert_only":"E3  DNABERT-only",
    "hybrid":      "E4  SNP + DNABERT",
    "kmer_only":   "E5  K-mer",
}

_FEATURE_LABELS = {
    "snp_only":    "SNP matrix (integer-encoded)",
    "dnabert_only":"DNABERT-2 mean-pooled embedding",
    "hybrid":      "SNP matrix + DNABERT-2 embedding (concatenated)",
    "kmer_only":   "Tetranucleotide k-mer frequency (alignment-free)",
}


def _bio_e2(m: dict, all_metrics: dict) -> str:
    f1 = m.get("f1_macro", 0.0)
    if f1 >= 0.70:
        return (
            "SNP features carry strong source-discriminating signal, consistent with "
            "lineage-specific mutation patterns accumulating differently across isolation "
            "environments."
        )
    if f1 >= 0.40:
        e3_f1 = all_metrics.get("dnabert_only", {}).get("f1_macro", 0)
        overlap_note = (
            " Partial overlap between food and food-animal classes is biologically "
            "expected given shared supply-chain transmission routes."
            if e3_f1 > 0 else ""
        )
        return (
            f"SNP features show partial source signal (Macro F1={f1:.3f}). "
            "Class imbalance or genomic similarity across certain source groups "
            "limits consistent classification." + overlap_note
        )
    return (
        f"SNP features yield low performance (Macro F1={f1:.3f}) on this dataset. "
        "Possible causes: severe class imbalance, too few isolates per source, or "
        "genomic signatures of source that are not captured by SNP positions alone."
    )


def _bio_e3(m: dict, all_metrics: dict) -> str:
    f1    = m.get("f1_macro", 0.0)
    e2_f1 = all_metrics.get("snp_only", {}).get("f1_macro", float("nan"))
    if not math.isnan(e2_f1) and f1 > e2_f1 + 0.02:
        return (
            f"DNABERT embeddings (F1={f1:.3f}) outperform explicit SNP features, "
            "suggesting that contextual nucleotide patterns around variant sites "
            "encode source-relevant information beyond individual SNP positions."
        )
    if not math.isnan(e2_f1) and f1 < e2_f1 - 0.02:
        return (
            f"Explicit SNP alignment (F1={e2_f1:.3f}) outperforms general DNABERT "
            f"embeddings (F1={f1:.3f}). Without Salmonella-specific fine-tuning, "
            "DNABERT representations are insufficiently specialised for source attribution "
            "on this small dataset."
        )
    if f1 <= 0.10:
        return (
            f"DNABERT embeddings (F1={f1:.3f}) dan SNP-only menghasilkan performa yang sama rendah. "
            "Dengan Macro F1 di bawah 0.10, tidak ada mode yang menunjukkan kemampuan source "
            "attribution. Dataset terlalu kecil dan label terlalu granular untuk kesimpulan statistik."
        )
    return (
        f"DNABERT embeddings (F1={f1:.3f}) are comparable to SNP-only features. "
        "The transformer captures sequence context that partially compensates for "
        "the lack of explicit positional SNP information."
    )


def _bio_e4(m: dict, all_metrics: dict) -> str:
    f1     = m.get("f1_macro", 0.0)
    e2_f1  = all_metrics.get("snp_only",    {}).get("f1_macro", float("nan"))
    e3_f1  = all_metrics.get("dnabert_only",{}).get("f1_macro", float("nan"))
    base   = max((v for v in [e2_f1, e3_f1] if not math.isnan(v)), default=float("nan"))
    if not math.isnan(base) and f1 > base + 0.02:
        return (
            f"Hybrid SNP+DNABERT achieves the best performance (F1={f1:.3f}), "
            "supporting the hypothesis that positional SNP variants and contextual "
            "sequence embeddings provide complementary genomic signals for source attribution."
        )
    if not math.isnan(base) and f1 >= base - 0.02:
        return (
            f"Hybrid features (F1={f1:.3f}) are comparable to the best single-mode, "
            "suggesting that SNP signal dominates and DNABERT adds marginal information "
            "on this dataset. Fine-tuning DNABERT could improve the hybrid gain."
        )
    return (
        f"Hybrid performance (F1={f1:.3f}) falls below individual modes. "
        "High-dimensional concatenation may introduce noise in this small-sample regime; "
        "dimensionality reduction before concatenation could help."
    )


def _bio_e5(m: dict, all_metrics: dict) -> str:
    f1     = m.get("f1_macro", 0.0)
    e2_f1  = all_metrics.get("snp_only",    {}).get("f1_macro", float("nan"))
    e3_f1  = all_metrics.get("dnabert_only",{}).get("f1_macro", float("nan"))
    base   = max((v for v in [e2_f1, e3_f1] if not math.isnan(v)), default=float("nan"))
    if not math.isnan(base) and f1 >= base - 0.05:
        return (
            f"K-mer frequency (F1={f1:.3f}) is competitive with alignment-based features, "
            "indicating that global nucleotide composition already encodes "
            "source-relevant genomic signatures without alignment."
        )
    return (
        f"K-mer frequency (F1={f1:.3f}) underperforms alignment-based features, "
        "suggesting that source signal resides in specific variant positions rather "
        "than genome-wide nucleotide composition."
    )


_BIO_FN = {
    "snp_only":    _bio_e2,
    "dnabert_only":_bio_e3,
    "hybrid":      _bio_e4,
    "kmer_only":   _bio_e5,
}


def _bio_e1(w_m: float, b_m: float, sil: float, ari: float) -> str:
    sep = b_m / max(w_m, 1) if not math.isnan(w_m) and not math.isnan(b_m) else float("nan")
    parts = []
    if not math.isnan(sep):
        if sep >= 5:
            parts.append(
                f"Ward-linkage clustering cleanly separates genomic lineages "
                f"(within={w_m:.0f} vs between={b_m:.0f} SNP, {sep:.1f}× separation)."
            )
        elif sep >= 2:
            parts.append(
                f"Clustering shows moderate genomic separation "
                f"({sep:.1f}× between/within SNP distance ratio)."
            )
        elif sep <= 1.05:
            parts.append(
                f"Separation ratio sangat rendah ({sep:.2f}×): isolat dalam kluster dan "
                f"antar-kluster hampir sama jauhnya (within={w_m:.0f} vs between={b_m:.0f} SNP). "
                "Kluster yang terbentuk tidak bermakna secara biologis."
            )
        else:
            parts.append(
                f"Low genomic separation ({sep:.1f}×); dataset may be genomically homogeneous."
            )
    if not math.isnan(ari):
        if ari > 0.5:
            parts.append(f"ARI={ari:.3f}: local clustering highly consistent with NCBI SNP clusters.")
        elif ari > 0.1:
            parts.append(f"ARI={ari:.3f}: partial correspondence with NCBI SNP cluster labels.")
        else:
            parts.append(f"ARI={ari:.3f}: local cluster boundaries differ from NCBI assignments.")
    return " ".join(parts) if parts else "Hierarchical clustering completed; see dendrogram."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_metrics(
    all_metrics: dict,
    bio_context: dict | None = None,
) -> dict:
    """
    Add bio_interpretation string to each mode's metrics dict.

    bio_context (optional dict) may contain:
        w_m    — mean within-cluster SNP distance
        b_m    — mean between-cluster SNP distance
        ari_e1 — ARI of hierarchical clustering vs NCBI snp_cluster
        sil_e1 — Silhouette of hierarchical clustering

    Returns the same all_metrics dict, mutated in place, for convenience.
    """
    ctx = bio_context or {}
    for mode, m in all_metrics.items():
        fn = _BIO_FN.get(mode)
        if fn is not None:
            m["bio_interpretation"] = fn(m, all_metrics)
    return all_metrics


def save_model_comparison(
    all_metrics: dict,
    cfg: dict,
    path: str,
    run_id: str | None = None,
    bio_context: dict | None = None,
) -> pd.DataFrame:
    """
    Write / append model_comparison.csv.

    Each call appends one row per mode (identified by run_id + mode).
    Existing rows for the same run_id are replaced.

    Returns the full DataFrame (including prior runs if file existed).
    """
    ctx    = bio_context or {}
    rid    = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    w_m    = ctx.get("w_m",    float("nan"))
    b_m    = ctx.get("b_m",    float("nan"))
    ari_e1 = ctx.get("ari_e1", float("nan"))
    sil_e1 = ctx.get("sil_e1", float("nan"))
    sep    = b_m / max(w_m, 1) if not math.isnan(w_m) and not math.isnan(b_m) else float("nan")

    rows = []

    # E1 synthetic row (no ML metrics)
    rows.append({
        "run_id":            rid,
        "experiment":        "E1  SNP Clustering",
        "feature_set":       "SNP pairwise distance",
        "model":             "Hierarchical (Ward)",
        "macro_f1":          float("nan"),
        "balanced_accuracy": float("nan"),
        "f1_weighted":       float("nan"),
        "silhouette":        round(sil_e1, 4) if not math.isnan(sil_e1) else float("nan"),
        "ari_vs_ncbi":       round(ari_e1, 4) if not math.isnan(ari_e1) else float("nan"),
        "within_snp_dist":   round(w_m,    1) if not math.isnan(w_m)    else float("nan"),
        "between_snp_dist":  round(b_m,    1) if not math.isnan(b_m)    else float("nan"),
        "separation_ratio":  round(sep,    2) if not math.isnan(sep)     else float("nan"),
        "split_type":        "-",
        "n_train":           "-",
        "n_test":            "-",
        "random_state":      cfg["ml"]["random_state"],
        "model_type":        cfg["ml"]["model"],
        "n_estimators":      cfg["ml"].get("n_estimators", "N/A"),
        "target_col":        cfg["ml"]["target_col"],
        "bio_interpretation": _bio_e1(w_m, b_m, sil_e1, ari_e1),
    })

    for mode, m in all_metrics.items():
        sil = m.get("silhouette", float("nan"))
        rows.append({
            "run_id":            rid,
            "experiment":        _MODE_LABELS.get(mode, mode),
            "feature_set":       _FEATURE_LABELS.get(mode, mode),
            "model":             cfg["ml"]["model"],
            "macro_f1":          round(m.get("f1_macro",          0.0), 4),
            "balanced_accuracy": round(m.get("balanced_accuracy", 0.0), 4),
            "f1_weighted":       round(m.get("f1_weighted",       0.0), 4),
            "silhouette":        round(sil, 4) if not math.isnan(sil) else float("nan"),
            "ari_vs_ncbi":       float("nan"),
            "within_snp_dist":   float("nan"),
            "between_snp_dist":  float("nan"),
            "separation_ratio":  float("nan"),
            "split_type":        m.get("split_type", "-"),
            "n_train":           len(m.get("train_ids", [])),
            "n_test":            len(m.get("test_ids",  [])),
            "random_state":      cfg["ml"]["random_state"],
            "model_type":        cfg["ml"]["model"],
            "n_estimators":      cfg["ml"].get("n_estimators", "N/A"),
            "target_col":        cfg["ml"]["target_col"],
            "bio_interpretation": m.get("bio_interpretation", ""),
        })

    new_df = pd.DataFrame(rows)
    path   = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_csv(path)
        # Replace rows from same run_id to allow re-runs to update in place
        existing = existing[existing["run_id"] != rid]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(path, index=False)
    print(f"[TRACKING] model_comparison.csv  → {path}  ({len(combined)} rows total)")
    return combined


# ---------------------------------------------------------------------------
# MLflow integration (opt-in)
# ---------------------------------------------------------------------------

def log_mlflow_all_modes(
    all_metrics: dict,
    cfg: dict,
    artifact_dir: str | None = None,
    bio_context: dict | None = None,
    run_name_prefix: str = "",
) -> None:
    """
    Log every feature mode as a separate MLflow run under the configured experiment.

    No-op if:
      • mlflow is not installed, OR
      • cfg["tracking"]["mlflow_enabled"] is False / absent

    Each run logs:
      params  — feature_set, model, n_estimators, target_col, group_col,
                split_method, test_size, random_state
      metrics — macro_f1, balanced_accuracy, f1_weighted, silhouette,
                ari_e1 (E1 only), separation_ratio (E1 only)
      tags    — bio_interpretation
      artifacts — confusion matrix + ROC curve PNGs for the mode (if artifact_dir given)
    """
    tracking_cfg = cfg.get("tracking", {})
    if not tracking_cfg.get("mlflow_enabled", False):
        return

    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("[TRACKING] mlflow not installed — skipping MLflow logging.")
        return

    ctx        = bio_context or {}
    experiment = tracking_cfg.get("experiment_name", cfg.get("project", {}).get("name", "SalmoTrace-BERT"))
    uri        = tracking_cfg.get("tracking_uri", "mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)

    # Common params logged to every run
    common_params = {
        "model":        cfg["ml"]["model"],
        "n_estimators": cfg["ml"].get("n_estimators", 100),
        "target_col":   cfg["ml"]["target_col"],
        "group_col":    "snp_cluster",
        "split_method": "StratifiedGroupKFold → GroupShuffleSplit → Stratified → Random",
        "test_size":    cfg["ml"].get("test_size", 0.2),
        "random_state": cfg["ml"]["random_state"],
        "organism":     cfg.get("metadata", {}).get("organism", "Salmonella enterica"),
        "reference":    cfg["data"]["reference_genome"],
    }

    for mode, m in all_metrics.items():
        run_name = f"{run_name_prefix}{mode}" if run_name_prefix else mode
        with mlflow.start_run(run_name=run_name):
            # Params
            mlflow.log_params(common_params)
            mlflow.log_param("feature_set", _FEATURE_LABELS.get(mode, mode))

            # Metrics
            mlflow.log_metric("macro_f1",          round(m.get("f1_macro",          0.0), 4))
            mlflow.log_metric("balanced_accuracy",  round(m.get("balanced_accuracy", 0.0), 4))
            mlflow.log_metric("f1_weighted",        round(m.get("f1_weighted",       0.0), 4))
            sil = m.get("silhouette", float("nan"))
            if not math.isnan(sil):
                mlflow.log_metric("silhouette", round(sil, 4))

            # E1 biological metrics from context
            ari = ctx.get("ari_e1", float("nan"))
            if not math.isnan(ari):
                mlflow.log_metric("ari_vs_ncbi_snp_cluster", round(ari, 4))
            sep = (ctx.get("b_m", float("nan")) / max(ctx.get("w_m", float("nan")), 1)
                   if not math.isnan(ctx.get("w_m", float("nan"))) else float("nan"))
            if not math.isnan(sep):
                mlflow.log_metric("snp_separation_ratio", round(sep, 2))

            # Tag with biological interpretation
            bio = m.get("bio_interpretation", "")
            if bio:
                mlflow.set_tag("bio_interpretation", bio[:500])  # MLflow tag limit

            # Artifact figures
            if artifact_dir:
                fig_dir = Path(artifact_dir) / "classification"
                for suffix in ["_confusion_matrix.png", "_roc_curve.png"]:
                    fig_path = fig_dir / f"{mode}{suffix}"
                    if fig_path.exists():
                        mlflow.log_artifact(str(fig_path), artifact_path="figures")

        print(f"[MLFLOW] Logged run: {run_name}  (experiment={experiment})")

    print(f"[MLFLOW] UI: mlflow ui --backend-store-uri {uri}")
