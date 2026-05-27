"""Forensic Interpretation Layer — per-isolate nearest-neighbor + source prediction."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.io import ensure_dir


def build_forensic_table(
    dist_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    target_col: str = "isolation_source",
    clf=None,
    feature_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Per-isolate forensic summary:
      - nearest genomic neighbor (by SNP distance)
      - SNP distance to nearest
      - known isolation source vs neighbor source
      - ML-predicted source + confidence (optional, needs clf + feature_df)
    """
    meta = (
        metadata_df.set_index("assembly_accession")
        if "assembly_accession" in metadata_df.columns
        else metadata_df
    )

    records = []
    for acc in dist_df.index:
        row = dist_df.loc[acc].drop(acc, errors="ignore")
        if row.empty:
            continue
        nearest = row.idxmin()
        snp_dist = float(row.min())
        source = meta.at[acc, target_col] if acc in meta.index else "unknown"
        nearest_source = meta.at[nearest, target_col] if nearest in meta.index else "unknown"

        rec: dict = {
            "assembly_accession": acc,
            "isolation_source": source,
            "nearest_neighbor": nearest,
            "snp_distance_to_nearest": round(snp_dist, 2),
            "nearest_source": nearest_source,
            "predicted_source": None,
            "prediction_confidence": None,
        }

        if clf is not None and feature_df is not None and acc in feature_df.index:
            x = feature_df.loc[[acc]].values
            try:
                rec["predicted_source"] = clf.predict(x)[0]
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(x)[0]
                    rec["prediction_confidence"] = round(float(proba.max()), 4)
            except ValueError:
                # clf was trained on a different feature set (e.g. hybrid SNP+DNABERT)
                # but feature_df only has DNABERT embeddings — skip prediction gracefully
                pass

        records.append(rec)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index("assembly_accession")
    return df


def generate_forensic_summary(
    forensic_df: pd.DataFrame,
    all_metrics: dict,
    out_path: str,
) -> None:
    """
    Write a human-readable forensic interpretation report.

    Includes:
      - Experiment comparison table (all modes in all_metrics)
      - Per-isolate block: nearest neighbor, SNP distance, source match, ML prediction
    """
    ensure_dir(str(Path(out_path).parent))

    _EXP_LABEL_MAP = {
        "dummy":            "E0  DummyClassifier",
        "snp_only":         "E2  SNP-only RF",
        "dnabert_only":     "E3  DNABERT RF",
        "hybrid":           "E4  Hybrid SNP+DNABERT RF",
        "kmer_only":        "E5  K-mer RF",
        "snp_lr":           "E6  SNP LR",
        "snp_svc":          "E7  SNP LinearSVC",
        "amr_lr":           "E8  AMR-only LR",
        "snp_amr_lr":       "E9  SNP+AMR LR",
        "dnabert_lr":       "E3b DNABERT LR",
        "dnabert_svc":      "E3c DNABERT LinearSVC",
        "kmer_amr_lr":      "E10 kmer+AMR LR",
        "kmer_amr_svc":     "E11 kmer+AMR LinearSVC",
        "snp_kmer_amr_lr":  "E12 SNP+kmer+AMR LR",
        "snp_kmer_amr_rf":  "E13 SNP+kmer+AMR RF",
    }

    lines = [
        "=" * 78,
        "  SalmoTrace-BERT — Forensic Interpretation Report",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 78,
        "",
        "[Ablation Study — Source Attribution (all feature modes)]",
        f"  {'Experiment':<32} {'Macro F1':>9} {'Bal. Acc':>9} {'F1 (wt.)':>9} {'Split':>12}",
        f"  {'-' * 73}",
    ]
    for key, m in all_metrics.items():
        label = _EXP_LABEL_MAP.get(key, key)
        lines.append(
            f"  {label:<32}"
            f" {m.get('f1_macro', 0.0):>9.4f}"
            f" {m.get('balanced_accuracy', 0.0):>9.4f}"
            f" {m.get('f1_weighted', 0.0):>9.4f}"
            f" {m.get('split_type', '-'):>12}"
        )

    lines += ["", "[Per-Isolate Forensic Summary]", ""]

    for acc, row in forensic_df.iterrows():
        match = row["nearest_source"] == row["isolation_source"]
        tag = "[same source]" if match else "[different source]"
        block = [
            f"  Isolate           : {acc}",
            f"  Known source      : {row['isolation_source']}",
            (
                f"  Nearest neighbor  : {row['nearest_neighbor']}"
                f"  (SNP dist = {row['snp_distance_to_nearest']:.0f})  {tag}"
            ),
            f"  Neighbor source   : {row['nearest_source']}",
        ]
        if row.get("predicted_source") is not None:
            conf = (
                f"  (confidence {row['prediction_confidence']:.1%})"
                if row.get("prediction_confidence") is not None else ""
            )
            block.append(f"  ML prediction     : {row['predicted_source']}{conf}")
        lines.extend(block)
        lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[FORENSIC] {out_path}")
