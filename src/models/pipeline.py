"""End-to-end ML pipeline with three feature modes and a comparison summary.

Ablation study:
  E2 — SNP-only      : SNP integer-encoded matrix → RandomForest/SVM
  E3 — DNABERT-only  : DNABERT-2 mean-pooled embeddings → RandomForest/SVM
  E4 — Hybrid        : SNP + DNABERT concatenated → RandomForest/SVM

Split strategy (anti-leakage):
  StratifiedGroupKFold on snp_cluster — preserves class balance while ensuring
  isolates from the same genomic cluster never span train and test.

  Fallback chain:
    StratifiedGroupKFold → GroupShuffleSplit → StratifiedShuffleSplit → ShuffleSplit

  Use use_groups=False for the "naive" comparison run only (notebook only).
  The reported main result always uses use_groups=True.

Reproducibility:
  _make_split returns integer indices (not sliced arrays) so that callers can
  record the exact assembly_accession IDs assigned to train and test sets.
  These IDs are saved inside the per-mode metrics dict as train_ids / test_ids.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold,
    StratifiedShuffleSplit, ShuffleSplit,
)
from sklearn.metrics import silhouette_score

from .trainer import (
    prepare_features, prepare_snp_features,
    prepare_hybrid_features, train_classifier,
    get_sample_index,
)
from .evaluator import evaluate, plot_confusion_matrix, plot_roc_curve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_groups(index: pd.Index, metadata_df: pd.DataFrame, group_col: str):
    """Return group array aligned to `index`, or None if group_col absent."""
    if group_col not in metadata_df.columns:
        return None
    meta = metadata_df.set_index("assembly_accession")
    return meta.reindex(index)[group_col].fillna("unknown").values


def _make_split(
    X: np.ndarray, y: np.ndarray, cfg: dict, groups=None
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Anti-leakage train/test split.  Returns integer indices into X/y.

    Priority (when groups is provided):
      1. StratifiedGroupKFold — keeps class balance AND group separation.
         Number of folds = min(5, n_unique_groups). Uses the first fold.
      2. GroupShuffleSplit — simpler group-aware split (fallback).
      3. StratifiedShuffleSplit — no group awareness (second fallback).
      4. ShuffleSplit — final fallback for tiny/single-class datasets.

    When groups=None, falls directly to step 3/4 (naive mode).

    Returns (train_idx, test_idx, split_type) where indices are integer arrays
    suitable for fancy indexing: X[train_idx], y[test_idx], index[train_idx].
    """
    test_size = cfg["ml"]["test_size"]
    rs        = cfg["ml"]["random_state"]

    # ── 1. StratifiedGroupKFold ──────────────────────────────────────────────
    if groups is not None:
        n_unique = len(np.unique(groups))
        if n_unique >= 2:
            n_splits = min(5, max(2, n_unique))
            try:
                sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rs)
                train_idx, test_idx = next(sgkf.split(X, y, groups=groups))
                if len(np.unique(y[test_idx])) >= 2:
                    ratio = len(test_idx) / len(y)
                    print(
                        f"  [SPLIT] StratifiedGroupKFold(n_splits={n_splits}) "
                        f"train={len(train_idx)} test={len(test_idx)} ({ratio:.0%})"
                    )
                    return train_idx, test_idx, f"sgkf_{n_splits}"
                print("  [WARN] SGKF: test set 1 kelas — coba GroupShuffleSplit")
            except Exception as exc:
                print(f"  [WARN] StratifiedGroupKFold gagal ({exc}) — coba GroupShuffleSplit")

    # ── 2. GroupShuffleSplit ─────────────────────────────────────────────────
    if groups is not None and len(np.unique(groups)) >= 2:
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            if len(np.unique(y[test_idx])) >= 2:
                ratio = len(test_idx) / len(y)
                print(
                    f"  [SPLIT] GroupShuffleSplit "
                    f"train={len(train_idx)} test={len(test_idx)} ({ratio:.0%})"
                )
                return train_idx, test_idx, "group_shuffle"
            print("  [WARN] GroupShuffleSplit: test set 1 kelas — fallback ke stratified")
        except Exception as exc:
            print(f"  [WARN] GroupShuffleSplit gagal ({exc}) — fallback ke stratified")

    # ── 3. StratifiedShuffleSplit ────────────────────────────────────────────
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        train_idx, test_idx = next(sss.split(X, y))
        print(f"  [SPLIT] StratifiedSplit train={len(train_idx)} test={len(test_idx)}")
        return train_idx, test_idx, "stratified"
    except ValueError:
        pass

    # ── 4. ShuffleSplit (final fallback) ─────────────────────────────────────
    ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
    train_idx, test_idx = next(ss.split(X))
    print(f"  [SPLIT] RandomSplit train={len(train_idx)} test={len(test_idx)}")
    return train_idx, test_idx, "random"


def _silhouette_for_test(X_test: np.ndarray, y_pred: np.ndarray) -> float:
    unique = np.unique(y_pred)
    if len(unique) < 2 or any(np.sum(y_pred == u) < 2 for u in unique):
        return float("nan")
    try:
        return float(silhouette_score(X_test, y_pred))
    except Exception:
        return float("nan")


def _run_single(
    X: np.ndarray,
    y: np.ndarray,
    label_names: list,
    cfg: dict,
    fig_dir: str,
    mode_name: str,
    groups=None,
    fig_suffix: str = "",
    index: pd.Index | None = None,
) -> tuple:
    """Train + evaluate one feature mode.  Returns (clf, metrics) or (None, empty)."""
    if len(set(y)) < 2:
        print(f"[SKIP] {mode_name}: hanya {len(set(y))} kelas setelah join.")
        return None, _empty_metrics()
    if len(y) < 4:
        print(f"[SKIP] {mode_name}: terlalu sedikit sampel ({len(y)}).")
        return None, _empty_metrics()

    model_type = cfg["ml"]["model"]
    scale      = model_type == "SVM"

    train_idx, test_idx, split_type = _make_split(X, y, cfg, groups=groups)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf, scaler = train_classifier(
        X_train, y_train,
        model_type=model_type,
        random_state=cfg["ml"]["random_state"],
        n_estimators=cfg["ml"]["n_estimators"],
        scale=scale,
    )
    if scaler is not None:
        X_test = scaler.transform(X_test)

    metrics = evaluate(clf, X_test, y_test, label_names)
    metrics["split_type"] = split_type
    metrics["silhouette"] = _silhouette_for_test(X_test, metrics["y_pred"])

    # Record exact accession IDs assigned to train / test for audit trail
    if index is not None:
        metrics["train_ids"] = index[train_idx].tolist()
        metrics["test_ids"]  = index[test_idx].tolist()

    plot_confusion_matrix(
        y_test, metrics["y_pred"], label_names,
        f"{fig_dir}{mode_name}{fig_suffix}_confusion_matrix.png",
    )
    plot_roc_curve(clf, X_test, y_test, f"{fig_dir}{mode_name}{fig_suffix}_roc_curve.png")
    return clf, metrics


def _empty_metrics() -> dict:
    return {
        "accuracy": 0.0, "balanced_accuracy": 0.0,
        "f1_weighted": 0.0, "f1_macro": 0.0,
        "silhouette": float("nan"), "report": "", "split_type": "none",
        "train_ids": [], "test_ids": [],
    }


def _print_comparison(all_metrics: dict, header: str = "Ablation Study") -> None:
    print("\n" + "=" * 72)
    print(f"  {header}")
    print("=" * 72)
    print(f"  {'Mode':<22} {'Macro F1':>9} {'Bal. Acc':>9} {'F1 (wt.)':>9} {'Silhouette':>11} {'Split':>12}")
    print(f"  {'-' * 68}")
    for mode, m in all_metrics.items():
        mf1  = m.get("f1_macro",          0.0)
        bacc = m.get("balanced_accuracy", 0.0)
        wf1  = m.get("f1_weighted",       0.0)
        sil  = m.get("silhouette",        float("nan"))
        spl  = m.get("split_type",        "-")
        sil_str = f"{sil:.4f}" if not (isinstance(sil, float) and np.isnan(sil)) else "   -  "
        print(f"  {mode:<22} {mf1:>9.4f} {bacc:>9.4f} {wf1:>9.4f} {sil_str:>11} {spl:>12}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_pipeline(
    embedding_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    cfg: dict,
    snp_encoded_df: pd.DataFrame | None = None,
    kmer_df: pd.DataFrame | None = None,
    use_groups: bool = True,
) -> tuple:
    """
    Run ablation study across up to five feature modes.

    Modes
    -----
    E2  snp_only      — integer-encoded SNP matrix
    E3  dnabert_only  — DNABERT-2 mean-pooled embeddings
    E4  hybrid        — SNP + DNABERT concatenated
    E5  kmer_only     — tetranucleotide k-mer frequencies (optional)

    Parameters
    ----------
    kmer_df    : optional k-mer frequency DataFrame (E5 baseline)
    use_groups : True → StratifiedGroupKFold on snp_cluster (main result).
                 False → naive stratified split (comparison only).

    Returns (best_clf, best_metrics, all_metrics_dict).
    'best' = highest balanced_accuracy across modes.
    Each mode's metrics dict includes train_ids and test_ids for reproducibility.
    """
    target_col = cfg["ml"].get("target_col", "snp_cluster")
    group_col  = "snp_cluster"
    fig_dir    = cfg["output"]["figures_dir"] + "classification/"
    fig_suffix = "" if use_groups else "_naive"

    if not use_groups:
        print("\n[INFO] Naive split mode — groups dinonaktifkan (hanya untuk perbandingan)")

    if target_col in metadata_df.columns:
        counts = metadata_df[target_col].value_counts()
        tiny = counts[counts < 2]
        if len(tiny):
            print(f"[WARN] Kelas < 2 sampel di '{target_col}': {tiny.to_dict()}")

    all_metrics: dict[str, dict] = {}
    best_clf     = None
    best_metrics = _empty_metrics()

    def _groups(feature_df):
        if not use_groups:
            return None
        idx = get_sample_index(feature_df, metadata_df, target_col)
        return _extract_groups(idx, metadata_df, group_col)

    # --- E3: DNABERT-only ---
    X, y, lnames = prepare_features(embedding_df, metadata_df, target_col)
    idx_e3 = get_sample_index(embedding_df, metadata_df, target_col)
    clf, m = _run_single(X, y, lnames, cfg, fig_dir, "dnabert_only",
                          groups=_groups(embedding_df), fig_suffix=fig_suffix, index=idx_e3)
    all_metrics["dnabert_only"] = m
    if clf is not None and m["balanced_accuracy"] > best_metrics["balanced_accuracy"]:
        best_clf, best_metrics = clf, m

    # --- E2: SNP-only ---
    if snp_encoded_df is not None and len(snp_encoded_df) > 0:
        X, y, lnames = prepare_snp_features(snp_encoded_df, metadata_df, target_col)
        idx_e2 = get_sample_index(snp_encoded_df, metadata_df, target_col)
        clf, m = _run_single(X, y, lnames, cfg, fig_dir, "snp_only",
                              groups=_groups(snp_encoded_df), fig_suffix=fig_suffix, index=idx_e2)
        all_metrics["snp_only"] = m
        if clf is not None and m["balanced_accuracy"] > best_metrics["balanced_accuracy"]:
            best_clf, best_metrics = clf, m

    # --- E4: Hybrid ---
    if snp_encoded_df is not None and len(snp_encoded_df) > 0:
        combined = snp_encoded_df.join(embedding_df, how="inner")
        X, y, lnames = prepare_hybrid_features(
            snp_encoded_df, embedding_df, metadata_df, target_col
        )
        idx_e4 = get_sample_index(combined, metadata_df, target_col)
        clf, m = _run_single(X, y, lnames, cfg, fig_dir, "hybrid",
                              groups=_groups(combined), fig_suffix=fig_suffix, index=idx_e4)
        all_metrics["hybrid"] = m
        if clf is not None and m["balanced_accuracy"] > best_metrics["balanced_accuracy"]:
            best_clf, best_metrics = clf, m

    # --- E5: k-mer only (optional alignment-free baseline) ---
    if kmer_df is not None and len(kmer_df) > 0:
        X, y, lnames = prepare_features(kmer_df, metadata_df, target_col)
        idx_e5 = get_sample_index(kmer_df, metadata_df, target_col)
        clf, m = _run_single(X, y, lnames, cfg, fig_dir, "kmer_only",
                              groups=_groups(kmer_df), fig_suffix=fig_suffix, index=idx_e5)
        all_metrics["kmer_only"] = m
        if clf is not None and m["balanced_accuracy"] > best_metrics["balanced_accuracy"]:
            best_clf, best_metrics = clf, m

    label = ("Group-aware split (snp_cluster)" if use_groups
             else "Naive stratified split (perbandingan)")
    _print_comparison(all_metrics, header=f"Ablation Study — {label}")

    # Tie-aware best-mode reporting
    all_ba = {
        mode: m["balanced_accuracy"]
        for mode, m in all_metrics.items()
        if m.get("balanced_accuracy", 0.0) > 0
    }
    if all_ba:
        max_ba = max(all_ba.values())
        best_modes = [m for m, v in all_ba.items() if abs(v - max_ba) < 1e-6]
        if len(best_modes) > 1:
            print(
                f"[INFO] Semua mode tied pada balanced_accuracy={max_ba:.4f}. "
                f"Modes: {best_modes}. Tidak ada mode yang secara statistik unggul."
            )
        else:
            print(f"[INFO] Mode terbaik: {best_modes[0]} (balanced_accuracy={max_ba:.4f})")

    return best_clf, best_metrics, all_metrics
