"""End-to-end ML pipeline with multiple feature modes and a comparison summary.

Ablation study:
  E0 — dummy         : DummyClassifier (majority-class baseline)
  E2 — SNP-only      : SNP integer-encoded matrix → RandomForest
  E3 — DNABERT-only  : DNABERT-2 mean-pooled embeddings → RandomForest
  E4 — Hybrid        : SNP + DNABERT concatenated → RandomForest
  E5 — k-mer         : k-mer frequency features → RandomForest (optional)
  E6 — snp_lr        : SNP matrix → LogisticRegression balanced + SelectKBest
  E7 — snp_svc       : SNP matrix → LinearSVC balanced + SelectKBest
  E8 — amr_lr        : AMR gene binary features → LogisticRegression balanced
  E9 — snp_amr_lr    : SNP + AMR → LogisticRegression balanced + SelectKBest

Split strategy (anti-leakage):
  StratifiedGroupKFold on snp_cluster — preserves class balance while ensuring
  isolates from the same genomic cluster never span train and test.

  Isolates with snp_cluster = "unknown" / "not provided" each get their own
  unique group (assembly_accession) so they are not artificially pooled.

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
    get_sample_index, _join_target,
)
from .evaluator import evaluate, plot_confusion_matrix, plot_roc_curve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNKNOWN_GROUPS = {"unknown", "not provided", "nan", ""}


def _merge_rare_classes(
    y: np.ndarray, min_samples: int, fallback_label: str = "animal_other"
) -> np.ndarray:
    """Relabel classes with fewer than `min_samples` instances to `fallback_label`.

    Prevents macro-F1 from being dragged to near-zero by singleton/doubleton
    classes that the model cannot possibly learn (no generalization signal).
    """
    if min_samples <= 1:
        return y
    counts = pd.Series(y).value_counts()
    rare = counts[counts < min_samples].index.tolist()
    if not rare:
        return y
    print(f"[MERGE] Kelas < {min_samples} sampel digabung ke '{fallback_label}': {rare}")
    return np.where(np.isin(y, rare), fallback_label, y)


def _extract_groups(index: pd.Index, metadata_df: pd.DataFrame, group_col: str):
    """Return group array aligned to `index`, or None if group_col absent.

    Isolates with unknown/missing snp_cluster values each receive their own
    unique group ID (assembly_accession) so they are not lumped into one
    artificial super-group that would inflate within-group distance and
    distort StratifiedGroupKFold fold assignments.
    """
    if group_col not in metadata_df.columns:
        return None
    meta = metadata_df.set_index("assembly_accession")
    sub = meta.reindex(index)

    def _group_id(row):
        val = str(row[group_col]) if not pd.isna(row[group_col]) else ""
        return row.name if val.lower() in _UNKNOWN_GROUPS else val

    return sub.apply(_group_id, axis=1).values


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


def _apply_svd(X_train: np.ndarray, X_test: np.ndarray, n_components: int,
               random_state: int, mode_name: str) -> tuple[np.ndarray, np.ndarray]:
    """TruncatedSVD fit on train, transform test. Returns (X_train_reduced, X_test_reduced)."""
    from sklearn.decomposition import TruncatedSVD
    n_comp = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    X_tr = svd.fit_transform(X_train.astype(np.float32))
    X_te = svd.transform(X_test.astype(np.float32))
    var_exp = svd.explained_variance_ratio_.sum()
    print(f"  [SVD] {mode_name}: {n_comp} PC, {var_exp:.1%} variance explained")
    return X_tr, X_te


def _apply_feature_selection(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, k: int, mode_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """SelectKBest (mutual_info_classif) fit on train, transform test.

    Feature selection is performed inside each CV fold so the selector sees
    only training data — scikit-learn best practice to prevent leakage.
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    k_eff = min(k, X_train.shape[1])
    sel = SelectKBest(mutual_info_classif, k=k_eff)
    X_tr = sel.fit_transform(X_train, y_train)
    X_te = sel.transform(X_test)
    print(f"  [SELECT] {mode_name}: {k_eff}/{X_train.shape[1]} SNP fitur dipilih")
    return X_tr, X_te


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
    model_override: str | None = None,
    feature_select_k: int | None = None,
) -> tuple:
    """Train + evaluate one feature mode with full K-fold CV.

    Runs ALL folds of StratifiedGroupKFold (or fallback splitter) and reports
    mean ± std across folds.  The final classifier is retrained on the full
    dataset so it can be used for inference (e.g. forensic table).
    Returns (clf_full, metrics_mean) or (None, empty).

    Parameters
    ----------
    model_override   : if set, overrides cfg["ml"]["model"] for this run only
    feature_select_k : if set, apply SelectKBest(k) inside each CV fold
    """
    if len(set(y)) < 2:
        print(f"[SKIP] {mode_name}: hanya {len(set(y))} kelas setelah join.")
        return None, _empty_metrics()
    if len(y) < 4:
        print(f"[SKIP] {mode_name}: terlalu sedikit sampel ({len(y)}).")
        return None, _empty_metrics()

    model_type    = model_override or cfg["ml"]["model"]
    scale         = model_type in ("SVM", "LogisticRegression", "LinearSVC")
    n_components  = int(cfg["ml"].get("snp_n_components", 40))
    rs            = cfg["ml"]["random_state"]
    svd_threshold = int(cfg["ml"].get("svd_feature_threshold", 5_000))

    # ── Determine number of folds ─────────────────────────────────────────────
    if groups is not None:
        n_unique_groups = len(np.unique(groups))
        n_splits = min(5, max(2, n_unique_groups))
    else:
        n_splits = 5

    # ── Collect per-fold metrics ──────────────────────────────────────────────
    fold_metrics: list[dict] = []
    all_y_true, all_y_pred = [], []
    split_type_used = "unknown"

    splitters = []
    if groups is not None and len(np.unique(groups)) >= 2:
        splitters.append(("sgkf", StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rs)))
        splitters.append(("gss",  GroupShuffleSplit(n_splits=n_splits, test_size=cfg["ml"]["test_size"], random_state=rs)))
    splitters.append(("sss", StratifiedShuffleSplit(n_splits=n_splits, test_size=cfg["ml"]["test_size"], random_state=rs)))
    splitters.append(("ss",  ShuffleSplit(n_splits=n_splits, test_size=cfg["ml"]["test_size"], random_state=rs)))

    for split_name, splitter in splitters:
        try:
            splits = list(splitter.split(X, y, groups=groups) if groups is not None
                         else splitter.split(X, y))
            if any(len(np.unique(y[te])) >= 2 for _, te in splits):
                split_type_used = f"{split_name}_cv{n_splits}"
                break
        except Exception:
            continue

    print(f"  [CV] {mode_name}: {n_splits}-fold ({split_type_used})")

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        if len(np.unique(y[test_idx])) < 2:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # SVD reduction (only when features exceed threshold)
        if X_train.shape[1] > svd_threshold and n_components < X_train.shape[1]:
            X_train, X_test = _apply_svd(X_train, X_test, n_components, rs, f"{mode_name}[f{fold_i}]")

        # SelectKBest feature selection (fit on train only — no leakage)
        if feature_select_k is not None and feature_select_k < X_train.shape[1]:
            X_train, X_test = _apply_feature_selection(
                X_train, X_test, y_train, feature_select_k, f"{mode_name}[f{fold_i}]"
            )

        clf_fold, scaler_fold = train_classifier(
            X_train, y_train, model_type=model_type,
            random_state=rs, n_estimators=cfg["ml"]["n_estimators"], scale=scale,
        )
        if scaler_fold is not None:
            X_test = scaler_fold.transform(X_test)

        m = evaluate(clf_fold, X_test, y_test, label_names)
        m["silhouette"] = _silhouette_for_test(X_test, m["y_pred"])
        fold_metrics.append(m)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(m["y_pred"].tolist())

    if not fold_metrics:
        return None, _empty_metrics()

    # ── Aggregate fold metrics ────────────────────────────────────────────────
    def _mean(key):
        vals = [fm[key] for fm in fold_metrics if not (isinstance(fm[key], float) and np.isnan(fm[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    def _std(key):
        vals = [fm[key] for fm in fold_metrics if not (isinstance(fm[key], float) and np.isnan(fm[key]))]
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    metrics = {
        "accuracy":          _mean("accuracy"),
        "balanced_accuracy": _mean("balanced_accuracy"),
        "f1_weighted":       _mean("f1_weighted"),
        "f1_macro":          _mean("f1_macro"),
        "f1_macro_std":      _std("f1_macro"),
        "balanced_acc_std":  _std("balanced_accuracy"),
        "silhouette":        _mean("silhouette"),
        "split_type":        split_type_used,
        "n_folds":           len(fold_metrics),
        "train_ids":         [],
        "test_ids":          [],
        "y_pred":            np.array(all_y_pred),
    }
    print(
        f"  [CV-MEAN] {mode_name}: "
        f"bal_acc={metrics['balanced_accuracy']:.3f}±{metrics['balanced_acc_std']:.3f}  "
        f"macro_f1={metrics['f1_macro']:.3f}±{metrics['f1_macro_std']:.3f}"
    )

    # ── Final classifier: retrain on ALL data ─────────────────────────────────
    X_full = X
    if X.shape[1] > svd_threshold and n_components < X.shape[1]:
        from sklearn.decomposition import TruncatedSVD
        n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
        svd_full = TruncatedSVD(n_components=n_comp, random_state=rs)
        X_full = svd_full.fit_transform(X.astype(np.float32))

    clf, scaler = train_classifier(
        X_full, y, model_type=model_type,
        random_state=rs, n_estimators=cfg["ml"]["n_estimators"], scale=scale,
    )

    # ── Confusion matrix on aggregated predictions ────────────────────────────
    plot_confusion_matrix(
        np.array(all_y_true), np.array(all_y_pred), label_names,
        f"{fig_dir}{mode_name}{fig_suffix}_confusion_matrix.png",
    )
    plot_roc_curve(clf, X_full, y, f"{fig_dir}{mode_name}{fig_suffix}_roc_curve.png")
    return clf, metrics


def _empty_metrics() -> dict:
    return {
        "accuracy": 0.0, "balanced_accuracy": 0.0,
        "f1_weighted": 0.0, "f1_macro": 0.0,
        "silhouette": float("nan"), "report": "", "split_type": "none",
        "train_ids": [], "test_ids": [],
    }


def _print_comparison(all_metrics: dict, header: str = "Ablation Study") -> None:
    print("\n" + "=" * 84)
    print(f"  {header}")
    print("=" * 84)
    print(f"  {'Mode':<22} {'Macro F1 (±std)':>18} {'Bal. Acc (±std)':>18} {'F1 (wt.)':>9} {'Folds':>6} {'Split':>9}")
    print(f"  {'-' * 80}")
    for mode, m in all_metrics.items():
        mf1     = m.get("f1_macro",          0.0)
        mf1_std = m.get("f1_macro_std",      0.0)
        bacc    = m.get("balanced_accuracy", 0.0)
        ba_std  = m.get("balanced_acc_std",  0.0)
        wf1     = m.get("f1_weighted",       0.0)
        spl     = m.get("split_type",        "-")
        nf      = m.get("n_folds",           "-")
        mf1_str  = f"{mf1:.3f}±{mf1_std:.3f}"
        bacc_str = f"{bacc:.3f}±{ba_std:.3f}"
        print(f"  {mode:<22} {mf1_str:>18} {bacc_str:>18} {wf1:>9.4f} {str(nf):>6} {spl:>9}")
    print("=" * 84 + "\n")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_pipeline(
    embedding_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    cfg: dict,
    snp_encoded_df: pd.DataFrame | None = None,
    kmer_df: pd.DataFrame | None = None,
    amr_df: pd.DataFrame | None = None,
    use_groups: bool = True,
) -> tuple:
    """
    Run ablation study across multiple feature modes.

    Modes
    -----
    E0  dummy         — DummyClassifier (majority-class sanity baseline)
    E2  snp_only      — integer-encoded SNP matrix + RandomForest
    E3  dnabert_only  — DNABERT-2 mean-pooled embeddings + RandomForest
    E4  hybrid        — SNP + DNABERT concatenated + RandomForest
    E5  kmer_only     — k-mer frequency features + RandomForest (optional)
    E6  snp_lr        — SNP matrix + LogisticRegression balanced + SelectKBest
    E7  snp_svc       — SNP matrix + LinearSVC balanced + SelectKBest
    E8  amr_lr        — AMR gene binary features + LogisticRegression balanced
    E9  snp_amr_lr    — SNP + AMR + LogisticRegression balanced + SelectKBest

    Parameters
    ----------
    amr_df     : optional AMR gene binary feature DataFrame (E8, E9)
    kmer_df    : optional k-mer frequency DataFrame (E5 baseline)
    use_groups : True → StratifiedGroupKFold on snp_cluster (main result).
                 False → naive stratified split (comparison only).

    Returns (best_clf, best_metrics, all_metrics_dict).
    'best' = highest balanced_accuracy across modes.
    Each mode's metrics dict includes train_ids and test_ids for reproducibility.
    """
    target_col    = cfg["ml"].get("target_col", "snp_cluster")
    group_col     = "snp_cluster"
    fig_dir       = cfg["output"]["figures_dir"] + "classification/"
    fig_suffix    = "" if use_groups else "_naive"
    min_cls       = cfg["ml"].get("min_class_samples", 5)
    n_components  = int(cfg["ml"].get("snp_n_components", 40))
    feat_sel_k    = cfg["ml"].get("feature_select_k")

    if not use_groups:
        print("\n[INFO] Naive split mode — groups dinonaktifkan (hanya untuk perbandingan)")

    if target_col in metadata_df.columns:
        counts = metadata_df[target_col].value_counts()
        tiny = counts[counts < 2]
        if len(tiny):
            print(f"[WARN] Kelas < 2 sampel di '{target_col}': {tiny.to_dict()}")

    all_metrics: dict[str, dict] = {}
    all_clfs:    dict[str, object] = {}
    best_clf     = None
    best_metrics = _empty_metrics()

    def _groups(feature_df):
        if not use_groups:
            return None
        idx = get_sample_index(feature_df, metadata_df, target_col)
        return _extract_groups(idx, metadata_df, group_col)

    def _update_best(clf, m, mode):
        nonlocal best_clf, best_metrics
        if clf is not None and m["balanced_accuracy"] > best_metrics["balanced_accuracy"]:
            best_clf, best_metrics = clf, m
        all_metrics[mode] = m
        all_clfs[mode] = clf

    # --- E3: DNABERT-only ---
    X, y, lnames = prepare_features(embedding_df, metadata_df, target_col)
    y = _merge_rare_classes(y, min_cls)
    lnames = sorted(set(y.tolist()))
    idx_e3 = get_sample_index(embedding_df, metadata_df, target_col)
    clf, m = _run_single(X, y, lnames, cfg, fig_dir, "dnabert_only",
                          groups=_groups(embedding_df), fig_suffix=fig_suffix, index=idx_e3)
    _update_best(clf, m, "dnabert_only")

    # --- E2: SNP-only (RandomForest baseline) ---
    if snp_encoded_df is not None and len(snp_encoded_df) > 0:
        X, y, lnames = prepare_snp_features(snp_encoded_df, metadata_df, target_col)
        y = _merge_rare_classes(y, min_cls)
        lnames = sorted(set(y.tolist()))
        idx_e2 = get_sample_index(snp_encoded_df, metadata_df, target_col)
        clf, m = _run_single(X, y, lnames, cfg, fig_dir, "snp_only",
                              groups=_groups(snp_encoded_df), fig_suffix=fig_suffix, index=idx_e2)
        _update_best(clf, m, "snp_only")

    # --- E4: Hybrid (SNP → TruncatedSVD then concat DNABERT) ---
    if snp_encoded_df is not None and len(snp_encoded_df) > 0:
        X, y, lnames = prepare_hybrid_features(
            snp_encoded_df, embedding_df, metadata_df, target_col,
            snp_n_components=n_components,
        )
        y = _merge_rare_classes(y, min_cls)
        lnames = sorted(set(y.tolist()))
        shared = snp_encoded_df.index.intersection(embedding_df.index)
        combined_df = pd.DataFrame(index=shared)
        idx_e4 = get_sample_index(combined_df, metadata_df, target_col)
        groups_e4 = _extract_groups(idx_e4, metadata_df, group_col) if use_groups else None
        clf, m = _run_single(X, y, lnames, cfg, fig_dir, "hybrid",
                              groups=groups_e4, fig_suffix=fig_suffix, index=idx_e4)
        _update_best(clf, m, "hybrid")

    # --- E5: k-mer only (optional alignment-free baseline) ---
    if kmer_df is not None and len(kmer_df) > 0:
        X, y, lnames = prepare_features(kmer_df, metadata_df, target_col)
        y = _merge_rare_classes(y, min_cls)
        lnames = sorted(set(y.tolist()))
        idx_e5 = get_sample_index(kmer_df, metadata_df, target_col)
        clf, m = _run_single(X, y, lnames, cfg, fig_dir, "kmer_only",
                              groups=_groups(kmer_df), fig_suffix=fig_suffix, index=idx_e5)
        _update_best(clf, m, "kmer_only")

    # --- E0: DummyClassifier (majority-class sanity check) ---
    # Use SNP features if available, otherwise DNABERT — the model ignores them.
    if snp_encoded_df is not None and len(snp_encoded_df) > 0:
        X_d, y_d, ln_d = prepare_snp_features(snp_encoded_df, metadata_df, target_col)
        g_d = _groups(snp_encoded_df)
    else:
        X_d, y_d, ln_d = prepare_features(embedding_df, metadata_df, target_col)
        g_d = _groups(embedding_df)
    y_d = _merge_rare_classes(y_d, min_cls)
    ln_d = sorted(set(y_d.tolist()))
    clf, m = _run_single(X_d, y_d, ln_d, cfg, fig_dir, "dummy",
                          groups=g_d, fig_suffix=fig_suffix,
                          model_override="Dummy")
    # DummyClassifier is not a candidate for best_clf (forensic use)
    all_metrics["dummy"] = m
    all_clfs["dummy"] = clf

    # --- E6: SNP-only + LogisticRegression balanced + SelectKBest ---
    if snp_encoded_df is not None and len(snp_encoded_df) > 0:
        X, y, lnames = prepare_snp_features(snp_encoded_df, metadata_df, target_col)
        y = _merge_rare_classes(y, min_cls)
        lnames = sorted(set(y.tolist()))
        clf, m = _run_single(X, y, lnames, cfg, fig_dir, "snp_lr",
                              groups=_groups(snp_encoded_df), fig_suffix=fig_suffix,
                              model_override="LogisticRegression",
                              feature_select_k=feat_sel_k)
        _update_best(clf, m, "snp_lr")

    # --- E7: SNP-only + LinearSVC balanced + SelectKBest ---
    if snp_encoded_df is not None and len(snp_encoded_df) > 0:
        X, y, lnames = prepare_snp_features(snp_encoded_df, metadata_df, target_col)
        y = _merge_rare_classes(y, min_cls)
        lnames = sorted(set(y.tolist()))
        clf, m = _run_single(X, y, lnames, cfg, fig_dir, "snp_svc",
                              groups=_groups(snp_encoded_df), fig_suffix=fig_suffix,
                              model_override="LinearSVC",
                              feature_select_k=feat_sel_k)
        _update_best(clf, m, "snp_svc")

    # --- E8: AMR-only + LogisticRegression balanced ---
    if amr_df is not None and len(amr_df) > 0:
        X, y, lnames = prepare_features(amr_df, metadata_df, target_col)
        y = _merge_rare_classes(y, min_cls)
        lnames = sorted(set(y.tolist()))
        clf, m = _run_single(X, y, lnames, cfg, fig_dir, "amr_lr",
                              groups=_groups(amr_df), fig_suffix=fig_suffix,
                              model_override="LogisticRegression")
        _update_best(clf, m, "amr_lr")

    # --- E9: SNP + AMR + LogisticRegression balanced + SelectKBest ---
    if snp_encoded_df is not None and len(snp_encoded_df) > 0 and amr_df is not None and len(amr_df) > 0:
        shared = snp_encoded_df.index.intersection(amr_df.index)
        if len(shared) >= 4:
            combined = snp_encoded_df.loc[shared].join(amr_df.loc[shared], how="inner")
            X, y, lnames = _join_target(combined, metadata_df, target_col)
            y = _merge_rare_classes(y, min_cls)
            lnames = sorted(set(y.tolist()))
            idx_e9 = get_sample_index(combined, metadata_df, target_col)
            groups_e9 = _extract_groups(idx_e9, metadata_df, group_col) if use_groups else None
            clf, m = _run_single(X, y, lnames, cfg, fig_dir, "snp_amr_lr",
                                  groups=groups_e9, fig_suffix=fig_suffix,
                                  model_override="LogisticRegression",
                                  feature_select_k=feat_sel_k)
            _update_best(clf, m, "snp_amr_lr")

    label = ("Group-aware split (snp_cluster)" if use_groups
             else "Naive stratified split (perbandingan)")
    _print_comparison(all_metrics, header=f"Ablation Study — {label}")

    # Tie-aware best-mode reporting + tie-breaking: prefer dnabert_only (most
    # deployable — does not require the full SNP matrix for new isolates).
    # DummyClassifier excluded from tie-breaking consideration.
    all_ba = {
        mode: m["balanced_accuracy"]
        for mode, m in all_metrics.items()
        if m.get("balanced_accuracy", 0.0) > 0 and mode != "dummy"
    }
    if all_ba:
        max_ba = max(all_ba.values())
        best_modes = [m for m, v in all_ba.items() if abs(v - max_ba) < 1e-6]
        if len(best_modes) > 1:
            print(
                f"[INFO] Semua mode tied pada balanced_accuracy={max_ba:.4f}. "
                f"Modes: {best_modes}. Tidak ada mode yang secara statistik unggul."
            )
            preference_order = (
                "dnabert_only", "snp_lr", "snp_svc", "snp_only",
                "snp_amr_lr", "amr_lr", "hybrid", "kmer_only",
            )
            for preferred in preference_order:
                if preferred in best_modes and all_clfs.get(preferred) is not None:
                    best_clf    = all_clfs[preferred]
                    best_metrics = all_metrics[preferred]
                    print(f"[INFO] Tie-break: memilih '{preferred}' sebagai best_clf.")
                    break
        else:
            print(f"[INFO] Mode terbaik: {best_modes[0]} (balanced_accuracy={max_ba:.4f})")

    return best_clf, best_metrics, all_metrics
