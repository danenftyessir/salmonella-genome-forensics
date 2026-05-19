"""Baseline Biology 2 — validate genomic clusters against biological metadata."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from utils.io import ensure_dir


_DEFAULT_COLS = ["isolation_source", "serovar", "snp_cluster", "country", "collection_date"]


def validate_clusters_vs_metadata(
    hclust_labels: pd.Series,
    metadata_df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compare genomic cluster assignments against metadata columns.

    Returns a DataFrame with ARI and NMI per column.
    ARI  = Adjusted Rand Index      (-1..1, 1 = perfect agreement)
    NMI  = Normalized Mutual Info   ( 0..1, 1 = perfect)
    """
    if columns is None:
        columns = [c for c in _DEFAULT_COLS if c in metadata_df.columns]

    meta = (
        metadata_df.set_index("assembly_accession")
        if "assembly_accession" in metadata_df.columns
        else metadata_df
    )
    shared = hclust_labels.index.intersection(meta.index)
    if len(shared) == 0:
        print("[WARN] validate_clusters: tidak ada index yang cocok.")
        return pd.DataFrame(columns=["metadata_col", "ARI", "NMI", "n_categories"])

    h = hclust_labels.loc[shared].astype(str)
    records = []
    for col in columns:
        if col not in meta.columns:
            continue
        m = meta.loc[shared, col].fillna("unknown").astype(str)
        try:
            ari = adjusted_rand_score(m, h)
            nmi = normalized_mutual_info_score(m, h, average_method="arithmetic")
        except Exception as exc:
            print(f"  [WARN] {col}: {exc}")
            ari = nmi = float("nan")
        records.append({
            "metadata_col": col,
            "ARI": round(ari, 4),
            "NMI": round(nmi, 4),
            "n_categories": int(m.nunique()),
        })

    result = pd.DataFrame(records)
    if not result.empty:
        print("\n[Cluster Validation] ARI / NMI vs metadata:")
        print(result.to_string(index=False))
        print()
    return result


def plot_cluster_composition(
    hclust_labels: pd.Series,
    metadata_df: pd.DataFrame,
    col: str,
    out_path: str,
    title: str | None = None,
) -> None:
    """
    Stacked bar chart showing the proportion of each metadata category
    within every genomic cluster.

    Answers: "does isolation_source / serovar correlate with genomic cluster?"
    """
    ensure_dir(out_path.rsplit("/", 1)[0] if "/" in out_path else ".")

    meta = (
        metadata_df.set_index("assembly_accession")
        if "assembly_accession" in metadata_df.columns
        else metadata_df
    )
    shared = hclust_labels.index.intersection(meta.index)
    if col not in meta.columns or len(shared) == 0:
        print(f"[SKIP] plot_cluster_composition: '{col}' tidak tersedia.")
        return

    df = pd.DataFrame({
        "cluster": hclust_labels.loc[shared].astype(str),
        col: meta.loc[shared, col].fillna("unknown").astype(str),
    })
    ct = df.groupby(["cluster", col]).size().unstack(fill_value=0)
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(max(6, len(ct_norm) * 1.4), 5))
    ct_norm.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", edgecolor="white")
    ax.set_xlabel("Genomic Cluster (hierarchical)", fontsize=11)
    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_title(title or f"Cluster Composition by {col}", fontsize=12)
    ax.legend(title=col, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
