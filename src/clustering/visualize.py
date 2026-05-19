"""Plot dendrogram, distance heatmap, and embedding scatter plots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from utils.io import ensure_dir


def plot_dendrogram(linkage_matrix: np.ndarray, labels: list, out_path: str):
    ensure_dir(out_path.rsplit("/", 1)[0])
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, ax=ax)
    ax.set_title("Hierarchical Clustering — SNP Distance")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_heatmap(dist_df: pd.DataFrame, out_path: str):
    ensure_dir(out_path.rsplit("/", 1)[0])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(dist_df, ax=ax, cmap="YlOrRd", square=True, linewidths=0.3)
    ax.set_title("Pairwise SNP Distance Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_scatter(coords_df: pd.DataFrame, labels: pd.Series, title: str, out_path: str):
    ensure_dir(out_path.rsplit("/", 1)[0])
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in labels.dropna().unique():
        mask = labels == label
        ax.scatter(
            coords_df.loc[mask, coords_df.columns[0]],
            coords_df.loc[mask, coords_df.columns[1]],
            label=label, s=60, alpha=0.8,
        )
    ax.set_xlabel(coords_df.columns[0])
    ax.set_ylabel(coords_df.columns[1])
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
