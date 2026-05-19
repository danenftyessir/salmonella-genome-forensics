"""Dimensionality reduction: PCA, UMAP, t-SNE."""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def run_pca(embedding_df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(embedding_df.values)
    explained = pca.explained_variance_ratio_
    print(f"PCA variance explained: {[f'{v:.1%}' for v in explained]}")
    return pd.DataFrame(coords, index=embedding_df.index, columns=[f"PC{i+1}" for i in range(n_components)])


def run_umap(embedding_df: pd.DataFrame, n_components: int = 2, n_neighbors: int = 15) -> pd.DataFrame:
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
    coords = reducer.fit_transform(embedding_df.values)
    return pd.DataFrame(coords, index=embedding_df.index, columns=["UMAP1", "UMAP2"])


def run_tsne(embedding_df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    perplexity = min(30, len(embedding_df) - 1)
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(embedding_df.values)
    return pd.DataFrame(coords, index=embedding_df.index, columns=["tSNE1", "tSNE2"])
