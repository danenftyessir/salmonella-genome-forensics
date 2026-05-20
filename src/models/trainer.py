"""Feature preparation and classifier training for three feature modes."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def _join_target(feature_df: pd.DataFrame, metadata_df: pd.DataFrame, target_col: str):
    """Inner-join feature_df with the target column from metadata, drop NaN rows."""
    merged = feature_df.join(
        metadata_df.set_index("assembly_accession")[target_col], how="inner"
    ).dropna(subset=[target_col])
    X = merged.drop(columns=[target_col]).values.astype(float)
    y = merged[target_col].values
    label_names = sorted(merged[target_col].unique().tolist())
    return X, y, label_names


def get_sample_index(
    feature_df: pd.DataFrame, metadata_df: pd.DataFrame, target_col: str
) -> pd.Index:
    """Return the sample index produced by the same inner join as _join_target."""
    merged = feature_df.join(
        metadata_df.set_index("assembly_accession")[target_col], how="inner"
    ).dropna(subset=[target_col])
    return merged.index


def prepare_features(
    embedding_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    target_col: str,
):
    """DNABERT-only feature set."""
    return _join_target(embedding_df, metadata_df, target_col)


def prepare_snp_features(
    snp_encoded_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    target_col: str,
):
    """SNP-only feature set (integer-encoded SNP matrix)."""
    return _join_target(snp_encoded_df, metadata_df, target_col)


def prepare_hybrid_features(
    snp_encoded_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    target_col: str,
):
    """Hybrid feature set: integer-encoded SNP matrix concatenated with DNABERT embeddings."""
    # Align on shared accessions
    combined = snp_encoded_df.join(embedding_df, how="inner")
    return _join_target(combined, metadata_df, target_col)


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------

def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "RandomForest",
    random_state: int = 42,
    n_estimators: int = 100,
    scale: bool = False,
) -> tuple:
    """
    Fit a classifier and return (clf, scaler).

    scaler is a fitted StandardScaler when scale=True (used for SVM),
    otherwise None.  The caller is responsible for applying scaler.transform
    to X_test before calling clf.predict / clf.predict_proba.
    """
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    if model_type == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )
    elif model_type == "SVM":
        clf = SVC(kernel="rbf", probability=True, random_state=random_state)
    else:
        raise ValueError(f"Model tidak dikenal: '{model_type}'. Pilih 'RandomForest' atau 'SVM'.")

    clf.fit(X_train, y_train)
    print(f"Model {model_type} selesai dilatih  (scale={scale}).")
    return clf, scaler
