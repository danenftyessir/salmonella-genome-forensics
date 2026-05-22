"""Feature preparation and classifier training for three feature modes."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
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
    snp_n_components: int = 40,
):
    """Hybrid: raw SNP (integer-encoded) concatenated with DNABERT embeddings.

    Dimensionality reduction (TruncatedSVD) is intentionally NOT performed
    here.  It is applied inside each CV fold in pipeline._run_single, fitted
    ONLY on the training fold, to prevent any information from the test fold
    leaking into the feature transformation (scikit-learn best practice).

    The snp_n_components argument is kept for API compatibility but is ignored;
    the effective n_components is read from cfg["ml"]["snp_n_components"] inside
    _run_single.
    """
    shared  = snp_encoded_df.index.intersection(embedding_df.index)
    snp_sub = snp_encoded_df.loc[shared]
    emb_sub = embedding_df.loc[shared]
    combined = snp_sub.join(emb_sub, how="inner")
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
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1,
            class_weight="balanced",
        )
    elif model_type == "SVM":
        clf = SVC(kernel="rbf", probability=True, random_state=random_state)
    elif model_type == "LogisticRegression":
        clf = LogisticRegression(
            class_weight="balanced", max_iter=5000, random_state=random_state,
        )
    elif model_type == "LinearSVC":
        clf = LinearSVC(
            class_weight="balanced", max_iter=5000, random_state=random_state,
        )
    elif model_type == "Dummy":
        clf = DummyClassifier(strategy="most_frequent", random_state=random_state)
    else:
        raise ValueError(
            f"Model tidak dikenal: '{model_type}'. "
            "Pilih 'RandomForest', 'SVM', 'LogisticRegression', 'LinearSVC', atau 'Dummy'."
        )

    clf.fit(X_train, y_train)
    print(f"Model {model_type} selesai dilatih  (scale={scale}).")
    return clf, scaler
