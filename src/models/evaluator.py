"""Evaluate classifier: accuracy, F1, balanced accuracy, confusion matrix, ROC curve."""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
)
from utils.io import ensure_dir


def evaluate(clf, X_test, y_test, label_names: list) -> dict:
    y_pred = clf.predict(X_test)
    acc          = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_weighted  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro     = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    report       = classification_report(y_test, y_pred, zero_division=0)
    print(
        f"Accuracy      : {acc:.4f}\n"
        f"Balanced Acc. : {balanced_acc:.4f}\n"
        f"F1 (weighted) : {f1_weighted:.4f}\n"
        f"F1 (macro)    : {f1_macro:.4f}\n"
        f"{report}"
    )
    return {
        "accuracy":          acc,
        "balanced_accuracy": balanced_acc,
        "f1_weighted":       f1_weighted,
        "f1_macro":          f1_macro,
        "report":            report,
        "y_pred":            y_pred,
    }


def plot_confusion_matrix(y_test, y_pred, label_names: list, out_path: str):
    ensure_dir(out_path.rsplit("/", 1)[0])
    cm   = confusion_matrix(y_test, y_pred, labels=label_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_roc_curve(clf, X_test, y_test, out_path: str):
    """Binary-only ROC curve. Skip for multiclass."""
    ensure_dir(out_path.rsplit("/", 1)[0])
    classes = clf.classes_
    if len(classes) != 2:
        print("[INFO] ROC curve hanya untuk klasifikasi biner, dilewati.")
        return
    if not hasattr(clf, "predict_proba"):
        print("[INFO] Classifier tidak punya predict_proba, ROC curve dilewati.")
        return
    y_prob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=classes[1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
