"""
Attention-based Multiple Instance Learning (MIL) for isolate-level classification.

Architecture:
    frozen DNABERT-2 → per-window embeddings (n_windows, 768)
    → AttentionMIL → attention-pooled (768,) + class logits

Key design: AttentionMIL is trained per CV fold on training-isolate window
embeddings only, so there is no leakage from test isolates into pooling weights.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


class AttentionMIL(nn.Module):
    """
    Attention-based MIL classifier.

    Input : (n_windows, hidden_size) — pre-computed DNABERT-2 window embeddings
    Output: (n_classes,) logits  +  (n_windows,) attention weights

    The attention sub-network learns which windows are informative for
    source attribution; weights sum to 1 (softmax) so pooled = weighted sum.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        attn_hidden: int = 256,
        n_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        X : (n_windows, hidden_size)
        Returns (logits (n_classes,), weights (n_windows,))
        """
        scores  = self.attn(X).squeeze(-1)                    # (n_windows,)
        weights = torch.softmax(scores, dim=0)                # (n_windows,)
        pooled  = (weights.unsqueeze(-1) * X).sum(dim=0)     # (hidden_size,)
        logits  = self.classifier(pooled)                     # (n_classes,)
        return logits, weights


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mil(
    window_embs_dict: dict,
    labels_dict: dict,
    label_names: list[str],
    mil_cfg: dict,
    device: str = "cpu",
) -> AttentionMIL:
    """
    Train AttentionMIL on pre-computed frozen-DNABERT window embeddings.

    Parameters
    ----------
    window_embs_dict : {acc: np.ndarray (n_windows, 768)}  — training isolates only
    labels_dict      : {acc: str}                          — source_binary label per isolate
    label_names      : sorted list of class strings (determines class index order)
    mil_cfg          : dict with keys: hidden_size, dropout, epochs,
                       learning_rate, early_stopping_patience
    device           : "cpu" or "cuda"

    Returns
    -------
    Trained AttentionMIL (eval mode).
    """
    hidden_size = mil_cfg.get("hidden_size", 256)
    dropout     = mil_cfg.get("dropout", 0.2)
    epochs      = mil_cfg.get("epochs", 30)
    lr          = mil_cfg.get("learning_rate", 1e-3)
    patience    = mil_cfg.get("early_stopping_patience", 5)

    n_classes = len(label_names)
    label2idx = {lbl: i for i, lbl in enumerate(label_names)}

    model = AttentionMIL(
        hidden_size=768,
        attn_hidden=hidden_size,
        n_classes=n_classes,
        dropout=dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Build list of (tensor, label_idx) pairs
    samples: list[tuple[torch.Tensor, int]] = []
    for acc, emb in window_embs_dict.items():
        if acc not in labels_dict:
            continue
        lbl = labels_dict[acc]
        if lbl not in label2idx:
            continue
        X = torch.tensor(emb, dtype=torch.float32, device=device)
        y = torch.tensor(label2idx[lbl], dtype=torch.long, device=device)
        samples.append((X, y))

    if not samples:
        raise ValueError("train_mil: tidak ada sampel valid untuk dilatih.")

    rng = np.random.default_rng(42)
    best_loss = float("inf")
    no_improve = 0

    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(samples))
        epoch_loss = 0.0
        for idx in order:
            X, y = samples[idx]
            optimizer.zero_grad()
            logits, _ = model(X)
            loss = criterion(logits.unsqueeze(0), y.unsqueeze(0))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(samples)
        if avg_loss < best_loss - 1e-4:
            best_loss  = avg_loss
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0:
            print(f"    [MIL] epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

        if no_improve >= patience:
            print(f"    [MIL] Early stopping at epoch {epoch + 1}")
            break

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_mil(
    model: AttentionMIL,
    window_embs: np.ndarray,
    label_names: list[str],
    device: str = "cpu",
) -> tuple[str, np.ndarray]:
    """
    Predict source label for one isolate.

    Parameters
    ----------
    model        : trained AttentionMIL
    window_embs  : (n_windows, 768) float32 array
    label_names  : class index → class name mapping

    Returns
    -------
    (predicted_label, attention_weights_array)
    """
    model.eval()
    with torch.no_grad():
        X = torch.tensor(window_embs, dtype=torch.float32, device=device)
        logits, weights = model(X)
        pred_idx = int(logits.argmax().item())
    return label_names[pred_idx], weights.cpu().numpy()


def predict_mil_proba(
    model: AttentionMIL,
    window_embs: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Return softmax probability vector (n_classes,)."""
    model.eval()
    with torch.no_grad():
        X = torch.tensor(window_embs, dtype=torch.float32, device=device)
        logits, _ = model(X)
        probs = torch.softmax(logits, dim=0).cpu().numpy()
    return probs


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------

def get_top_windows(
    attention_weights: np.ndarray,
    windows: list[str],
    top_k: int = 10,
) -> list[dict]:
    """
    Return the top-k most attended windows with their weights.

    Used for biological interpretability: high-attention windows are
    likely flanking genomic regions most discriminative for source attribution.
    """
    if len(windows) == 0 or len(attention_weights) == 0:
        return []
    top_k  = min(top_k, len(windows))
    top_idx = np.argsort(attention_weights)[::-1][:top_k]
    return [
        {"rank": r + 1, "window_index": int(i), "attention": float(attention_weights[i]),
         "sequence_snippet": windows[i][:40] + "..." if len(windows[i]) > 40 else windows[i]}
        for r, i in enumerate(top_idx)
    ]
