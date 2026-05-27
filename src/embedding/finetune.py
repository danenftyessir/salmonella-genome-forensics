"""
LoRA fine-tuning pipeline for DNABERT-2 + AttentionMIL (end-to-end).

Architecture:
    raw_window_seqs → DNABERT-2 + LoRA adapters → (n_windows, 768)
    → AttentionMIL → (n_classes,) logits

Unlike the frozen-DNABERT path, here DNABERT weights are partially trainable
(only LoRA adapter parameters) so we cannot pre-cache embeddings.

Requirements:
    pip install peft

Enable via config:
    dnabert_finetune:
      enabled: true
"""

from __future__ import annotations

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .mil import AttentionMIL


# ---------------------------------------------------------------------------
# Model loading with LoRA
# ---------------------------------------------------------------------------

def load_lora_model(cfg: dict):
    """
    Load DNABERT-2 and wrap it with LoRA adapters via PEFT.

    Returns (tokenizer, lora_model).
    Falls back gracefully if peft is not installed.
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        raise ImportError(
            "peft is required for LoRA fine-tuning. "
            "Install with: pip install peft"
        )

    from .model import load_model as _load_base

    dna_cfg  = cfg["dnabert"]
    lora_cfg = cfg.get("lora", {})

    tokenizer, base_model = _load_base(dna_cfg["model_id"], dna_cfg["cache_dir"])

    # Auto-detect valid target modules from model named_modules
    target_modules = lora_cfg.get("target_modules", ["Wqkv", "query", "key", "value", "dense"])
    available = {name.split(".")[-1] for name, _ in base_model.named_modules()}
    valid_targets = [m for m in target_modules if m in available]
    if not valid_targets:
        # Fallback: use all Linear layers (safe but more params)
        valid_targets = None  # PEFT will auto-target all linear layers

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=valid_targets,
        bias="none",
        # FEATURE_EXTRACTION: no classification head in base model
        task_type="FEATURE_EXTRACTION",
    )

    lora_model = get_peft_model(base_model, peft_config)
    lora_model.print_trainable_parameters()
    return tokenizer, lora_model


# ---------------------------------------------------------------------------
# End-to-end pipeline model
# ---------------------------------------------------------------------------

class LoRaMILPipeline(nn.Module):
    """
    End-to-end LoRA-DNABERT + AttentionMIL model.

    forward(window_seqs, tokenizer, device) → (logits, attention_weights)
    """

    def __init__(self, lora_model, mil_module: AttentionMIL) -> None:
        super().__init__()
        self.encoder = lora_model
        self.mil     = mil_module

    def encode_windows(
        self,
        window_seqs: list[str],
        tokenizer,
        device: str,
        max_length: int = 512,
    ) -> torch.Tensor:
        """Tokenize + forward through LoRA-DNABERT → (n_windows, 768)."""
        all_embs = []
        for seq in window_seqs:
            inputs = tokenizer(
                seq,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            input_ids      = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            outputs = self.encoder(input_ids)
            hidden  = outputs[0]                           # (1, seq_len, 768)
            mask    = attention_mask.unsqueeze(-1).float()
            emb     = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embs.append(emb.squeeze(0))                # (768,)

        return torch.stack(all_embs)                       # (n_windows, 768)

    def forward(self, window_seqs: list[str], tokenizer, device: str):
        window_embs       = self.encode_windows(window_seqs, tokenizer, device)
        logits, weights   = self.mil(window_embs)
        return logits, weights


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lora_mil(
    train_windows: dict,
    y_train: dict,
    label_names: list[str],
    cfg: dict,
) -> LoRaMILPipeline:
    """
    Fine-tune DNABERT-2 (LoRA) + AttentionMIL end-to-end on source attribution.

    Parameters
    ----------
    train_windows : {acc: [window_seq, ...]}  — training isolates only
    y_train       : {acc: label_str}          — source_binary per isolate
    label_names   : sorted list of class strings
    cfg           : full project config dict

    Returns
    -------
    Trained LoRaMILPipeline (eval mode), ready for predict_lora_mil().
    """
    ft_cfg   = cfg.get("dnabert_finetune", {})
    mil_cfg  = cfg.get("mil", {})
    device   = cfg["dnabert"].get("device", "cpu")

    lr        = ft_cfg.get("learning_rate", 2e-5)
    epochs    = ft_cfg.get("epochs", 8)
    patience  = ft_cfg.get("early_stopping_patience", 2)
    dropout   = ft_cfg.get("dropout", 0.2)

    n_classes  = len(label_names)
    label2idx  = {lbl: i for i, lbl in enumerate(label_names)}

    tokenizer, lora_model = load_lora_model(cfg)
    lora_model = lora_model.to(device)

    mil_module = AttentionMIL(
        hidden_size=768,
        attn_hidden=mil_cfg.get("hidden_size", 256),
        n_classes=n_classes,
        dropout=dropout,
    ).to(device)

    pipeline   = LoRaMILPipeline(lora_model, mil_module).to(device)
    optimizer  = Adam(
        filter(lambda p: p.requires_grad, pipeline.parameters()),
        lr=lr,
        weight_decay=ft_cfg.get("weight_decay", 0.01),
    )
    criterion  = nn.CrossEntropyLoss()

    # Build sample list
    samples: list[tuple[list[str], int]] = []
    for acc, wins in train_windows.items():
        if acc not in y_train or not wins:
            continue
        lbl = y_train[acc]
        if lbl not in label2idx:
            continue
        samples.append((wins, label2idx[lbl]))

    if not samples:
        raise ValueError("train_lora_mil: tidak ada sampel valid.")

    rng        = np.random.default_rng(cfg.get("ml", {}).get("random_state", 42))
    best_loss  = float("inf")
    no_improve = 0

    pipeline.train()
    for epoch in range(epochs):
        order      = rng.permutation(len(samples))
        epoch_loss = 0.0
        steps      = 0

        for idx in order:
            wins, y_idx = samples[idx]
            y_t = torch.tensor([y_idx], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits, _ = pipeline(wins, tokenizer, device)
            loss      = criterion(logits.unsqueeze(0), y_t)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps      += 1

        avg_loss = epoch_loss / max(steps, 1)
        if avg_loss < best_loss - 1e-4:
            best_loss  = avg_loss
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 2 == 0:
            print(f"    [LoRA-MIL] epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

        if no_improve >= patience:
            print(f"    [LoRA-MIL] Early stopping at epoch {epoch + 1}")
            break

    pipeline.eval()
    return pipeline


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_lora_mil(
    pipeline: LoRaMILPipeline,
    windows: list[str],
    label_names: list[str],
    cfg: dict,
) -> str:
    """Predict source label for one isolate using LoRA-MIL pipeline."""
    device = cfg["dnabert"].get("device", "cpu")
    tokenizer_ref = getattr(pipeline, "_tokenizer", None)
    if tokenizer_ref is None:
        # Re-load tokenizer (lightweight, cached by HuggingFace)
        from .model import load_model as _load_base
        dna_cfg = cfg["dnabert"]
        tokenizer_ref, _ = _load_base(dna_cfg["model_id"], dna_cfg["cache_dir"])

    pipeline.eval()
    with torch.no_grad():
        logits, _ = pipeline(windows, tokenizer_ref, device)
        pred_idx  = int(logits.argmax().item())
    return label_names[pred_idx]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_lora(pipeline: LoRaMILPipeline, save_dir: str) -> None:
    """Save LoRA adapters + MIL weights to disk."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    pipeline.encoder.save_pretrained(os.path.join(save_dir, "lora_adapter"))
    torch.save(pipeline.mil.state_dict(), os.path.join(save_dir, "mil_head.pt"))
    print(f"[SAVE] LoRA-MIL → {save_dir}")


def load_lora_pipeline(save_dir: str, cfg: dict) -> LoRaMILPipeline:
    """Reload a saved LoRA-MIL pipeline."""
    import os
    from peft import PeftModel
    from .model import load_model as _load_base

    dna_cfg = cfg["dnabert"]
    mil_cfg = cfg.get("mil", {})
    tokenizer, base_model = _load_base(dna_cfg["model_id"], dna_cfg["cache_dir"])
    lora_model = PeftModel.from_pretrained(base_model, os.path.join(save_dir, "lora_adapter"))

    n_classes  = 2  # binary by default; adjust if needed
    mil_module = AttentionMIL(
        hidden_size=768,
        attn_hidden=mil_cfg.get("hidden_size", 256),
        n_classes=n_classes,
        dropout=mil_cfg.get("dropout", 0.2),
    )
    mil_module.load_state_dict(torch.load(os.path.join(save_dir, "mil_head.pt"), map_location="cpu"))

    pipeline = LoRaMILPipeline(lora_model, mil_module)
    pipeline.eval()
    return pipeline
