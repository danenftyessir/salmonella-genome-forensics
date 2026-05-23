"""Encode a single sequence window or all windows of one isolate."""

import numpy as np
import torch


def embed_sequence(seq: str, tokenizer, model, device: str = "cpu") -> np.ndarray:
    """
    Return 768-dim embedding for one DNA sequence window.

    DNABERT-2 official API: pass only input_ids (no attention_mask needed
    for single short sequences). Mean-pool over sequence-length dimension,
    excluding CLS and SEP special tokens.
    """
    # Tokenize; DNABERT-2 uses BPE, not k-mer
    inputs = tokenizer(
        seq,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    input_ids = inputs["input_ids"].to(device)          # (1, seq_len)
    attention_mask = inputs["attention_mask"].to(device) # (1, seq_len)

    with torch.no_grad():
        outputs = model(input_ids)

    # outputs[0] == last_hidden_state: (1, seq_len, 768)
    hidden = outputs[0]

    # Masked mean pooling: exclude padding tokens from average
    mask = attention_mask.unsqueeze(-1).float()          # (1, seq_len, 1)
    emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (1, 768)
    return emb.squeeze(0).cpu().numpy()


def embed_isolate(windows: list, tokenizer, model, device: str = "cpu") -> np.ndarray:
    """Mean-pool embeddings across all windows -> single 768-dim vector per isolate."""
    vecs = [embed_sequence(w, tokenizer, model, device) for w in windows]
    return np.mean(vecs, axis=0)


def embed_isolate_stat(windows: list, tokenizer, model, device: str = "cpu") -> np.ndarray:
    """Concatenate mean + max + std across window embeddings -> 2304-dim per isolate."""
    vecs = [embed_sequence(w, tokenizer, model, device) for w in windows]
    arr = np.stack(vecs)  # (n_windows, 768)
    return np.concatenate([arr.mean(axis=0), arr.max(axis=0), arr.std(axis=0)])


def embed_isolate_windows(windows: list, tokenizer, model, device: str = "cpu") -> np.ndarray:
    """Return (n_windows, 768) per-window embeddings — NOT pooled. Used by MIL."""
    vecs = [embed_sequence(w, tokenizer, model, device) for w in windows]
    return np.stack(vecs)
