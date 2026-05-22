"""Global reproducibility seed — call once at the very start of any entry point."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """
    Fix all sources of randomness so that results are reproducible across runs
    given the same data, config, and seed.

    Covers:
      • Python built-in hash seed (PYTHONHASHSEED)
      • Python random module
      • NumPy random generator
      • PyTorch CPU and CUDA seeds (if torch is installed)
      • cuDNN deterministic mode (if CUDA is available)

    Call this before any data splitting, model training, UMAP/PCA, or embedding.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"[SEED] Global seed set to {seed}")
