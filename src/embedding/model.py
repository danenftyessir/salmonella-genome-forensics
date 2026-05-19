"""Load DNABERT-2-117M tokenizer and model from HuggingFace cache."""

import sys
import types
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig


def _stub_triton() -> None:
    # DNABERT-2 remote code imports triton at module level for FlashAttention.
    # transformers' check_imports() fails on Windows/CPU before we can set
    # attn_implementation="eager".  Injecting a stub lets the check pass;
    # the eager attention path never calls any triton code at runtime.
    for name in ("triton", "triton.language", "triton.ops", "triton.ops.blocksparse"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    # wire sub-modules as attributes so `import triton; triton.language` works
    sys.modules["triton"].language = sys.modules["triton.language"]


def load_model(model_id: str = "zhihan1996/DNABERT-2-117M", cache_dir: str = "weights/dnabert2/"):
    """Return (tokenizer, model) ready for CPU inference."""
    _stub_triton()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    # Newer transformers raises AttributeError (not None) for missing config
    # attributes. DNABERT-2's BertConfig omits pad_token_id; inject it from
    # the tokenizer so BertEmbeddings.__init__ can read it.
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    model = AutoModel.from_pretrained(
        model_id,
        config=config,
        cache_dir=cache_dir,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
    )
    model.eval()
    print(f"Model dimuat: {model_id}  (params: {sum(p.numel() for p in model.parameters()):,})")
    return tokenizer, model
