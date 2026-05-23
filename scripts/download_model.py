"""
Download dan verifikasi bobot model DNABERT-2-117M ke weights/dnabert2/.

DNABERT-2 adalah model transformer PyTorch dari HuggingFace.
Framework: PyTorch (bukan Keras/TensorFlow).

Jalankan sekali dari root project:
    python scripts/download_model.py
"""

import os
import sys
import time

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "weights", "dnabert2")
MODEL_ID  = "zhihan1996/DNABERT-2-117M"

os.makedirs(CACHE_DIR, exist_ok=True)


def check_existing() -> bool:
    """Return True jika bobot sudah ada di cache."""
    needed = ["config.json", "tokenizer_config.json"]
    for f in needed:
        if not any(f in p for p in os.listdir(CACHE_DIR) if os.path.isfile(os.path.join(CACHE_DIR, p))):
            pass
    # Cek rekursif
    for root, dirs, files in os.walk(CACHE_DIR):
        for fname in files:
            if fname == "config.json":
                return True
    return False


def download():
    import torch
    from transformers import AutoTokenizer, AutoModel

    print("=" * 60)
    print("  DNABERT-2 — Download Model Weights")
    print("=" * 60)
    print(f"  Model    : {MODEL_ID}")
    print(f"  Cache    : {CACHE_DIR}")
    print(f"  Backend  : PyTorch {torch.__version__}  (bukan Keras)")
    print(f"  Device   : CPU")
    print("=" * 60)

    if check_existing():
        print("\n[INFO] Bobot sudah ada di cache. Melakukan verifikasi load...")
    else:
        print("\n[1] Mengunduh tokenizer (~2 MB)...")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,   # wajib: model pakai custom BertModel dari Hub
    )
    print(f"    Tokenizer OK  (vocab_size={tokenizer.vocab_size})")

    print("[2] Mengunduh model weights (~450 MB, sabar)...")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager",    # disable FlashAttention (triton tidak ada di Windows CPU)
        low_cpu_mem_usage=False,        # disable meta-device lazy init (transformers v5)
    )
    model.eval()
    elapsed = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model OK  ({n_params:,} parameter, {elapsed:.1f} detik)")

    # Verifikasi inference dengan urutan pendek
    print("\n[3] Verifikasi inference...")
    test_seq = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    import torch
    inputs = tokenizer(test_seq, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    print(f"    Input  : '{test_seq[:30]}...' ({len(test_seq)} bp)")
    print(f"    Tokens : {input_ids.shape}  -> {input_ids.shape[1]} token BPE")

    with torch.no_grad():
        outputs = model(input_ids)

    hidden = outputs[0]   # (1, seq_len, 768)
    emb = hidden[0].mean(dim=0)  # (768,)
    print(f"    Output : hidden_state {tuple(hidden.shape)}  -> embedding {tuple(emb.shape)}")
    print(f"    Embedding norm : {emb.norm().item():.4f}  (>0 = OK)")

    print("\n" + "=" * 60)
    print("  Setup selesai! Model siap dipakai sebagai feature extractor.")
    print(f"  Cache tersimpan di: {CACHE_DIR}")
    print("=" * 60)

    # Tulis marker agar pipeline tahu model sudah siap
    marker = os.path.join(CACHE_DIR, "READY")
    with open(marker, "w") as f:
        import datetime
        f.write(f"Downloaded: {datetime.datetime.now()}\nModel: {MODEL_ID}\nParams: {n_params:,}\n")


if __name__ == "__main__":
    try:
        download()
    except ImportError as e:
        print(f"\n[ERROR] Package belum terinstall: {e}")
        print("Jalankan dulu: pip install torch transformers einops")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
