"""
Reference-based core-SNP extraction.

Pipeline:
  1. Build k-mer index for reference genome (unique 21-mers, vectorised NumPy).
  2. Align each assembly's contigs to the reference via diagonal k-mer chaining.
  3. Fill a pre-allocated (n_isolates x ref_len) uint8 matrix with base calls
     (0 = uncalled, 65/67/71/84 = A/C/G/T ASCII).
  4. Keep only core positions (callable in >= min_core_fraction of isolates).
  5. Remove invariant sites (no variation across isolates).

Memory design:
  The old implementation used a Python dict {ref_pos → base} per isolate, which
  costs ~130 bytes/entry x 4.8M positions x 50 isolates ≈ 31 GB — silently causing
  OOM kills.  The current design uses a pre-allocated uint8 numpy matrix:
    50 isolates x 4.8M positions x 1 byte ≈ 240 MB total — manageable.

Algorithmic accuracy (why this beats the reference-free approach):
  - All positions are in reference coordinate space (GCF_000006945.2).
  - Contig reordering and genome rearrangements do NOT create spurious SNPs.
  - Only positions with biologically meaningful variation are retained.
  - Expected output: 5 k – 50 k real SNPs vs 4.5 M artefacts from the old method.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


BASE_TO_INT: dict[str, int] = {"A": 0, "T": 1, "G": 2, "C": 3, "N": -1}

# ASCII codes for the four canonical bases
_A, _C, _G, _T = ord("A"), ord("C"), ord("G"), ord("T")
_VALID_BASES = {_A, _C, _G, _T}

# Lookup table: ASCII index → 2-bit code (-1 for invalid/N)
_ENC = np.full(256, -1, dtype=np.int8)
_ENC[_A] = 0; _ENC[_C] = 1; _ENC[_G] = 2; _ENC[_T] = 3


# ── 2-bit rolling hash helpers ──────────────────────────────────────────────

def _compute_rolling_hashes(seq_up: str, k: int) -> np.ndarray:
    """
    Rolling 2-bit polynomial hashes for all windows of length k in seq_up.

    Hash: h[i] = enc[i]*4^(k-1) + enc[i+1]*4^(k-2) + ... + enc[i+k-1]
    This is a bijective mapping ACGT^k → [0, 4^k), so no intra-space collisions.
    For k=21: max_hash = 4^21-1 ≈ 4.4e12 << int64 max ≈ 9.2e18 → no overflow.

    Returns int64 array of length max(0, len-k+1).
    Windows containing N (or unknown bases) → -1.
    """
    n = len(seq_up)
    if n < k:
        return np.array([], dtype=np.int64)

    arr = np.frombuffer(seq_up.encode("ascii", errors="replace"), dtype=np.uint8)
    enc = _ENC[arr]                        # -1 for non-ACGT

    # Prefix-sum to count N-windows
    has_n   = enc < 0
    n_cum   = np.concatenate(([0], np.cumsum(has_n)))
    n_win   = n_cum[k:] - n_cum[:n - k + 1]

    # Polynomial hash via convolution (direct, exact integer arithmetic)
    weights = np.int64(4) ** np.arange(k - 1, -1, -1, dtype=np.int64)
    hashes  = np.convolve(enc.clip(0).astype(np.int64), weights, mode="valid")
    hashes[n_win > 0] = -1                 # invalidate N-windows
    return hashes


def _build_kmer_index(ref_seq: str, k: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sorted k-mer index for the reference (unique k-mers only).
    Repetitive k-mers are excluded to prevent multi-mapping.

    Returns (sorted_hashes : int64,  positions : int32).
    Lookup: idx = np.searchsorted(sorted_hashes, query); hit if sorted_hashes[idx]==query.
    """
    hashes = _compute_rolling_hashes(ref_seq.upper(), k)
    if len(hashes) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int32)

    positions = np.arange(len(hashes), dtype=np.int32)
    valid     = hashes >= 0
    h_v, p_v  = hashes[valid], positions[valid]

    order = np.argsort(h_v, kind="stable")
    h_s   = h_v[order]
    p_s   = p_v[order]

    # Mark BOTH members of any pair of identical hashes (repetitive regions)
    dup_right = np.concatenate(([False], h_s[1:] == h_s[:-1]))
    dup_left  = np.concatenate((h_s[:-1] == h_s[1:], [False]))
    keep      = ~(dup_left | dup_right)

    return h_s[keep], p_s[keep]


# ── Per-isolate alignment (fills one row of the master matrix) ───────────────

def _fill_isolate_row(
    contigs:     list[str],
    ref_hashes:  np.ndarray,
    ref_pos_arr: np.ndarray,
    row:         np.ndarray,    # uint8 view of shape (ref_len,), modified in-place
    k:           int = 21,
    min_votes:   int = 5,
) -> None:
    """
    Align each contig to the reference via diagonal k-mer chaining.
    Writes base ASCII codes (65/67/71/84) directly into `row`; 0 = uncalled.

    Strategy:
    - Compute rolling k-mer hashes for the contig (vectorised).
    - Binary-search each hash against the reference index (vectorised).
    - Group hits by diagonal d = contig_pos - ref_pos; the dominant diagonal
      is the consistent linear alignment of this contig to the reference.
    - Record centre-base of every anchor k-mer (consensus positions, one write
      per k-mer → full coverage for dense overlapping anchors).
    - Fill inter-anchor gaps of matching length ≤ 50 bp (SNP positions where
      the variant breaks the k-mer but flanking anchors bracket it).
    """
    ref_len = len(row)

    for raw_ctg in contigs:
        ctg = raw_ctg.upper()
        if len(ctg) < k:
            continue

        ctg_arr  = np.frombuffer(ctg.encode("ascii", errors="replace"), dtype=np.uint8)
        ctg_hash = _compute_rolling_hashes(ctg, k)
        if len(ctg_hash) == 0:
            continue

        ctg_pos  = np.arange(len(ctg_hash), dtype=np.int64)
        valid_v  = ctg_hash >= 0
        if not valid_v.any():
            continue

        q_hash = ctg_hash[valid_v]
        q_cpos = ctg_pos[valid_v]

        # Vectorised lookup in reference index
        idx = np.searchsorted(ref_hashes, q_hash)
        hit = (idx < len(ref_hashes)) & (ref_hashes[idx] == q_hash)
        if not hit.any():
            continue

        ca = q_cpos[hit]
        ra = ref_pos_arr[idx[hit]].astype(np.int64)

        # Dominant alignment diagonal
        diag_vals, diag_counts = np.unique(ca - ra, return_counts=True)
        best_diag = diag_vals[diag_counts.argmax()]
        if int(diag_counts.max()) < min_votes:
            continue

        on_diag = (ca - ra) == best_diag
        ca = ca[on_diag]; ra = ra[on_diag]
        order = np.argsort(ca)
        ca = ca[order]; ra = ra[order]

        # ── Centre-base recording (consensus positions, vectorised) ────────
        mid   = k // 2
        r_mid = (ra + mid).astype(np.int64)
        c_mid = (ca + mid).astype(np.int64)

        # Clip to valid range
        in_range  = (c_mid >= 0) & (c_mid < len(ctg_arr)) & \
                    (r_mid >= 0) & (r_mid < ref_len)
        r_mid     = r_mid[in_range]
        c_mid     = c_mid[in_range]
        bases_mid = ctg_arr[c_mid]
        valid_b   = (bases_mid == _A) | (bases_mid == _C) | \
                    (bases_mid == _G) | (bases_mid == _T)

        # Only write where not yet called (first-write-wins per contig order)
        not_yet = row[r_mid[valid_b]] == 0
        rp_write = r_mid[valid_b][not_yet]
        cb_write = bases_mid[valid_b][not_yet]
        row[rp_write] = cb_write

        # ── Gap filling: SNP positions between consecutive anchors ─────────
        if len(ca) < 2:
            continue

        ri1, ri2 = ra[:-1], ra[1:]
        ci1, ci2 = ca[:-1], ca[1:]
        ref_gap  = ri2 - (ri1 + k)
        ctg_gap  = ci2 - (ci1 + k)

        snp_gaps = np.where((ref_gap == ctg_gap) & (ref_gap > 0) & (ref_gap <= 50))[0]

        for ai in snp_gaps.tolist():
            rgs      = int(ri1[ai]) + k
            rge      = int(ri2[ai])
            cgs      = int(ci1[ai]) + k
            gap_size = rge - rgs

            for j in range(gap_size):
                rp = rgs + j
                cp = cgs + j
                if rp < ref_len and cp < len(ctg_arr):
                    b = int(ctg_arr[cp])
                    if b in _VALID_BASES and row[rp] == 0:
                        row[rp] = b


# ── Public API ───────────────────────────────────────────────────────────────

def build_core_snp_matrix(
    genomes: dict[str, str | list[str]],
    reference_seq: str,
    k: int = 21,
    min_core_fraction: float = 0.95,
) -> pd.DataFrame:
    """
    Reference-based core-SNP matrix.

    Parameters
    ----------
    genomes           : {accession: full_seq_str} OR {accession: [contig, ...]}
                        Pass individual contigs (list form) for best accuracy;
                        concatenated strings work but may miss inter-contig SNPs.
    reference_seq     : Reference genome sequence (chromosome + plasmids cat.).
    k                 : K-mer length for alignment anchoring (default 21).
    min_core_fraction : Fraction of isolates where a position must be callable
                        (default 0.95 = soft-core; use 1.0 for strict core).

    Returns
    -------
    DataFrame (isolates x variable SNP positions).
    Column labels are integer reference positions.
    Bases encoded as characters A/T/G/C/N (N = uncalled at that position).

    Memory usage: ~(n_isolates x ref_len x 1 byte) + overhead ≈ 240 MB for
    50 isolates against a 4.8 Mb reference — no Python dicts per position.
    """
    ref_seq_up = reference_seq.upper()
    ref_len    = len(ref_seq_up)
    n_iso      = len(genomes)
    accessions = list(genomes.keys())

    print(f"[SNP] Building {k}-mer index for reference ({ref_len:,} bp)...")
    ref_hashes, ref_pos_arr = _build_kmer_index(ref_seq_up, k)
    print(f"[SNP] Unique {k}-mers in reference: {len(ref_hashes):,}")

    if len(ref_hashes) == 0:
        print("[WARN] Tidak ada unique k-mer di referensi — periksa file referensi.")
        return pd.DataFrame()

    # Pre-allocate (n_iso, ref_len) uint8 matrix — 0 = uncalled
    # For 50 isolates x 4.8M bp: 50 x 4.8M x 1 byte ≈ 240 MB
    print(f"[SNP] Alokasi matrix {n_iso} x {ref_len:,} uint8 "
          f"({n_iso * ref_len / 1e6:.0f} MB)...")
    mat = np.zeros((n_iso, ref_len), dtype=np.uint8)

    for i, acc in enumerate(accessions):
        raw    = genomes[acc]
        contigs = raw if isinstance(raw, list) else [raw]
        _fill_isolate_row(contigs, ref_hashes, ref_pos_arr, mat[i], k=k)
        n_called = int(np.count_nonzero(mat[i]))
        print(f"  [{i+1:02d}/{n_iso}] {acc}: {n_called:,} positions called "
              f"({n_called / ref_len:.1%} of reference)")

    # Core positions: callable in >= fraction of isolates (mat[row, col] != 0)
    n_called_per_pos = np.count_nonzero(mat, axis=0)   # shape (ref_len,)
    min_count        = max(1, int(np.ceil(min_core_fraction * n_iso)))
    core_mask        = n_called_per_pos >= min_count
    core_idx         = np.where(core_mask)[0]
    print(f"[SNP] Core positions (>={min_core_fraction:.0%} callable, "
          f"n>={min_count}): {len(core_idx):,}")

    if len(core_idx) == 0:
        print("[WARN] Tidak ada core positions — coba kurangi min_core_fraction di config.yaml")
        return pd.DataFrame(index=accessions)

    # Extract core columns and filter invariant (all same base, ignoring 0)
    core_mat = mat[:, core_idx]           # shape (n_iso, n_core)

    # Vectorised variable-site detection:
    # Replace 0 (uncalled) with 255 for min computation (no valid base has code 255)
    mat_min  = np.where(core_mat > 0, core_mat, np.uint8(255))
    col_min  = mat_min.min(axis=0)         # min non-zero base code per column
    col_max  = core_mat.max(axis=0)        # max (0 if fully uncalled, else a base)
    variable = col_max > col_min           # True if at least 2 distinct non-zero bases
    del mat_min, mat                       # free memory

    var_idx = core_idx[variable]
    var_mat = core_mat[:, variable]        # shape (n_iso, n_var)
    print(f"[SNP] Variable positions: {var_mat.shape[1]:,}")

    if var_mat.shape[1] == 0:
        print("[WARN] Semua core positions invariant — dataset mungkin terlalu klonal.")
        return pd.DataFrame(index=accessions)

    # Decode uint8 → character (0 → 'N', 65/67/71/84 → A/C/G/T)
    decode = np.full(256, ord("N"), dtype=np.uint8)
    decode[_A] = _A; decode[_C] = _C; decode[_G] = _G; decode[_T] = _T

    char_mat = decode[var_mat]             # shape (n_iso, n_var), uint8 ASCII
    # Convert to Python strings for DataFrame construction
    data = {
        int(pos): [chr(b) for b in char_mat[:, j].tolist()]
        for j, pos in enumerate(var_idx.tolist())
    }

    df = pd.DataFrame(data, index=accessions)
    df.index.name = "assembly_accession"
    print(f"[SNP] Final core-SNP matrix: {df.shape[0]} isolates x {df.shape[1]:,} "
          f"variable positions")
    return df


# ── Legacy fallback ──────────────────────────────────────────────────────────

def build_snp_matrix(genomes: dict[str, str]) -> pd.DataFrame:
    """
    Reference-free SNP matrix (DEPRECATED — biologically unreliable).

    Truncates all genomes to the shortest length and compares raw sequence
    positions without alignment.  Positions between different assemblies do NOT
    correspond to the same genomic locus, producing millions of artefactual
    SNPs caused by contig reordering and genome rearrangements.

    Use build_core_snp_matrix(genomes, reference_seq) instead.
    """
    warnings.warn(
        "build_snp_matrix() is reference-free and produces biologically unreliable "
        "SNP counts (typically millions of artefactual positions).  "
        "Use build_core_snp_matrix(genomes, reference_seq) for reference-based "
        "core-SNP analysis.",
        DeprecationWarning,
        stacklevel=2,
    )
    accessions = list(genomes.keys())
    seqs       = list(genomes.values())
    min_len    = min(len(s) for s in seqs)
    seqs_c     = [s[:min_len].upper() for s in seqs]
    snp_pos    = [
        p for p in range(min_len)
        if len({s[p] for s in seqs_c} - {"N"}) > 1
    ]
    matrix = {acc: [seq[p] for p in snp_pos] for acc, seq in zip(accessions, seqs_c)}
    df = pd.DataFrame(matrix, index=snp_pos).T
    df.index.name = "assembly_accession"
    print(f"[LEGACY] {len(snp_pos):,} raw SNP positions (reference-free, unreliable)")
    return df


# ── Encoding ─────────────────────────────────────────────────────────────────

def encode_snp_matrix(snp_df: pd.DataFrame, method: str = "integer") -> pd.DataFrame:
    """Encode character SNP matrix to integers: A=0 T=1 G=2 C=3 N=-1."""
    if method != "integer":
        raise ValueError(f"Encoding '{method}' not supported.  Use 'integer'.")
    encoded = snp_df.replace(BASE_TO_INT)
    encoded = encoded.apply(pd.to_numeric, errors="coerce").fillna(-1).astype(int)
    encoded.index.name = "assembly_accession"
    print(f"SNP encoded ({method}): {encoded.shape}")
    return encoded
