#!/usr/bin/env python3
"""
Post-Snippy QC filter: parse core.txt stats, drop outlier isolates,
re-run snippy-core on clean isolates → core_clean.aln.

Usage (from project root in WSL):
    python scripts/snippy_qc_filter.py
    python scripts/snippy_qc_filter.py --max-snp 20000 --max-unaligned 500000 --max-het 1000
    python scripts/snippy_qc_filter.py --dry-run   # preview without re-running

Defaults (tuned for Salmonella enterica vs GCF_000006945.2):
    --max-snp       10000    isolates with >10k SNPs vs reference are likely wrong serovar/species
    --max-unaligned 500000   >500 kbp unaligned = poor reference coverage
    --max-het       1000     high het in a haploid bacterium = mixed/contaminated assembly

Output:
    data/processed/snippy/qc_report.tsv      full per-isolate table (PASS/FAIL + reason)
    data/processed/snippy/passing_isolates.txt  one isolate dir per line
    data/processed/snippy/core_clean.*          clean core alignment files (after re-run)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

DOCKER_IMG = "staphb/snippy:4.6.0"

DEFAULT_MAX_SNP       = 10_000
DEFAULT_MAX_UNALIGNED = 500_000
DEFAULT_MAX_HET       = 1_000


# ── core.txt parser ──────────────────────────────────────────────────────────

def parse_core_txt(path: Path) -> pd.DataFrame:
    """
    Parse snippy-core core.txt (tab-separated) into a DataFrame.

    Expected columns (case-insensitive):
        ID  LENGTH  ALIGNED  UNALIGNED  %ALIGNED  HET  SNP  INS  DEL

    The 'Reference' row is excluded from QC since it is always perfect.
    """
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip().upper().lstrip("%") for c in df.columns]

    # Normalise common column-name variants
    rename = {
        "VARIANTS": "SNP",
        "SNPS":     "SNP",
        "HETERO":   "HET",
    }
    df.rename(columns=rename, inplace=True)

    # Drop Reference row
    df = df[~df["ID"].isin(["Reference", "Ref", "reference"])].copy()

    for col in ("SNP", "UNALIGNED", "HET", "INS", "DEL"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df.reset_index(drop=True)


# ── QC filter ────────────────────────────────────────────────────────────────

def apply_qc(df: pd.DataFrame, max_snp: int, max_unaligned: int, max_het: int) -> pd.DataFrame:
    df = df.copy()
    reasons = []
    for _, row in df.iterrows():
        r = []
        if row.get("SNP",       0) > max_snp:
            r.append(f"snp={row['SNP']:,} > {max_snp:,}")
        if row.get("UNALIGNED", 0) > max_unaligned:
            r.append(f"unaligned={row['UNALIGNED']:,} > {max_unaligned:,}")
        if row.get("HET",       0) > max_het:
            r.append(f"het={row['HET']:,} > {max_het:,}")
        reasons.append("; ".join(r))

    df["FAIL_REASON"] = reasons
    df["QC_PASS"]     = df["FAIL_REASON"] == ""
    return df


# ── snippy-core via Docker ───────────────────────────────────────────────────

def run_snippy_core(project_root: Path, ref: str, isolate_dirs: list[str], prefix: str) -> None:
    """
    Run snippy-core inside the staphb/snippy Docker container.
    All paths are relative to project_root (= /data inside the container).
    """
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{project_root}:/data",
        "-w", "/data",
        DOCKER_IMG,
        "snippy-core",
        "--ref",    ref,
        "--prefix", prefix,
        *isolate_dirs,
    ]

    n = len(isolate_dirs)
    print(f"\n[SNIPPY-CORE] Running on {n} clean isolates ...")
    print(f"[SNIPPY-CORE] Output prefix: {prefix}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] snippy-core exited with code {result.returncode}")
        sys.exit(1)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter Snippy outliers and re-run snippy-core on clean isolates"
    )
    ap.add_argument("--max-snp",       type=int, default=DEFAULT_MAX_SNP,
                    help=f"Max SNPs per isolate vs reference (default {DEFAULT_MAX_SNP:,})")
    ap.add_argument("--max-unaligned", type=int, default=DEFAULT_MAX_UNALIGNED,
                    help=f"Max unaligned bases (default {DEFAULT_MAX_UNALIGNED:,})")
    ap.add_argument("--max-het",       type=int, default=DEFAULT_MAX_HET,
                    help=f"Max heterozygous calls (default {DEFAULT_MAX_HET:,})")
    ap.add_argument("--snippy-dir",    default="data/processed/snippy",
                    help="Snippy output directory (relative to project root)")
    ap.add_argument("--ref",           default="data/raw/genomes/GCF_000006945.2.fna",
                    help="Reference FASTA path (relative to project root)")
    ap.add_argument("--dry-run",       action="store_true",
                    help="Show QC report only — do not re-run snippy-core")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    snippy_dir   = project_root / args.snippy_dir
    core_txt     = snippy_dir / "core.txt"

    if not core_txt.exists():
        print(f"[ERROR] core.txt not found: {core_txt}")
        print("        Run scripts/run_snippy.sh first.")
        sys.exit(1)

    # ── Parse & filter ────────────────────────────────────────────────────────
    df = parse_core_txt(core_txt)
    df = apply_qc(df, args.max_snp, args.max_unaligned, args.max_het)

    n_total = len(df)
    n_pass  = int(df["QC_PASS"].sum())
    n_fail  = n_total - n_pass

    print(f"\n{'='*62}")
    print("SNIPPY POST-QC FILTER REPORT")
    print(f"{'='*62}")
    print(f"core.txt        : {core_txt}")
    print(f"Total isolates  : {n_total}")
    print(f"Passing QC      : {n_pass}")
    print(f"Failing QC      : {n_fail}")
    print(f"\nThresholds applied:")
    print(f"  max_snp             = {args.max_snp:,}")
    print(f"  max_unaligned_bases = {args.max_unaligned:,}")
    print(f"  max_het             = {args.max_het:,}")

    if n_fail:
        print(f"\nFailed isolates ({n_fail}):")
        for _, row in df[~df["QC_PASS"]].iterrows():
            print(f"  {row['ID']:<35}  {row['FAIL_REASON']}")

    # Warn on snp=0 (not filtered, but flag for manual review)
    zero_snp = df[(df["QC_PASS"]) & (df.get("SNP", pd.Series(dtype=int)) == 0)]
    if len(zero_snp):
        print(f"\n[WARN] {len(zero_snp)} passing isolat dengan SNP=0 (identik/hampir identik dengan reference):")
        for _, row in zero_snp.iterrows():
            print(f"  {row['ID']}")
        print("  Pertimbangkan cek metadata: mungkin reference strain atau duplikat.")

    print(f"{'='*62}\n")

    # ── Save QC report ────────────────────────────────────────────────────────
    qc_report = snippy_dir / "qc_report.tsv"
    df.to_csv(qc_report, sep="\t", index=False)
    print(f"QC report       : {qc_report}")

    # ── Collect passing isolate dirs ──────────────────────────────────────────
    passing_dirs_abs  = []
    passing_dirs_rel  = []   # relative to project_root (for Docker /data mount)
    missing           = []

    for acc in df[df["QC_PASS"]]["ID"]:
        abs_dir = snippy_dir / acc
        rel_dir = str(abs_dir.relative_to(project_root))
        if abs_dir.is_dir():
            passing_dirs_abs.append(str(abs_dir))
            passing_dirs_rel.append(rel_dir)
        else:
            missing.append(acc)

    if missing:
        print(f"[WARN] {len(missing)} passing isolat tidak ada direktorinya (dilewati):")
        for m in missing:
            print(f"  {m}")

    pass_list = snippy_dir / "passing_isolates.txt"
    pass_list.write_text("\n".join(passing_dirs_abs) + "\n")
    print(f"Passing dirs    : {pass_list}  ({len(passing_dirs_rel)} isolat)")

    if args.dry_run:
        print("\n[DRY RUN] snippy-core tidak dijalankan. Hapus --dry-run untuk eksekusi.")
        return

    if not passing_dirs_rel:
        print("[ERROR] Tidak ada isolat yang lolos QC. Periksa threshold dan coba lagi.")
        sys.exit(1)

    # ── Re-run snippy-core on clean isolates ──────────────────────────────────
    clean_prefix = str(Path(args.snippy_dir) / "core_clean")
    run_snippy_core(
        project_root=project_root,
        ref=args.ref,
        isolate_dirs=passing_dirs_rel,
        prefix=clean_prefix,
    )

    # ── Report output ─────────────────────────────────────────────────────────
    print("\n[SNIPPY-CORE] Output files:")
    for ext in ("aln", "full.aln", "tab", "txt"):
        f = project_root / f"{clean_prefix}.{ext}"
        if f.exists():
            print(f"  {f.stat().st_size:>12,} bytes  {clean_prefix}.{ext}")

    print(f"\n[OK] Gunakan core_clean.aln di Stage 3 notebook.")
    print(f"     Update config.yaml: snippy_core_aln: {clean_prefix}.aln")


if __name__ == "__main__":
    main()
