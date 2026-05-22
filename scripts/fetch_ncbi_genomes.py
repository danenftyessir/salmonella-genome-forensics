"""
fetch_ncbi_genomes.py
Download additional Salmonella Typhimurium genomes + metadata from NCBI Pathogen Detection.

Usage:
    python scripts/fetch_ncbi_genomes.py [--target 80] [--nfc-ratio 0.6] [--dry-run]

  --target      Total new isolates to download (default 80)
  --nfc-ratio   Fraction that must be non_food_chain (default 0.6 = 60%)
  --dry-run     Print selection plan, no download

Steps:
  1. Stream NCBI Pathogen Detection metadata TSV
  2. Filter: serovar=Typhimurium, has assembly, isolation_source not empty, quality OK
  3. Join with SNP-cluster file
  4. Split into food_chain / non_food_chain pools
  5. Select with nfc-ratio quota + cluster-diversity cap (max 3/cluster)
  6. Download FASTA via Entrez esummary -> HTTPS FTP
  7. Append rows to metadata_salmonella.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import ssl
import sys
import time
from collections import defaultdict
from pathlib import Path

# ── SSL fix for Windows ───────────────────────────────────────────────────────
ssl._create_default_https_context = ssl._create_unverified_context

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from Bio import Entrez

# ── Config ────────────────────────────────────────────────────────────────────
Entrez.email = "pscl19623@gmail.com"

BASE_DIR   = Path(__file__).resolve().parents[1]
GENOME_DIR = BASE_DIR / "data/raw/genomes"
META_CSV   = BASE_DIR / "data/raw/metadata/metadata_salmonella.csv"

PDG_BASE    = "https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Salmonella/PDG000000002.819"
META_URL    = f"{PDG_BASE}/Metadata/PDG000000002.819.metadata.tsv"
CLUSTER_URL = f"{PDG_BASE}/Clusters/PDG000000002.819.reference_target.all_isolates.tsv"

TARGET_SEROVAR = "typhimurium"

# Assembly quality thresholds
MIN_LEN_BP  = 4_000_000
MAX_LEN_BP  = 6_000_000
MAX_CONTIGS = 500

# ── Source classification ─────────────────────────────────────────────────────
# Keywords (lowercase substring match) that signal food-chain-associated sources
FOOD_CHAIN_KEYWORDS = {
    "cow", "cattle", "bovine", "beef", "dairy",
    "chicken", "broiler", "poultry",
    "swine", "pork", "pig",
    "turkey",
    "milk", "raw milk",
    "sprout", "produce", "vegetable", "lettuce", "tomato", "pepper",
    "egg",
    "meat", "carcass", "abattoir", "slaughter",
    "food", "grocery", "retail",
    "pet food", "feed",
}

# Keywords that signal non-food-chain sources
NON_FOOD_CHAIN_KEYWORDS = {
    # human clinical
    "human", "clinical", "patient", "hospital", "stool", "blood",
    "feces", "faeces", "diarrhea", "diarrhoea", "urine", "intestine",
    "colon", "cecum", "cecal", "rectal", "wound", "gastro",
    # environment
    "water", "river", "pond", "lake", "creek", "stream", "soil",
    "drain", "floor", "surface", "sediment", "sewage", "wastewater",
    "environment", "environmental",
    # wildlife
    "bird", "wild bird", "wildlife", "sparrow", "finch", "redpoll",
    "rodent", "rat ", "mouse", "deer", "fox", "raccoon", "turtle",
    # companion animals
    "cat", "feline", "dog", "canine", "kennel", "pet",
    # other non-food animals
    "llama", "camel", "rabbit", "horse", "equine", "reptile", "lizard",
}


def classify_source(isolation_source: str) -> str:
    """Return 'food_chain', 'non_food_chain', or 'ambiguous'."""
    s = isolation_source.lower()
    is_food = any(kw in s for kw in FOOD_CHAIN_KEYWORDS)
    is_nfc  = any(kw in s for kw in NON_FOOD_CHAIN_KEYWORDS)
    if is_food and not is_nfc:
        return "food_chain"
    if is_nfc and not is_food:
        return "non_food_chain"
    if is_nfc:   # both match — conservative: treat as non_food_chain
        return "non_food_chain"
    return "ambiguous"


# ── Load existing data ────────────────────────────────────────────────────────

def load_existing(meta_csv: Path) -> tuple[set[str], set[str]]:
    """Return (existing_accessions, existing_snp_clusters)."""
    accs: set[str] = set()
    clusters: set[str] = set()
    if not meta_csv.exists():
        return accs, clusters
    with open(meta_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            accs.add(row["assembly_accession"].strip())
            cl = row.get("snp_cluster", "").strip()
            if cl and cl not in ("unknown", "NULL", ""):
                clusters.add(cl)
    return accs, clusters


# ── Stream NCBI metadata ──────────────────────────────────────────────────────

def stream_metadata(existing_accs: set[str]) -> list[dict]:
    print("[INFO] Streaming metadata from NCBI ...")
    r = requests.get(META_URL, stream=True, timeout=60, verify=False)
    r.raise_for_status()

    candidates: list[dict] = []
    header: list[str] | None = None
    buf = ""
    total_seen = 0

    for chunk in r.iter_content(chunk_size=65536):
        buf += chunk.decode("utf-8", errors="replace")
        lines = buf.split("\n")
        buf = lines[-1]

        for line in lines[:-1]:
            line = line.rstrip("\r")
            if not line:
                continue
            if header is None:
                header = line.lstrip("#").split("\t")
                continue

            total_seen += 1
            parts = line.split("\t")
            if len(parts) < len(header):
                continue

            row = dict(zip(header, parts))

            asm = row.get("asm_acc", "").strip()
            if not asm or asm.upper() in ("NULL", "NA", ""):
                continue
            if asm in existing_accs:
                continue

            sv = row.get("serovar", "").strip().lower()
            if TARGET_SEROVAR not in sv:
                continue

            src = row.get("isolation_source", "").strip()
            if not src or src.lower() in ("missing", "not provided", "not collected",
                                           "unknown", "na", "n/a", "null", ""):
                continue

            try:
                n_contig = int(row.get("asm_stats_n_contig", 9999) or 9999)
                length   = int(row.get("asm_stats_length_bp", 0) or 0)
            except ValueError:
                continue
            if n_contig > MAX_CONTIGS:
                continue
            if length > 0 and (length < MIN_LEN_BP or length > MAX_LEN_BP):
                continue

            candidates.append({
                "asm_acc":          asm,
                "target_acc":       row.get("target_acc", "").strip(),
                "isolation_source": src,
                "geo_loc_name":     row.get("geo_loc_name", "").strip(),
                "collection_date":  row.get("collection_date", "").strip(),
                "serovar":          row.get("serovar", "Typhimurium").strip(),
                "n_contig":         n_contig,
                "length_bp":        length,
                "source_class":     classify_source(src),
            })

    print(f"[INFO] Parsed {total_seen:,} rows -> {len(candidates)} candidates after filter")
    food = sum(1 for c in candidates if c["source_class"] == "food_chain")
    nfc  = sum(1 for c in candidates if c["source_class"] == "non_food_chain")
    amb  = sum(1 for c in candidates if c["source_class"] == "ambiguous")
    print(f"[INFO]   food_chain={food}  non_food_chain={nfc}  ambiguous={amb}")
    return candidates


# ── Load SNP clusters ─────────────────────────────────────────────────────────

def load_snp_clusters() -> dict[str, str]:
    print("[INFO] Loading SNP cluster assignments ...")
    r = requests.get(CLUSTER_URL, stream=True, timeout=60, verify=False)
    r.raise_for_status()

    lookup: dict[str, str] = {}
    header: list[str] | None = None
    buf = ""

    for chunk in r.iter_content(chunk_size=65536):
        buf += chunk.decode("utf-8", errors="replace")
        lines = buf.split("\n")
        buf = lines[-1]

        for line in lines[:-1]:
            line = line.rstrip("\r")
            if not line:
                continue
            if header is None:
                header = line.split("\t")
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            row = dict(zip(header, parts))
            ta  = row.get("target_acc", "").strip()
            pds = row.get("PDS_acc", "").strip()
            if ta and pds and pds.upper() not in ("NULL", "NA", ""):
                lookup[ta] = pds

    print(f"[INFO] Loaded {len(lookup):,} cluster assignments")
    return lookup


# ── Candidate selection ───────────────────────────────────────────────────────

def _pick_pool(
    pool: list[dict],
    n: int,
    existing_clusters: set[str],
    cluster_counts: dict[str, int],
    max_per_cluster: int = 3,
) -> list[dict]:
    """Pick up to n from pool with cluster-diversity cap."""
    # Sort: new clusters first, then fewest contigs
    def sort_key(c):
        is_new = 0 if c["snp_cluster"] not in existing_clusters else 1
        return (is_new, c["n_contig"])

    pool_sorted = sorted(pool, key=sort_key)
    picked = []
    for c in pool_sorted:
        cl = c["snp_cluster"]
        if cluster_counts[cl] >= max_per_cluster:
            continue
        cluster_counts[cl] += 1
        picked.append(c)
        if len(picked) >= n:
            break
    return picked


def select_candidates(
    candidates: list[dict],
    cluster_lookup: dict[str, str],
    existing_clusters: set[str],
    target_n: int,
    nfc_ratio: float,
) -> list[dict]:
    # Attach SNP cluster
    for c in candidates:
        c["snp_cluster"] = cluster_lookup.get(c["target_acc"], "unknown")

    food_pool = [c for c in candidates if c["source_class"] == "food_chain"]
    nfc_pool  = [c for c in candidates if c["source_class"] == "non_food_chain"]
    amb_pool  = [c for c in candidates if c["source_class"] == "ambiguous"]

    n_nfc  = int(target_n * nfc_ratio)
    n_food = target_n - n_nfc

    print(f"[INFO] Quota: {n_nfc} non_food_chain + {n_food} food_chain "
          f"(pool sizes: nfc={len(nfc_pool)}, food={len(food_pool)}, amb={len(amb_pool)})")

    cluster_counts: dict[str, int] = defaultdict(int)

    nfc_sel  = _pick_pool(nfc_pool,  n_nfc,  existing_clusters, cluster_counts)
    food_sel = _pick_pool(food_pool, n_food, existing_clusters, cluster_counts)

    # Fill shortfall with ambiguous pool
    shortfall = target_n - len(nfc_sel) - len(food_sel)
    amb_sel   = _pick_pool(amb_pool, shortfall, existing_clusters, cluster_counts) if shortfall > 0 else []

    selected = nfc_sel + food_sel + amb_sel
    print(f"[INFO] Selected {len(selected)} isolates  "
          f"(nfc={len(nfc_sel)}, food={len(food_sel)}, amb={len(amb_sel)})")
    print(f"[INFO]   Unique SNP clusters: {len({c['snp_cluster'] for c in selected})}")

    src_dist: dict[str, int] = defaultdict(int)
    for c in selected:
        src_dist[c["isolation_source"]] += 1
    print("[INFO]   Source distribution (top 20):")
    for src, cnt in sorted(src_dist.items(), key=lambda x: -x[1])[:20]:
        print(f"           {src:<45} {cnt}")

    return selected


# ── Download FASTA ────────────────────────────────────────────────────────────

def get_ftp_url(asm_acc: str) -> str | None:
    try:
        handle = Entrez.esearch(db="assembly",
                                term=f"{asm_acc}[Assembly Accession]",
                                retmax=1)
        rec = Entrez.read(handle)
        handle.close()
        if not rec["IdList"]:
            return None
        uid = rec["IdList"][0]
        time.sleep(0.35)

        handle = Entrez.esummary(db="assembly", id=uid)
        doc_set = Entrez.read(handle)
        handle.close()
        docs = doc_set.get("DocumentSummarySet", {}).get("DocumentSummary", [])
        if not docs:
            return None
        ftp = str(docs[0].get("FtpPath_GenBank", ""))
        return ftp or None
    except Exception as e:
        print(f"  [WARN] Entrez error for {asm_acc}: {e}")
        return None


def download_fasta(asm_acc: str, ftp_path: str, out_dir: Path) -> bool:
    # NCBI FTP is also served via HTTPS
    https_path = ftp_path.replace("ftp://ftp.ncbi.nlm.nih.gov",
                                  "https://ftp.ncbi.nlm.nih.gov")
    basename   = https_path.rstrip("/").split("/")[-1]
    fna_gz_url = f"{https_path}/{basename}_genomic.fna.gz"
    out_path   = out_dir / f"{asm_acc}.fna"

    if out_path.exists():
        print(f"  [SKIP] Already have {out_path.name}")
        return True

    try:
        r = requests.get(fna_gz_url, stream=True, timeout=120, verify=False)
        r.raise_for_status()
        content = b""
        for chunk in r.iter_content(chunk_size=65536):
            content += chunk
        fna_bytes = gzip.decompress(content)
        out_path.write_bytes(fna_bytes)
        print(f"  [OK]   {out_path.name}  ({len(fna_bytes)/1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {asm_acc}: {e}")
        if out_path.exists():
            out_path.unlink()
        return False


# ── Write metadata ────────────────────────────────────────────────────────────

FIELDNAMES = ["assembly_accession", "isolate_name", "serovar",
              "isolation_source", "geo_loc_name", "collection_date",
              "snp_cluster", "amr_genes"]


def append_metadata(rows: list[dict]) -> None:
    if not rows:
        return
    with open(META_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for r in rows:
            writer.writerow(r)
    print(f"[INFO] Appended {len(rows)} rows to {META_CSV.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",    type=int,   default=80,
                        help="New isolates to download (default 80)")
    parser.add_argument("--nfc-ratio", type=float, default=0.6,
                        help="Fraction that must be non_food_chain (default 0.6)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Plan only, no download or CSV write")
    args = parser.parse_args()

    GENOME_DIR.mkdir(parents=True, exist_ok=True)

    existing_accs, existing_clusters = load_existing(META_CSV)
    print(f"[INFO] {len(existing_accs)} existing isolates, "
          f"{len(existing_clusters)} known SNP clusters")

    candidates  = stream_metadata(existing_accs)
    cluster_lut = load_snp_clusters()
    selected    = select_candidates(
        candidates, cluster_lut, existing_clusters,
        target_n=args.target, nfc_ratio=args.nfc_ratio,
    )

    if args.dry_run:
        print("\n[DRY RUN] First 25 selected:")
        for c in selected[:25]:
            print(f"  {c['asm_acc']:<25} [{c['source_class'][:4]}] "
                  f"src={c['isolation_source'][:40]:<42} cl={c['snp_cluster']}")
        if len(selected) > 25:
            print(f"  ... and {len(selected)-25} more")
        return

    new_rows: list[dict] = []
    batch: list[dict] = []

    for i, c in enumerate(selected, 1):
        acc = c["asm_acc"]
        print(f"\n[{i:3d}/{len(selected)}] {acc}  [{c['source_class'][:4]}]  "
              f"src={c['isolation_source'][:45]}")

        ftp = get_ftp_url(acc)
        if not ftp:
            print(f"  [SKIP] No FTP URL found")
            continue

        ok = download_fasta(acc, ftp, GENOME_DIR)
        if not ok:
            continue

        row = {
            "assembly_accession": acc,
            "isolate_name":       acc,
            "serovar":            c["serovar"],
            "isolation_source":   c["isolation_source"],
            "geo_loc_name":       c["geo_loc_name"],
            "collection_date":    c["collection_date"],
            "snp_cluster":        c["snp_cluster"],
            "amr_genes":          "",
        }
        new_rows.append(row)
        batch.append(row)

        if len(batch) >= 10:
            append_metadata(batch)
            batch = []

    if batch:
        append_metadata(batch)

    print(f"\n[DONE] Downloaded {len(new_rows)} new isolates.")
    print(f"       FASTA  -> {GENOME_DIR}")
    print(f"       CSV    -> {META_CSV}")

    existing_accs2, _ = load_existing(META_CSV)
    print(f"       Total isolates in dataset: {len(existing_accs2)}")


if __name__ == "__main__":
    main()
