"""
Download dataset Salmonella untuk pipeline SalmoTrace-BERT.

Langkah:
  1. Download cluster_list.tsv (42 MB) -> dict {PDT_id: SNP_cluster}
  2. Stream metadata.tsv (777 MB), filter Typhimurium + asm_acc valid
  3. Pilih 50 isolat beragam sumber
  4. Download FASTA tiap isolat via NCBI Datasets REST API v2
  5. Download reference genome GCF_000006945.2
  6. Simpan metadata_salmonella.csv

Jalankan dari root project:
    python scripts/download_data.py
"""

import os, sys, csv, time, io, zipfile, gzip, json, re
import requests

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENOMES_DIR  = os.path.join(ROOT, "data", "raw", "genomes")
META_DIR     = os.path.join(ROOT, "data", "raw", "metadata")
META_OUT     = os.path.join(META_DIR, "metadata_salmonella.csv")
REF_ACC      = "GCF_000006945.2"
TARGET_N     = 50

PDG_VER      = "PDG000000002.4067"
FTP_META     = f"https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Salmonella/latest_snps/Metadata/{PDG_VER}.metadata.tsv"
FTP_CLUSTER  = f"https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Salmonella/latest_snps/Clusters/{PDG_VER}.reference_target.cluster_list.tsv"
NCBI_DS      = "https://api.ncbi.nlm.nih.gov/datasets/v2"

os.makedirs(GENOMES_DIR, exist_ok=True)
os.makedirs(META_DIR,    exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Download cluster_list.tsv -> {pdt_id: pds_cluster}
# ──────────────────────────────────────────────────────────────────────────────

def load_cluster_map() -> dict:
    print("[1] Download cluster_list.tsv (42 MB)...")
    r = requests.get(FTP_CLUSTER, timeout=120)
    r.raise_for_status()
    cluster_map = {}
    reader = csv.DictReader(io.StringIO(r.text), delimiter="\t")
    for row in reader:
        pdt = row.get("target_acc", "").strip()
        pds = row.get("PDS_acc", "").strip()
        if pdt and pds:
            cluster_map[pdt] = pds
    print(f"    -> {len(cluster_map):,} entri cluster dimuat")
    return cluster_map


# ──────────────────────────────────────────────────────────────────────────────
# 2. Stream metadata.tsv, filter Typhimurium + valid asm_acc
# ──────────────────────────────────────────────────────────────────────────────

def stream_isolates(cluster_map: dict) -> list:
    print(f"[2] Stream metadata (target {TARGET_N} isolat Typhimurium)...")

    isolates    = []
    src_count   = {}   # {source: count} — batasi per source agar beragam
    MAX_PER_SRC = 15

    with requests.get(FTP_META, stream=True, timeout=120) as resp:
        resp.raise_for_status()

        # Baca stream sebagai teks baris per baris
        buf  = b""
        header = None

        for chunk in resp.iter_content(chunk_size=256 * 1024):
            buf += chunk

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    line_str = line.decode("utf-8", errors="replace").rstrip("\r")
                except Exception:
                    continue

                if header is None:
                    # Baris pertama = header (dimulai dengan #label)
                    header = line_str.lstrip("#").split("\t")
                    # Normalisasi: buang "#" dari "label"
                    if header[0].startswith("#"):
                        header[0] = header[0][1:]
                    continue

                fields = line_str.split("\t")
                if len(fields) < len(header):
                    continue

                row = dict(zip(header, fields))

                # Filter 1: harus punya assembly accession GCA/GCF
                asm = row.get("asm_acc", "").strip()
                if not asm or not re.match(r"GC[AF]_\d{9}\.\d+", asm):
                    continue

                # Filter 2: serovar Typhimurium
                serovar = row.get("serovar", "").strip()
                if "typhimurium" not in serovar.lower():
                    continue

                # Ambil PDT ID dari kolom "label" (setelah strip #)
                label_raw = row.get("label", "") or row.get("#label", "")
                pdt_id    = label_raw.split("|")[0].strip() if "|" in label_raw else ""

                src        = row.get("isolation_source", "unknown").strip() or "unknown"
                src_key    = src.lower()[:30]
                if src_count.get(src_key, 0) >= MAX_PER_SRC:
                    continue

                geo   = row.get("geo_loc_name", "unknown").strip() or "unknown"
                date  = row.get("collection_date", "unknown").strip() or "unknown"
                amr   = row.get("AMR_genotypes", "").strip()
                name  = row.get("isolate_identifiers", asm).strip() or asm

                snp_cluster = cluster_map.get(pdt_id, "")
                # Fallback: gunakan minsame tier sebagai cluster proxy
                if not snp_cluster:
                    ms = row.get("minsame", "").strip()
                    snp_cluster = f"dist_{ms}" if ms and ms != "NULL" else "unknown"

                src_count[src_key] = src_count.get(src_key, 0) + 1
                isolates.append({
                    "assembly_accession": asm,
                    "isolate_name":       name,
                    "serovar":            serovar,
                    "isolation_source":   src,
                    "geo_loc_name":       geo,
                    "collection_date":    date,
                    "snp_cluster":        snp_cluster,
                    "amr_genes":          amr,
                })

                if len(isolates) >= TARGET_N:
                    print(f"    -> Target {TARGET_N} isolat tercapai, stop streaming.")
                    return isolates

    print(f"    -> {len(isolates)} isolat ditemukan")
    return isolates


# ──────────────────────────────────────────────────────────────────────────────
# 3. Download satu genome FASTA via NCBI Datasets v2
# ──────────────────────────────────────────────────────────────────────────────

def download_fasta(accession: str, out_dir: str) -> bool:
    out_path = os.path.join(out_dir, f"{accession}.fna")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10_000:
        size_kb = os.path.getsize(out_path) // 1024
        print(f"    [SKIP] {accession}  ({size_kb:,} KB sudah ada)")
        return True

    url = (
        f"{NCBI_DS}/genome/accession/{accession}/download"
        "?include_annotation_type=GENOME_FASTA&hydrated=FULLY_HYDRATED"
    )
    try:
        r = requests.get(url, headers={"Accept": "application/zip"}, timeout=180)
        if r.status_code == 404:
            print(f"    [404]  {accession}")
            return False
        r.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            fna_files = [n for n in zf.namelist() if n.endswith(".fna")]
            if not fna_files:
                print(f"    [WARN] Tidak ada .fna dalam zip: {accession}")
                return False
            with zf.open(fna_files[0]) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

        size_kb = os.path.getsize(out_path) // 1024
        print(f"    [OK]   {accession}  ({size_kb:,} KB)")
        return True

    except Exception as e:
        print(f"    [ERR]  {accession}: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 4. Download reference genome (GCF_000006945.2 — Typhimurium LT2)
# ──────────────────────────────────────────────────────────────────────────────

def download_reference():
    ref_path = os.path.join(GENOMES_DIR, f"{REF_ACC}.fna")
    if os.path.exists(ref_path) and os.path.getsize(ref_path) > 1_000_000:
        print(f"[5] Reference {REF_ACC} sudah ada ({os.path.getsize(ref_path)//1024:,} KB)")
        return

    print(f"[5] Mengunduh reference genome {REF_ACC}...")
    ok = download_fasta(REF_ACC, GENOMES_DIR)
    if not ok:
        # Fallback: NCBI FTP direct
        _ftp_download_ref(ref_path)


def _ftp_download_ref(out_path: str):
    ftp_url = (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/006/945/"
        "GCF_000006945.2_ASM694v2/"
        "GCF_000006945.2_ASM694v2_genomic.fna.gz"
    )
    print(f"    Fallback FTP: {ftp_url}")
    r = requests.get(ftp_url, timeout=180, stream=True)
    r.raise_for_status()
    gz_bytes = io.BytesIO(r.content)
    with gzip.open(gz_bytes, "rb") as src, open(out_path, "wb") as dst:
        dst.write(src.read())
    print(f"    [OK] Reference via FTP ({os.path.getsize(out_path)//1024:,} KB)")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Simpan metadata CSV
# ──────────────────────────────────────────────────────────────────────────────

def save_metadata(isolates: list, downloaded: set):
    rows = [r for r in isolates if r["assembly_accession"] in downloaded]
    if not rows:
        print("[WARN] Tidak ada isolat berhasil diunduh.")
        return
    fields = ["assembly_accession", "isolate_name", "serovar",
              "isolation_source", "geo_loc_name", "collection_date",
              "snp_cluster", "amr_genes"]
    with open(META_OUT, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
        csv.DictWriter(f, fieldnames=fields).writerows(rows)
    print(f"[6] Metadata disimpan: {META_OUT}  ({len(rows)} isolat)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SalmoTrace-BERT — Download Dataset")
    print("=" * 60)

    cluster_map = load_cluster_map()
    isolates    = stream_isolates(cluster_map)

    if not isolates:
        print("[ERROR] Tidak ada isolat ditemukan.")
        sys.exit(1)

    # Tampilkan ringkasan sumber
    src_summary = {}
    for iso in isolates:
        k = iso["isolation_source"][:40]
        src_summary[k] = src_summary.get(k, 0) + 1
    print("\n    Distribusi isolation_source:")
    for k, v in sorted(src_summary.items(), key=lambda x: -x[1])[:10]:
        print(f"      {v:>3}x  {k}")

    # Download FASTA
    print(f"\n[3/4] Mengunduh {len(isolates)} genome FASTA...")
    downloaded = set()
    for i, iso in enumerate(isolates, 1):
        acc = iso["assembly_accession"]
        print(f"  [{i:>2}/{len(isolates)}]", end=" ")
        if download_fasta(acc, GENOMES_DIR):
            downloaded.add(acc)
        time.sleep(0.35)   # hormati rate limit NCBI (~3 req/s)

    print(f"\n  Berhasil: {len(downloaded)}/{len(isolates)} genome")

    download_reference()
    save_metadata(isolates, downloaded)

    print("\n" + "=" * 60)
    print("  Download selesai!")
    print(f"  Genome : {GENOMES_DIR}")
    print(f"  Meta   : {META_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
