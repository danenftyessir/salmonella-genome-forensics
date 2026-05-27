# SalmoTrace-BERT

**Pipeline forensik genomik untuk atribusi sumber biner *Salmonella enterica***  
menggunakan fitur SNP, frekuensi k-mer, dan embedding DNABERT-2.

> IF3211 — Komputasi Domain Spesifik · Institut Teknologi Bandung · 2026

---

## Team

| No | Nama | NIM |
|:--:|------|:---:|
| 1 | Danendra Shafi Athallah | 13523136 |
| 2 | Muhammad Raihaan Perdana | 13523124 |
| 3 | M. Abizzar Gamadrian | 13523155 |
| 4 | Kenzie Raffa Ardhana | 18223127 |

---

## Overview

SalmoTrace-BERT adalah pipeline Python end-to-end untuk **atribusi sumber biner** isolat *Salmonella enterica* — mengklasifikasikan apakah suatu isolat berasal dari **rantai pangan** (`food_chain_associated`, FCA) atau **non-rantai pangan** (`non_food_chain_associated`, NFCA) berdasarkan data whole-genome sequencing (WGS).

Pipeline membandingkan tiga representasi fitur genomik secara sistematis dalam studi ablasi:

| Mode | Fitur | Model | Macro F1 |
|:----:|-------|-------|:--------:|
| SNP | Matriks SNP berbasis referensi (8.157 posisi) | Random Forest | 0,634 |
| DNABERT-2 | Embedding beku 768-dim (mean/attention pooling) | RF / MIL | 0,515 – 0,622 |
| **K-mer** | **Frekuensi 5-mer bebas-alignment (1.024 fitur)** | **Random Forest** | **0,655** ✓ |
| Hibrida | SNP + DNABERT / K-mer + AMR | RF / LR | 0,586 – 0,635 |
| Dummy | Stratified baseline | — | 0,282 |

Evaluasi menggunakan **StratifiedGroupKFold 5-lipat** yang dikelompokkan berdasarkan `snp_cluster` NCBI untuk mencegah kebocoran dari isolat berkerabat dekat (klonal).

---

## Struktur Repositori

```
salmonella-genome-forensics/
├── config.yaml                   # Semua parameter pipeline (single source of truth)
├── requirements.txt              # Dependensi Python
├── main.py                       # Entry point CLI
├── notebooks/
│   └── salmogen_trace_pipeline.ipynb   # Pipeline interaktif (notebook)
├── scripts/
│   ├── download_data.py          # Unduh metadata NCBI
│   ├── fetch_ncbi_genomes.py     # Unduh FASTA per accession
│   ├── download_model.py         # Unduh checkpoint DNABERT-2
│   ├── run_snippy.sh             # Panggil SNP caller Snippy (opsional)
│   └── snippy_qc_filter.py      # Filter output Snippy pasca-QC
├── src/
│   ├── utils/       config, checkpoint, seed, io, seq
│   ├── data/        loader, validator, preprocessor, metadata, kmer, amr
│   ├── snp/         extractor, distance, filter, snippy_parser, stage3
│   ├── embedding/   encoder (DNABERT-2), mil (Attention-MIL), pipeline
│   ├── clustering/  hierarchical, metrics, reduction, validation, visualize
│   ├── models/      trainer, evaluator, pipeline
│   └── report/      generator, forensic interpreter
├── data/
│   ├── raw/          ← TIDAK di-git (unduh manual, lihat bagian Data)
│   └── interim/      ← dibuat oleh pipeline
├── artifacts/
│   ├── matrices/     snp_matrix.parquet, distance_matrix.csv
│   ├── embeddings/   dnabert_embeddings.npz  ← TIDAK di-git
│   ├── models/       rf_model.joblib          ← TIDAK di-git
│   └── reports/      metrics.json, manifest.json, model_comparison.csv,
│                     forensic_report.txt, limitations_report.txt
```

---

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd salmonella-genome-forensics

# 2. Buat virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install dependensi
pip install -r requirements.txt

# PyTorch CPU (jika belum ter-install otomatis)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Data

### Unduh genom

```bash
# Unduh metadata dari NCBI Pathogen Detection
python scripts/download_data.py

# Unduh FASTA per accession ke data/raw/genomes/
python scripts/fetch_ncbi_genomes.py
```

Atau unduh manual:
1. Buka [NCBI Pathogen Detection](https://www.ncbi.nlm.nih.gov/pathogens/)
2. Cari *Salmonella enterica*, unduh metadata CSV → `data/raw/metadata/metadata_salmonella.csv`
3. Unduh FASTA tiap `assembly_accession` → `data/raw/genomes/<accession>.fna`
4. Unduh referensi `GCF_000006945.2` → `data/raw/genomes/GCF_000006945.2.fna`

### Unduh model DNABERT-2

```bash
python scripts/download_model.py
# Atau otomatis via HuggingFace: zhihan1996/DNABERT-2-117M
```

---

## Menjalankan Pipeline

```bash
# Pipeline penuh (semua tahap, dengan checkpoint caching)
python main.py

# Paksa rekomputasi dari awal (abaikan semua cache)
# Edit config.yaml: pipeline.force_recompute: true
python main.py

# Atau jalankan secara interaktif via notebook
jupyter notebook notebooks/salmogen_trace_pipeline.ipynb
```

Di notebook, set `FORCE_RECOMPUTE = False` untuk mode demo cepat (load artefak cache), atau `FORCE_RECOMPUTE = True` untuk menjalankan ulang semua tahap dari awal.

---

## Tahapan Pipeline & Caching

Setiap tahap menulis artefak checkpoint. Pada re-run, tahap dilewati jika artefak sudah ada dan `force_recompute: false`.

```
Tahap 1 — QC & Normalisasi Label
    → data/interim/clean_metadata.parquet

Tahap 2 — Validasi Kualitas Genom
    → data/interim/sequence_quality.csv

Tahap 3 — Ekstraksi Matriks SNP
    → artifacts/matrices/snp_matrix.parquet
    → artifacts/matrices/distance_matrix.csv

Tahap 4 — Embedding DNABERT-2
    → artifacts/embeddings/dnabert_embeddings.npz  (tidak di-git)

Tahap 5 — Komputasi K-mer
    → artifacts/matrices/kmer_features.parquet

Tahap 6 — Pengelompokan Hierarkis
    → artifacts/matrices/snp_clusters.parquet

Tahap 7 — Klasifikasi & Evaluasi (Studi Ablasi)
    → artifacts/models/rf_model.joblib             (tidak di-git)
    → artifacts/reports/metrics.json
    → artifacts/reports/model_comparison.csv
    → artifacts/reports/manifest.json

Tahap 8 — Laporan Forensik
    → artifacts/reports/forensic_report.txt
    → artifacts/reports/limitations_report.txt
```

---

## Konfigurasi

Semua parameter ada di `config.yaml`. Parameter kunci:

```yaml
project:
  random_state: 42        # seed tunggal untuk semua operasi acak

data:
  reference_genome: GCF_000006945.2   # S. Typhimurium LT2

genome_qc:
  min_genome_bp: 4000000
  max_genome_bp: 6000000
  max_n_fraction: 0.05
  max_contig_count: 500

snp:
  k: 21                         # panjang k-mer untuk anchoring alignment
  min_core_fraction: 0.80       # posisi yang dapat dipanggil di ≥80% isolat

kmer:
  k: 5                          # pentanukleotida (4^5 = 1.024 fitur)

ml:
  n_estimators: 200
  random_state: 42
```

---

## Hasil Utama

Dataset: 168 isolat *S. enterica* (84 FCA / 84 NFCA) dari NCBI Pathogen Detection, semua lulus QC.  
Referensi: GCF_000006945.2 (*S.* Typhimurium LT2). Evaluasi: StratifiedGroupKFold 5-lipat.

| Fitur | Model | Macro F1 | Balanced Acc |
|-------|-------|:--------:|:------------:|
| K-mer (5-mer) | Random Forest | **0,655** | **0,676** |
| SNP (8.157 pos) | Random Forest | 0,634 | 0,666 |
| DNABERT-2 + MIL | Attention-MIL | 0,622 | 0,637 |
| SNP + DNABERT | Random Forest | 0,635 | 0,665 |
| DNABERT-2 beku | Random Forest | 0,515 | 0,563 |
| Dummy baseline | — | 0,282 | 0,500 |

**Temuan kunci:** frekuensi 5-mer bebas-alignment melampaui SNP berbasis referensi, menunjukkan bahwa komposisi nukleotida global mengkodekan sinyal sumber tanpa alignment eksplisit. DNABERT-2 berfungsi sebagai lapisan representasi perantara; tanpa fine-tuning tugas-spesifik, performa RF langsung terbatas namun meningkat signifikan (+0,107) dengan mekanisme Attention MIL.

> **Catatan keterbatasan:** pipeline ini merupakan simulasi forensik genomik pada data publik. Performa terbatas oleh ukuran dataset kecil (168 isolat) dan homogenitas genomik inherent isolat *Salmonella* publik.

---

## Dependensi Utama

| Paket | Fungsi |
|-------|--------|
| `scikit-learn` | Model ML, split, metrik |
| `torch` + `transformers` | DNABERT-2 feature extraction |
| `numpy`, `pandas` | Manipulasi data |
| `scipy` | Pengelompokan hierarkis |
| `umap-learn` | Reduksi dimensi UMAP |
| `joblib` | Serialisasi model |
| `pyarrow` | Parquet I/O |

Lihat `requirements.txt` untuk daftar lengkap dengan versi.
