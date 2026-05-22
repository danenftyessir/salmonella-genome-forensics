#!/usr/bin/env bash
# Run Snippy whole-genome SNP calling for all isolate FASTA files,
# then produce core-genome SNP alignment via snippy-core.
#
# Backend: Docker (staphb/snippy:4.6.0) — no local Perl/conda needed.
#
# Requirements:
#   Docker Desktop running (Windows) with WSL2 integration enabled
#
# Usage (from project root in WSL):
#   bash scripts/run_snippy.sh
#
# Override CPUs:
#   SNIPPY_CPUS=8 bash scripts/run_snippy.sh
#
# Output (in data/processed/snippy/):
#   core.aln       — multi-FASTA, variable SNP positions only  ← used by Stage 3
#   core.full.aln  — multi-FASTA, full core-genome alignment
#   core.tab       — TSV with CHR, POS, REF, and per-isolate bases
#   core.vcf       — merged VCF
#   core.txt       — summary statistics

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
DOCKER_IMG="staphb/snippy:4.6.0"
REF="data/raw/genomes/GCF_000006945.2.fna"
GENOMES_DIR="data/raw/genomes"
OUT_DIR="data/processed/snippy"
CORE_PREFIX="${OUT_DIR}/core"
CPUS="${SNIPPY_CPUS:-4}"

# Absolute path of project root → used as Docker bind-mount source
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Docker wrapper functions ──────────────────────────────────────────────────
# All paths passed to these functions must be relative to PROJECT_ROOT,
# since the container sees PROJECT_ROOT mounted at /data.

_docker_snippy() {
    docker run --rm \
        -v "${PROJECT_ROOT}:/data" \
        -w /data \
        "${DOCKER_IMG}" \
        snippy "$@"
}

_docker_snippy_core() {
    docker run --rm \
        -v "${PROJECT_ROOT}:/data" \
        -w /data \
        "${DOCKER_IMG}" \
        snippy-core "$@"
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "[ERROR] docker command not found."
    echo "        Install Docker Desktop for Windows and enable WSL2 integration."
    exit 1
fi

if ! docker info &>/dev/null; then
    echo "[ERROR] Docker daemon is not running."
    echo "        Start Docker Desktop on Windows first."
    exit 1
fi

# Pull image if not cached locally
if ! docker image inspect "${DOCKER_IMG}" &>/dev/null; then
    echo "[DOCKER] Pulling ${DOCKER_IMG} ..."
    docker pull "${DOCKER_IMG}"
fi

if [[ ! -f "${PROJECT_ROOT}/${REF}" ]]; then
    echo "[ERROR] Reference not found: ${PROJECT_ROOT}/${REF}"
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/${OUT_DIR}"

# ── Per-isolat SNP calling ────────────────────────────────────────────────────
echo "[SNIPPY] Docker image : ${DOCKER_IMG}"
echo "[SNIPPY] Project root : ${PROJECT_ROOT}"
echo "[SNIPPY] Reference    : ${REF}"
echo "[SNIPPY] Output dir   : ${OUT_DIR}"
echo "[SNIPPY] CPUs         : ${CPUS}"
echo ""

ISOLATE_DIRS=()

for fna in "${PROJECT_ROOT}/${GENOMES_DIR}"/*.fna; do
    acc=$(basename "$fna" .fna)

    # Skip reference genome
    [[ "$acc" == "GCF_000006945.2" ]] && continue

    iso_dir="${OUT_DIR}/${acc}"

    if [[ -d "${PROJECT_ROOT}/${iso_dir}" ]] && \
       [[ -f "${PROJECT_ROOT}/${iso_dir}/snps.vcf" ]]; then
        echo "[SKIP]  $acc  (snps.vcf sudah ada)"
    else
        echo "[RUN]   snippy → $acc ..."
        _docker_snippy \
            --cpus    "$CPUS"       \
            --outdir  "$iso_dir"    \
            --ref     "$REF"        \
            --ctgs    "${GENOMES_DIR}/${acc}.fna" \
            --force                 \
            --quiet
        echo "[DONE]  $acc"
    fi

    ISOLATE_DIRS+=("$iso_dir")
done

echo ""
echo "[SNIPPY] ${#ISOLATE_DIRS[@]} isolat selesai."

# ── Core SNP alignment ────────────────────────────────────────────────────────
echo ""
echo "[SNIPPY-CORE] Membangun core alignment → ${CORE_PREFIX}.aln ..."

_docker_snippy_core \
    --ref    "$REF"          \
    --prefix "$CORE_PREFIX"  \
    "${ISOLATE_DIRS[@]}"

echo ""
echo "[SNIPPY-CORE] Selesai. Output:"
for ext in aln full.aln tab txt; do
    f="${PROJECT_ROOT}/${CORE_PREFIX}.${ext}"
    [[ -f "$f" ]] && echo "  $(wc -c < "$f") bytes  ${CORE_PREFIX}.${ext}"
done
