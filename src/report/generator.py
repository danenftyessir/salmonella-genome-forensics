"""Generate outputs/reports/summary_report.txt from all pipeline results."""

from __future__ import annotations

import os
from datetime import datetime
from utils.io import ensure_dir


def generate_report(
    metadata_df,
    snp_df,
    dist_df,
    cluster_result: dict,
    ml_metrics: dict,
    cfg: dict,
    out_path: str = None,
    all_metrics: dict | None = None,
    cluster_validation_df=None,
):
    if out_path is None:
        out_path = os.path.join(cfg["output"]["reports_dir"], "summary_report.txt")
    ensure_dir(os.path.dirname(out_path))

    n_isolates = len(metadata_df)
    n_snps = snp_df.shape[1]
    dist_vals = dist_df.values
    max_dist = dist_vals.max()
    mean_dist = dist_vals[dist_vals > 0].mean()

    lines = [
        "=" * 66,
        "  SalmoTrace-BERT — Pipeline Summary Report",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 66,
        "",
        "[1] Dataset",
        f"    Isolat valid     : {n_isolates}",
        f"    Sumber data      : NCBI Pathogen Detection / Datasets",
        f"    Referensi genome : {cfg['data']['reference_genome']}",
        "",
        "[2] SNP Analysis (E1 — SNP Distance Clustering)",
        f"    Posisi SNP total : {n_snps}",
        f"    Jarak SNP maks   : {max_dist:.0f}",
        f"    Jarak SNP rata²  : {mean_dist:.2f}",
        "",
        "[3] Hierarchical Clustering",
        f"    Metode           : {cfg['clustering']['method']}",
        f"    Jumlah kluster   : {cluster_result.get('n_clusters', 'N/A')}",
        "",
    ]

    # --- Baseline Biology 2: cluster validation ---
    if cluster_validation_df is not None and not cluster_validation_df.empty:
        lines += ["[3b] Cluster Validation vs Metadata (ARI / NMI)"]
        lines.append(f"    {'Metadata Col':<22} {'ARI':>8} {'NMI':>8} {'#Cat':>6}")
        lines.append(f"    {'-' * 46}")
        for _, r in cluster_validation_df.iterrows():
            lines.append(
                f"    {str(r['metadata_col']):<22} {r['ARI']:>8.4f} {r['NMI']:>8.4f} {int(r['n_categories']):>6}"
            )
        lines.append("")

    lines += [
        "[4] DNABERT Embedding",
        f"    Model            : {cfg['dnabert']['model_id']}",
        f"    Dimensi          : {cfg['dnabert']['embedding_dim']}",
        f"    Device           : {cfg['dnabert']['device']}",
        "",
    ]

    # --- Multi-mode ML comparison ---
    lines += ["[5] ML Classification — Ablation Study (Source Attribution)"]
    lines.append(
        f"    {'Experiment':<32} {'Macro F1':>9} {'Bal. Acc':>9} {'F1 (wt.)':>9} {'Silhouette':>11} {'Split':>10}"
    )
    lines.append(f"    {'-' * 82}")
    _exp_labels = {
        "snp_only":     "E2  SNP-only Random Forest",
        "dnabert_only": "E3  DNABERT Embedding",
        "hybrid":       "E4  Hybrid SNP + DNABERT",
    }
    import math
    if all_metrics:
        for key in ["snp_only", "dnabert_only", "hybrid"]:
            m = all_metrics.get(key, {})
            label = _exp_labels.get(key, key)
            sil = m.get("silhouette", float("nan"))
            sil_str = f"{sil:.4f}" if not math.isnan(sil) else "   -  "
            lines.append(
                f"    {label:<32}"
                f" {m.get('f1_macro', 0.0):>9.4f}"
                f" {m.get('balanced_accuracy', 0.0):>9.4f}"
                f" {m.get('f1_weighted', 0.0):>9.4f}"
                f" {sil_str:>11}"
                f" {m.get('split_type', '-'):>10}"
            )
    else:
        lines.append(
            f"    {'Best model':<32} {ml_metrics.get('f1_macro', 0.0):>9.4f}"
            f" {ml_metrics.get('balanced_accuracy', 0.0):>9.4f}"
            f" {ml_metrics.get('f1_weighted', 0.0):>9.4f}"
        )
    lines += [
        "",
        "[6] Output Files",
        f"    Figures          : outputs/figures/",
        f"    Reports          : artifacts/reports/",
        "",
        "[7] Batasan & Disclaimer",
        "    Penelitian ini merupakan simulasi forensik genomik menggunakan data",
        "    publik NCBI Pathogen Detection. MBG diposisikan sebagai konteks aktual",
        "    keamanan pangan; analisis ini bukan investigasi epidemiologis resmi.",
        "",
        "    Batasan utama:",
        "    (a) Dataset bukan isolat MBG langsung — klaim dibatasi pada dataset.",
        "    (b) DNABERT-2 dipakai sebagai feature extractor tanpa fine-tuning.",
        "    (c) SNP distance menunjukkan kedekatan genomik, bukan bukti epidemiologi.",
        "    (d) Source attribution hanya berlaku pada dataset & label yang dianalisis.",
        "    (e) Hasil tidak dapat digunakan untuk menetapkan sumber kontaminasi nyata",
        "        tanpa validasi epidemiologi lapangan yang independen.",
        "    (f) Dataset mencakup isolat 2001–2013, seluruhnya dari USA. Tidak",
        "        merepresentasikan isolat MBG 2025 maupun kondisi Indonesia.",
        "        Hasil hanya sebagai demonstrasi pipeline forensik genomik.",
        "    (g) Jumlah SNP absolut antar-isolat sangat tinggi akibat pendekatan",
        "        reference-free pairwise (bukan core-SNP terhadap referensi tunggal).",
        "        Pasangan dengan jarak SNP terkecil disebut 'relatif paling mirip",
        "        dalam dataset', bukan sebagai kandidat common source secara biologis.",
        "",
        "=" * 66,
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {out_path}")
