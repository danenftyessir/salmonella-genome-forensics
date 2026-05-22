"""Entry point utama pipeline SalmoTrace-BERT dengan checkpoint per tahap."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils.seed import set_global_seed
from utils.config import load_config
from utils.checkpoint import (
    load_or_compute,
    save_parquet, load_parquet,
    save_csv_ckpt, load_csv_ckpt,
    save_model, load_model as ckpt_load_model,
    save_json,
    save_manifest,
)
from utils.tracking import enrich_metrics, save_model_comparison, log_mlflow_all_modes

# Stage 1 — metadata
from data import (
    load_metadata,
    filter_organism, drop_missing_accession,
    normalize_isolation_source, remove_ambiguous_sources,
    select_dominant_serovars, filter_metadata, check_class_balance,
    add_source_group,
)

# Stage 2 — genome QC
from data import (
    load_genomes, genome_qc_report, filter_by_qc, validate_accessions,
)

# Stage 3 — SNP
from snp import (
    build_snp_matrix, encode_snp_matrix,
    compute_distance_matrix, filter_snp_positions,
)
from snp.filter import remove_high_n_columns

# Stage 4 — windows + DNABERT
from data import extract_windows, extract_snp_context_windows, extract_kmer_features, extract_amr_features
from data.preprocess import save_windows
from embedding import generate_embeddings

# Clustering + visualisation + validation
from clustering import (
    hierarchical_clustering, assign_clusters,
    run_pca, run_umap,
    plot_dendrogram, plot_heatmap, plot_scatter,
    validate_clusters_vs_metadata, plot_cluster_composition,
)

# Stage 5 — ML
from models import run_pipeline

# Report
from report import generate_report, build_forensic_table, generate_forensic_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _abs(cfg_path: str, root: str) -> str:
    """Return absolute path by joining root with a config-relative path."""
    return os.path.join(root, cfg_path)


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main(config_path: str = "config.yaml") -> None:
    cfg  = load_config(config_path)
    ROOT = os.path.dirname(os.path.abspath(config_path))
    FORCE   = cfg["pipeline"].get("force_recompute", False)

    # Fix all random sources before any computation
    set_global_seed(cfg.get("project", {}).get("random_state",
                    cfg["ml"].get("random_state", 42)))
    ART     = cfg["artifacts"]
    meta_cfg  = cfg.get("metadata", {})
    qc_cfg    = cfg.get("genome_qc", {})
    pre_cfg   = cfg["preprocessing"]
    target_col = cfg["ml"]["target_col"]

    # ── Stage 1 · Metadata ──────────────────────────────────────────────────
    _section("Stage 1 — Metadata Preprocessing")

    def _build_metadata():
        df = load_metadata(_abs(cfg["data"]["metadata_path"], ROOT))
        df = drop_missing_accession(df)
        df = filter_organism(df, meta_cfg.get("organism", "Salmonella enterica"))
        df = normalize_isolation_source(df)
        df = remove_ambiguous_sources(df)
        df = select_dominant_serovars(df, top_n=meta_cfg.get("top_serovars", 3))
        df = add_source_group(df)
        df = filter_metadata(
            df,
            min_isolates=cfg["pipeline"]["n_isolates_min"],
            max_isolates=cfg["pipeline"]["n_isolates_max"],
        )
        return df

    metadata_df = load_or_compute(
        path=_abs(ART["metadata_clean_path"], ROOT),
        compute_fn=_build_metadata,
        save_fn=save_parquet,
        load_fn=load_parquet,
        force=FORCE,
    )
    check_class_balance(metadata_df, col=target_col)

    # ── Stage 2 · Genome QC ─────────────────────────────────────────────────
    _section("Stage 2 — FASTA Loading & Genome QC")

    accessions = metadata_df["assembly_accession"].tolist()
    genomes, contig_counts = load_genomes(_abs(cfg["data"]["raw_genomes_dir"], ROOT), accessions)

    qc_df = load_or_compute(
        path=_abs(ART["sequence_quality_path"], ROOT),
        compute_fn=lambda: genome_qc_report(
            genomes, contig_counts,
            min_bp=qc_cfg.get("min_genome_bp", 4_000_000),
            max_bp=qc_cfg.get("max_genome_bp", 6_000_000),
            max_n_frac=qc_cfg.get("max_n_fraction", 0.05),
            max_contigs=qc_cfg.get("max_contig_count", 500),
        ),
        save_fn=save_csv_ckpt,
        load_fn=load_csv_ckpt,
        force=FORCE,
    )

    genomes     = filter_by_qc(genomes, qc_df)
    metadata_df = validate_accessions(metadata_df, genomes)

    if len(genomes) < cfg["pipeline"]["n_isolates_min"]:
        raise RuntimeError(f"Terlalu sedikit isolat lulus QC: {len(genomes)}")

    # ── Stage 3 · SNP matrix ────────────────────────────────────────────────
    _section("Stage 3 — SNP Matrix")

    def _build_snp():
        df = build_snp_matrix(genomes)
        df = remove_high_n_columns(df)
        df = filter_snp_positions(df)
        return df

    snp_df = load_or_compute(
        path=_abs(ART["snp_matrix_path"], ROOT),
        compute_fn=_build_snp,
        save_fn=save_parquet,
        load_fn=load_parquet,
        force=FORCE,
    )

    snp_encoded_df = load_or_compute(
        path=_abs(ART["snp_encoded_path"], ROOT),
        compute_fn=lambda: encode_snp_matrix(snp_df, method="integer"),
        save_fn=save_parquet,
        load_fn=load_parquet,
        force=FORCE,
    )

    dist_df = load_or_compute(
        path=_abs(ART["distance_matrix_path"], ROOT),
        compute_fn=lambda: compute_distance_matrix(snp_df),
        save_fn=save_csv_ckpt,
        load_fn=load_csv_ckpt,
        force=FORCE,
    )

    snp_positions = [int(c) for c in snp_df.columns]

    # ── Stage 3 cont. · Hierarchical clustering ─────────────────────────────
    iso_labels  = dist_df.index.tolist()
    linkage_mat = hierarchical_clustering(dist_df, method=cfg["clustering"]["method"])
    n_clusters  = min(5, len(iso_labels) - 1)
    hclust_labels = assign_clusters(linkage_mat, iso_labels, n_clusters=n_clusters)

    fig_cluster = _abs(cfg["output"]["figures_dir"] + "clustering/", ROOT)
    plot_dendrogram(linkage_mat, iso_labels, os.path.join(fig_cluster, "dendrogram.png"))
    plot_heatmap(dist_df, os.path.join(fig_cluster, "heatmap.png"))
    cluster_result = {"n_clusters": int(hclust_labels.nunique())}

    # ── Baseline Biology 2 · Cluster validation vs metadata ─────────────────
    _section("Baseline Biology 2 — Cluster Validation vs Metadata")

    cluster_val_df = validate_clusters_vs_metadata(hclust_labels, metadata_df)
    for col in ["isolation_source", "serovar"]:
        plot_cluster_composition(
            hclust_labels, metadata_df, col,
            os.path.join(fig_cluster, f"cluster_composition_{col}.png"),
        )

    # Compute E1 biological metrics for tracking (silhouette + ARI + separation)
    from clustering import compute_silhouette
    from sklearn.metrics import adjusted_rand_score
    import numpy as _np

    _bio_ctx: dict = {}
    try:
        _bio_ctx["sil_e1"] = compute_silhouette(dist_df, hclust_labels.values, metric="precomputed")
    except Exception:
        _bio_ctx["sil_e1"] = float("nan")
    try:
        if "snp_cluster" in metadata_df.columns:
            _meta_s  = metadata_df.set_index("assembly_accession")
            _shared  = hclust_labels.index.intersection(_meta_s.index)
            _bio_ctx["ari_e1"] = float(adjusted_rand_score(
                _meta_s.reindex(_shared)["snp_cluster"].fillna("unknown").astype(str).values,
                hclust_labels.loc[_shared].values.astype(str),
            ))
    except Exception:
        _bio_ctx["ari_e1"] = float("nan")
    try:
        _n = len(dist_df)
        _within, _between = [], []
        _cl_arr = hclust_labels.to_dict()
        for _i in range(_n):
            for _j in range(_i + 1, _n):
                _d = float(dist_df.iloc[_i, _j])
                if _cl_arr.get(dist_df.index[_i]) == _cl_arr.get(dist_df.index[_j]):
                    _within.append(_d)
                else:
                    _between.append(_d)
        _bio_ctx["w_m"] = float(_np.mean(_within))  if _within  else float("nan")
        _bio_ctx["b_m"] = float(_np.mean(_between)) if _between else float("nan")
    except Exception:
        _bio_ctx["w_m"] = _bio_ctx["b_m"] = float("nan")

    # ── Stage 4 · DNABERT windows + embeddings ──────────────────────────────
    _section("Stage 4 — DNABERT Window Extraction & Embedding")

    use_snp_ctx = pre_cfg.get("use_snp_context_windows", True)
    max_n_win   = pre_cfg.get("max_n_fraction_window", 0.10)
    max_wins    = pre_cfg["max_windows_per_isolate"]

    if use_snp_ctx and snp_positions:
        windows = extract_snp_context_windows(
            genomes,
            snp_positions=snp_positions,
            flank=pre_cfg["snp_context_flank"],
            max_windows=max_wins,
            max_n_frac=max_n_win,
        )
        save_windows(windows, _abs(cfg["data"].get("snp_context_dir", "data/processed/snp_context/"), ROOT))
    else:
        windows = extract_windows(
            genomes,
            window_size=pre_cfg["window_size"],
            max_windows=max_wins,
            max_n_frac=max_n_win,
        )
        save_windows(windows, _abs(cfg["data"].get("windows_dir", "data/processed/windows/"), ROOT))

    # generate_embeddings has its own NPZ cache check via cfg['artifacts']['embeddings_path']
    embedding_df = generate_embeddings(windows, cfg)

    labels_for_plot = metadata_df.set_index("assembly_accession")[target_col]
    fig_emb = _abs(cfg["output"]["figures_dir"] + "embedding/", ROOT)
    pca_df  = run_pca(embedding_df)
    umap_df = run_umap(embedding_df)
    plot_scatter(pca_df,  labels_for_plot, "PCA — DNABERT Embeddings",  os.path.join(fig_emb, "pca_scatter.png"))
    plot_scatter(umap_df, labels_for_plot, "UMAP — DNABERT Embeddings", os.path.join(fig_emb, "umap_scatter.png"))

    # ── E5 · k-mer features (optional) ──────────────────────────────────────
    kmer_cfg = cfg.get("kmer", {})
    kmer_df  = None
    if kmer_cfg.get("enabled", False):
        _section("E5 — K-mer Feature Extraction")
        kmer_df = extract_kmer_features(genomes, k=kmer_cfg.get("k", 4))

    # ── AMR gene features ────────────────────────────────────────────────────
    _section("AMR Gene Feature Extraction")
    amr_df = extract_amr_features(metadata_df)

    # ── Stage 5 · ML classification ─────────────────────────────────────────
    _section("Stage 5 — ML Classification (SNP / DNABERT / Hybrid)")

    model_path = _abs(ART["model_path"], ROOT)
    if os.path.exists(model_path) and not FORCE:
        print(f"[LOAD]    rf_model.joblib  ← {model_path}")
        best_clf = ckpt_load_model(model_path)
        metrics_path = _abs(ART["metrics_path"], ROOT)
        if os.path.exists(metrics_path):
            from utils.checkpoint import load_json
            loaded = load_json(metrics_path)
            best_metrics = loaded.get("best", {})
            all_metrics  = loaded.get("all_modes", {})
        else:
            best_clf = best_metrics = all_metrics = None
    else:
        best_clf, best_metrics, all_metrics = run_pipeline(
            embedding_df, metadata_df, cfg,
            snp_encoded_df=snp_encoded_df,
            kmer_df=kmer_df,
            amr_df=amr_df,
        )
        save_model(best_clf, model_path)
        save_json(
            {"best": best_metrics, "all_modes": all_metrics},
            _abs(ART["metrics_path"], ROOT),
        )

    # ── Experiment tracking ──────────────────────────────────────────────────
    if all_metrics:
        _section("Experiment Tracking")
        all_metrics = enrich_metrics(all_metrics, _bio_ctx)
        # Re-save metrics.json with bio_interpretation included
        save_json(
            {"best": best_metrics, "all_modes": all_metrics},
            _abs(ART["metrics_path"], ROOT),
        )
        save_model_comparison(
            all_metrics, cfg,
            path=_abs(cfg.get("tracking", {}).get("comparison_path",
                      "artifacts/reports/model_comparison.csv"), ROOT),
            bio_context=_bio_ctx,
        )
        log_mlflow_all_modes(
            all_metrics, cfg,
            artifact_dir=_abs(cfg["output"]["figures_dir"], ROOT),
            bio_context=_bio_ctx,
        )

    # ── Report + Forensic + Manifest ────────────────────────────────────────
    _section("Report & Manifest")

    generate_report(
        metadata_df, snp_df, dist_df, cluster_result,
        best_metrics or {}, cfg,
        all_metrics=all_metrics or {},
        cluster_validation_df=cluster_val_df,
    )

    # Forensic interpretation layer — per-isolate nearest-neighbor summary
    if best_clf is not None:
        forensic_df = build_forensic_table(
            dist_df, metadata_df, target_col=target_col,
            clf=best_clf, feature_df=embedding_df,
        )
        generate_forensic_summary(
            forensic_df, all_metrics or {},
            _abs(ART.get("forensic_path", "artifacts/reports/forensic_report.txt"), ROOT),
        )

    save_manifest(
        cfg, metadata_df, snp_df, all_metrics or {},
        _abs(ART["manifest_path"], ROOT),
    )

    # ── Limitations report ───────────────────────────────────────────────────
    _generate_limitations_report(
        metadata_df, qc_df, cfg, _bio_ctx,
        all_metrics or {}, {},   # naive_metrics is notebook-only; not run in main.py
        _abs(ART.get("limitations_path", "artifacts/reports/limitations_report.txt"), ROOT),
    )

    print("\nPipeline selesai.")


def _generate_limitations_report(
    metadata_df, qc_df, cfg, bio_ctx, all_metrics, naive_metrics, out_path
):
    """Generate limitations_report.txt with auto-detected risks."""
    import math, textwrap
    from pathlib import Path

    warnings, notes, mitigated = [], [], []

    def _w(m): warnings.append(m)
    def _n(m): notes.append(m)
    def _ok(m): mitigated.append(m)

    n_iso = len(metadata_df)
    ref   = cfg["data"]["reference_genome"]

    # 1. Dataset disclaimer
    _n("Dataset bukan isolat MBG. Penelitian ini menggunakan data publik NCBI Pathogen "
       "Detection untuk mensimulasikan forensik genomik. MBG diposisikan sebagai konteks "
       "aktual keamanan pangan, bukan target investigasi langsung.")

    # 2. Class imbalance
    src_counts = metadata_df["isolation_source"].value_counts()
    tiny = src_counts[src_counts < 3]
    if len(tiny):
        _w(f"CLASS IMBALANCE: {len(tiny)} kelas < 3 sampel: {tiny.to_dict()}. "
           "Estimasi F1 per kelas tidak stabil.")
    else:
        _ok(f"Class balance: semua kelas ≥ 3 sampel. "
            f"Kelas terbesar: '{src_counts.index[0]}' ({src_counts.iloc[0]}).")

    # 3. Leakage
    if naive_metrics:
        max_delta = float("-inf")
        leakage_modes = []
        for mode in ["snp_only", "dnabert_only", "hybrid", "kmer_only"]:
            ga = all_metrics.get(mode, {}).get("f1_macro", float("nan"))
            nv = naive_metrics.get(mode, {}).get("f1_macro", float("nan"))
            if not math.isnan(ga) and not math.isnan(nv):
                d = nv - ga
                max_delta = max(max_delta, d)
                if d > 0.15:
                    leakage_modes.append(f"{mode} ΔF1={d:+.3f}")
        if leakage_modes:
            _w(f"LEAKAGE TINGGI: {', '.join(leakage_modes)}. Gunakan hasil group-aware.")
        elif max_delta > 0.05:
            _w(f"LEAKAGE MODERAT: ΔF1 maks={max_delta:+.3f}.")
        else:
            _ok(f"Leakage rendah: ΔF1 maks={max_delta:+.3f}. Group-aware split efektif.")
    else:
        _n("Leakage assessment tidak tersedia pada run ini (naive_metrics tidak dihitung).")

    # 4. Dataset size
    if n_iso < 30:
        _w(f"DATASET KECIL: {n_iso} isolat. Estimasi metrik tidak stabil.")
    elif n_iso < 60:
        _n(f"Dataset moderat: {n_iso} isolat. Cukup untuk demo pipeline.")
    else:
        _ok(f"Dataset: {n_iso} isolat — ukuran memadai.")

    # 5. Separation
    w_m = bio_ctx.get("w_m", float("nan"))
    b_m = bio_ctx.get("b_m", float("nan"))
    sep = b_m / max(w_m, 1) if not math.isnan(w_m) and not math.isnan(b_m) else float("nan")
    if not math.isnan(sep):
        if sep < 2:
            _w(f"GENOMIK HOMOGEN: separation={sep:.1f}×. Dataset mungkin terlalu homogen.")
        elif sep < 4:
            _n(f"Separasi moderat: {sep:.1f}×.")
        else:
            _ok(f"Separasi baik: {sep:.1f}×.")

    # 6. DNABERT
    _n(f"DNABERT-2 digunakan tanpa fine-tuning ({cfg['dnabert']['model_id']}). "
       "Embedding bersifat general.")

    # 7. Reference
    _ok(f"Reference tetap: {ref}. Pilihan didokumentasikan di config.yaml.")

    # 8. QC
    if qc_df is not None and "qc_pass" in qc_df.columns:
        n_fail = int((~qc_df["qc_pass"]).sum())
        if n_fail > 0:
            _n(f"QC: {n_fail} isolat dibuang karena gagal threshold kualitas.")
        else:
            _ok(f"QC: semua {n_iso} isolat lulus threshold kualitas.")

    # 9. Metadata noise
    _ok(f"Label metadata dinormalisasi: {metadata_df['isolation_source'].nunique()} "
        "kategori unik. Entri ambigu dihapus.")

    # 10. Geo bias
    top_geo = (metadata_df["geo_loc_name"].str.split(":").str[0].str.strip()
               .value_counts())
    if len(top_geo):
        dom = top_geo.index[0]
        dom_pct = top_geo.iloc[0] / n_iso * 100
        if dom_pct > 60:
            _w(f"BIAS GEOGRAFIS: {dom_pct:.0f}% dari '{dom}'. Klaim dibatasi pada dataset.")
        else:
            _n(f"Distribusi geografis: {len(top_geo)} wilayah. Dominan: '{dom}' ({dom_pct:.0f}%).")

    # Compile
    SEP = "─" * 66
    L = ["=" * 66, "  RISK ASSESSMENT — SalmoTrace-BERT",
         f"  Dataset: {n_iso} isolat | Ref: {ref}", "=" * 66, ""]

    def _section(items, label):
        if not items:
            return
        L.append(f"[{label}]"); L.append(SEP)
        for i, msg in enumerate(items, 1):
            L.append(textwrap.fill(f"{i}. {msg}", 70,
                                   initial_indent="  ", subsequent_indent="     "))
        L.append("")

    _section(warnings,  "PERINGATAN AKTIF")
    _section(notes,     "CATATAN PENTING")
    _section(mitigated, "RISIKO DIMITIGASI")
    L += [SEP,
          "  DISCLAIMER: Penelitian ini adalah simulasi forensik genomik berbasis",
          "  data publik. Hasil tidak dapat menetapkan sumber kontaminasi nyata",
          "  tanpa validasi epidemiologi lapangan yang independen.", SEP]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(L), encoding="utf-8")
    print(f"[LIMITATIONS] {out_path}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path)
