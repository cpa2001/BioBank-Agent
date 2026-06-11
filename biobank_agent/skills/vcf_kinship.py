"""Kinship estimation via Genomic Relationship Matrix."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from biobank_agent.registry import skill
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE
from biobank_agent.utils.wgs import wgs_results_dir

logger = logging.getLogger(__name__)


@skill(
    name="vcf_kinship",
    description=(
        "Estimate pairwise kinship coefficients from WGS genotype data using "
        "the Genomic Relationship Matrix (GRM). Flags potentially related sample "
        "pairs and generates a kinship heatmap."
    ),
    parameters={
        "maf_threshold": {
            "type": "number",
            "description": "Minimum MAF for GRM computation (default 0.05)",
            "default": 0.05,
        },
        "max_snps": {
            "type": "integer",
            "description": "Number of common SNPs to use (default 10000)",
            "default": 10000,
        },
        "kinship_threshold": {
            "type": "number",
            "description": "Flag pairs above this kinship value (default 0.1)",
            "default": 0.1,
        },
        "region": {
            "type": "string",
            "description": "Genomic region (e.g. chr22) for fast computation",
            "default": "",
        },
    },
    required=[],
)
def vcf_kinship(
    maf_threshold: float = 0.05,
    max_snps: int = 10000,
    kinship_threshold: float = 0.1,
    region: str = "",
    *,
    ctx=None,
) -> dict:
    from biobank_agent.utils.bcftools import merge_vcfs, get_tmp_dir
    from biobank_agent.utils.vcf_genotypes import build_genotype_matrix, get_sample_vcf_paths

    sample_paths = get_sample_vcf_paths(ctx)
    if len(sample_paths) < 2:
        return {"error": "Need at least 2 samples for kinship estimation."}

    qc_data = None
    if hasattr(ctx, "state") and hasattr(ctx.state, "custom_data"):
        qc_data = ctx.state.custom_data.get("vcf_qc_results")
    if qc_data and qc_data.get("pass_samples"):
        pass_set = set(qc_data["pass_samples"])
        sample_paths = {k: v for k, v in sample_paths.items() if k in pass_set}
        logger.info("Using %d QC-pass samples for kinship", len(sample_paths))

    tmp_dir = get_tmp_dir(ctx)
    merged = tmp_dir / "merged_kinship.vcf.gz"

    merge_vcfs(list(sample_paths.values()), merged, region=region or None)

    G, vids, samples = build_genotype_matrix(
        merged,
        maf_min=maf_threshold,
        max_variants=max_snps,
        snv_only=True,
    )

    if G.shape[1] < 10:
        return {"error": f"Only {G.shape[1]} variants passed MAF filter; need more for kinship."}

    n_samples, n_snps = G.shape

    G_norm = (G - G.mean(axis=0)) / (G.std(axis=0) + 1e-10)
    GRM = G_norm @ G_norm.T / n_snps

    kinship = GRM / 2.0

    related_pairs = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            k = float(kinship[i, j])
            if k > kinship_threshold:
                relationship = "unknown"
                if k > 0.354:
                    relationship = "identical/MZ twin"
                elif k > 0.177:
                    relationship = "1st degree (parent-child/full sibling)"
                elif k > 0.0884:
                    relationship = "2nd degree (half-sibling/grandparent)"
                related_pairs.append({
                    "sample1": samples[i],
                    "sample2": samples[j],
                    "kinship": round(k, 4),
                    "relationship_estimate": relationship,
                })

    figures = []
    report_dir = wgs_results_dir(ctx, "popgen")

    fig, ax = nature_figure(width="single", height_ratio=1.0)
    import matplotlib.pyplot as plt

    pheno_df = None
    try:
        pheno_df = ctx.dm.query("SELECT * FROM biomarkers")
    except Exception:
        pass

    labels = list(samples)
    if pheno_df is not None and "phenotype_group" in pheno_df.columns:
        id_col = getattr(ctx.dm, "subject_id_col", "sample_id")
        grp_map = dict(zip(pheno_df[id_col], pheno_df["phenotype_group"]))
        labels = [f"{s} ({grp_map.get(s, '?')})" for s in samples]

    kinship_plot = kinship.copy()
    np.fill_diagonal(kinship_plot, np.nan)
    off_diag = kinship[np.triu_indices(n_samples, k=1)]
    heatmap_vmax = max(0.25, float(off_diag.max()) * 1.1) if len(off_diag) > 0 else 0.25

    im = ax.imshow(kinship_plot, cmap="RdBu_r", vmin=-0.05, vmax=heatmap_vmax, aspect="auto")
    ax.set_xticks(range(n_samples))
    ax.set_yticks(range(n_samples))
    ax.set_xticklabels(labels, rotation=90, fontsize=4)
    ax.set_yticklabels(labels, fontsize=4)
    ax.set_title(f"Kinship Matrix (n={n_snps:,} SNPs)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Kinship coefficient")
    fig.tight_layout()
    paths = save_figure(fig, "vcf_kinship_heatmap", report_dir)
    figures.extend(paths)

    if hasattr(ctx, "state") and hasattr(ctx.state, "figures"):
        ctx.state.figures.extend(figures)

    off_diag = kinship[np.triu_indices(n_samples, k=1)]

    return {
        "n_samples": n_samples,
        "n_snps_used": n_snps,
        "n_related_pairs": len(related_pairs),
        "kinship_threshold": kinship_threshold,
        "related_pairs": related_pairs,
        "kinship_summary": {
            "mean_off_diagonal": round(float(off_diag.mean()), 4),
            "max_off_diagonal": round(float(off_diag.max()), 4),
            "min_off_diagonal": round(float(off_diag.min()), 4),
        },
        "figures": [str(p) for p in figures],
        "result_dir": str(report_dir),
    }
