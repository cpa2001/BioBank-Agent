"""Genotype-level PCA for population structure analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from biobank_agent.registry import skill
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE
from biobank_agent.utils.wgs import wgs_results_dir

logger = logging.getLogger(__name__)

_AUTOSOMES = [f"chr{i}" for i in range(1, 23)]


def _expand_chromosomes(spec: str) -> list[str]:
    if not spec or spec.lower() in ("all", "chr1-22"):
        return _AUTOSOMES
    parts = []
    for token in spec.replace(" ", "").split(","):
        if "-" in token and token.startswith("chr"):
            prefix, rest = "chr", token[3:]
            if "-" in rest:
                a, b = rest.split("-", 1)
                for i in range(int(a), int(b) + 1):
                    parts.append(f"chr{i}")
            else:
                parts.append(token)
        else:
            parts.append(token)
    return parts


@skill(
    name="vcf_pca",
    description=(
        "Run genotype-level PCA on common variants across all WGS samples. "
        "Uses bcftools merge + sklearn PCA (no plink2 required). "
        "Generates scree plot and PC scatter plots colored by phenotype group. "
        "Stores principal components in session state for use as GWAS covariates."
    ),
    parameters={
        "maf_threshold": {
            "type": "number",
            "description": "Minimum cohort MAF for variant inclusion (default 0.05)",
            "default": 0.05,
        },
        "n_components": {
            "type": "integer",
            "description": "Number of principal components (default 10)",
            "default": 10,
        },
        "max_variants": {
            "type": "integer",
            "description": "Max SNPs for PCA; random subsample if exceeded (default 50000)",
            "default": 50000,
        },
        "chromosomes": {
            "type": "string",
            "description": "Chromosomes to include (default chr1-22)",
            "default": "chr1-22",
        },
        "region": {
            "type": "string",
            "description": "Specific region (overrides chromosomes), e.g. chr22:10000000-50000000",
            "default": "",
        },
    },
    required=[],
)
def vcf_pca(
    maf_threshold: float = 0.05,
    n_components: int = 10,
    max_variants: int = 50000,
    chromosomes: str = "chr1-22",
    region: str = "",
    *,
    ctx=None,
) -> dict:
    from biobank_agent.utils.bcftools import merge_vcfs, get_tmp_dir
    from biobank_agent.utils.vcf_genotypes import build_genotype_matrix, get_sample_vcf_paths

    sample_paths = get_sample_vcf_paths(ctx)
    if not sample_paths:
        return {"error": "No sample VCF paths found."}

    qc_data = None
    if hasattr(ctx, "state") and hasattr(ctx.state, "custom_data"):
        qc_data = ctx.state.custom_data.get("vcf_qc_results")

    if qc_data and qc_data.get("pass_samples"):
        pass_set = set(qc_data["pass_samples"])
        sample_paths = {k: v for k, v in sample_paths.items() if k in pass_set}
        logger.info("Using %d QC-pass samples for PCA", len(sample_paths))

    if len(sample_paths) < 3:
        return {"error": f"Need at least 3 samples for PCA, have {len(sample_paths)}."}

    tmp_dir = get_tmp_dir(ctx)

    chrom_list = _expand_chromosomes("" if region else chromosomes)
    regions_to_process = [region] if region else chrom_list

    all_G = []
    all_vids = []
    sample_order = None

    for rgn in regions_to_process:
        merged_path = tmp_dir / f"merged_pca_{rgn.replace(':', '_')}.vcf.gz"
        try:
            merge_vcfs(list(sample_paths.values()), merged_path, region=rgn)
        except Exception as e:
            logger.warning("Merge failed for %s: %s", rgn, e)
            continue

        G_chr, vids_chr, samples = build_genotype_matrix(
            merged_path,
            maf_min=maf_threshold,
            maf_max=1.0 - maf_threshold,
            max_variants=max_variants - len(all_vids) if max_variants else 0,
            snv_only=True,
        )

        if G_chr.shape[1] == 0:
            continue

        if sample_order is None:
            sample_order = samples
        all_G.append(G_chr)
        all_vids.extend(vids_chr)

        if max_variants and len(all_vids) >= max_variants:
            break

    if not all_G:
        return {"error": "No common variants found after merging and MAF filtering."}

    G = np.hstack(all_G)
    n_samples, n_snps = G.shape

    if max_variants and n_snps > max_variants:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_snps, max_variants, replace=False)
        idx.sort()
        G = G[:, idx]
        all_vids = [all_vids[i] for i in idx]
        n_snps = max_variants

    G_std = (G - G.mean(axis=0)) / (G.std(axis=0) + 1e-10)

    n_comp = min(n_components, n_samples - 1, n_snps)
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(G_std)
    explained = pca.explained_variance_ratio_

    pheno_df = None
    try:
        pheno_df = ctx.dm.query("SELECT * FROM biomarkers")
    except Exception:
        pass

    pc_records = []
    for i, sid in enumerate(sample_order):
        rec = {"sample_id": sid}
        for j in range(n_comp):
            rec[f"PC{j+1}"] = round(float(pcs[i, j]), 6)
        if pheno_df is not None:
            id_col = getattr(ctx.dm, "subject_id_col", "sample_id")
            match = pheno_df[pheno_df[id_col] == sid]
            if not match.empty and "phenotype_group" in match.columns:
                rec["phenotype_group"] = str(match.iloc[0]["phenotype_group"])
        pc_records.append(rec)

    figures = []
    report_dir = wgs_results_dir(ctx, "popgen")

    # Scree plot
    fig, ax = nature_figure(width="single", height_ratio=0.7)
    x = range(1, n_comp + 1)
    ax.bar(x, explained * 100, color=PALETTE[0], alpha=0.8, width=0.7)
    ax.plot(x, np.cumsum(explained) * 100, "o-", color=PALETTE[1], markersize=3, linewidth=1)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title(f"PCA Scree Plot (n={n_snps:,} SNPs)")
    ax.set_xticks(list(x))
    fig.tight_layout()
    paths = save_figure(fig, "vcf_pca_scree", report_dir)
    figures.extend(paths)

    # PC1 vs PC2 scatter
    pc_df = pd.DataFrame(pc_records)

    for pc_pair, suffix in [("PC1", "PC2"), ("PC1", "PC3")]:
        if suffix not in pc_df.columns:
            continue
        fig2, ax2 = nature_figure(width="single", height_ratio=0.9)
        if "phenotype_group" in pc_df.columns:
            groups = sorted(pc_df["phenotype_group"].dropna().unique())
            for i, grp in enumerate(groups):
                mask = pc_df["phenotype_group"] == grp
                ax2.scatter(
                    pc_df.loc[mask, pc_pair], pc_df.loc[mask, suffix],
                    c=PALETTE[i % len(PALETTE)], label=str(grp),
                    s=40, alpha=0.8, edgecolors="white", linewidth=0.3,
                )
            ax2.legend(fontsize=6, frameon=False)
        else:
            ax2.scatter(pc_df[pc_pair], pc_df[suffix], c=PALETTE[0], s=40, alpha=0.8)

        ax2.set_xlabel(f"{pc_pair} ({explained[int(pc_pair[2:])-1]*100:.1f}%)")
        ax2.set_ylabel(f"{suffix} ({explained[int(suffix[2:])-1]*100:.1f}%)")
        ax2.set_title(f"Genotype PCA: {pc_pair} vs {suffix}")
        fig2.tight_layout()
        name = f"vcf_pca_{pc_pair.lower()}_{suffix.lower()}"
        paths2 = save_figure(fig2, name, report_dir)
        figures.extend(paths2)

    if hasattr(ctx, "state") and hasattr(ctx.state, "figures"):
        ctx.state.figures.extend(figures)

    if hasattr(ctx, "state") and hasattr(ctx.state, "custom_data"):
        ctx.state.custom_data["pca_result"] = {
            "pcs": pc_df,
            "explained_variance": [round(float(v), 6) for v in explained],
            "sample_order": sample_order,
            "n_snps": n_snps,
        }

    return {
        "n_samples": n_samples,
        "n_snps_used": n_snps,
        "n_components": n_comp,
        "maf_threshold": maf_threshold,
        "explained_variance_ratio": [round(float(v), 4) for v in explained],
        "cumulative_variance_pct": round(float(np.sum(explained)) * 100, 1),
        "pc_coordinates": pc_records[:10],
        "figures": [str(p) for p in figures],
        "result_dir": str(report_dir),
        "pca_stored_in_state": True,
    }
