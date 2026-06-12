"""Gene-level burden testing for rare variants."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from biobank_agent.registry import skill
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE
from biobank_agent.utils.stats import benjamini_hochberg
from biobank_agent.utils.wgs import wgs_results_dir

logger = logging.getLogger(__name__)


@skill(
    name="vcf_burden_test",
    description=(
        "Gene-level collapsing burden test for rare variants comparing two "
        "phenotype groups. Uses Fisher's exact test on per-gene carrier counts "
        "and a weighted sum test (Madsen-Browning style). Focuses on vitiligo "
        "candidate genes by default."
    ),
    parameters={
        "case_group": {
            "type": "string",
            "description": "Phenotype group code for cases (default 'J')",
            "default": "J",
        },
        "control_group": {
            "type": "string",
            "description": "Phenotype group code for controls (default 'V')",
            "default": "V",
        },
        "maf_max": {
            "type": "number",
            "description": "Rare variant MAF threshold (default 0.05 for small cohorts)",
            "default": 0.05,
        },
        "min_variants_per_gene": {
            "type": "integer",
            "description": "Minimum rare variants per gene to include (default 2)",
            "default": 2,
        },
        "gene_list": {
            "type": "string",
            "description": "Comma-separated gene names (empty = vitiligo candidates)",
            "default": "",
        },
    },
    required=[],
)
def vcf_burden_test(
    case_group: str = "J",
    control_group: str = "V",
    maf_max: float = 0.05,
    min_variants_per_gene: int = 2,
    gene_list: str = "",
    *,
    ctx=None,
) -> dict:
    from biobank_agent.skills.vcf_annotation import VITILIGO_GENES_HG38
    from biobank_agent.utils.bcftools import merge_vcfs, get_tmp_dir
    from biobank_agent.utils.vcf_genotypes import get_sample_vcf_paths
    import cyvcf2

    pheno_df = ctx.dm.query("SELECT * FROM biomarkers")
    id_col = getattr(ctx.dm, "subject_id_col", "sample_id")

    case_samples = pheno_df.loc[pheno_df["phenotype_group"] == case_group, id_col].tolist()
    ctrl_samples = pheno_df.loc[pheno_df["phenotype_group"] == control_group, id_col].tolist()

    if len(case_samples) < 2 or len(ctrl_samples) < 2:
        return {"error": f"Need >=2 samples per group. Cases={len(case_samples)}, Controls={len(ctrl_samples)}."}

    sample_paths = get_sample_vcf_paths(ctx)

    qc_data = None
    if hasattr(ctx, "state") and hasattr(ctx.state, "custom_data"):
        qc_data = ctx.state.custom_data.get("vcf_qc_results")
    if qc_data and qc_data.get("pass_samples"):
        pass_set = set(qc_data["pass_samples"])
        case_samples = [s for s in case_samples if s in pass_set]
        ctrl_samples = [s for s in ctrl_samples if s in pass_set]
        logger.info("After QC filter: %d cases, %d controls", len(case_samples), len(ctrl_samples))

    all_sample_ids = [s for s in case_samples + ctrl_samples if s in sample_paths]
    all_vcfs = [sample_paths[s] for s in all_sample_ids]

    if gene_list:
        target_genes = {}
        for g in gene_list.split(","):
            g = g.strip().upper()
            if g in VITILIGO_GENES_HG38:
                target_genes[g] = VITILIGO_GENES_HG38[g]
    else:
        target_genes = dict(VITILIGO_GENES_HG38)

    tmp_dir = get_tmp_dir(ctx)
    gene_results = []
    skipped_regions: list[dict[str, str]] = []

    genes_by_chrom: dict[str, list[tuple[str, int, int]]] = {}
    for gene_name, (chrom, start, end) in target_genes.items():
        genes_by_chrom.setdefault(chrom, []).append((gene_name, start, end))

    chrom_merged_cache: dict[str, str] = {}

    for chrom, gene_list_on_chrom in genes_by_chrom.items():
        chrom_start = min(s for _, s, _ in gene_list_on_chrom)
        chrom_end = max(e for _, _, e in gene_list_on_chrom)
        chrom_region = f"{chrom}:{chrom_start}-{chrom_end}"
        merged = tmp_dir / f"merged_burden_{chrom}.vcf.gz"

        try:
            merge_vcfs(all_vcfs, merged, region=chrom_region)
        except Exception as e:
            logger.warning("Merge failed for %s: %s", chrom, e)
            skipped_regions.append({"region": chrom_region, "stage": "merge", "reason": str(e)})
            continue
        chrom_merged_cache[chrom] = str(merged)

        for gene_name, gstart, gend in gene_list_on_chrom:
            gene_region = f"{chrom}:{gstart}-{gend}"

            vcf = cyvcf2.VCF(str(merged))
            samples_in_vcf = list(vcf.samples)

            case_idx = [samples_in_vcf.index(s) for s in case_samples if s in samples_in_vcf]
            ctrl_idx = [samples_in_vcf.index(s) for s in ctrl_samples if s in samples_in_vcf]

            if not case_idx or not ctrl_idx:
                vcf.close()
                continue

            n_total = len(case_idx) + len(ctrl_idx)
            rare_variants = []
            case_carrier_set = set()
            ctrl_carrier_set = set()

            for v in vcf(gene_region):
                if not v.ALT or v.ALT[0] == ".":
                    continue
                gt_types = v.gt_types
                valid = gt_types != 3
                n_valid = valid.sum()
                if n_valid < n_total * 0.5:
                    continue

                alt_count = gt_types[valid].sum()
                maf = alt_count / (2 * n_valid) if n_valid > 0 else 0

                if maf > maf_max or maf == 0:
                    continue

                rare_variants.append({
                    "pos": v.POS,
                    "ref": v.REF,
                    "alt": v.ALT[0],
                    "maf": maf,
                    "gt_types": gt_types.copy(),
                })

                for idx in case_idx:
                    if gt_types[idx] in (1, 2):
                        case_carrier_set.add(idx)
                for idx in ctrl_idx:
                    if gt_types[idx] in (1, 2):
                        ctrl_carrier_set.add(idx)

            vcf.close()

            n_rare = len(rare_variants)
            if n_rare < min_variants_per_gene:
                skipped_regions.append({
                    "region": gene_region,
                    "stage": "variant_filter",
                    "reason": (
                        f"Only {n_rare} rare variant(s) remained; "
                        f"min_variants_per_gene={min_variants_per_gene}."
                    ),
                })
                continue

            n_case_carriers = len(case_carrier_set)
            n_case_noncarriers = len(case_idx) - n_case_carriers
            n_ctrl_carriers = len(ctrl_carrier_set)
            n_ctrl_noncarriers = len(ctrl_idx) - n_ctrl_carriers

            table = np.array([[n_case_carriers, n_case_noncarriers],
                              [n_ctrl_carriers, n_ctrl_noncarriers]])
            or_val, p_burden = sp_stats.fisher_exact(table)

            p_weighted = 1.0
            if n_rare >= 2:
                case_scores = np.zeros(len(case_idx))
                ctrl_scores = np.zeros(len(ctrl_idx))
                for rv in rare_variants:
                    w = 1.0 / max(np.sqrt(rv["maf"] * (1 - rv["maf"])), 0.01)
                    for i, idx in enumerate(case_idx):
                        if rv["gt_types"][idx] in (1, 2):
                            case_scores[i] += w * rv["gt_types"][idx]
                    for i, idx in enumerate(ctrl_idx):
                        if rv["gt_types"][idx] in (1, 2):
                            ctrl_scores[i] += w * rv["gt_types"][idx]

                if case_scores.std() > 0 or ctrl_scores.std() > 0:
                    try:
                        _, p_weighted = sp_stats.mannwhitneyu(
                            case_scores, ctrl_scores, alternative="two-sided"
                        )
                    except Exception:
                        p_weighted = 1.0

            gene_results.append({
                "gene": gene_name,
                "region": f"{chrom}:{gstart}-{gend}",
                "n_rare_variants": n_rare,
                "maf_threshold": maf_max,
                "n_case_carriers": n_case_carriers,
                "n_case_total": len(case_idx),
                "n_ctrl_carriers": n_ctrl_carriers,
                "n_ctrl_total": len(ctrl_idx),
                "odds_ratio": round(or_val, 3) if or_val != float("inf") else "Inf",
                "p_burden": float(p_burden),
                "p_weighted": float(p_weighted),
            })

    if not gene_results:
        return {
            "error": "No genes had enough rare variants for burden testing.",
            "skipped_regions": skipped_regions,
        }

    p_burden_arr = np.array([r["p_burden"] for r in gene_results])
    p_weighted_arr = np.array([r["p_weighted"] for r in gene_results])
    fdr_burden = benjamini_hochberg(p_burden_arr)
    fdr_weighted = benjamini_hochberg(p_weighted_arr)

    for i, gr in enumerate(gene_results):
        gr["p_burden_fdr"] = float(fdr_burden[i])
        gr["p_weighted_fdr"] = float(fdr_weighted[i])
        gr["significant_burden"] = fdr_burden[i] < 0.05
        gr["significant_weighted"] = fdr_weighted[i] < 0.05

    gene_results.sort(key=lambda x: x["p_burden"])

    figures = []
    report_dir = wgs_results_dir(ctx, "association")

    # Lollipop plot
    fig, ax = nature_figure(width="double", height_ratio=0.5)
    names = [gr["gene"] for gr in gene_results]
    p_vals = [-np.log10(max(gr["p_burden"], 1e-20)) for gr in gene_results]
    n_vars = [gr["n_rare_variants"] for gr in gene_results]
    sig = [gr["significant_burden"] for gr in gene_results]

    colors_lollipop = [PALETTE[1] if s else PALETTE[0] for s in sig]
    sizes = [max(20, min(100, n * 5)) for n in n_vars]

    y_pos = range(len(names))
    ax.hlines(y_pos, 0, p_vals, colors=colors_lollipop, linewidth=1, alpha=0.6)
    ax.scatter(p_vals, y_pos, s=sizes, c=colors_lollipop, alpha=0.8, edgecolors="white", linewidth=0.3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel(r"$-\log_{10}(P_{burden})$")
    ax.set_title(f"Burden Test: {case_group} vs {control_group} (rare MAF<{maf_max})")

    bonferroni_line = -np.log10(0.05 / len(gene_results)) if gene_results else 1
    ax.axvline(x=bonferroni_line, color="red", linewidth=0.5, linestyle="--", alpha=0.5, label="Bonferroni")

    fig.tight_layout()
    paths = save_figure(fig, "vcf_burden_test", report_dir)
    figures.extend(paths)

    if hasattr(ctx, "state") and hasattr(ctx.state, "figures"):
        ctx.state.figures.extend(figures)

    if hasattr(ctx, "state") and hasattr(ctx.state, "custom_data"):
        ctx.state.custom_data["burden_results"] = {
            "gene_results": gene_results,
            "significant_genes": [gr["gene"] for gr in gene_results if gr["significant_burden"]],
            "skipped_regions": skipped_regions,
        }

    n_sig = sum(1 for gr in gene_results if gr["significant_burden"])

    return {
        "case_group": case_group,
        "control_group": control_group,
        "n_genes_tested": len(gene_results),
        "n_genes_significant_burden": n_sig,
        "maf_threshold": maf_max,
        "gene_results": gene_results,
        "top_gene": gene_results[0]["gene"] if gene_results else "",
        "figures": [str(p) for p in figures],
        "result_dir": str(report_dir),
        "skipped_regions": skipped_regions,
    }
