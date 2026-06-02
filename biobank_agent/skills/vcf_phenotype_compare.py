"""VCF-phenotype comparison — variant burden differences across phenotype groups."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from biobank_agent.registry import skill
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE
from biobank_agent.utils.wgs import wgs_results_dir

logger = logging.getLogger(__name__)


def _stats_cache_dir() -> Path | None:
    value = (
        os.getenv("WGS_STATS_CACHE_DIR", "").strip()
        or os.getenv("BIOBANK_WGS_STATS_CACHE_DIR", "").strip()
        or os.getenv("WGS_VCF_CACHE_DIR", "").strip()
        or os.getenv("BIOBANK_WGS_VCF_CACHE_DIR", "").strip()
    )
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _vcf_signature(path: str | Path) -> dict:
    p = Path(path)
    try:
        stat = p.stat()
        return {"path": str(p.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        return {"path": str(p), "missing": True}


def _sample_stats_cache_key(sample_id: str, vcf_path: str, region: str | None, max_variants: int) -> str:
    payload = {
        "version": 1,
        "operation": "phenotype_sample_variant_stats",
        "sample_id": sample_id,
        "vcf": _vcf_signature(vcf_path),
        "region": region or "",
        "max_variants": int(max_variants or 0),
    }
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _compute_sample_variant_stats(
    vcf_dm,
    sample_ids: list[str],
    region: Optional[str],
    max_variants: int,
) -> pd.DataFrame:
    rows = []
    vcf_by_id = {f["filename"]: f["vcf_path"] for f in vcf_dm.list_vcf_files()}
    cache_dir = _stats_cache_dir()
    for sid in sample_ids:
        try:
            vcf_path = vcf_by_id.get(sid)
            if not vcf_path:
                continue
            cache_path = None
            if cache_dir is not None:
                cache_key = _sample_stats_cache_key(sid, vcf_path, region, max_variants)
                cache_path = cache_dir / f"sample_variant_stats_{cache_key}.json"
                if cache_path.exists():
                    try:
                        rows.append(json.loads(cache_path.read_text(encoding="utf-8")))
                        continue
                    except Exception as exc:
                        logger.debug("Ignoring unreadable phenotype stats cache %s: %s", cache_path, exc)

            vdf, gdf = None, None
            from biobank_agent.data.vcf_loader import extract_variants_from_vcf
            vdf, gdf = extract_variants_from_vcf(
                vcf_path, sid, max_variants=max_variants, region=region,
            )
            if vdf is None or vdf.empty:
                continue

            n_total = len(vdf)
            snv_mask = (vdf["ref"].str.len() == 1) & (vdf["alt"].str.len() == 1)
            n_snv = int(snv_mask.sum())
            n_indel = n_total - n_snv

            ti_tv = _ti_tv_ratio(vdf[snv_mask]) if n_snv > 0 else float("nan")

            mean_dp = float(gdf["dp"].mean()) if "dp" in gdf.columns and not gdf["dp"].isnull().all() else float("nan")
            mean_gq = float(gdf["gq"].mean()) if "gq" in gdf.columns and not gdf["gq"].isnull().all() else float("nan")

            het_count = int((gdf["gt_numeric"] == 1).sum()) if "gt_numeric" in gdf.columns else 0
            hom_alt_count = int((gdf["gt_numeric"] == 2).sum()) if "gt_numeric" in gdf.columns else 0

            row = {
                "sample_id": sid,
                "n_variants": n_total,
                "n_snv": n_snv,
                "n_indel": n_indel,
                "ti_tv_ratio": round(ti_tv, 3) if not np.isnan(ti_tv) else None,
                "mean_dp": round(mean_dp, 1) if not np.isnan(mean_dp) else None,
                "mean_gq": round(mean_gq, 1) if not np.isnan(mean_gq) else None,
                "n_het": het_count,
                "n_hom_alt": hom_alt_count,
                "het_hom_ratio": round(het_count / hom_alt_count, 3) if hom_alt_count > 0 else None,
            }
            rows.append(row)
            if cache_path is not None:
                tmp = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
                try:
                    tmp.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
                    os.replace(tmp, cache_path)
                finally:
                    tmp.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Failed stats for %s: %s", sid, e)

    return pd.DataFrame(rows)


def _ti_tv_ratio(snv_df: pd.DataFrame) -> float:
    transitions = {"AG", "GA", "CT", "TC"}
    ref = snv_df["ref"].str.upper()
    alt = snv_df["alt"].str.upper()
    pairs = ref + alt
    ti = int(pairs.isin(transitions).sum())
    tv = len(pairs) - ti
    return ti / tv if tv > 0 else float("nan")


def _kruskal_wallis(groups: dict[str, np.ndarray]) -> dict:
    arrays = [a for a in groups.values() if len(a) >= 2]
    if len(arrays) < 2:
        return {"statistic": None, "p_value": None, "test": "kruskal_wallis", "note": "insufficient groups"}
    stat, p = stats.kruskal(*arrays)
    return {"statistic": round(float(stat), 4), "p_value": float(p), "test": "kruskal_wallis"}


@skill(
    name="vcf_phenotype_comparison",
    description=(
        "Compare WGS variant features across phenotype groups. Computes per-sample "
        "variant counts, SNV/indel ratio, Ti/Tv ratio, depth, genotype quality, "
        "and het/hom ratio, then tests for between-group differences "
        "(Kruskal-Wallis). Generates grouped bar and box plots."
    ),
    parameters={
        "region": {
            "type": "string",
            "description": "Genomic region (e.g. chr1:1-50000000) or empty for capped scan",
            "default": "",
        },
        "group_by": {
            "type": "string",
            "description": "Column name for grouping (default: phenotype_group)",
            "default": "phenotype_group",
        },
        "max_variants_per_sample": {
            "type": "integer",
            "description": "Max variants per sample (default 50000; 0=unlimited)",
            "default": 50000,
        },
        "sample_ids": {
            "type": "string",
            "description": "Comma-separated sample IDs, or empty for all",
            "default": "",
        },
    },
    required=[],
)
def vcf_phenotype_comparison(
    region: str = "",
    group_by: str = "phenotype_group",
    max_variants_per_sample: int = 50000,
    sample_ids: str = "",
    *,
    ctx=None,
) -> dict:
    from biobank_agent.skills.vcf_query import _build_vcf_dm

    dm = ctx.dm
    vcf_dm = _build_vcf_dm(ctx)

    pheno_df = dm.query("SELECT * FROM biomarkers")
    if pheno_df.empty:
        return {"error": "No phenotype data in biomarkers view."}
    if group_by not in pheno_df.columns:
        return {"error": f"Column '{group_by}' not found in phenotype data."}

    id_col = getattr(dm, "subject_id_col", ctx.settings.subject_id_col)

    vcf_files = vcf_dm.list_vcf_files()
    if not vcf_files:
        return {"error": "No VCF files found."}

    available_sids = {f["filename"] for f in vcf_files}

    if sample_ids:
        target_sids = [s.strip() for s in sample_ids.split(",") if s.strip()]
    else:
        target_sids = [
            row[id_col] for _, row in pheno_df.iterrows()
            if row[id_col] in available_sids
        ]

    if not target_sids:
        return {"error": "No matching samples between phenotype data and VCF files."}

    sample_stats = _compute_sample_variant_stats(
        vcf_dm, target_sids, region=region or None, max_variants=max_variants_per_sample,
    )

    if sample_stats.empty:
        return {"error": "No variant statistics could be computed."}

    merged = sample_stats.merge(
        pheno_df[[id_col, group_by]],
        left_on="sample_id",
        right_on=id_col,
        how="left",
    )

    metrics = ["n_variants", "n_snv", "n_indel", "ti_tv_ratio", "mean_dp", "mean_gq", "het_hom_ratio"]
    group_results = {}
    test_results = {}

    for metric in metrics:
        if metric not in merged.columns or merged[metric].isnull().all():
            continue
        groups_dict = {}
        for g in sorted(merged[group_by].dropna().unique()):
            vals = merged.loc[merged[group_by] == g, metric].dropna().values
            if len(vals) > 0:
                groups_dict[str(g)] = vals

        if groups_dict:
            group_results[metric] = {
                g: {
                    "n": len(v),
                    "mean": round(float(np.mean(v)), 2),
                    "std": round(float(np.std(v)), 2),
                    "median": round(float(np.median(v)), 2),
                }
                for g, v in groups_dict.items()
            }
            test_results[metric] = _kruskal_wallis(groups_dict)

    figures = []
    report_dir = wgs_results_dir(ctx, "qc")

    plot_metrics = [m for m in ["n_variants", "ti_tv_ratio", "mean_dp"] if m in group_results]
    if plot_metrics:
        fig, axes = nature_figure(width="double", nrows=1, ncols=len(plot_metrics), height_ratio=0.5)
        if not hasattr(axes, "__len__"):
            axes = [axes]

        group_names = sorted(merged[group_by].dropna().unique())
        for ax_idx, metric in enumerate(plot_metrics):
            ax = axes[ax_idx]
            data_arrays = []
            labels = []
            for g in group_names:
                vals = merged.loc[merged[group_by] == g, metric].dropna().values
                data_arrays.append(vals)
                labels.append(str(g))
            if data_arrays:
                bp = ax.boxplot(data_arrays, patch_artist=True, widths=0.6)
                for patch, color in zip(bp["boxes"], PALETTE[: len(data_arrays)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_xticklabels(labels, fontsize=6)
                kw = test_results.get(metric, {})
                p_str = f"p={kw.get('p_value', 'NA'):.3g}" if kw.get("p_value") is not None else ""
                ax.set_title(f"{metric}\n{p_str}", fontsize=7)
        fig.tight_layout()
        paths = save_figure(fig, "vcf_phenotype_comparison", report_dir)
        figures.extend(paths)
        if hasattr(ctx, "state") and hasattr(ctx.state, "figures"):
            ctx.state.figures.extend(paths)

    return {
        "n_samples_analyzed": len(sample_stats),
        "n_samples_requested": len(target_sids),
        "region": region or "capped scan",
        "group_by": group_by,
        "group_stats": group_results,
        "statistical_tests": test_results,
        "result_dir": str(report_dir),
        "per_sample_preview": sample_stats.head(10).to_dict(orient="records"),
        "figures": [str(p) for p in figures],
    }
