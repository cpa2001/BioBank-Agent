"""VCF query skills for the VirtualCell WGS cohort."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from biobank_agent.registry import skill
from biobank_agent.utils.wgs import wgs_environment_status, wgs_results_dir


def _get_vcf_dirs() -> list[Path]:
    dirs: list[Path] = []
    for env in ("VC_WGS_VCF_DIR", "VC_VIRTUAL_VCF_DIR"):
        val = os.getenv(env, "").strip()
        if val:
            dirs.append(Path(val))
    for candidate in (
        Path("data/vc_wgs_vcf"),
        Path("data/VirtualCell_WGS_vcf"),
        Path("data/BW_WGS_vcf"),
        Path("input/Files/ResultData/VirtualCell_WGS_vcf"),
        Path("input/Files/ResultData/BW_WGS_vcf"),
        Path("/Files/ResultData/VirtualCell_WGS_vcf"),
        Path("/Files/ResultData/BW_WGS_vcf"),
    ):
        dirs.append(candidate)
    out: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        key = str(d.expanduser())
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _build_vcf_dm(ctx: Any = None):
    from biobank_agent.data.vcf_loader import VCFDataManager

    vcf_dirs = _get_vcf_dirs()
    if not vcf_dirs:
        settings = getattr(ctx, "settings", None)
        if settings:
            data_dir = Path(getattr(settings, "data_dir", "."))
            vcf_dirs = [
                data_dir.parent / "bw_wgs_vcf",
                data_dir.parent / "vc_wgs_vcf",
                data_dir / "VirtualCell_WGS_vcf",
                Path("input/Files/ResultData/VirtualCell_WGS_vcf"),
                Path("input/Files/ResultData/BW_WGS_vcf"),
                Path("/Files/ResultData/BW_WGS_vcf"),
                Path("/Files/ResultData/VirtualCell_WGS_vcf"),
            ]
    conn = getattr(getattr(ctx, "dm", None), "conn", None)
    return VCFDataManager(vcf_dirs=vcf_dirs, conn=conn)


@skill(
    name="vcf_sample_list",
    description=(
        "List all WGS VCF samples in the VirtualCell cohort with file paths, "
        "index status, and file sizes."
    ),
    parameters={
        "check_index": {
            "type": "boolean",
            "description": "Also verify .tbi index files exist for each VCF",
            "default": True,
        },
    },
    required=[],
)
def vcf_sample_list(check_index: bool = True, *, ctx=None) -> dict:
    vcf_dm = _build_vcf_dm(ctx)
    files = vcf_dm.list_vcf_files()
    if not files:
        return {"error": "No VCF files found. Check VC_WGS_VCF_DIR env var.", "n_samples": 0}

    summary = {
        "n_samples": len(files),
        "total_size_gb": round(sum(f["size_mb"] for f in files) / 1024, 2),
        "samples": files,
    }
    if check_index:
        n_indexed = sum(1 for f in files if f.get("has_index"))
        summary["n_indexed"] = n_indexed
        summary["n_missing_index"] = len(files) - n_indexed

    return summary


@skill(
    name="wgs_environment_check",
    description=(
        "Check local WGS workflow dependencies and report whether the current "
        "environment can run exploratory or standard WGS analysis modules."
    ),
    parameters={},
    required=[],
)
def wgs_environment_check(*, ctx=None) -> dict:
    status = wgs_environment_status()
    vcf_dm = _build_vcf_dm(ctx)
    files = vcf_dm.list_vcf_files()
    status["vcf_samples"] = {
        "n_samples": len(files),
        "n_indexed": sum(1 for f in files if f.get("has_index")),
        "total_size_gb": round(sum(f.get("size_mb", 0) for f in files) / 1024, 2),
    }
    return status


@skill(
    name="vcf_variant_query",
    description=(
        "Query variants from one or more WGS VCF samples by genomic region. "
        "Returns variant table and per-sample genotypes."
    ),
    parameters={
        "sample_ids": {
            "type": "string",
            "description": "Comma-separated sample IDs (filename stems), or empty for first sample",
            "default": "",
        },
        "region": {
            "type": "string",
            "description": "Genomic region in CHROM:START-END format (e.g. chr1:1000000-2000000)",
            "default": "",
        },
        "max_variants": {
            "type": "integer",
            "description": "Max variants per sample to return (0=unlimited)",
            "default": 1000,
        },
    },
    required=[],
)
def vcf_variant_query(
    sample_ids: str = "",
    region: str = "",
    max_variants: int = 1000,
    *,
    ctx=None,
) -> dict:
    vcf_dm = _build_vcf_dm(ctx)

    sid_list = [s.strip() for s in sample_ids.split(",") if s.strip()] if sample_ids else None
    if not sid_list:
        files = vcf_dm.list_vcf_files()
        if files:
            sid_list = [files[0]["filename"]]

    variants_df = vcf_dm.query_variants(
        sample_ids=sid_list,
        region=region or None,
        max_variants_per_sample=max_variants,
    )

    if variants_df.empty:
        return {"error": "No variants found for the given parameters.", "n_variants": 0}

    gts_df = vcf_dm.conn.execute("SELECT * FROM vcf_genotypes").df()

    return {
        "n_variants": len(variants_df),
        "n_genotype_records": len(gts_df),
        "samples_queried": sid_list,
        "region": region or "whole genome",
        "variants_preview": variants_df.head(20).to_dict(orient="records"),
        "genotypes_preview": gts_df.head(20).to_dict(orient="records"),
    }


@skill(
    name="vcf_cohort_stats",
    description=(
        "Compute cohort-wide WGS variant statistics from one or more samples: "
        "SNV/indel counts, per-chromosome breakdown. Generates a bar chart."
    ),
    parameters={
        "sample_ids": {
            "type": "string",
            "description": "Comma-separated sample IDs, or empty for first sample",
            "default": "",
        },
        "region": {
            "type": "string",
            "description": "Genomic region to analyze (e.g. chr1:1-50000000), or empty for all",
            "default": "",
        },
        "max_variants_per_sample": {
            "type": "integer",
            "description": "Cap variants per sample for speed (0=unlimited, slow for WGS)",
            "default": 50000,
        },
    },
    required=[],
)
def vcf_cohort_stats(
    sample_ids: str = "",
    region: str = "",
    max_variants_per_sample: int = 50000,
    *,
    ctx=None,
) -> dict:
    vcf_dm = _build_vcf_dm(ctx)

    sid_list = [s.strip() for s in sample_ids.split(",") if s.strip()] if sample_ids else None
    if not sid_list:
        files = vcf_dm.list_vcf_files()
        if files:
            sid_list = [files[0]["filename"]]

    variants_df = vcf_dm.query_variants(
        sample_ids=sid_list,
        region=region or None,
        max_variants_per_sample=max_variants_per_sample,
    )

    if variants_df.empty:
        return {"error": "No variants found.", "n_variants": 0}

    total = len(variants_df)
    snvs = int((variants_df["ref"].str.len() == 1).sum() & (variants_df["alt"].str.len() == 1).sum())
    chrom_counts = variants_df["chrom"].value_counts().to_dict()

    result = {
        "samples_analyzed": sid_list,
        "region": region or "capped scan",
        "total_variants": total,
        "snvs": snvs,
        "indels": total - snvs,
        "per_chromosome": dict(sorted(chrom_counts.items())),
    }

    report_dir = getattr(ctx, "report_dir", None)
    if report_dir is None:
        settings = getattr(ctx, "settings", None)
        if settings:
            report_dir = getattr(settings, "reports_dir", Path("./reports"))
    if report_dir:
        try:
            fig_paths = _plot_chrom_bar(chrom_counts, sid_list, wgs_results_dir(ctx, "qc"))
            result["figures"] = [str(p) for p in fig_paths]
            state = getattr(ctx, "state", None)
            if state and hasattr(state, "figures"):
                state.figures.extend(fig_paths)
        except Exception:
            pass

    return result


def _plot_chrom_bar(chrom_counts: dict, samples: list[str], report_dir: Path) -> list[Path]:
    from biobank_agent.utils.plotting import nature_figure, save_figure

    chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    counts = [chrom_counts.get(c, 0) for c in chroms]

    fig, ax = nature_figure(width="double", height_ratio=0.5)
    if hasattr(ax, "__len__"):
        ax = ax.flat[0]

    ax.bar(range(len(chroms)), counts, color="#4C72B0", edgecolor="none", width=0.8)
    ax.set_xticks(range(len(chroms)))
    ax.set_xticklabels([c.replace("chr", "") for c in chroms], fontsize=5)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Variant count")
    label = ", ".join(samples[:3])
    if len(samples) > 3:
        label += f" +{len(samples)-3}"
    ax.set_title(f"Variants per chromosome ({label})")
    fig.tight_layout()

    return save_figure(fig, "vcf_cohort_chrom_stats", report_dir)
