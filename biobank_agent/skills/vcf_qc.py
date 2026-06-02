"""VCF quality control — per-sample and per-variant QC metrics."""

from __future__ import annotations

import hashlib
import logging
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from biobank_agent.registry import skill
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE, SEMANTIC_PALETTE
from biobank_agent.utils.wgs import wgs_results_dir

logger = logging.getLogger(__name__)


def _qc_cache_dir() -> Path | None:
    value = (
        os.getenv("WGS_QC_CACHE_DIR", "").strip()
        or os.getenv("BIOBANK_WGS_QC_CACHE_DIR", "").strip()
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


def _qc_cache_key(vcf_path: str, sample_id: str, region: str | None, max_variants: int) -> str:
    payload = {
        "version": 1,
        "operation": "per_sample_vcf_qc",
        "sample_id": sample_id,
        "vcf": _vcf_signature(vcf_path),
        "region": region or "",
        "max_variants": int(max_variants or 0),
    }
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _compute_per_sample_qc(vcf_path: str, sample_id: str, region: str | None = None,
                            max_variants: int = 200000) -> dict:
    """Compute QC metrics for a single sample VCF."""
    cache_dir = _qc_cache_dir()
    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"sample_qc_{_qc_cache_key(vcf_path, sample_id, region, max_variants)}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.debug("Ignoring unreadable QC cache %s: %s", cache_path, exc)

    import cyvcf2

    vcf = cyvcf2.VCF(vcf_path)
    n_total = 0
    n_pass = 0
    n_missing = 0
    n_het = 0
    n_hom_alt = 0
    n_hom_ref = 0
    dp_vals = []
    gq_vals = []
    ti = 0
    tv = 0
    transitions = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}

    iterator = vcf(region) if region else vcf
    for variant in iterator:
        if max_variants and n_total >= max_variants:
            break
        n_total += 1

        filt = variant.FILTER
        if filt is None or filt == "PASS":
            n_pass += 1

        gt = variant.gt_types[0] if len(variant.gt_types) > 0 else 3
        if gt == 3:
            n_missing += 1
        elif gt == 0:
            n_hom_ref += 1
        elif gt == 1:
            n_het += 1
        elif gt == 2:
            n_hom_alt += 1

        try:
            dp = variant.format("DP")
            if dp is not None and dp[0][0] >= 0:
                dp_vals.append(int(dp[0][0]))
        except Exception:
            pass

        try:
            gq = variant.format("GQ")
            if gq is not None and gq[0][0] >= 0:
                gq_vals.append(int(gq[0][0]))
        except Exception:
            pass

        if (len(variant.REF) == 1 and variant.ALT and
                len(variant.ALT) == 1 and len(variant.ALT[0]) == 1):
            pair = (variant.REF.upper(), variant.ALT[0].upper())
            if pair in transitions:
                ti += 1
            else:
                tv += 1

    vcf.close()

    called = n_total - n_missing
    call_rate = called / n_total if n_total > 0 else 0.0
    het_ratio = n_het / (n_het + n_hom_ref + n_hom_alt) if (n_het + n_hom_ref + n_hom_alt) > 0 else 0.0
    variant_only_vcf = (n_hom_ref == 0 and (n_het + n_hom_alt) > 0)
    ti_tv_ratio = ti / tv if tv > 0 else float("nan")

    result = {
        "sample_id": sample_id,
        "n_total_variants": n_total,
        "n_pass": n_pass,
        "n_missing": n_missing,
        "n_het": n_het,
        "n_hom_alt": n_hom_alt,
        "n_hom_ref": n_hom_ref,
        "call_rate": float(round(call_rate, 4)),
        "het_ratio": float(round(het_ratio, 4)),
        "ti_tv_ratio": float(round(ti_tv_ratio, 3)) if not np.isnan(ti_tv_ratio) else None,
        "mean_dp": float(round(np.mean(dp_vals), 1)) if dp_vals else None,
        "median_dp": float(round(np.median(dp_vals), 1)) if dp_vals else None,
        "mean_gq": float(round(np.mean(gq_vals), 1)) if gq_vals else None,
        "median_gq": float(round(np.median(gq_vals), 1)) if gq_vals else None,
        "variant_only_vcf": variant_only_vcf,
    }
    if cache_path is not None:
        tmp = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cache_path)
        finally:
            tmp.unlink(missing_ok=True)
    return result


@skill(
    name="vcf_qc",
    description=(
        "Per-sample and per-variant quality control for WGS VCF data. "
        "Computes call rate, heterozygosity ratio, Ti/Tv, mean depth and "
        "genotype quality per sample. Flags outlier samples (>3 SD). "
        "Optionally writes QC-filtered VCFs."
    ),
    parameters={
        "min_qual": {
            "type": "number",
            "description": "Minimum QUAL score threshold (default 30)",
            "default": 30.0,
        },
        "min_dp": {
            "type": "integer",
            "description": "Minimum per-sample depth (default 10)",
            "default": 10,
        },
        "min_gq": {
            "type": "integer",
            "description": "Minimum genotype quality (default 20)",
            "default": 20,
        },
        "min_call_rate": {
            "type": "number",
            "description": "Minimum per-sample call rate (default 0.90)",
            "default": 0.90,
        },
        "sample_ids": {
            "type": "string",
            "description": "Comma-separated sample IDs (empty = all)",
            "default": "",
        },
        "output_filtered": {
            "type": "boolean",
            "description": "Write QC-filtered VCFs (default false)",
            "default": False,
        },
        "region": {
            "type": "string",
            "description": "Genomic region for fast QC (e.g. chr22)",
            "default": "",
        },
        "max_variants_per_sample": {
            "type": "integer",
            "description": "Max variants to scan per sample (default 200000; 0=unlimited)",
            "default": 200000,
        },
    },
    required=[],
)
def vcf_qc(
    min_qual: float = 30.0,
    min_dp: int = 10,
    min_gq: int = 20,
    min_call_rate: float = 0.90,
    sample_ids: str = "",
    output_filtered: bool = False,
    region: str = "",
    max_variants_per_sample: int = 200000,
    *,
    ctx=None,
) -> dict:
    from biobank_agent.skills.vcf_query import _build_vcf_dm
    from biobank_agent.utils.vcf_genotypes import get_sample_vcf_paths

    vcf_dm = _build_vcf_dm(ctx)
    vcf_files = vcf_dm.list_vcf_files()
    if not vcf_files:
        return {"error": "No VCF files found."}

    path_map = {f["filename"]: f["vcf_path"] for f in vcf_files}

    if sample_ids:
        targets = [s.strip() for s in sample_ids.split(",") if s.strip()]
    else:
        targets = list(path_map.keys())

    pheno_df = None
    try:
        pheno_df = ctx.dm.query("SELECT * FROM biomarkers")
    except Exception:
        pass

    qc_results = []
    for sid in targets:
        if sid not in path_map:
            logger.warning("Sample %s not found in VCF files", sid)
            continue
        try:
            qc = _compute_per_sample_qc(
                path_map[sid], sid,
                region=region or None,
                max_variants=max_variants_per_sample or 0,
            )
            qc_results.append(qc)
        except Exception as e:
            logger.warning("QC failed for %s: %s", sid, e)

    if not qc_results:
        return {"error": "No QC results could be computed."}

    df = pd.DataFrame(qc_results)

    outlier_flags = {}
    metrics_for_outlier = ["call_rate", "het_ratio", "ti_tv_ratio", "mean_dp", "mean_gq"]
    for metric in metrics_for_outlier:
        if metric not in df.columns or df[metric].isnull().all():
            continue
        vals = df[metric].dropna()
        if len(vals) < 3:
            continue
        mean, std = vals.mean(), vals.std()
        if std < 1e-10:
            continue
        for idx, row in df.iterrows():
            v = row[metric]
            if pd.notna(v) and abs(v - mean) > 3 * std:
                sid = row["sample_id"]
                if sid not in outlier_flags:
                    outlier_flags[sid] = []
                direction = "high" if v > mean else "low"
                outlier_flags[sid].append(f"{metric}_{direction}")

    # Adaptive call rate threshold: use user threshold or cohort-based
    cohort_call_rates = df["call_rate"].dropna()
    adaptive_min_call_rate = min_call_rate
    if len(cohort_call_rates) > 3:
        cohort_mean_cr = cohort_call_rates.mean()
        cohort_std_cr = cohort_call_rates.std()
        if cohort_mean_cr < min_call_rate:
            adaptive_min_call_rate = max(0.5, cohort_mean_cr - 3 * cohort_std_cr)

    for qc in qc_results:
        sid = qc["sample_id"]
        flags = outlier_flags.get(sid, [])
        qc_pass = (
            qc["call_rate"] >= adaptive_min_call_rate
            and len(flags) == 0
        )
        qc["qc_pass"] = qc_pass
        qc["flags"] = flags

    n_pass = sum(1 for q in qc_results if q["qc_pass"])
    n_fail = len(qc_results) - n_pass

    hwe_fail_count = None
    hwe_n_tested = 0
    if n_pass >= 3 and region:
        try:
            from biobank_agent.utils.bcftools import merge_vcfs, get_tmp_dir
            import cyvcf2

            pass_ids = [q["sample_id"] for q in qc_results if q["qc_pass"]]
            pass_vcfs = [path_map[sid] for sid in pass_ids if sid in path_map]
            if len(pass_vcfs) >= 3:
                tmp_dir = get_tmp_dir(ctx)
                merged_hwe = tmp_dir / "merged_hwe.vcf.gz"
                merge_vcfs(pass_vcfs, merged_hwe, region=region or None)

                n_hwe_fail = 0
                vcf = cyvcf2.VCF(str(merged_hwe))
                for v in vcf(region):
                    if not v.ALT or v.ALT[0] == ".":
                        continue
                    gt = v.gt_types
                    n_aa = int((gt == 0).sum())
                    n_ab = int((gt == 1).sum())
                    n_bb = int((gt == 2).sum())
                    n_total = n_aa + n_ab + n_bb
                    if n_total < 3:
                        continue
                    hwe_n_tested += 1
                    p = (2 * n_aa + n_ab) / (2 * n_total)
                    q_val = 1 - p
                    exp_aa = p * p * n_total
                    exp_ab = 2 * p * q_val * n_total
                    exp_bb = q_val * q_val * n_total
                    chi2 = 0
                    for obs, exp in [(n_aa, exp_aa), (n_ab, exp_ab), (n_bb, exp_bb)]:
                        if exp > 0:
                            chi2 += (obs - exp) ** 2 / exp
                    if chi2 > 10.83:
                        n_hwe_fail += 1
                    if hwe_n_tested >= 50000:
                        break
                vcf.close()
                hwe_fail_count = n_hwe_fail
        except Exception as e:
            logger.warning("HWE test skipped: %s", e)

    figures = []
    report_dir = wgs_results_dir(ctx, "qc")

    if pheno_df is not None and "phenotype_group" in pheno_df.columns:
        id_col = getattr(ctx.dm, "subject_id_col", "sample_id")
        df = df.merge(pheno_df[[id_col, "phenotype_group"]],
                       left_on="sample_id", right_on=id_col, how="left")

    # Figure 1: call_rate vs het_ratio scatter
    fig, ax = nature_figure(width="single", height_ratio=0.9)
    if "phenotype_group" in df.columns:
        for i, grp in enumerate(sorted(df["phenotype_group"].dropna().unique())):
            mask = df["phenotype_group"] == grp
            ax.scatter(df.loc[mask, "call_rate"], df.loc[mask, "het_ratio"],
                       c=PALETTE[i % len(PALETTE)], label=str(grp), s=30, alpha=0.8, edgecolors="none")
        ax.legend(fontsize=6, frameon=False)
    else:
        ax.scatter(df["call_rate"], df["het_ratio"], c=PALETTE[0], s=30, alpha=0.8, edgecolors="none")

    for sid in outlier_flags:
        row = df[df["sample_id"] == sid]
        if not row.empty:
            ax.annotate(sid, (row["call_rate"].values[0], row["het_ratio"].values[0]),
                        fontsize=4, color="red", alpha=0.7)

    ax.axvline(x=adaptive_min_call_rate, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Call rate")
    ax.set_ylabel("Het / (Het + HomAlt) ratio")
    ax.set_title("Sample QC: Call Rate vs Heterozygosity")
    fig.tight_layout()
    paths = save_figure(fig, "vcf_qc_scatter", report_dir)
    figures.extend(paths)

    # Figure 2: Ti/Tv boxplot by phenotype group
    if "phenotype_group" in df.columns and df["ti_tv_ratio"].notna().sum() > 0:
        fig2, ax2 = nature_figure(width="single", height_ratio=0.7)
        groups = sorted(df["phenotype_group"].dropna().unique())
        data = [df.loc[df["phenotype_group"] == g, "ti_tv_ratio"].dropna().values for g in groups]
        if any(len(d) > 0 for d in data):
            bp = ax2.boxplot(data, patch_artist=True, widths=0.6)
            for patch, color in zip(bp["boxes"], PALETTE[:len(groups)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax2.set_xticklabels(groups, fontsize=6)
            ax2.set_ylabel("Ti/Tv ratio")
            ax2.set_title("Ti/Tv Ratio by Phenotype Group")
            fig2.tight_layout()
            paths2 = save_figure(fig2, "vcf_qc_titv", report_dir)
            figures.extend(paths2)

    if hasattr(ctx, "state") and hasattr(ctx.state, "figures"):
        ctx.state.figures.extend(figures)

    if hasattr(ctx, "state") and hasattr(ctx.state, "custom_data"):
        ctx.state.custom_data["vcf_qc_results"] = {
            "per_sample": qc_results,
            "pass_samples": [q["sample_id"] for q in qc_results if q["qc_pass"]],
            "fail_samples": [q["sample_id"] for q in qc_results if not q["qc_pass"]],
            "hwe_fail_count": hwe_fail_count,
        }

    cohort_stats = {}
    for metric in metrics_for_outlier:
        vals = df[metric].dropna()
        if len(vals) > 0:
            cohort_stats[metric] = {
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
                "min": round(float(vals.min()), 4),
                "max": round(float(vals.max()), 4),
            }

    filtered_vcf_dir = None
    if output_filtered:
        from biobank_agent.utils.bcftools import filter_vcf as bcf_filter, get_tmp_dir
        filt_dir = get_tmp_dir(ctx) / "qc_filtered"
        filt_dir.mkdir(parents=True, exist_ok=True)
        expr = f"QUAL>={min_qual} && FMT/DP>={min_dp} && FMT/GQ>={min_gq}"
        for qc in qc_results:
            if qc["qc_pass"] and qc["sample_id"] in path_map:
                try:
                    bcf_filter(
                        path_map[qc["sample_id"]],
                        filt_dir / f"{qc['sample_id']}.qc.vcf.gz",
                        include_expr=expr,
                        region=region or None,
                    )
                except Exception as e:
                    logger.warning("Filter failed for %s: %s", qc["sample_id"], e)
        filtered_vcf_dir = str(filt_dir)

    return {
        "n_samples_analyzed": len(qc_results),
        "n_samples_pass": n_pass,
        "n_samples_fail": n_fail,
        "qc_thresholds": {
            "min_qual": min_qual,
            "min_dp": min_dp,
            "min_gq": min_gq,
            "min_call_rate": min_call_rate,
        },
        "cohort_stats": cohort_stats,
        "per_sample_qc": qc_results[:10],
        "outlier_samples": list(outlier_flags.keys()),
        "outlier_details": outlier_flags,
        "figures": [str(p) for p in figures],
        "result_dir": str(report_dir),
        "filtered_vcf_dir": filtered_vcf_dir,
        "filtered_vcf_region": region or "full genome",
        "region": region or "full genome",
        "hwe_n_tested": hwe_n_tested,
        "hwe_fail_count": hwe_fail_count,
        "variant_only_vcf_note": (
            "Single-sample VCFs contain only variant sites (no hom-ref calls). "
            "het_ratio = n_het/(n_het+n_hom_ref+n_hom_alt) is ~1.0 in variant-only VCFs. "
            "For cohort-level heterozygosity, use merged multi-sample VCFs."
        ) if any(q.get("variant_only_vcf") for q in qc_results) else None,
    }
