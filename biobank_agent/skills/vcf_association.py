"""Case-control association testing for WGS variants."""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from biobank_agent.registry import skill
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE, SEMANTIC_PALETTE
from biobank_agent.utils.stats import fisher_exact, benjamini_hochberg, bonferroni
from biobank_agent.utils.wgs import (
    external_cache_dir,
    external_tool_log_path,
    file_signature,
    find_executable,
    link_or_copy,
    run_external,
    stable_hash,
    tool_version,
    wgs_environment_status,
    wgs_results_dir,
)

logger = logging.getLogger(__name__)

_CHROM_ORDER = {f"chr{i}": i for i in range(1, 23)}
_CHROM_ORDER.update({"chrX": 23, "chrY": 24, "chrM": 25})

_CHROM_LENGTHS_HG38 = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
    "chr22": 50818468,
}


def _workflow_mode_from_ctx(ctx) -> str:
    try:
        custom = getattr(getattr(ctx, "state", None), "custom_data", {}) or {}
        context = custom.get("plan_context") or {}
        mode = str(context.get("workflow_mode") or custom.get("workflow_mode") or "").strip().lower()
        if mode:
            return mode
    except Exception:
        pass
    return str(os.getenv("BIOBANK_WGS_WORKFLOW_MODE", "") or os.getenv("WGS_WORKFLOW_MODE", "")).strip().lower()


def _exploratory_fallback_allowed(ctx, explicit: bool = False) -> bool:
    if explicit:
        return True
    mode = _workflow_mode_from_ctx(ctx)
    if mode in {"exploratory", "degraded", "fallback"}:
        return True
    value = str(os.getenv("BIOBANK_ALLOW_EXPLORATORY_FALLBACK", "") or "").strip().lower()
    return value in {"1", "true", "yes", "y"}


def _pause_for_standard_gwas_downgrade(
    *,
    ctx,
    reason: str,
    plink_runs: list[dict],
    allow_exploratory_fallback: bool,
) -> dict | None:
    if _exploratory_fallback_allowed(ctx, explicit=allow_exploratory_fallback):
        return None
    try:
        from biobank_agent.skills.pause_and_ask import pause_and_ask
    except Exception:
        pause_and_ask = None
    question = (
        "Standard PLINK2 GWAS could not be completed for this VCF/phenotype input. "
        "Should I stop so you can fix the input/tool setup, or continue with exploratory Fisher exact results?"
    )
    options = "fix input/tool setup,continue exploratory,provide corrected phenotype/VCF path"
    if pause_and_ask is not None:
        payload = pause_and_ask(
            question=question,
            reason=reason,
            category="tool_mismatch",
            options=options,
            ctx=ctx,
        )
    else:
        payload = {
            "status": "ok",
            "awaiting_user": True,
            "question": question,
            "reason": reason,
            "category": "tool_mismatch",
            "options": [o.strip() for o in options.split(",")],
        }
    payload.update({
        "standard_gwas_downgrade": True,
        "analysis_mode": "standard_gwas_blocked",
        "plink2_runs": plink_runs,
    })
    return payload


def _genomic_inflation(p_values: np.ndarray) -> float:
    p_clean = p_values[(p_values > 0) & (p_values < 1)]
    if len(p_clean) < 10:
        return float("nan")
    chi2_obs = sp_stats.chi2.ppf(1 - p_clean, df=1)
    return float(np.median(chi2_obs) / 0.4549364)


def _safe_prefix(text: str) -> str:
    return (
        str(text)
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def _pc_columns(df: pd.DataFrame, n_pcs: int) -> list[str]:
    """Names of the first ``n_pcs`` principal-component columns (PC1, PC2, …)."""
    pcs = [c for c in df.columns if str(c).upper().startswith("PC") and str(c)[2:].isdigit()]
    pcs.sort(key=lambda c: int(str(c)[2:]))
    return pcs[: max(0, int(n_pcs))]


def _pcs_from_ctx(ctx) -> "pd.DataFrame | None":
    """Per-sample principal components from a prior vcf_pca step (population-structure
    covariates), if present in session state (``pca_result.pcs``)."""
    try:
        pca = ctx.state.custom_data.get("pca_result") if hasattr(ctx, "state") else None
        pcs = (pca or {}).get("pcs")
        if pcs is not None and hasattr(pcs, "empty") and not pcs.empty:
            return pcs
    except Exception:
        pass
    return None


def _write_plink2_inputs(
    pheno_df: pd.DataFrame,
    case_samples: list[str],
    ctrl_samples: list[str],
    id_col: str,
    out_dir: Path,
    pcs_df: pd.DataFrame | None = None,
    n_pcs: int = 10,
) -> tuple[Path, Path | None]:
    """Write PLINK2 phenotype/covariate files for a binary case-control GWAS.

    Population-structure correction: when ``pcs_df`` (from ``vcf_pca``) is supplied,
    the top ``n_pcs`` principal components are merged in as covariates alongside
    age/sex."""
    out_dir.mkdir(parents=True, exist_ok=True)
    case_set = set(case_samples)
    ctrl_set = set(ctrl_samples)
    pheno_rows = [
        {"#FID": sid, "IID": sid, "PHENO": 2 if sid in case_set else 1}
        for sid in case_samples + ctrl_samples
    ]
    pheno_path = out_dir / "plink2_case_control.pheno.tsv"
    pd.DataFrame(pheno_rows).to_csv(pheno_path, sep="\t", index=False)

    covar_cols: list[str] = []
    covar_df = pheno_df.copy()
    if id_col not in covar_df.columns:
        return pheno_path, None
    if "age" in covar_df.columns:
        covar_df["age"] = pd.to_numeric(
            covar_df["age"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )
        covar_cols.append("age")
    if "is_male" in covar_df.columns:
        covar_df["is_male"] = pd.to_numeric(covar_df["is_male"], errors="coerce")
        covar_cols.append("is_male")
    elif "sex" in covar_df.columns:
        sex = covar_df["sex"].astype(str).str.upper().str[0]
        covar_df["is_male"] = (sex == "M").astype(int)
        covar_cols.append("is_male")

    # Merge principal components as covariates for population-structure correction.
    if pcs_df is not None and not pcs_df.empty:
        pc_id = id_col if id_col in pcs_df.columns else (
            "sample_id" if "sample_id" in pcs_df.columns else None)
        pc_cols = _pc_columns(pcs_df, n_pcs)
        if pc_id is not None and pc_cols:
            pcs_small = pcs_df[[pc_id] + pc_cols].rename(columns={pc_id: id_col})
            covar_df = covar_df.merge(pcs_small, on=id_col, how="left")
            covar_cols.extend(pc_cols)

    if not covar_cols:
        return pheno_path, None

    records: list[dict[str, Any]] = []
    for _, row in covar_df[covar_df[id_col].isin(case_set | ctrl_set)].iterrows():
        sid = str(row[id_col])
        rec: dict[str, Any] = {"#FID": sid, "IID": sid}
        missing = False
        for col in covar_cols:
            value = row.get(col)
            if pd.isna(value):
                missing = True
                break
            rec[col] = value
        if not missing:
            records.append(rec)
    if len(records) < 3:
        return pheno_path, None

    covar_path = out_dir / "plink2_case_control.covar.tsv"
    pd.DataFrame(records).to_csv(covar_path, sep="\t", index=False)
    return pheno_path, covar_path


def _parse_plink2_glm(path: Path) -> list[dict]:
    """Parse PLINK2 --glm output into the agent's association record schema."""
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception:
        return []
    if df.empty or "P" not in df.columns:
        return []
    if "TEST" in df.columns:
        df = df[df["TEST"].astype(str).str.upper() == "ADD"].copy()

    results = []
    for _, row in df.iterrows():
        pval = pd.to_numeric(row.get("P"), errors="coerce")
        pos = pd.to_numeric(row.get("POS"), errors="coerce")
        if pd.isna(pval) or pd.isna(pos):
            continue
        chrom = str(row.get("#CHROM", row.get("CHROM", "")))
        if chrom and not chrom.startswith("chr") and chrom not in {"X", "Y", "MT", "M"}:
            chrom = f"chr{chrom}"
        effect = pd.to_numeric(row.get("OR", row.get("BETA", np.nan)), errors="coerce")
        freq = pd.to_numeric(row.get("A1_FREQ", np.nan), errors="coerce")
        results.append({
            "chrom": chrom,
            "pos": int(pos),
            "ref": str(row.get("REF", "")),
            "alt": str(row.get("ALT", row.get("A1", ""))),
            "rsid": str(row.get("ID", ".")),
            "p_value": float(pval),
            "odds_ratio": float(effect) if not pd.isna(effect) else float("nan"),
            "maf": float(freq) if not pd.isna(freq) else float("nan"),
            "source": "plink2_glm",
        })
    return results


def _run_plink2_association(
    merged_vcf: Path,
    pheno_df: pd.DataFrame,
    case_samples: list[str],
    ctrl_samples: list[str],
    id_col: str,
    report_dir: Path,
    label: str,
    pcs_df: pd.DataFrame | None = None,
    n_pcs: int = 10,
    ctx=None,
) -> dict:
    """Run PLINK2 logistic/Firth association on one merged VCF."""
    plink2 = find_executable("plink2")
    if not plink2:
        return {"ok": False, "reason": "plink2 executable not found"}

    pheno_path, covar_path = _write_plink2_inputs(
        pheno_df, case_samples, ctrl_samples, id_col, report_dir,
        pcs_df=pcs_df, n_pcs=n_pcs,
    )
    out_prefix = report_dir / f"plink2_assoc_{_safe_prefix(label)}"
    cmd = [
        plink2,
        "--vcf", str(merged_vcf),
        "--double-id",
        "--allow-extra-chr",
        "--set-missing-var-ids", "@:#:$r:$a",
        "--new-id-max-allele-len", "50", "missing",
        "--pheno", str(pheno_path),
        "--pheno-name", "PHENO",
        "--glm", "hide-covar", "omit-ref", "firth-fallback",
        "--out", str(out_prefix),
    ]
    if covar_path is not None:
        covar_names = [
            c for c in pd.read_csv(covar_path, sep="\t", nrows=0).columns
            if c not in {"#FID", "FID", "IID"}
        ]
        if covar_names:
            cmd.extend(["--covar", str(covar_path), "--covar-name", ",".join(covar_names)])

    cache_dir = external_cache_dir()
    if cache_dir is not None:
        key = stable_hash({
            "version": 1,
            "tool": "plink2_glm",
            "analysis": {
                "label": label,
                "case_samples": case_samples,
                "ctrl_samples": ctrl_samples,
                "id_col": id_col,
                "covar_columns": (
                    [
                        c for c in pd.read_csv(covar_path, sep="\t", nrows=0).columns
                        if c not in {"#FID", "FID", "IID"}
                    ]
                    if covar_path is not None
                    else []
                ),
            },
            "merged_vcf": file_signature(merged_vcf, include_path=False),
            "pheno_rows": (
                pd.read_csv(pheno_path, sep="\t").to_dict(orient="records")
                if pheno_path.exists()
                else []
            ),
            "covar_rows": (
                pd.read_csv(covar_path, sep="\t").to_dict(orient="records")
                if covar_path is not None and covar_path.exists()
                else []
            ),
            "plink2": plink2,
        })
        cache_prefix = cache_dir / f"plink2_assoc_{key}"
        cached_outputs = sorted(cache_dir.glob(f"{cache_prefix.name}*"))
        cached_glm = [p for p in cached_outputs if ".glm." in p.name]
        cached_records_path = cache_prefix.with_suffix(".records.json")
        if cached_records_path.exists():
            try:
                records = json.loads(cached_records_path.read_text(encoding="utf-8"))
            except Exception:
                records = []
            outputs = []
            for cached in cached_outputs:
                dest = report_dir / cached.name.replace(cache_prefix.name, out_prefix.name, 1)
                link_or_copy(cached, dest)
                outputs.append(str(dest))
            return {
                "ok": bool(records),
                "reason": "" if records else "cached PLINK2 records were empty",
                "run": {
                    "cmd": cmd,
                    "returncode": 0,
                    "stdout": "<cached_records>",
                    "stderr": "",
                    "ok": True,
                    "cache_hit": True,
                },
                "outputs": outputs,
                "records": records,
                "plink2_version": tool_version(plink2, "--version"),
                "pheno_file": str(pheno_path),
                "covar_file": str(covar_path) if covar_path else None,
                "cache_hit": True,
            }
        if cached_glm:
            outputs = []
            for cached in cached_outputs:
                dest = report_dir / cached.name.replace(cache_prefix.name, out_prefix.name, 1)
                link_or_copy(cached, dest)
                outputs.append(str(dest))
            records = []
            for glm in sorted(report_dir.glob(f"{out_prefix.name}*.glm.*")):
                records.extend(_parse_plink2_glm(glm))
            if records:
                tmp = cached_records_path.with_suffix(f".records.{os.getpid()}.tmp")
                try:
                    tmp.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
                    os.replace(tmp, cached_records_path)
                finally:
                    tmp.unlink(missing_ok=True)
            return {
                "ok": bool(records),
                "reason": "" if records else "cached PLINK2 outputs contained no parseable ADD test rows",
                "run": {
                    "cmd": cmd,
                    "returncode": 0,
                    "stdout": "<cached>",
                    "stderr": "",
                    "ok": True,
                    "cache_hit": True,
                },
                "outputs": outputs,
                "records": records,
                "plink2_version": tool_version(plink2, "--version"),
                "pheno_file": str(pheno_path),
                "covar_file": str(covar_path) if covar_path else None,
                "cache_hit": True,
            }
    else:
        cache_prefix = None

    log_path = external_tool_log_path(ctx, report_dir, f"plink2_assoc_{_safe_prefix(label)}") if ctx is not None else None
    run = run_external(
        cmd,
        timeout=1800,
        settings=getattr(ctx, "settings", None) if ctx is not None else None,
        line_sink=getattr(ctx, "emit_line", None) if ctx is not None else None,
        log_path=log_path,
    )
    outputs = sorted(str(p) for p in report_dir.glob(f"{out_prefix.name}*"))
    if not run["ok"]:
        return {
            "ok": False,
            "reason": "plink2 --glm failed",
            "run": run,
            "outputs": outputs,
            "plink2_version": tool_version(plink2, "--version"),
        }

    records = []
    for glm in sorted(report_dir.glob(f"{out_prefix.name}*.glm.*")):
        records.extend(_parse_plink2_glm(glm))
    if cache_prefix is not None and records:
        for out in report_dir.glob(f"{out_prefix.name}*"):
            dest = cache_dir / out.name.replace(out_prefix.name, cache_prefix.name, 1)
            link_or_copy(out, dest)
        tmp = cache_prefix.with_suffix(f".records.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cache_prefix.with_suffix(".records.json"))
        finally:
            tmp.unlink(missing_ok=True)
    return {
        "ok": bool(records),
        "reason": "" if records else "plink2 completed but no parseable ADD test rows were produced",
        "run": run,
        "outputs": outputs,
        "records": records,
        "plink2_version": tool_version(plink2, "--version"),
        "pheno_file": str(pheno_path),
        "covar_file": str(covar_path) if covar_path else None,
    }


def _association_cache_key(
    all_results: list[dict],
    case_group: str,
    control_group: str,
    case_samples: list[str],
    ctrl_samples: list[str],
    analysis_mode: str,
) -> str:
    compact = [
        (
            r.get("chrom"),
            int(r.get("pos", 0) or 0),
            str(r.get("ref", "")),
            str(r.get("alt", "")),
            float(r.get("p_value", 1.0) or 1.0),
            float(r.get("odds_ratio", 0.0) or 0.0) if str(r.get("odds_ratio", "")).lower() != "nan" else "nan",
        )
        for r in all_results
    ]
    return stable_hash({
        "version": 1,
        "operation": "vcf_association_postprocess",
        "case_group": case_group,
        "control_group": control_group,
        "case_samples": case_samples,
        "ctrl_samples": ctrl_samples,
        "analysis_mode": analysis_mode,
        "records": compact,
    })


def _association_return(
    *,
    case_group: str,
    control_group: str,
    case_samples: list[str],
    ctrl_samples: list[str],
    all_results: list[dict],
    top_hits: list[dict],
    n_sig_bonf: int,
    n_sig_fdr: int,
    lambda_gc: float,
    lambda_gc_warning: str | None,
    analysis_mode: str,
    standard_gwas_available: bool,
    standard_result_note: str | None,
    all_table: Path,
    top_table: Path,
    plink_runs: list[dict],
    skipped_regions: list[dict],
    figures: list[str | Path],
    report_dir: Path,
) -> dict:
    return {
        "case_group": case_group,
        "control_group": control_group,
        "n_cases": len(case_samples),
        "n_controls": len(ctrl_samples),
        "n_variants_tested": len(all_results),
        "n_significant_bonferroni": n_sig_bonf,
        "n_significant_fdr": n_sig_fdr,
        "lambda_gc": round(lambda_gc, 3) if not np.isnan(lambda_gc) else None,
        "lambda_gc_warning": lambda_gc_warning,
        "analysis_mode": analysis_mode,
        "standard_gwas_available": standard_gwas_available,
        "standard_gwas_note": standard_result_note,
        "test_method": "PLINK2 --glm logistic/Firth fallback" if analysis_mode.startswith("plink2") else "Fisher's exact (2x2 allele table)",
        "top_hits": [{k: v for k, v in h.items()
                      if k in ("chrom", "pos", "ref", "alt", "rsid", "p_value", "p_fdr", "odds_ratio", "maf")}
                     for h in top_hits],
        "result_tables": [str(all_table), str(top_table)],
        "plink2_runs": plink_runs,
        "skipped_regions": skipped_regions,
        "figures": [str(p) for p in figures],
        "result_dir": str(report_dir),
        "power_warning": (
            f"With n_cases={len(case_samples)} and n_controls={len(ctrl_samples)}, "
            f"statistical power is very limited. Results should be interpreted as "
            f"exploratory candidate-level associations, not genome-wide significant findings."
        ),
    }


@skill(
    name="vcf_association",
    description=(
        "Case-control association test for biallelic SNPs between two phenotype "
        "groups. Uses Fisher's exact test (appropriate for small samples). "
        "Applies Bonferroni and FDR correction. Generates Manhattan plot, QQ plot, "
        "and reports genomic inflation factor (lambda GC)."
    ),
    parameters={
        "case_group": {
            "type": "string",
            "description": "Phenotype group code for cases (default 'J' = Juvenile)",
            "default": "J",
        },
        "control_group": {
            "type": "string",
            "description": "Phenotype group code for controls (default 'V' = Vitiligo)",
            "default": "V",
        },
        "maf_min": {
            "type": "number",
            "description": "Minimum cohort MAF to include (default 0.01)",
            "default": 0.01,
        },
        "region": {
            "type": "string",
            "description": "Restrict to a region (e.g. chr6:28000000-34000000 for HLA)",
            "default": "",
        },
        "chromosomes": {
            "type": "string",
            "description": "Chromosomes to test (default chr1-22)",
            "default": "chr1-22",
        },
        "max_variants": {
            "type": "integer",
            "description": "Max variants to test per chromosome (default 0 = unlimited)",
            "default": 0,
        },
        "allow_exploratory_fallback": {
            "type": "boolean",
            "description": "If true, continue with built-in exploratory Fisher tests when standard PLINK2 GWAS cannot run.",
            "default": False,
        },
    },
    required=[],
)
def vcf_association(
    case_group: str = "J",
    control_group: str = "V",
    maf_min: float = 0.01,
    region: str = "",
    chromosomes: str = "chr1-22",
    max_variants: int = 0,
    allow_exploratory_fallback: bool = False,
    *,
    ctx=None,
) -> dict:
    from biobank_agent.utils.bcftools import merge_vcfs, get_tmp_dir
    from biobank_agent.utils.vcf_genotypes import build_allele_counts, get_sample_vcf_paths
    from biobank_agent.skills.vcf_pca import _expand_chromosomes

    pheno_df = ctx.dm.query("SELECT * FROM biomarkers")
    if pheno_df.empty:
        return {"error": "No phenotype data."}

    id_col = getattr(ctx.dm, "subject_id_col", "sample_id")
    if "phenotype_group" not in pheno_df.columns:
        return {"error": "No phenotype_group column in biomarkers."}

    case_samples = pheno_df.loc[pheno_df["phenotype_group"] == case_group, id_col].tolist()
    ctrl_samples = pheno_df.loc[pheno_df["phenotype_group"] == control_group, id_col].tolist()

    if len(case_samples) < 2:
        return {"error": f"Case group '{case_group}' has <2 samples ({len(case_samples)})."}
    if len(ctrl_samples) < 2:
        return {"error": f"Control group '{control_group}' has <2 samples ({len(ctrl_samples)})."}

    sample_paths = get_sample_vcf_paths(ctx)

    qc_data = None
    if hasattr(ctx, "state") and hasattr(ctx.state, "custom_data"):
        qc_data = ctx.state.custom_data.get("vcf_qc_results")
    if qc_data and qc_data.get("pass_samples"):
        pass_set = set(qc_data["pass_samples"])
        case_samples = [s for s in case_samples if s in pass_set]
        ctrl_samples = [s for s in ctrl_samples if s in pass_set]
        logger.info("After QC filter: %d cases, %d controls", len(case_samples), len(ctrl_samples))

    case_vcfs = [sample_paths[s] for s in case_samples if s in sample_paths]
    ctrl_vcfs = [sample_paths[s] for s in ctrl_samples if s in sample_paths]
    all_vcfs = case_vcfs + ctrl_vcfs

    if not all_vcfs:
        return {"error": "No VCF files found for case/control samples."}

    tmp_dir = get_tmp_dir(ctx)
    regions_to_process = [region] if region else _expand_chromosomes(chromosomes)

    all_results = []
    plink_results: list[dict] = []
    plink_runs: list[dict] = []
    skipped_regions: list[dict] = []
    report_dir = wgs_results_dir(ctx, "association")
    env_status = wgs_environment_status()
    can_try_plink2 = bool(
        env_status["executables"].get("plink2")
        and all(Path(p).exists() for p in all_vcfs)
    )
    standard_requested = _workflow_mode_from_ctx(ctx) == "standard"
    if standard_requested and not env_status["executables"].get("plink2"):
        paused = _pause_for_standard_gwas_downgrade(
            ctx=ctx,
            reason=(
                "workflow_mode=standard was requested, but PLINK2 is not available. "
                "Running the built-in Fisher exact fallback would be an exploratory downgrade."
            ),
            plink_runs=[],
            allow_exploratory_fallback=bool(allow_exploratory_fallback),
        )
        if paused is not None:
            return paused
    for rgn in regions_to_process:
        merged = tmp_dir / f"merged_assoc_{rgn.replace(':', '_')}.vcf.gz"
        try:
            merge_vcfs(all_vcfs, merged, region=rgn)
        except Exception as e:
            logger.warning("Merge failed for %s: %s", rgn, e)
            skipped_regions.append({"region": str(rgn), "stage": "merge", "reason": str(e)})
            continue

        case_ids = [s for s in case_samples if s in sample_paths]
        ctrl_ids = [s for s in ctrl_samples if s in sample_paths]

        if can_try_plink2:
            plink_run = _run_plink2_association(
                merged,
                pheno_df,
                case_ids,
                ctrl_ids,
                id_col,
                report_dir,
                rgn or "all",
                pcs_df=_pcs_from_ctx(ctx),
                n_pcs=10,
                ctx=ctx,
            )
            plink_runs.append({k: v for k, v in plink_run.items() if k != "records"})
            if plink_run.get("records"):
                plink_results.extend(plink_run["records"])
                continue

        counts = build_allele_counts(
            merged, case_ids, ctrl_ids,
            maf_min=maf_min, region=None,
            max_variants=max_variants,
        )
        if not counts:
            skipped_regions.append({
                "region": str(rgn),
                "stage": "variant_filter",
                "reason": "No variants remained after allele-count, MAF, and genotype filters.",
            })
            continue

        for rec in counts:
            table = np.array([[rec["case_alt"], rec["case_ref"]],
                              [rec["ctrl_alt"], rec["ctrl_ref"]]])
            try:
                result = fisher_exact(table)
                rec["p_value"] = result["p_value"]
                rec["odds_ratio"] = result["odds_ratio"]
            except Exception:
                rec["p_value"] = 1.0
                rec["odds_ratio"] = 1.0

        all_results.extend(counts)

    analysis_mode = "exploratory_fisher_exact"
    standard_result_note = None
    if plink_results:
        all_results = plink_results
        analysis_mode = "plink2_glm_logistic_firth_fallback"
    elif env_status["executables"].get("plink2"):
        if not can_try_plink2:
            standard_result_note = (
                "PLINK2 is installed but was not run because not all source VCF paths "
                "exist locally; exploratory Fisher exact results were used."
            )
        elif plink_runs:
            reasons = [r.get("reason", "") for r in plink_runs if r.get("reason")]
            standard_result_note = "; ".join(reasons[:3]) or "PLINK2 produced no parseable association rows."

    if standard_result_note:
        paused = _pause_for_standard_gwas_downgrade(
            ctx=ctx,
            reason=standard_result_note,
            plink_runs=plink_runs,
            allow_exploratory_fallback=bool(allow_exploratory_fallback),
        )
        if paused is not None:
            return paused

    if not all_results:
        return {"error": "No variants tested. Check regions and MAF threshold."}

    p_values = np.array([r["p_value"] for r in all_results])
    p_bonf = bonferroni(p_values)
    p_fdr = benjamini_hochberg(p_values)

    for i, rec in enumerate(all_results):
        rec["p_bonferroni"] = float(p_bonf[i])
        rec["p_fdr"] = float(p_fdr[i])

    lambda_gc = _genomic_inflation(p_values)

    n_sig_bonf = int((p_bonf < 0.05).sum())
    n_sig_fdr = int((p_fdr < 0.05).sum())

    all_results.sort(key=lambda x: x["p_value"])
    top_hits = all_results[:20]
    all_table = report_dir / "association_results.tsv"
    top_table = report_dir / "top_hits.tsv"
    figures: list[str | Path] = []

    lambda_gc_warning = None
    if not np.isnan(lambda_gc):
        if lambda_gc > 1.2:
            lambda_gc_warning = (
                f"lambda_gc={lambda_gc:.2f} indicates substantial inflation. "
                "Consider PCA covariate correction or mixed-model approach. "
                "Results may contain systematic false positives."
            )
        elif lambda_gc > 1.1:
            lambda_gc_warning = (
                f"lambda_gc={lambda_gc:.2f} shows mild inflation. "
                "PCA covariate correction recommended before interpreting top hits."
            )

    assoc_cache_dir = external_cache_dir()
    assoc_cache_prefix: Path | None = None
    if assoc_cache_dir is not None:
        assoc_cache_prefix = assoc_cache_dir / f"vcf_assoc_{_association_cache_key(all_results, case_group, control_group, case_samples, ctrl_samples, analysis_mode)}"
        cached_summary = assoc_cache_prefix.with_suffix(".summary.json")
        cached_files = {
            assoc_cache_prefix.with_suffix(".association_results.tsv"): all_table,
            assoc_cache_prefix.with_suffix(".top_hits.tsv"): top_table,
            assoc_cache_prefix.with_suffix(".manhattan.pdf"): report_dir / "vcf_association_manhattan.pdf",
            assoc_cache_prefix.with_suffix(".manhattan.svg"): report_dir / "vcf_association_manhattan.svg",
            assoc_cache_prefix.with_suffix(".qq.pdf"): report_dir / "vcf_association_qq.pdf",
            assoc_cache_prefix.with_suffix(".qq.svg"): report_dir / "vcf_association_qq.svg",
        }
        if cached_summary.exists() and all(src.exists() for src in cached_files):
            for src, dest in cached_files.items():
                link_or_copy(src, dest)
            figures = [
                cached_files[assoc_cache_prefix.with_suffix(".manhattan.pdf")],
                cached_files[assoc_cache_prefix.with_suffix(".manhattan.svg")],
                cached_files[assoc_cache_prefix.with_suffix(".qq.pdf")],
                cached_files[assoc_cache_prefix.with_suffix(".qq.svg")],
            ]
            if hasattr(ctx, "state") and hasattr(ctx.state, "figures"):
                ctx.state.figures.extend(figures)
            if hasattr(ctx, "state") and hasattr(ctx.state, "custom_data"):
                ctx.state.custom_data["gwas_results"] = {
                    "case_group": case_group,
                    "control_group": control_group,
                    "all_results": top_hits,
                    "n_tested": len(all_results),
                    "lambda_gc": lambda_gc,
                    "analysis_mode": analysis_mode,
                    "plink2_runs": plink_runs,
                    "skipped_regions": skipped_regions,
                }
            return _association_return(
                case_group=case_group,
                control_group=control_group,
                case_samples=case_samples,
                ctrl_samples=ctrl_samples,
                all_results=all_results,
                top_hits=top_hits,
                n_sig_bonf=n_sig_bonf,
                n_sig_fdr=n_sig_fdr,
                lambda_gc=lambda_gc,
                lambda_gc_warning=lambda_gc_warning,
                analysis_mode=analysis_mode,
                standard_gwas_available=env_status["modules"]["standard_gwas"] == "READY",
                standard_result_note=standard_result_note,
                all_table=all_table,
                top_table=top_table,
                plink_runs=plink_runs,
                skipped_regions=skipped_regions,
                figures=figures,
                report_dir=report_dir,
            )

    pd.DataFrame(all_results).to_csv(all_table, sep="\t", index=False)
    pd.DataFrame(top_hits).to_csv(top_table, sep="\t", index=False)

    # Manhattan plot with proportional chromosome positions
    fig, ax = nature_figure(width="double", height_ratio=0.4)
    chroms_present = sorted(
        {rec["chrom"] for rec in all_results},
        key=lambda c: _CHROM_ORDER.get(c, 99),
    )
    chrom_offsets = {}
    cumulative = 0
    for c in chroms_present:
        chrom_offsets[c] = cumulative
        cumulative += _CHROM_LENGTHS_HG38.get(c, 1e8)

    x_positions = []
    y_values = []
    colors = []
    chrom_midpoints = {}

    for rec in all_results:
        c = rec["chrom"]
        x = chrom_offsets[c] + rec["pos"]
        x_positions.append(x)
        pval = max(rec["p_value"], 1e-300)
        y_values.append(-np.log10(pval))
        cidx = _CHROM_ORDER.get(c, 0) % 2
        colors.append(PALETTE[0] if cidx == 0 else PALETTE[4])

    for c in chroms_present:
        chrom_midpoints[c] = chrom_offsets[c] + _CHROM_LENGTHS_HG38.get(c, 1e8) / 2

    ax.scatter(x_positions, y_values, c=colors, s=3, alpha=0.5, edgecolors="none", rasterized=True)

    bonf_threshold = -np.log10(0.05 / len(all_results)) if len(all_results) > 0 else 7.3
    suggestive = -np.log10(1e-5)
    ax.axhline(y=bonf_threshold, color="red", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.axhline(y=suggestive, color="blue", linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_ylabel(r"$-\log_{10}(P)$")
    ax.set_title(f"Manhattan Plot: {case_group} vs {control_group} "
                 f"(n={len(case_samples)}+{len(ctrl_samples)}, "
                 f"{len(all_results):,} variants)")
    ax.set_xlabel("Chromosome")
    if chrom_midpoints:
        tick_labels = [c.replace("chr", "") for c in chroms_present]
        ax.set_xticks([chrom_midpoints[c] for c in chroms_present])
        ax.set_xticklabels(tick_labels, fontsize=5)
    fig.tight_layout()
    paths = save_figure(fig, "vcf_association_manhattan", report_dir)
    figures.extend(paths)

    # QQ plot
    fig2, ax2 = nature_figure(width="single", height_ratio=0.9)
    n_tests = len(p_values)
    expected = -np.log10(np.arange(1, n_tests + 1) / (n_tests + 1))
    observed = -np.log10(np.sort(p_values)[::-1])
    observed = np.clip(observed, 0, 300)

    ax2.scatter(expected, observed, c=PALETTE[0], s=5, alpha=0.5, edgecolors="none", rasterized=True)
    max_val = max(expected.max(), observed.max()) * 1.05
    ax2.plot([0, max_val], [0, max_val], "k--", linewidth=0.5, alpha=0.5)
    ax2.set_xlabel(r"Expected $-\log_{10}(P)$")
    ax2.set_ylabel(r"Observed $-\log_{10}(P)$")
    lambda_str = f"{lambda_gc:.2f}" if not np.isnan(lambda_gc) else "N/A"
    ax2.set_title(f"QQ Plot ($\\lambda_{{GC}}$ = {lambda_str})")
    fig2.tight_layout()
    paths2 = save_figure(fig2, "vcf_association_qq", report_dir)
    figures.extend(paths2)

    if hasattr(ctx, "state") and hasattr(ctx.state, "figures"):
        ctx.state.figures.extend(figures)

    if hasattr(ctx, "state") and hasattr(ctx.state, "custom_data"):
        ctx.state.custom_data["gwas_results"] = {
            "case_group": case_group,
            "control_group": control_group,
            "all_results": all_results,
            "n_tested": len(all_results),
            "lambda_gc": lambda_gc,
            "analysis_mode": analysis_mode,
            "plink2_runs": plink_runs,
            "skipped_regions": skipped_regions,
        }

    if assoc_cache_prefix is not None:
        cache_files = {
            all_table: assoc_cache_prefix.with_suffix(".association_results.tsv"),
            top_table: assoc_cache_prefix.with_suffix(".top_hits.tsv"),
            report_dir / "vcf_association_manhattan.pdf": assoc_cache_prefix.with_suffix(".manhattan.pdf"),
            report_dir / "vcf_association_manhattan.svg": assoc_cache_prefix.with_suffix(".manhattan.svg"),
            report_dir / "vcf_association_qq.pdf": assoc_cache_prefix.with_suffix(".qq.pdf"),
            report_dir / "vcf_association_qq.svg": assoc_cache_prefix.with_suffix(".qq.svg"),
        }
        for src, dest in cache_files.items():
            if src.exists():
                link_or_copy(src, dest)
        summary_tmp = assoc_cache_prefix.with_suffix(f".summary.{os.getpid()}.tmp")
        try:
            summary_tmp.write_text(json.dumps({"n": len(all_results)}, ensure_ascii=False), encoding="utf-8")
            os.replace(summary_tmp, assoc_cache_prefix.with_suffix(".summary.json"))
        finally:
            summary_tmp.unlink(missing_ok=True)

    return _association_return(
        case_group=case_group,
        control_group=control_group,
        case_samples=case_samples,
        ctrl_samples=ctrl_samples,
        all_results=all_results,
        top_hits=top_hits,
        n_sig_bonf=n_sig_bonf,
        n_sig_fdr=n_sig_fdr,
        lambda_gc=lambda_gc,
        lambda_gc_warning=lambda_gc_warning,
        analysis_mode=analysis_mode,
        standard_gwas_available=env_status["modules"]["standard_gwas"] == "READY",
        standard_result_note=standard_result_note,
        all_table=all_table,
        top_table=top_table,
        plink_runs=plink_runs,
        skipped_regions=skipped_regions,
        figures=figures,
        report_dir=report_dir,
    )
