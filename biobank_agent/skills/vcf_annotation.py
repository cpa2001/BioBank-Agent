"""Variant annotation using built-in curated gene coordinates."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from biobank_agent.registry import skill
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE
from biobank_agent.utils.wgs import (
    external_cache_dir,
    external_tool_log_path,
    file_signature,
    find_executable,
    link_or_copy,
    run_external,
    snpeff_database_ready,
    stable_hash,
    tool_version,
    vep_cache_ready,
    wgs_environment_status,
    wgs_results_dir,
)

logger = logging.getLogger(__name__)

VITILIGO_GENES_HG38 = {
    "TYR":       ("chr11", 88911696,  88921223),
    "OCA2":      ("chr15", 27622010,  27819535),
    "MC1R":      ("chr16", 89919740,  89920776),
    "NLRP1":     ("chr17", 5484000,   5531000),
    "PAX3":      ("chr2",  217288628, 217374533),
    "MITF":      ("chr3",  69788846,  70009099),
    "SLC45A2":   ("chr5",  33951717,  33987793),
    "PTPN22":    ("chr1",  113813717, 113871753),
    "CTLA4":     ("chr2",  203867788, 203873965),
    "HLA":       ("chr6",  28477797,  33448354),
    "IRF4":      ("chr6",  396321,    411447),
    "IL2RA":     ("chr10", 6010466,   6062370),
    "TYRP1":     ("chr9",  12693384,  12710448),
    "DCT":       ("chr13", 94793016,  94856846),
    "SOX10":     ("chr22", 37972312,  37984561),
    "FOXD3":     ("chr1",  63323657,  63327071),
    "ASIP":      ("chr20", 34095920,  34096796),
    "IFNG":      ("chr12", 68154768,  68159740),
    "CXCL10":    ("chr4",  76013416,  76016388),
    "TNF":       ("chr6",  31575565,  31578336),
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


def _annotation_fallback_allowed(ctx, explicit: bool = False) -> bool:
    if explicit:
        return True
    if _workflow_mode_from_ctx(ctx) in {"exploratory", "degraded", "fallback"}:
        return True
    value = str(os.getenv("BIOBANK_ALLOW_EXPLORATORY_FALLBACK", "") or "").strip().lower()
    return value in {"1", "true", "yes", "y"}


def _pause_for_standard_annotation_downgrade(
    *,
    ctx,
    reason: str,
    standard_annotation_runs: list[dict],
    allow_exploratory_fallback: bool,
) -> dict | None:
    if _annotation_fallback_allowed(ctx, explicit=allow_exploratory_fallback):
        return None
    try:
        from biobank_agent.skills.pause_and_ask import pause_and_ask
    except Exception:
        pause_and_ask = None
    question = (
        "Standard VEP/SnpEff/ANNOVAR annotation could not be completed. "
        "Should I stop so you can fix the annotation setup, or continue with built-in hg38 gene-body coordinates?"
    )
    options = "fix annotation setup,continue exploratory,provide corrected VEP/SnpEff/ANNOVAR cache path"
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
        "standard_annotation_downgrade": True,
        "annotation_mode": "standard_annotation_blocked",
        "standard_annotation_runs": standard_annotation_runs,
    })
    return payload


def _query_gene_variants(vcf_path: str, gene: str, chrom: str,
                          start: int, end: int, min_qual: float = 0.0,
                          max_variants: int = 10000) -> list[dict]:
    import cyvcf2

    vcf = cyvcf2.VCF(vcf_path)
    region = f"{chrom}:{start}-{end}"
    variants = []

    try:
        for v in vcf(region):
            if max_variants and len(variants) >= max_variants:
                break
            if v.QUAL is not None and v.QUAL < min_qual:
                continue
            if not v.ALT or v.ALT[0] == ".":
                continue

            gt_types = v.gt_types
            samples_with_alt = []
            for i, gt in enumerate(gt_types):
                if gt in (1, 2):
                    samples_with_alt.append(vcf.samples[i])

            if not samples_with_alt:
                continue

            variants.append({
                "chrom": v.CHROM,
                "pos": v.POS,
                "ref": v.REF,
                "alt": ",".join(v.ALT),
                "rsid": v.ID or ".",
                "qual": round(v.QUAL, 1) if v.QUAL else None,
                "gene": gene,
                "n_carriers": len(samples_with_alt),
                "carrier_samples": samples_with_alt,
                "gt_types": [int(g) for g in gt_types],
            })
    except Exception as e:
        logger.warning("Failed querying %s (%s): %s", gene, region, e)

    vcf.close()
    return variants


def _parse_snpeff_stats(stats_path: Path) -> dict:
    """Parse the stable summary fields from SnpEff's HTML stats report."""
    if not stats_path.exists():
        return {}
    text = stats_path.read_text(encoding="utf-8", errors="ignore")
    summary: dict[str, int] = {}
    for label in ("Number of variants", "Number of effects"):
        marker = f">{label}<"
        idx = text.find(marker)
        if idx == -1:
            continue
        window = text[idx: idx + 500]
        import re
        match = re.search(r"<td[^>]*>\s*([0-9,]+)\s*</td>", window)
        if match:
            summary[label.lower().replace(" ", "_")] = int(match.group(1).replace(",", ""))
    return summary


def _parse_snpeff_annotations(vcf_path: Path, max_rows: int = 20000) -> tuple[list[dict], dict[str, int]]:
    """Extract ANN records from a SnpEff-annotated VCF into a compact table."""
    import cyvcf2

    rows: list[dict] = []
    effect_counts: dict[str, int] = {}
    gene_counts: dict[str, int] = {}
    vcf = cyvcf2.VCF(str(vcf_path))
    try:
        for variant in vcf:
            ann_raw = variant.INFO.get("ANN")
            if ann_raw is None:
                continue
            for ann in str(ann_raw).split(","):
                fields = ann.split("|")
                if len(fields) < 5:
                    continue
                effects = fields[1].split("&") if fields[1] else [""]
                gene = fields[3] or fields[4] or ""
                for effect in effects:
                    if effect:
                        effect_counts[effect] = effect_counts.get(effect, 0) + 1
                if gene:
                    gene_counts[gene] = gene_counts.get(gene, 0) + 1
                if len(rows) < max_rows:
                    rows.append({
                        "chrom": variant.CHROM,
                        "pos": variant.POS,
                        "id": variant.ID or ".",
                        "ref": variant.REF,
                        "alt": ",".join(variant.ALT or []),
                        "allele": fields[0] if len(fields) > 0 else "",
                        "effect": fields[1] if len(fields) > 1 else "",
                        "impact": fields[2] if len(fields) > 2 else "",
                        "gene": gene,
                        "feature_type": fields[5] if len(fields) > 5 else "",
                        "feature_id": fields[6] if len(fields) > 6 else "",
                        "hgvs_c": fields[9] if len(fields) > 9 else "",
                        "hgvs_p": fields[10] if len(fields) > 10 else "",
                    })
    finally:
        vcf.close()
    return rows, effect_counts


def _run_snpeff_annotation(
    merged_vcf: Path,
    report_dir: Path,
    genome: str = "hg38",
    label: str = "candidate_regions",
    ctx=None,
) -> dict:
    """Run SnpEff on a merged VCF when the genome database is installed."""
    snpeff = find_executable("snpEff")
    if not snpeff:
        return {"ok": False, "reason": "snpEff executable not found"}
    if not snpeff_database_ready(genome):
        return {"ok": False, "reason": f"SnpEff database '{genome}' is not installed locally"}

    safe_label = label.replace(":", "_").replace("/", "_").replace("\\", "_")
    annotated_vcf = report_dir / f"snpeff_annotated_{safe_label}.vcf"
    stats_html = report_dir / f"snpeff_summary_{safe_label}.html"
    cache_dir = external_cache_dir()
    cache_prefix: Path | None = None
    if cache_dir is not None:
        key = stable_hash({
            "version": 1,
            "tool": "snpeff",
            "genome": genome,
            "label": label,
            "merged_vcf": file_signature(merged_vcf, include_path=False),
            "snpeff": snpeff,
        })
        cache_prefix = cache_dir / f"snpeff_{key}"
        cached_vcf = cache_prefix.with_suffix(".vcf")
        cached_html = cache_prefix.with_suffix(".html")
        if cached_vcf.exists() and cached_html.exists():
            link_or_copy(cached_vcf, annotated_vcf)
            link_or_copy(cached_html, stats_html)
            rows, effect_counts = _parse_snpeff_annotations(annotated_vcf)
            annotation_table = report_dir / f"snpeff_annotations_{safe_label}.tsv"
            pd.DataFrame(rows).to_csv(annotation_table, sep="\t", index=False)
            effect_table = report_dir / f"snpeff_effect_counts_{safe_label}.tsv"
            pd.DataFrame(
                [{"effect": k, "count": v} for k, v in sorted(effect_counts.items(), key=lambda x: x[1], reverse=True)]
            ).to_csv(effect_table, sep="\t", index=False)
            return {
                "ok": True,
                "reason": "reused cached SnpEff annotation",
                "run": {
                    "cmd": [snpeff, "-noLog", "-stats", str(stats_html), genome, str(merged_vcf)],
                    "returncode": 0,
                    "stdout": "<cached>",
                    "stderr": "",
                    "ok": True,
                    "cache_hit": True,
                },
                "outputs": [str(annotated_vcf), str(stats_html), str(annotation_table), str(effect_table)],
                "annotation_table": str(annotation_table),
                "effect_table": str(effect_table),
                "effect_counts": effect_counts,
                "n_annotated_rows": len(rows),
                "snpeff_stats": _parse_snpeff_stats(stats_html),
                "snpeff_version": tool_version(snpeff, "-version"),
                "cache_hit": True,
            }

    cmd = [
        snpeff,
        "-noLog",
        "-stats", str(stats_html),
        genome,
        str(merged_vcf),
    ]
    run = run_external(
        cmd,
        timeout=3600,
        stdout_path=annotated_vcf,
        settings=getattr(ctx, "settings", None) if ctx is not None else None,
        log_path=external_tool_log_path(ctx, report_dir, f"snpeff_{safe_label}") if ctx is not None else None,
    )
    outputs = [str(annotated_vcf), str(stats_html)]
    if not run["ok"]:
        return {
            "ok": False,
            "reason": "SnpEff annotation failed",
            "run": run,
            "outputs": outputs,
            "snpeff_version": tool_version(snpeff, "-version"),
        }

    rows, effect_counts = _parse_snpeff_annotations(annotated_vcf)
    if cache_prefix is not None and annotated_vcf.exists() and stats_html.exists():
        link_or_copy(annotated_vcf, cache_prefix.with_suffix(".vcf"))
        link_or_copy(stats_html, cache_prefix.with_suffix(".html"))
    annotation_table = report_dir / f"snpeff_annotations_{safe_label}.tsv"
    pd.DataFrame(rows).to_csv(annotation_table, sep="\t", index=False)
    effect_table = report_dir / f"snpeff_effect_counts_{safe_label}.tsv"
    pd.DataFrame(
        [{"effect": k, "count": v} for k, v in sorted(effect_counts.items(), key=lambda x: x[1], reverse=True)]
    ).to_csv(effect_table, sep="\t", index=False)

    return {
        "ok": True,
        "reason": "",
        "run": run,
        "outputs": outputs + [str(annotation_table), str(effect_table)],
        "annotation_table": str(annotation_table),
        "effect_table": str(effect_table),
        "effect_counts": effect_counts,
        "n_annotated_rows": len(rows),
        "snpeff_stats": _parse_snpeff_stats(stats_html),
        "snpeff_version": tool_version(snpeff, "-version"),
    }


def _run_vep_annotation(
    merged_vcf: Path,
    report_dir: Path,
    assembly: str = "GRCh38",
    ctx=None,
) -> dict:
    """Run VEP if a local cache is present; avoid online DB dependence."""
    vep = find_executable("vep")
    if not vep:
        return {"ok": False, "reason": "VEP executable not found"}
    if not vep_cache_ready(assembly):
        return {"ok": False, "reason": f"VEP {assembly} cache is not installed locally"}
    out_vcf = report_dir / "vep_annotated.vcf"
    cmd = [
        vep,
        "--offline",
        "--cache",
        "--assembly", assembly,
        "--vcf",
        "--symbol",
        "--canonical",
        "--force_overwrite",
        "-i", str(merged_vcf),
        "-o", str(out_vcf),
    ]
    run = run_external(
        cmd,
        timeout=3600,
        settings=getattr(ctx, "settings", None) if ctx is not None else None,
        line_sink=getattr(ctx, "emit_line", None) if ctx is not None else None,
        log_path=external_tool_log_path(ctx, report_dir, "vep_annotation") if ctx is not None else None,
    )
    return {
        "ok": bool(run["ok"]),
        "reason": "" if run["ok"] else "VEP annotation failed",
        "run": run,
        "outputs": [str(out_vcf)],
        "vep_version": tool_version(vep, "--help"),
    }


@skill(
    name="vcf_annotation",
    description=(
        "Annotate WGS variants with built-in curated gene coordinates for "
        "vitiligo-relevant genes (TYR, OCA2, MC1R, HLA, NLRP1, PAX3, MITF, "
        "SLC45A2, PTPN22, CTLA4, and more). Reports variants per gene with "
        "carrier information across samples. No VEP/ANNOVAR required."
    ),
    parameters={
        "genes": {
            "type": "string",
            "description": "Comma-separated gene names (empty = all vitiligo candidates)",
            "default": "",
        },
        "min_qual": {
            "type": "number",
            "description": "Minimum QUAL score (default 30)",
            "default": 30.0,
        },
        "max_variants_per_gene": {
            "type": "integer",
            "description": "Max variants per gene (default 5000)",
            "default": 5000,
        },
        "sample_ids": {
            "type": "string",
            "description": "Comma-separated sample IDs (empty = all)",
            "default": "",
        },
        "allow_exploratory_fallback": {
            "type": "boolean",
            "description": "If true, continue with built-in hg38 gene-body coordinates when standard annotation cannot run.",
            "default": False,
        },
    },
    required=[],
)
def vcf_annotation(
    genes: str = "",
    min_qual: float = 30.0,
    max_variants_per_gene: int = 5000,
    sample_ids: str = "",
    allow_exploratory_fallback: bool = False,
    *,
    ctx=None,
) -> dict:
    from biobank_agent.utils.bcftools import merge_vcfs, get_tmp_dir
    from biobank_agent.utils.vcf_genotypes import get_sample_vcf_paths

    if genes:
        target_genes = {g.strip().upper(): VITILIGO_GENES_HG38.get(g.strip().upper())
                        for g in genes.split(",")}
        target_genes = {k: v for k, v in target_genes.items() if v is not None}
        unknown = [g.strip() for g in genes.split(",")
                   if g.strip().upper() not in VITILIGO_GENES_HG38]
    else:
        target_genes = dict(VITILIGO_GENES_HG38)
        unknown = []

    if not target_genes:
        return {"error": f"No valid gene names. Available: {list(VITILIGO_GENES_HG38.keys())}"}

    sample_paths = get_sample_vcf_paths(ctx)
    if sample_ids:
        targets = [s.strip() for s in sample_ids.split(",")]
        sample_paths = {k: v for k, v in sample_paths.items() if k in targets}

    qc_data = None
    if hasattr(ctx, "state") and hasattr(ctx.state, "custom_data"):
        qc_data = ctx.state.custom_data.get("vcf_qc_results")
    if qc_data and qc_data.get("pass_samples") and not sample_ids:
        pass_set = set(qc_data["pass_samples"])
        sample_paths = {k: v for k, v in sample_paths.items() if k in pass_set}
        logger.info("Using %d QC-pass samples for annotation", len(sample_paths))

    if not sample_paths:
        return {"error": "No VCF files found."}

    tmp_dir = get_tmp_dir(ctx)

    chroms_needed = set()
    for chrom, start, end in target_genes.values():
        chroms_needed.add(chrom)

    gene_hits = {}
    total_variants = 0
    report_dir = wgs_results_dir(ctx, "annotation")
    env_status = wgs_environment_status()
    standard_annotation_runs: list[dict] = []
    skipped_regions: list[dict[str, str]] = []
    merged_for_standard: Path | None = None
    can_try_standard_annotation = all(Path(p).exists() for p in sample_paths.values())
    standard_requested = _workflow_mode_from_ctx(ctx) == "standard"
    if standard_requested and env_status["modules"].get("standard_annotation") != "READY":
        paused = _pause_for_standard_annotation_downgrade(
            ctx=ctx,
            reason=(
                "workflow_mode=standard was requested, but no local VEP/SnpEff/ANNOVAR "
                "annotation setup is ready. Built-in hg38 gene-body coordinates would be an exploratory downgrade."
            ),
            standard_annotation_runs=[],
            allow_exploratory_fallback=bool(allow_exploratory_fallback),
        )
        if paused is not None:
            return paused

    for chrom in sorted(chroms_needed):
        genes_on_chrom = {g: coords for g, coords in target_genes.items()
                          if coords[0] == chrom}

        chrom_start = min(c[1] for c in genes_on_chrom.values())
        chrom_end = max(c[2] for c in genes_on_chrom.values())
        region = f"{chrom}:{chrom_start}-{chrom_end}"

        merged = tmp_dir / f"merged_annot_{chrom}.vcf.gz"
        try:
            merge_vcfs(list(sample_paths.values()), merged, region=region)
        except Exception as e:
            logger.warning("Merge failed for %s: %s", chrom, e)
            skipped_regions.append({"region": region, "stage": "merge", "reason": str(e)})
            continue
        if merged_for_standard is None:
            merged_for_standard = merged

        if (
            can_try_standard_annotation
            and Path(merged).exists()
            and env_status.get("annotation_databases", {}).get("snpeff_hg38")
        ):
            snpeff_run = _run_snpeff_annotation(merged, report_dir, genome="hg38", label=chrom, ctx=ctx)
            standard_annotation_runs.append(snpeff_run)
        elif (
            can_try_standard_annotation
            and Path(merged).exists()
            and env_status.get("annotation_databases", {}).get("vep_GRCh38_cache")
        ):
            vep_run = _run_vep_annotation(merged, report_dir, assembly="GRCh38", ctx=ctx)
            standard_annotation_runs.append(vep_run)

        for gene_name, (_, gstart, gend) in genes_on_chrom.items():
            variants = _query_gene_variants(
                str(merged), gene_name, chrom, gstart, gend,
                min_qual=min_qual, max_variants=max_variants_per_gene,
            )
            if variants:
                gene_hits[gene_name] = variants
                total_variants += len(variants)

    pheno_df = None
    try:
        pheno_df = ctx.dm.query("SELECT * FROM biomarkers")
    except Exception:
        pass

    gene_summary = []
    for gene_name in sorted(target_genes.keys()):
        coords = target_genes[gene_name]
        variants = gene_hits.get(gene_name, [])
        n_snv = sum(1 for v in variants
                    if len(v["ref"]) == 1 and len(v["alt"]) == 1)
        n_indel = len(variants) - n_snv

        gene_summary.append({
            "gene": gene_name,
            "region": f"{coords[0]}:{coords[1]}-{coords[2]}",
            "n_variants": len(variants),
            "n_snv": n_snv,
            "n_indel": n_indel,
            "top_variants": variants[:5] if variants else [],
        })

    figures = []

    genes_with_hits = [gs for gs in gene_summary if gs["n_variants"] > 0]
    if genes_with_hits:
        fig, ax = nature_figure(width="double", height_ratio=0.5)
        names = [gs["gene"] for gs in genes_with_hits]
        snv_counts = [gs["n_snv"] for gs in genes_with_hits]
        indel_counts = [gs["n_indel"] for gs in genes_with_hits]
        x = range(len(names))
        bar_width = 0.35

        ax.bar([i - bar_width/2 for i in x], snv_counts, bar_width,
               label="SNV", color=PALETTE[0], alpha=0.8)
        ax.bar([i + bar_width/2 for i in x], indel_counts, bar_width,
               label="Indel", color=PALETTE[1], alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Variant count")
        ax.set_title(f"Variants in Candidate Genes (QUAL>={min_qual})")
        ax.legend(fontsize=6, frameon=False)
        fig.tight_layout()
        paths = save_figure(fig, "vcf_annotation_gene_counts", report_dir)
        figures.extend(paths)

    if hasattr(ctx, "state") and hasattr(ctx.state, "figures"):
        ctx.state.figures.extend(figures)

    if hasattr(ctx, "state") and hasattr(ctx.state, "custom_data"):
        ctx.state.custom_data["annotation_results"] = {
            "gene_hits": {g: len(v) for g, v in gene_hits.items()},
            "gene_coords": target_genes,
            "standard_annotation_runs": standard_annotation_runs,
            "skipped_regions": skipped_regions,
        }

    standard_ok = [r for r in standard_annotation_runs if r.get("ok")]
    if standard_ok:
        mode = "snpeff_hg38" if "snpeff_version" in standard_ok[0] else "vep_GRCh38_cache"
        note = "Functional annotations were generated with an installed standard annotation tool."
    else:
        mode = "built_in_hg38_candidate_gene_coordinates"
        if env_status["executables"].get("snpEff") or env_status["executables"].get("vep"):
            if not can_try_standard_annotation:
                note = (
                    "Standard annotation executables are installed, but source VCF paths do not all "
                    "exist locally in this run. Built-in hg38 gene-body coordinates were used for "
                    "the reported carrier summaries."
                )
            else:
                note = (
                    "Standard annotation executables are installed, but no successful standard annotation "
                    "run was available for these merged candidate regions. Built-in hg38 gene-body "
                    "coordinates were used for the reported carrier summaries."
                )
        else:
            note = (
                "Annotation uses built-in hg38 gene body coordinates. "
                "Exon/intron distinction and functional effect prediction require VEP/SnpEff/ANNOVAR."
            )

    if standard_requested and not standard_ok:
        paused = _pause_for_standard_annotation_downgrade(
            ctx=ctx,
            reason=note,
            standard_annotation_runs=[
                {k: v for k, v in r.items() if k not in {"run"}}
                for r in standard_annotation_runs
            ],
            allow_exploratory_fallback=bool(allow_exploratory_fallback),
        )
        if paused is not None:
            return paused

    return {
        "n_genes_queried": len(target_genes),
        "n_genes_with_variants": len(gene_hits),
        "n_total_variants": total_variants,
        "gene_summary": gene_summary,
        "unknown_genes": unknown,
        "available_genes": list(VITILIGO_GENES_HG38.keys()),
        "figures": [str(p) for p in figures],
        "result_dir": str(report_dir),
        "annotation_mode": mode,
        "standard_annotation_available": env_status["modules"]["standard_annotation"] == "READY",
        "standard_annotation_runs": [
            {k: v for k, v in r.items() if k not in {"run"}}
            for r in standard_annotation_runs
        ],
        "skipped_regions": skipped_regions,
        "note": note,
    }
