"""Juvenile hair-whitening multi-omics mechanism skills.

These skills turn short mechanistic prompts into auditable intermediate
artifacts.  They prefer real upstream WGS/scRNA/scATAC/Stereo results when
present in session state and otherwise emit explicit fallback evidence rather
than fabricating discoveries.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from biobank_agent.data.virtualcell_multimodal import (
    h5ad_sample_summaries,
    inspect_h5ad_metadata,
    link_wgs_to_bwhair,
    load_bwhair_manifest,
    load_default_wgs_manifest,
)
from biobank_agent.registry import skill


PIGMENT_IMMUNE_GENES = {
    "TYR": ("chr11", 88911696, 88921223, "melanin synthesis"),
    "OCA2": ("chr15", 27622010, 27819535, "melanosome pigmentation"),
    "SLC45A2": ("chr5", 33951717, 33987793, "melanosome pigmentation"),
    "MC1R": ("chr16", 89919740, 89920776, "melanocortin signalling"),
    "HLA": ("chr6", 28477797, 33448354, "antigen presentation"),
    "NLRP1": ("chr17", 5484000, 5531000, "inflammasome autoimmunity"),
    "PAX3": ("chr2", 217288628, 217374533, "melanocyte development"),
    "SOX10": ("chr22", 37972312, 37984561, "melanocyte lineage"),
    "MITF": ("chr3", 69788846, 70009099, "melanocyte transcription"),
    "TYRP1": ("chr9", 12693384, 12710448, "melanin synthesis"),
    "DCT": ("chr13", 94793016, 94856846, "melanin synthesis"),
    "IRF4": ("chr6", 396321, 411447, "pigmentation and immune regulation"),
    "TNF": ("chr6", 31575565, 31578336, "inflammatory signalling"),
    "IFNG": ("chr12", 68154768, 68159740, "T-cell immune signalling"),
    "CXCL10": ("chr4", 76013416, 76016388, "immune chemotaxis"),
}


TF_MOTIF_PRIORS = {
    "SOX10": ["SOX10", "SOX9"],
    "MITF": ["MITF", "TFEB"],
    "PAX3": ["PAX3"],
    "IRF4": ["IRF4", "STAT1"],
    "HLA": ["NF-kB", "IRF1", "STAT1"],
    "TNF": ["NF-kB", "AP-1"],
    "IFNG": ["STAT1", "IRF1"],
    "CXCL10": ["NF-kB", "STAT1", "IRF1"],
    "TYR": ["MITF", "SOX10"],
    "OCA2": ["MITF", "SOX10"],
    "SLC45A2": ["MITF", "SOX10"],
    "MC1R": ["MITF", "CREB"],
}


def _out_dir(ctx: Any, name: str) -> Path:
    report_dir = getattr(ctx, "report_dir", None)
    if report_dir is None:
        settings = getattr(ctx, "settings", None)
        report_dir = getattr(settings, "reports_dir", Path("./reports")) if settings else Path("./reports")
    out = Path(report_dir) / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _state(ctx: Any) -> dict:
    try:
        custom = getattr(getattr(ctx, "state", None), "custom_data", None)
        if isinstance(custom, dict):
            return custom
    except Exception:
        pass
    return {}


def _record(ctx: Any, key: str, value: Any) -> None:
    custom = _state(ctx)
    if custom is not None:
        custom[key] = value


def _read_table(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.DataFrame()


def _variant_id(row: dict) -> str:
    chrom = str(row.get("chrom") or row.get("#CHROM") or row.get("CHROM") or "")
    pos = str(row.get("pos") or row.get("POS") or "")
    ref = str(row.get("ref") or row.get("REF") or "")
    alt = str(row.get("alt") or row.get("ALT") or row.get("A1") or "")
    return f"{chrom}:{pos}:{ref}>{alt}".strip(":>")


def _nearest_gene(chrom: str, pos: int) -> tuple[str, int, str]:
    best_gene = ""
    best_distance = 10**18
    best_annotation = ""
    for gene, (g_chrom, start, end, annotation) in PIGMENT_IMMUNE_GENES.items():
        if str(chrom) != str(g_chrom):
            continue
        distance = 0 if start <= pos <= end else min(abs(pos - start), abs(pos - end))
        if distance < best_distance:
            best_gene = gene
            best_distance = distance
            best_annotation = annotation
    if not best_gene:
        return "", -1, ""
    return best_gene, int(best_distance), best_annotation


def _candidate_variants_from_state(ctx: Any, max_variants: int) -> tuple[list[dict], list[str]]:
    custom = _state(ctx)
    warnings: list[str] = []
    variants: list[dict] = []

    gwas = custom.get("gwas_results") if isinstance(custom, dict) else None
    gwas_rows = []
    if isinstance(gwas, dict):
        gwas_rows = list(gwas.get("all_results") or gwas.get("top_hits") or [])
    for row in gwas_rows[: max_variants * 2]:
        if not isinstance(row, dict):
            continue
        try:
            chrom = str(row.get("chrom") or row.get("#CHROM") or row.get("CHROM") or "")
            pos = int(float(row.get("pos") or row.get("POS") or 0))
        except Exception:
            continue
        gene, distance, annotation = _nearest_gene(chrom, pos)
        p_value = row.get("p_value", row.get("P", 1.0))
        try:
            p_float = float(p_value)
        except Exception:
            p_float = 1.0
        variants.append({
            "variant_id": _variant_id(row),
            "chrom": chrom,
            "pos": pos,
            "ref": str(row.get("ref") or row.get("REF") or ""),
            "alt": str(row.get("alt") or row.get("ALT") or ""),
            "rsid": str(row.get("rsid") or row.get("ID") or "."),
            "p_value": p_float,
            "effect": row.get("odds_ratio", row.get("OR", row.get("beta", ""))),
            "maf": row.get("maf", row.get("A1_FREQ", "")),
            "gene": gene,
            "distance_to_gene": distance,
            "evidence": "session_gwas_top_hit",
            "mechanism_context": annotation,
        })

    annot = custom.get("annotation_results") if isinstance(custom, dict) else None
    if isinstance(annot, dict):
        coords = annot.get("gene_coords") or {}
        hits = annot.get("gene_hits") or {}
        for gene, n_hits in hits.items():
            if len(variants) >= max_variants:
                break
            coords_row = coords.get(gene) if isinstance(coords, dict) else None
            if not coords_row:
                coords_row = PIGMENT_IMMUNE_GENES.get(str(gene).upper(), ("", 0, 0, ""))[:3]
            try:
                chrom, start, end = coords_row[:3]
                pos = int((int(start) + int(end)) / 2)
            except Exception:
                continue
            annotation = PIGMENT_IMMUNE_GENES.get(str(gene).upper(), ("", 0, 0, "candidate gene"))[3]
            variants.append({
                "variant_id": f"{chrom}:{pos}:candidate:{gene}",
                "chrom": chrom,
                "pos": pos,
                "ref": "",
                "alt": "",
                "rsid": ".",
                "p_value": "",
                "effect": "",
                "maf": "",
                "gene": str(gene).upper(),
                "distance_to_gene": 0,
                "evidence": f"session_annotation_gene_hit_count={n_hits}",
                "mechanism_context": annotation,
            })

    if not variants:
        warnings.append(
            "No upstream association or annotation results were found in session state; "
            "using curated pigmentation/autoimmune loci as hypothesis anchors."
        )
        for gene, (chrom, start, end, annotation) in PIGMENT_IMMUNE_GENES.items():
            pos = int((start + end) / 2)
            variants.append({
                "variant_id": f"{chrom}:{pos}:candidate:{gene}",
                "chrom": chrom,
                "pos": pos,
                "ref": "",
                "alt": "",
                "rsid": ".",
                "p_value": "",
                "effect": "",
                "maf": "",
                "gene": gene,
                "distance_to_gene": 0,
                "evidence": "curated_locus_fallback",
                "mechanism_context": annotation,
            })

    deduped: list[dict] = []
    seen: set[str] = set()
    for row in sorted(
        variants,
        key=lambda r: (
            float(r["p_value"]) if isinstance(r.get("p_value"), (int, float)) or str(r.get("p_value", "")).replace(".", "", 1).isdigit() else 1.0,
            int(r.get("distance_to_gene", 10**9) if r.get("distance_to_gene", -1) != -1 else 10**9),
        ),
    ):
        key = str(row.get("variant_id") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= max_variants:
            break
    return deduped, warnings


def _load_candidates(ctx: Any) -> list[dict]:
    custom = _state(ctx)
    if isinstance(custom, dict):
        existing = custom.get("jh_candidate_variants")
        if isinstance(existing, dict):
            return list(existing.get("candidate_variants") or [])
        if isinstance(existing, list):
            return existing
    return []


def _load_regulatory_hits(ctx: Any) -> list[dict]:
    custom = _state(ctx)
    if isinstance(custom, dict):
        existing = custom.get("regulatory_variant_annotation")
        if isinstance(existing, dict):
            return list(existing.get("regulatory_hits") or [])
    return []


@skill(
    name="jh_variant_discovery",
    description=(
        "Identify Juvenile hair-whitening candidate variants or locus anchors from "
        "session WGS association/annotation results, with curated fallback loci."
    ),
    parameters={
        "case_group": {"type": "string", "description": "Primary WGS case group code", "default": "J"},
        "control_groups": {"type": "string", "description": "Comma-separated comparison group codes", "default": "V,S"},
        "max_variants": {"type": "integer", "description": "Maximum candidate variants/loci to retain", "default": 30},
    },
    required=[],
)
def jh_variant_discovery(case_group: str = "J", control_groups: str = "V,S", max_variants: int = 30, *, ctx=None) -> dict:
    wgs = load_default_wgs_manifest()
    link = link_wgs_to_bwhair()
    variants, warnings = _candidate_variants_from_state(ctx, max(1, int(max_variants or 30)))
    out = _out_dir(ctx, "06_MultiOmics")
    table = out / "jh_candidate_variants.tsv"
    pd.DataFrame(variants).to_csv(table, sep="\t", index=False)

    payload = {
        "case_group": case_group,
        "control_groups": [g.strip() for g in str(control_groups).split(",") if g.strip()],
        "n_wgs_samples": int(len(wgs)),
        "wgs_phenotype_counts": wgs["phenotype"].value_counts().to_dict(),
        "n_linked_donors": int(((link["has_wgs"] == True) & (link["n_h5ad_files"] > 0)).sum()),  # noqa: E712
        "n_candidate_variants": len(variants),
        "candidate_variants": variants,
        "candidate_table": str(table),
        "warnings": warnings,
        "claim_boundary": "exploratory candidate prioritization; not causal proof",
    }
    _write_json(out / "jh_variant_discovery.json", payload)
    _record(ctx, "jh_candidate_variants", payload)
    return payload

@skill(
    name="regulatory_variant_annotation",
    description="Annotate candidate variants with nearby pigmentation, immune, HLA and regulatory context.",
    parameters={
        "window_bp": {"type": "integer", "description": "Distance window for near-gene regulatory annotation", "default": 100000},
    },
    required=[],
)
def regulatory_variant_annotation(window_bp: int = 100000, *, ctx=None) -> dict:
    candidates = _load_candidates(ctx)
    if not candidates:
        candidates, _ = _candidate_variants_from_state(ctx, 30)
    rows: list[dict] = []
    for item in candidates:
        chrom = str(item.get("chrom", ""))
        try:
            pos = int(item.get("pos") or 0)
        except Exception:
            pos = 0
        gene, distance, annotation = _nearest_gene(chrom, pos)
        source_gene = str(item.get("gene") or gene or "")
        distance_value = int(item.get("distance_to_gene", distance if distance >= 0 else -1) or -1)
        regulatory_class = "gene_body" if distance_value == 0 else ("proximal_regulatory" if 0 < distance_value <= int(window_bp or 0) else "distal_or_unmapped")
        if chrom == "chr6" and 28_000_000 <= pos <= 34_000_000:
            regulatory_class = "HLA_region"
            source_gene = source_gene or "HLA"
            annotation = "antigen presentation and immune regulation"
        rows.append({
            **item,
            "gene": source_gene,
            "regulatory_class": regulatory_class,
            "distance_to_gene": distance_value,
            "priority_context": annotation or item.get("mechanism_context", ""),
            "databases_used": "built_in_hg38_loci",
        })
    out = _out_dir(ctx, "06_MultiOmics")
    table = out / "regulatory_variant_annotation.tsv"
    pd.DataFrame(rows).to_csv(table, sep="\t", index=False)
    payload = {
        "n_regulatory_hits": len(rows),
        "window_bp": int(window_bp or 0),
        "regulatory_hits": rows,
        "annotation_table": str(table),
        "warnings": ["External regulatory databases were not required for this fallback annotation."],
    }
    _write_json(out / "regulatory_variant_annotation.json", payload)
    _record(ctx, "regulatory_variant_annotation", payload)
    return payload


@skill(
    name="tf_binding_disruption",
    description=(
        "Assess whether candidate regulatory variants are plausible TF binding "
        "site disruptions using motif priors and explicit database fallbacks."
    ),
    parameters={
        "motif_database": {"type": "string", "description": "Motif database preference, e.g. auto,JASPAR,HOCOMOCO", "default": "auto"},
        "top_n": {"type": "integer", "description": "Maximum TFBS rows to return", "default": 30},
    },
    required=[],
)
def tf_binding_disruption(motif_database: str = "auto", top_n: int = 30, *, ctx=None) -> dict:
    hits = _load_regulatory_hits(ctx)
    if not hits:
        hits = regulatory_variant_annotation(ctx=ctx)["regulatory_hits"]
    rows: list[dict] = []
    for item in hits:
        gene = str(item.get("gene") or "").upper()
        tfs = TF_MOTIF_PRIORS.get(gene) or TF_MOTIF_PRIORS.get(str(item.get("priority_context", "")).upper(), [])
        if not tfs and str(item.get("regulatory_class")) == "HLA_region":
            tfs = TF_MOTIF_PRIORS["HLA"]
        if not tfs:
            tfs = ["MITF", "SOX10"] if gene in {"TYR", "OCA2", "SLC45A2", "MC1R"} else ["unknown"]
        for idx, tf in enumerate(tfs[:3]):
            evidence = "motif_prior_fallback"
            delta = round(0.35 / (idx + 1), 4) if tf != "unknown" else 0.0
            rows.append({
                "variant_id": item.get("variant_id"),
                "chrom": item.get("chrom"),
                "pos": item.get("pos"),
                "gene": gene,
                "tf": tf,
                "predicted_binding_delta": delta,
                "direction": "possible_loss_or_gain" if delta else "unknown",
                "motif_database": motif_database,
                "evidence": evidence,
                "limitation": "Sequence-level motif scanning requires reference FASTA plus JASPAR/HOCOMOCO/FIMO or motifbreakR.",
            })
    rows = rows[: max(1, int(top_n or 30))]
    out = _out_dir(ctx, "06_MultiOmics")
    table = out / "tf_binding_disruption.tsv"
    pd.DataFrame(rows).to_csv(table, sep="\t", index=False)
    payload = {
        "n_tfbs_candidates": len(rows),
        "motif_database": motif_database,
        "tf_binding_hits": rows,
        "tf_binding_table": str(table),
        "warnings": ["TF binding deltas are motif-prior placeholders unless an external motif scanner/database is configured."],
    }
    _write_json(out / "tf_binding_disruption.json", payload)
    _record(ctx, "tf_binding_disruption", payload)
    return payload


@skill(
    name="scatac_peak_overlap",
    description="Check candidate variants against scATAC metadata/peak availability and summarize overlap readiness.",
    parameters={
        "max_files": {"type": "integer", "description": "Maximum scATAC h5ad files to inspect in backed mode", "default": 1},
    },
    required=[],
)
def scatac_peak_overlap(max_files: int = 1, *, ctx=None) -> dict:
    candidates = _load_candidates(ctx)
    if not candidates:
        candidates, _ = _candidate_variants_from_state(ctx, 30)
    h5ad = load_bwhair_manifest()
    scatac = h5ad[h5ad["modality"] == "scatac"].copy()
    inspections = h5ad_sample_summaries(
        modality="scatac",
        sample_query="BWhair_",
        max_files=max(0, int(max_files or 1)),
        include_obs_summary=True,
    )
    var_columns = set()
    for item in inspections:
        var_columns.update(str(c) for c in item.get("var_columns", []))
    peak_like = sorted(c for c in var_columns if any(token in c.lower() for token in ("chrom", "start", "end", "peak", "interval")))
    rows = []
    for item in candidates:
        rows.append({
            "variant_id": item.get("variant_id"),
            "chrom": item.get("chrom"),
            "pos": item.get("pos"),
            "gene": item.get("gene"),
            "overlap_status": "requires_peak_coordinates" if not peak_like else "peak_coordinate_columns_detected",
            "peak_coordinate_columns": ",".join(peak_like),
            "evidence": "scatac_h5ad_metadata",
        })
    out = _out_dir(ctx, "06_MultiOmics")
    table = out / "scatac_peak_overlap.tsv"
    pd.DataFrame(rows).to_csv(table, sep="\t", index=False)
    payload = {
        "n_scatac_files": int(len(scatac)),
        "n_existing_scatac_files": int(scatac["file_exists"].sum()) if "file_exists" in scatac else 0,
        "n_candidate_variants": len(candidates),
        "peak_coordinate_columns": peak_like,
        "overlap_rows": rows,
        "overlap_table": str(table),
        "inspections": inspections,
        "warnings": [] if peak_like else ["No explicit peak coordinate columns were detected in backed metadata; exact variant-peak overlap is deferred."],
    }
    _write_json(out / "scatac_peak_overlap.json", payload)
    _record(ctx, "scatac_peak_overlap", payload)
    return payload


@skill(
    name="scatac_accessibility_differential",
    description="Compare BWhair scATAC accessibility readiness for Juvenile white-vs-black hair and cell-type specificity.",
    parameters={
        "case_hair_state": {"type": "string", "description": "Case hair state code", "default": "W"},
        "control_hair_state": {"type": "string", "description": "Control hair state code", "default": "B"},
    },
    required=[],
)
def scatac_accessibility_differential(case_hair_state: str = "W", control_hair_state: str = "B", *, ctx=None) -> dict:
    wgs = load_default_wgs_manifest()
    h5ad = load_bwhair_manifest()
    juvenile_donors = set(wgs[wgs["phenotype"] == "Juvenile_White"]["donor"].astype(str))
    spatial = h5ad[(h5ad["modality"] == "spatial") & (h5ad["donor"].isin(juvenile_donors))].copy()
    coverage = spatial.groupby("hair_state", dropna=False).agg(
        n_files=("filename", "count"),
        n_donors=("donor", "nunique"),
        cells=("cells", "sum"),
        existing_files=("file_exists", "sum"),
    ).reset_index()
    scrna_meta = h5ad_sample_summaries(
        modality="scatac",
        sample_query="BWhair_",
        max_files=1,
        include_obs_summary=True,
    )
    celltype_columns = []
    for meta in scrna_meta:
        for col in meta.get("obs_columns", []):
            if any(token in col.lower() for token in ("cell", "type", "cluster", "annotation", "leiden")):
                celltype_columns.append(col)
    case_cells = int(coverage.loc[coverage["hair_state"] == case_hair_state, "cells"].sum()) if not coverage.empty else 0
    ctrl_cells = int(coverage.loc[coverage["hair_state"] == control_hair_state, "cells"].sum()) if not coverage.empty else 0
    out = _out_dir(ctx, "06_MultiOmics")
    coverage_table = out / "juvenile_hair_state_coverage.tsv"
    coverage.to_csv(coverage_table, sep="\t", index=False)
    payload = {
        "case_hair_state": case_hair_state,
        "control_hair_state": control_hair_state,
        "n_juvenile_donors": len(juvenile_donors),
        "case_manifest_cells": case_cells,
        "control_manifest_cells": ctrl_cells,
        "coverage_table": str(coverage_table),
        "celltype_columns_detected": sorted(set(celltype_columns)),
        "status": "metadata_ready_matrix_deferred",
        "warnings": [
            "scATAC differential accessibility requires peak-by-cell matrix processing and cell-type labels; this skill records readiness and exact fallback status."
        ],
    }
    _write_json(out / "scatac_accessibility_differential.json", payload)
    _record(ctx, "scatac_accessibility_differential", payload)
    return payload


@skill(
    name="scrna_expression_differential",
    description="Summarize scRNA expression differential-analysis readiness for variant-linked genes.",
    parameters={
        "genes": {"type": "string", "description": "Comma-separated genes; empty uses candidate variant genes", "default": ""},
    },
    required=[],
)
def scrna_expression_differential(genes: str = "", *, ctx=None) -> dict:
    candidates = _load_candidates(ctx)
    gene_list = [g.strip().upper() for g in str(genes or "").split(",") if g.strip()]
    if not gene_list:
        gene_list = sorted({str(v.get("gene", "")).upper() for v in candidates if v.get("gene")})
    gene_list = [g for g in gene_list if g][:30]
    meta = h5ad_sample_summaries(
        modality="scrna",
        sample_query="BWhair_",
        max_files=1,
        include_obs_summary=True,
    )
    var_columns = set()
    obs_columns = set()
    n_vars = None
    for item in meta:
        var_columns.update(str(c) for c in item.get("var_columns", []))
        obs_columns.update(str(c) for c in item.get("obs_columns", []))
        n_vars = item.get("n_vars")
    gene_symbol_columns = sorted(c for c in var_columns if any(token in c.lower() for token in ("gene", "symbol", "name", "feature")))
    celltype_columns = sorted(c for c in obs_columns if any(token in c.lower() for token in ("cell", "type", "cluster", "annotation", "leiden")))
    rows = [
        {
            "gene": gene,
            "expression_status": "requires_matrix_scan",
            "gene_symbol_columns": ",".join(gene_symbol_columns),
            "celltype_columns": ",".join(celltype_columns),
        }
        for gene in gene_list
    ]
    out = _out_dir(ctx, "06_MultiOmics")
    table = out / "scrna_expression_differential.tsv"
    pd.DataFrame(rows).to_csv(table, sep="\t", index=False)
    payload = {
        "n_genes": len(gene_list),
        "genes": gene_list,
        "n_scrna_vars": n_vars,
        "gene_symbol_columns": gene_symbol_columns,
        "celltype_columns_detected": celltype_columns,
        "expression_rows": rows,
        "expression_table": str(table),
        "warnings": ["Expression differential testing is deferred until backed/chunked matrix extraction is enabled for the selected gene symbols."],
    }
    _write_json(out / "scrna_expression_differential.json", payload)
    _record(ctx, "scrna_expression_differential", payload)
    return payload


@skill(
    name="atac_expression_coupling",
    description="Integrate variant-linked accessibility readiness with scRNA gene-expression readiness.",
    parameters={
        "top_n": {"type": "integer", "description": "Maximum coupled mechanism rows", "default": 30},
    },
    required=[],
)
def atac_expression_coupling(top_n: int = 30, *, ctx=None) -> dict:
    custom = _state(ctx)
    candidates = _load_candidates(ctx)
    overlap = (custom.get("scatac_peak_overlap") or {}).get("overlap_rows", []) if isinstance(custom, dict) else []
    expression = (custom.get("scrna_expression_differential") or {}).get("expression_rows", []) if isinstance(custom, dict) else []
    expr_by_gene = {str(r.get("gene", "")).upper(): r for r in expression if isinstance(r, dict)}
    rows: list[dict] = []
    for item in candidates:
        gene = str(item.get("gene", "")).upper()
        peak = next((r for r in overlap if r.get("variant_id") == item.get("variant_id")), {})
        expr = expr_by_gene.get(gene, {})
        evidence_count = int(bool(peak)) + int(bool(expr)) + int(bool(item.get("gene")))
        rows.append({
            "variant_id": item.get("variant_id"),
            "gene": gene,
            "accessibility_evidence": peak.get("overlap_status", "not_evaluated"),
            "expression_evidence": expr.get("expression_status", "not_evaluated"),
            "coupling_direction": "unknown_until_matrix_effects_available",
            "evidence_score": evidence_count,
        })
    rows = sorted(rows, key=lambda r: r["evidence_score"], reverse=True)[: max(1, int(top_n or 30))]
    out = _out_dir(ctx, "06_MultiOmics")
    table = out / "atac_expression_coupling.tsv"
    pd.DataFrame(rows).to_csv(table, sep="\t", index=False)
    payload = {
        "n_coupled_rows": len(rows),
        "coupling_rows": rows,
        "coupling_table": str(table),
        "warnings": ["Coupling direction is not assigned without quantitative peak accessibility and gene expression contrasts."],
    }
    _write_json(out / "atac_expression_coupling.json", payload)
    _record(ctx, "atac_expression_coupling", payload)
    return payload


@skill(
    name="spatial_celltype_localization",
    description="Inspect Stereo-seq coordinate and cell-type metadata readiness for variant-linked cell-state localization.",
    parameters={
        "max_files": {"type": "integer", "description": "Maximum spatial h5ad files to inspect", "default": 3},
        "sample_query": {"type": "string", "description": "Optional sample/donor filter", "default": "J"},
    },
    required=[],
)
def spatial_celltype_localization(max_files: int = 3, sample_query: str = "J", *, ctx=None) -> dict:
    summaries = h5ad_sample_summaries(
        modality="spatial",
        sample_query=sample_query,
        max_files=max(1, int(max_files or 3)),
        include_obs_summary=True,
    )
    rows = []
    for item in summaries:
        obs_cols = [str(c) for c in item.get("obs_columns", [])]
        obsm_keys = [str(c) for c in item.get("obsm_keys", [])]
        coordinate_cols = [c for c in obs_cols if c.lower() in {"x", "y", "spatial_x", "spatial_y"} or "coord" in c.lower()]
        spatial_keys = [k for k in obsm_keys if "spatial" in k.lower()]
        celltype_cols = [c for c in obs_cols if any(token in c.lower() for token in ("cell", "type", "cluster", "annotation", "leiden"))]
        rows.append({
            "sample_id": item.get("sample_id"),
            "donor": item.get("donor"),
            "hair_state": item.get("hair_state"),
            "status": item.get("status"),
            "coordinate_columns": ",".join(coordinate_cols),
            "spatial_obsm_keys": ",".join(spatial_keys),
            "celltype_columns": ",".join(celltype_cols),
            "n_obs": item.get("n_obs"),
        })
    out = _out_dir(ctx, "06_MultiOmics")
    table = out / "spatial_celltype_localization.tsv"
    pd.DataFrame(rows).to_csv(table, sep="\t", index=False)
    payload = {
        "n_spatial_files_inspected": len(rows),
        "localization_rows": rows,
        "localization_table": str(table),
        "warnings": [
            "Spatial localization testing requires coordinate keys and cell-type labels in Stereo h5ad obs/obsm; this step records availability."
        ],
    }
    _write_json(out / "spatial_celltype_localization.json", payload)
    _record(ctx, "spatial_celltype_localization", payload)
    return payload


@skill(
    name="spatial_cell_interaction",
    description="Summarize spatial cell-cell interaction analysis readiness for variant-linked cell types.",
    parameters={
        "neighborhood_radius": {"type": "number", "description": "Neighborhood radius for future spatial interaction graph", "default": 50.0},
    },
    required=[],
)
def spatial_cell_interaction(neighborhood_radius: float = 50.0, *, ctx=None) -> dict:
    custom = _state(ctx)
    loc_rows = (custom.get("spatial_celltype_localization") or {}).get("localization_rows", []) if isinstance(custom, dict) else []
    rows = []
    for item in loc_rows:
        has_coords = bool(item.get("coordinate_columns") or item.get("spatial_obsm_keys"))
        has_celltypes = bool(item.get("celltype_columns"))
        rows.append({
            "sample_id": item.get("sample_id"),
            "donor": item.get("donor"),
            "hair_state": item.get("hair_state"),
            "interaction_status": "ready_for_neighbor_graph" if has_coords and has_celltypes else "requires_coordinates_or_celltypes",
            "neighborhood_radius": float(neighborhood_radius or 0.0),
        })
    out = _out_dir(ctx, "06_MultiOmics")
    table = out / "spatial_cell_interaction.tsv"
    pd.DataFrame(rows).to_csv(table, sep="\t", index=False)
    payload = {
        "n_samples": len(rows),
        "interaction_rows": rows,
        "interaction_table": str(table),
        "warnings": ["Interaction changes are readiness calls unless neighbor graphs are computed from coordinates and cell labels."],
    }
    _write_json(out / "spatial_cell_interaction.json", payload)
    _record(ctx, "spatial_cell_interaction", payload)
    return payload


@skill(
    name="multiomics_mechanism_prioritization",
    description="Rank variant-to-mechanism hypotheses across WGS, TF, ATAC, RNA and spatial evidence.",
    parameters={
        "top_n": {"type": "integer", "description": "Maximum prioritized hypotheses", "default": 20},
    },
    required=[],
)
def multiomics_mechanism_prioritization(top_n: int = 20, *, ctx=None) -> dict:
    custom = _state(ctx)
    candidates = _load_candidates(ctx)
    tf_rows = (custom.get("tf_binding_disruption") or {}).get("tf_binding_hits", []) if isinstance(custom, dict) else []
    peak_rows = (custom.get("scatac_peak_overlap") or {}).get("overlap_rows", []) if isinstance(custom, dict) else []
    coupling_rows = (custom.get("atac_expression_coupling") or {}).get("coupling_rows", []) if isinstance(custom, dict) else []
    tf_by_variant: dict[str, list[dict]] = {}
    for row in tf_rows:
        tf_by_variant.setdefault(str(row.get("variant_id")), []).append(row)
    peak_by_variant = {str(r.get("variant_id")): r for r in peak_rows}
    coupling_by_variant = {str(r.get("variant_id")): r for r in coupling_rows}
    rows = []
    for item in candidates:
        vid = str(item.get("variant_id"))
        gene = str(item.get("gene") or "")
        p = item.get("p_value")
        p_score = 0.0
        try:
            p_float = float(p)
            if p_float > 0:
                p_score = min(5.0, -math.log10(p_float))
        except Exception:
            pass
        evidence_score = p_score
        evidence_labels = [str(item.get("evidence", ""))]
        if tf_by_variant.get(vid):
            evidence_score += 1.0
            evidence_labels.append("tf_prior")
        if peak_by_variant.get(vid):
            evidence_score += 1.0
            evidence_labels.append("scatac_metadata")
        if coupling_by_variant.get(vid):
            evidence_score += 1.0
            evidence_labels.append("atac_rna_coupling_ready")
        if gene in PIGMENT_IMMUNE_GENES:
            evidence_score += 1.0
            evidence_labels.append(PIGMENT_IMMUNE_GENES[gene][3])
        rows.append({
            "variant_id": vid,
            "gene": gene,
            "mechanism_hypothesis": (
                f"{vid} may affect {gene} through regulatory accessibility, TF binding, "
                "or downstream melanocyte/immune cell-state changes."
            ),
            "evidence_score": round(evidence_score, 4),
            "evidence_labels": ";".join([e for e in evidence_labels if e]),
            "claim_boundary": "hypothesis_for_follow_up",
        })
    rows = sorted(rows, key=lambda r: r["evidence_score"], reverse=True)[: max(1, int(top_n or 20))]
    out = _out_dir(ctx, "06_MultiOmics")
    table = out / "multiomics_mechanism_prioritization.tsv"
    pd.DataFrame(rows).to_csv(table, sep="\t", index=False)
    payload = {
        "n_prioritized_hypotheses": len(rows),
        "prioritized_hypotheses": rows,
        "prioritization_table": str(table),
        "claim_boundary": "exploratory mechanism ranking; requires independent validation",
    }
    _write_json(out / "multiomics_mechanism_prioritization.json", payload)
    _record(ctx, "multiomics_mechanism_prioritization", payload)
    return payload


@skill(
    name="workflow_gap_detector",
    description="Audit the mechanism workflow for missing data, databases, MCP tools and candidate generated skills.",
    parameters={
        "trusted_harness_mode": {"type": "boolean", "description": "Whether automatic skill activation is permitted by the harness", "default": False},
    },
    required=[],
)
def workflow_gap_detector(trusted_harness_mode: bool = False, *, ctx=None) -> dict:
    custom = _state(ctx)
    gaps: list[dict] = []
    checks = {
        "candidate_variants": "jh_candidate_variants",
        "regulatory_annotation": "regulatory_variant_annotation",
        "tf_binding": "tf_binding_disruption",
        "scatac_overlap": "scatac_peak_overlap",
        "scatac_differential": "scatac_accessibility_differential",
        "scrna_expression": "scrna_expression_differential",
        "atac_expression_coupling": "atac_expression_coupling",
        "spatial_localization": "spatial_celltype_localization",
        "spatial_interaction": "spatial_cell_interaction",
        "mechanism_prioritization": "multiomics_mechanism_prioritization",
    }
    for name, key in checks.items():
        if key not in custom:
            gaps.append({"gap": name, "severity": "missing_step", "recommendation": f"Run `{key}` or equivalent skill."})
    tf = custom.get("tf_binding_disruption") or {}
    if tf.get("warnings"):
        gaps.append({
            "gap": "motif_database",
            "severity": "database_or_mcp_needed",
            "recommendation": "Configure JASPAR/HOCOMOCO/FIMO/motifbreakR MCP or local database for sequence-level TF binding deltas.",
        })
    atac = custom.get("scatac_peak_overlap") or {}
    if atac.get("warnings"):
        gaps.append({
            "gap": "scatac_peak_coordinates",
            "severity": "data_parser_needed",
            "recommendation": "Generate a reviewed skill for backed scATAC peak coordinate extraction and interval overlap.",
        })
    expr = custom.get("scrna_expression_differential") or {}
    if expr.get("warnings"):
        gaps.append({
            "gap": "scrna_matrix_contrast",
            "severity": "analysis_skill_needed",
            "recommendation": "Generate a reviewed skill for chunked gene-expression extraction and per-cell-type W-vs-B differential testing.",
        })
    spatial = custom.get("spatial_cell_interaction") or {}
    if spatial.get("warnings"):
        gaps.append({
            "gap": "spatial_neighbor_graph",
            "severity": "analysis_skill_needed",
            "recommendation": "Generate a reviewed skill for Stereo coordinate neighbor graphs and cell-type interaction statistics.",
        })

    generated_skill_proposals = [
        {
            "name": "scatac_peak_interval_overlap",
            "description": "Extract scATAC peak coordinates from h5ad var metadata and overlap candidate variants.",
            "activation_policy": "auto_allowed" if trusted_harness_mode else "review_required",
        },
        {
            "name": "chunked_singlecell_differential",
            "description": "Run chunked per-cell-type W-vs-B differential accessibility/expression tests from backed h5ad matrices.",
            "activation_policy": "auto_allowed" if trusted_harness_mode else "review_required",
        },
        {
            "name": "stereo_neighbor_interaction",
            "description": "Build spatial neighbor graphs from Stereo h5ad coordinates and summarize cell-cell interaction shifts.",
            "activation_policy": "auto_allowed" if trusted_harness_mode else "review_required",
        },
    ]
    out = _out_dir(ctx, "06_MultiOmics")
    payload = {
        "n_gaps": len(gaps),
        "gaps": gaps,
        "generated_skill_proposals": generated_skill_proposals,
        "trusted_harness_mode": bool(trusted_harness_mode),
        "policy": "Generated skills are proposals unless trusted_harness_mode and ctx approval permit activation.",
    }
    _write_json(out / "workflow_gap_detector.json", payload)
    _record(ctx, "workflow_gap_detector", payload)
    return payload
