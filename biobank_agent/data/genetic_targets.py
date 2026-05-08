"""Rare-variant burden to therapeutic target hypothesis primitives.

Design source:
    - GeneBass-like rare-variant burden summary statistics.
    - Concept: direct genetic evidence drives target ranking, while external
      annotations remain context for downstream triage.
    - Reference doc: docs/related_works/GENETIC_TARGET_PRIORITIZATION.md
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_BURDEN_COLUMNS = ["gene", "phenotype", "annotation", "beta", "p_value"]
P_VALUE_FLOOR = 1e-300

PHENOTYPE_ALIASES = {
    "bmi": ["body mass index", "weight adjusted for height", "obesity"],
    "body mass index": ["bmi", "weight adjusted for height", "obesity"],
    "ldl": ["ldl cholesterol", "low density lipoprotein"],
    "ldl cholesterol": ["ldl", "low density lipoprotein"],
    "t2d": ["type 2 diabetes", "diabetes mellitus type 2"],
    "type 2 diabetes": ["t2d", "diabetes mellitus type 2"],
}

GENETIC_TARGET_CAVEATS = [
    "Uses caller-supplied summary-level rare-variant burden statistics; it does not rerun genotype QC, burden tests, or sample filters.",
    "Loss-of-function sign gives a first-pass therapeutic direction, not proof of drug efficacy or safety.",
    "Exploratory hits require orthogonal validation such as replication, colocalization, model-organism evidence, or target-trial evidence.",
    "UK Biobank exome discovery can be ancestry-skewed; direction and prioritization should be checked in diverse cohorts.",
]


@dataclass(frozen=True)
class BurdenAssociation:
    """One gene-phenotype-annotation burden association."""

    source_index: int
    gene: str
    phenotype: str
    annotation: str
    beta: float
    p_value: float
    raw: dict[str, Any] = field(default_factory=dict)
    pathways: list[str] = field(default_factory=list)

    @property
    def therapeutic_direction(self) -> str:
        return therapeutic_direction_from_beta(self.beta)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["therapeutic_direction"] = self.therapeutic_direction
        return payload


def load_burden_rows(
    *,
    burden_rows: Any = None,
    burden_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load GeneBass-like burden rows from an in-memory payload and/or file.

    Supported files are CSV, TSV, JSON list/object, and JSONL. In-memory rows may
    be a list of dictionaries or a JSON string containing that list.
    """

    rows: list[dict[str, Any]] = []
    if isinstance(burden_rows, str):
        if burden_rows.strip():
            parsed = _normalize_rows_input(burden_rows)
            rows.extend(parsed)
    elif burden_rows is not None:
        if not (isinstance(burden_rows, list) and len(burden_rows) == 0):
            parsed = _normalize_rows_input(burden_rows)
            rows.extend(parsed)

    if burden_path:
        path = Path(burden_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Burden statistics file not found: {path}")
        rows.extend(_read_rows_file(path))

    return rows


def _empty_target_result(result_base: dict[str, Any], *, recommended_next_steps: list[str]) -> dict[str, Any]:
    return {
        **result_base,
        "n_discovery_rows": 0,
        "n_discovery_genes": 0,
        "targets": [],
        "mechanism_clusters": [],
        "priority_counts": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
        "direction_counts": {"activate": 0, "inhibit": 0, "uncertain": 0},
        "recommended_next_steps": recommended_next_steps,
    }


def _multiple_testing_scope(
    *,
    family_size: int | None,
    n_selected: int,
) -> str:
    if family_size:
        return "filtered_with_family_size" if int(family_size) > n_selected else "full_family"
    return "provided_rows_only"


def _normalize_family_size(
    family_size: int | None,
    *,
    n_selected: int,
) -> tuple[int | None, list[str]]:
    if family_size is None:
        return None, []
    try:
        value = int(family_size)
    except (TypeError, ValueError):
        return None, ["family_size was not an integer; using supplied rows as the multiple-testing scope."]
    if value <= 0:
        return None, ["family_size must be positive; using supplied rows as the multiple-testing scope."]
    if value < n_selected:
        return None, [
            f"family_size={value} is smaller than n_matched_rows={n_selected}; "
            "using supplied rows as the multiple-testing scope."
        ]
    return value, []


def _tier_interpretation(scope: str) -> str:
    if scope == "full_family":
        return "Bonferroni and BY-FDR tiers are computed against the supplied full test family."
    if scope == "filtered_with_family_size":
        return (
            "Bonferroni uses the declared full family size; BY-FDR q-values are conservative "
            "approximations because only filtered rows were supplied."
        )
    return (
        "Bonferroni and BY-FDR tiers are limited to the supplied rows; provide family_size "
        "or the full phenotype-family table for publication-grade tiering."
    )


def _base_result(
    *,
    phenotype: str,
    burden_rows: list[dict[str, Any]],
    associations: list[BurdenAssociation],
    selected: list[BurdenAssociation],
    row_errors: list[str],
    warnings: list[str],
    status: str,
    discovery_p: float,
    fdr_alpha: float,
    n_family: int,
    prefiltered: bool,
    multiple_testing_scope: str,
) -> dict[str, Any]:
    return {
        "skill": "genetic_target_hypothesis",
        "phenotype": phenotype,
        "status": status,
        "source_data_type": "rare_variant_burden_summary_statistics",
        "design_principle": "genetics_drives_ranking_annotations_are_context",
        "required_columns": list(REQUIRED_BURDEN_COLUMNS),
        "n_input_rows": len(burden_rows),
        "n_parsed_rows": len(associations),
        "n_matched_rows": len(selected),
        "n_family_tests": n_family,
        "prefiltered": bool(prefiltered),
        "multiple_testing_scope": multiple_testing_scope,
        "tier_interpretation": _tier_interpretation(multiple_testing_scope),
        "discovery_threshold": discovery_p,
        "fdr_alpha": fdr_alpha,
        "bonferroni_threshold": fdr_alpha / max(n_family, 1),
        "row_errors": row_errors[:20],
        "warnings": warnings,
        "caveats": list(GENETIC_TARGET_CAVEATS),
        "sources": [
            {
                "title": "Karczewski et al. GeneBass rare-variant burden testing",
                "doi": "10.1016/j.xgen.2022.100168",
            },
        ],
    }


def _normalize_rows_input(burden_rows: Any) -> list[dict[str, Any]]:
    if isinstance(burden_rows, str):
        parsed = json.loads(burden_rows)
    else:
        parsed = burden_rows
    if isinstance(parsed, dict):
        parsed = parsed.get("rows", parsed.get("data", []))
    if not isinstance(parsed, list):
        raise ValueError("burden_rows must be a list of objects or JSON list")
    rows = []
    for row in parsed:
        if not isinstance(row, dict):
            raise ValueError("each burden row must be an object")
        rows.append(dict(row))
    return rows


def build_genetic_target_hypotheses(
    *,
    phenotype: str,
    burden_rows: list[dict[str, Any]],
    discovery_p: float = 1e-4,
    fdr_alpha: float = 0.05,
    family_size: int | None = None,
    top_n: int = 20,
    prefiltered: bool = False,
) -> dict[str, Any]:
    """Rank therapeutic target hypotheses from rare-variant burden rows."""

    phenotype = (phenotype or "").strip()
    if not phenotype:
        raise ValueError("phenotype is required")
    discovery_p = _clamp_float(discovery_p, 1e-300, 1.0, default=1e-4)
    fdr_alpha = _clamp_float(fdr_alpha, 1e-12, 1.0, default=0.05)
    top_n = max(1, min(int(top_n or 20), 200))

    associations, row_errors = _coerce_burden_associations(burden_rows)
    if not associations:
        result_base = _base_result(
            phenotype=phenotype,
            burden_rows=burden_rows,
            associations=[],
            selected=[],
            row_errors=row_errors,
            warnings=["No valid burden association rows were parsed; refusing to rank targets."],
            status="INVALID_INPUT",
            discovery_p=discovery_p,
            fdr_alpha=fdr_alpha,
            n_family=1,
            prefiltered=prefiltered,
            multiple_testing_scope="provided_rows_only",
        )
        return _empty_target_result(
            result_base,
            recommended_next_steps=[
                "Provide at least one valid row with gene, annotation, beta, and p_value.",
                "Check column names and numeric parsing before rerunning target prioritization.",
            ],
        )

    selected, warnings, block_status = _select_phenotype_rows(
        phenotype,
        associations,
        prefiltered=prefiltered,
    )
    normalized_family_size, family_warnings = _normalize_family_size(family_size, n_selected=len(selected))
    warnings = warnings + family_warnings
    n_family = max(int(normalized_family_size or 0), len(selected), 1)
    multiple_testing_scope = _multiple_testing_scope(family_size=normalized_family_size, n_selected=len(selected))

    if block_status:
        result_base = _base_result(
            phenotype=phenotype,
            burden_rows=burden_rows,
            associations=associations,
            selected=selected,
            row_errors=row_errors,
            warnings=warnings,
            status=block_status,
            discovery_p=discovery_p,
            fdr_alpha=fdr_alpha,
            n_family=n_family,
            prefiltered=prefiltered,
            multiple_testing_scope=multiple_testing_scope,
        )
        return _empty_target_result(
            result_base,
            recommended_next_steps=[
                "Confirm that the selected phenotype name matches the burden table.",
                "Set prefiltered=true only when the supplied rows have already been filtered to the requested phenotype.",
                "Provide the full phenotype-family burden table when exact multiple-testing tiers are required.",
            ],
        )

    q_values = benjamini_yekutieli([a.p_value for a in selected], n_tests=n_family)
    q_by_index = {a.source_index: q for a, q in zip(selected, q_values)}

    discovery = [a for a in selected if a.p_value <= discovery_p]
    result_base = _base_result(
        phenotype=phenotype,
        burden_rows=burden_rows,
        associations=associations,
        selected=selected,
        row_errors=row_errors,
        warnings=warnings,
        status="PASS" if discovery else "NO_DISCOVERY",
        discovery_p=discovery_p,
        fdr_alpha=fdr_alpha,
        n_family=n_family,
        prefiltered=prefiltered,
        multiple_testing_scope=multiple_testing_scope,
    )

    if not discovery:
        return _empty_target_result(
            result_base,
            recommended_next_steps=[
                "Confirm that the selected phenotype name matches the burden table.",
                "Provide unfiltered GeneBass-like rows if FDR tiers are required.",
                "Relax discovery_p only for exploratory ideation, not confirmatory claims.",
            ],
        )

    targets = _rank_gene_targets(
        discovery=discovery,
        q_by_index=q_by_index,
        bonferroni_threshold=result_base["bonferroni_threshold"],
        fdr_alpha=fdr_alpha,
        top_n=top_n,
    )
    clusters = _mechanism_clusters(discovery)
    priority_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    direction_counts = {"activate": 0, "inhibit": 0, "uncertain": 0}
    for target in targets:
        priority_counts[target["priority"]] = priority_counts.get(target["priority"], 0) + 1
        direction_counts[target["therapeutic_direction"]] = direction_counts.get(target["therapeutic_direction"], 0) + 1

    return {
        **result_base,
        "status": "PASS",
        "n_discovery_rows": len(discovery),
        "n_discovery_genes": len({a.gene for a in discovery}),
        "priority_counts": priority_counts,
        "direction_counts": direction_counts,
        "targets": targets,
        "mechanism_clusters": clusters,
        "recommended_next_steps": [
            "Replicate top targets in ancestry-diverse cohorts before program prioritization.",
            "Run tissue and cell-type expression review for feasibility and safety context.",
            "Check druggability, mouse knockout, clinical-trial, and adverse-phenotype evidence.",
            "Use colocalization or functional validation to separate causal targets from burden artifacts.",
        ],
    }


def benjamini_yekutieli(p_values: list[float], n_tests: int | None = None) -> list[float]:
    """Benjamini-Yekutieli adjusted q-values for dependent test families."""

    if not p_values:
        return []
    observed_n = len(p_values)
    n = max(int(n_tests or observed_n), observed_n, 1)
    harmonic = sum(1.0 / i for i in range(1, n + 1))
    order = sorted(range(observed_n), key=lambda i: p_values[i])
    adjusted = [1.0] * observed_n
    running_min = 1.0
    for rank in range(observed_n, 0, -1):
        idx = order[rank - 1]
        raw = p_values[idx] * n * harmonic / rank
        running_min = min(running_min, raw)
        adjusted[idx] = min(1.0, running_min)
    return adjusted


def therapeutic_direction_from_beta(beta: float) -> str:
    """Infer first-pass therapeutic direction from loss-of-function beta sign."""

    if beta < 0:
        return "inhibit"
    if beta > 0:
        return "activate"
    return "uncertain"


def record_genetic_targets_to_action_graph(memory: Any, result: dict[str, Any]) -> None:
    """Persist target hypotheses into Action Graph when memory is available."""

    if memory is None or not hasattr(memory, "upsert_node"):
        return
    phenotype = str(result.get("phenotype", "phenotype"))
    set_id = f"genetic_target_hypothesis:{_slug(phenotype)}"
    memory.upsert_node(
        "therapeutic_hypothesis_set",
        set_id,
        payload={
            "phenotype": phenotype,
            "n_discovery_genes": result.get("n_discovery_genes", 0),
            "design_principle": result.get("design_principle"),
            "status": result.get("status"),
        },
        score=1.0 if result.get("status") == "PASS" else 0.3,
    )
    memory.upsert_node("phenotype", _slug(phenotype), payload={"label": phenotype}, score=0.7)
    memory.link_nodes(
        "therapeutic_hypothesis_set",
        set_id,
        "phenotype",
        _slug(phenotype),
        relation="prioritizes_for",
        weight=0.8,
    )
    for target in result.get("targets", [])[:50]:
        gene = str(target.get("gene", "")).upper()
        if not gene:
            continue
        target_id = f"{set_id}:{gene}"
        score = float(target.get("score", 0.0))
        memory.upsert_node("gene", gene, payload={"gene": gene}, score=max(score, 0.1))
        memory.upsert_node("therapeutic_target_hypothesis", target_id, payload=target, score=score)
        memory.link_nodes(
            "therapeutic_hypothesis_set",
            set_id,
            "therapeutic_target_hypothesis",
            target_id,
            relation="contains",
            weight=score,
        )
        memory.link_nodes(
            "therapeutic_target_hypothesis",
            target_id,
            "gene",
            gene,
            relation="targets_gene",
            weight=score,
            evidence={"direction": target.get("therapeutic_direction")},
        )
        for evidence_row in target.get("evidence_rows", [])[:10]:
            source_index = evidence_row.get("source_index", "unknown")
            evidence_id = f"{target_id}:burden_row:{source_index}"
            memory.upsert_node(
                "rare_variant_burden_evidence",
                evidence_id,
                payload={
                    **evidence_row,
                    "tier": target.get("tier"),
                    "multiple_testing_scope": result.get("multiple_testing_scope"),
                },
                score=score,
            )
            memory.link_nodes(
                "therapeutic_target_hypothesis",
                target_id,
                "rare_variant_burden_evidence",
                evidence_id,
                relation="supported_by_burden_row",
                weight=score,
                evidence={
                    "tier": target.get("tier"),
                    "multiple_testing_scope": result.get("multiple_testing_scope"),
                },
            )


def write_genetic_target_report(result: dict[str, Any], report_dir: str | Path) -> dict[str, str]:
    """Write Markdown and CSV artifacts for a target hypothesis run."""

    out_dir = Path(report_dir) / "genetic_target_hypotheses"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(str(result.get("phenotype", "phenotype")))
    md_path = out_dir / f"{slug}_targets.md"
    csv_path = out_dir / f"{slug}_targets.csv"

    targets = list(result.get("targets", []) or [])
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "rank",
            "gene",
            "therapeutic_direction",
            "score",
            "priority",
            "tier",
            "best_p_value",
            "best_beta",
            "variant_support",
            "pathway_convergence",
            "known_drugs",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in targets:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    lines = [
        f"# Genetic Target Hypotheses: {result.get('phenotype', '')}",
        "",
        f"**Status:** {result.get('status', '')}",
        f"**Design principle:** {result.get('design_principle', '')}",
        f"**Discovery rows:** {result.get('n_discovery_rows', 0)}",
        f"**Discovery genes:** {result.get('n_discovery_genes', 0)}",
        f"**Bonferroni threshold:** {result.get('bonferroni_threshold', 0):.3e}",
        f"**Multiple-testing scope:** {result.get('multiple_testing_scope', 'unknown')}",
        f"**Tier interpretation:** {result.get('tier_interpretation', '')}",
        "",
        "## Top Targets",
        "",
        "| Rank | Gene | Direction | Score | Priority | Tier | Best P | Evidence |",
        "|------|------|-----------|------:|----------|------|-------:|----------|",
    ]
    for row in targets[:25]:
        lines.append(
            "| {rank} | {gene} | {direction} | {score:.3f} | {priority} | {tier} | {p:.2e} | {evidence} |".format(
                rank=row.get("rank", ""),
                gene=row.get("gene", ""),
                direction=row.get("therapeutic_direction", ""),
                score=float(row.get("score", 0.0)),
                priority=row.get("priority", ""),
                tier=row.get("tier", ""),
                p=float(row.get("best_p_value", 1.0)),
                evidence=row.get("variant_support", ""),
            )
        )
    lines.extend(["", "## Caveats", ""])
    for caveat in result.get("caveats", []):
        lines.append(f"- {caveat}")
    lines.extend(["", "## Recommended Next Steps", ""])
    for step in result.get("recommended_next_steps", []):
        lines.append(f"- {step}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"markdown": str(md_path), "csv": str(csv_path)}


def _rank_gene_targets(
    *,
    discovery: list[BurdenAssociation],
    q_by_index: dict[int, float],
    bonferroni_threshold: float,
    fdr_alpha: float,
    top_n: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[BurdenAssociation]] = {}
    for assoc in discovery:
        grouped.setdefault(assoc.gene, []).append(assoc)

    pathway_counts: dict[str, int] = {}
    gene_pathways: dict[str, set[str]] = {}
    for gene, rows in grouped.items():
        pathways = {p for row in rows for p in row.pathways}
        gene_pathways[gene] = pathways
        for pathway in pathways:
            pathway_counts[pathway] = pathway_counts.get(pathway, 0) + 1
    convergence_raw = {
        gene: sum(max(0, pathway_counts[p] - 1) for p in pathways)
        for gene, pathways in gene_pathways.items()
    }
    max_convergence = max(convergence_raw.values() or [0])

    best_rows = {gene: min(rows, key=lambda row: row.p_value) for gene, rows in grouped.items()}
    max_abs_beta = max(abs(row.beta) for row in best_rows.values()) or 1.0
    denom = max(-math.log10(max(bonferroni_threshold, 1e-300)), -math.log10(1e-4), 1.0)

    targets: list[dict[str, Any]] = []
    for gene, rows in grouped.items():
        best = best_rows[gene]
        best_q = q_by_index.get(best.source_index, 1.0)
        tier = _tier(rows, q_by_index, bonferroni_threshold, fdr_alpha)
        directions = [row.therapeutic_direction for row in rows]
        direction = _consensus_direction(directions, best.therapeutic_direction)
        variant_support = _variant_support(rows)
        p_score = min(1.0, -math.log10(max(best.p_value, 1e-300)) / denom)
        effect_score = min(1.0, abs(best.beta) / max_abs_beta)
        variant_score = _variant_support_score(variant_support)
        pathway_score = 0.0 if max_convergence <= 0 else convergence_raw[gene] / max_convergence
        orthogonal_bonus = _orthogonal_bonus(rows, direction)
        score = min(
            1.0,
            0.35 * p_score
            + 0.25 * effect_score
            + 0.25 * variant_score
            + 0.15 * pathway_score
            + orthogonal_bonus,
        )
        target = {
            "gene": gene,
            "rank": 0,
            "therapeutic_direction": direction,
            "score": round(score, 6),
            "priority": _priority(score),
            "tier": tier,
            "best_p_value": best.p_value,
            "best_q_value": best_q,
            "best_beta": best.beta,
            "best_annotation": best.annotation,
            "variant_support": variant_support,
            "pathway_convergence": round(pathway_score, 6),
            "pathways": sorted(gene_pathways[gene]),
            "known_drugs": _context_value(rows, ["known_drugs", "drugs", "approved_drugs"]),
            "tissue_context": _context_value(rows, ["tissue", "tissue_expression", "gtex_tissue", "top_tissue"]),
            "clinical_trials": _context_value(rows, ["clinical_trials", "trials", "clinicaltrials"]),
            "literature_count": _context_value(rows, ["literature_count", "pubmed_count", "publication_count"]),
            "evidence_rows": [_evidence_row(row, q_by_index) for row in sorted(rows, key=lambda r: r.p_value)],
            "claim_status": "direction_uncertain" if direction == "uncertain" else "hypothesis_generating",
            "score_components": {
                "p_value_strength": round(p_score, 6),
                "effect_size": round(effect_score, 6),
                "variant_class_support": round(variant_score, 6),
                "pathway_convergence": round(pathway_score, 6),
                "orthogonal_bonus": round(orthogonal_bonus, 6),
            },
        }
        targets.append(target)

    targets.sort(key=lambda row: (-row["score"], row["best_p_value"], row["gene"]))
    for rank, target in enumerate(targets, 1):
        target["rank"] = rank
    return targets[:top_n]


def _coerce_burden_associations(rows: list[dict[str, Any]]) -> tuple[list[BurdenAssociation], list[str]]:
    associations: list[BurdenAssociation] = []
    errors: list[str] = []
    for idx, row in enumerate(rows):
        try:
            gene = str(_pick(row, ["gene", "gene_symbol", "symbol", "gene_name"]) or "").strip().upper()
            phenotype = str(_pick(row, ["phenotype", "trait", "description", "phenotype_description"]) or "").strip()
            annotation = normalize_annotation(_pick(row, ["annotation", "variant_class", "consequence", "mask"]))
            beta = _as_float(_pick(row, ["beta", "effect", "estimate"]))
            p_value = _as_float(_pick(row, ["p_value", "pvalue", "pval", "p", "p.value"]))
            if not gene:
                raise ValueError("missing gene")
            if beta is None:
                raise ValueError("missing beta")
            if p_value is None or not (0 <= p_value <= 1):
                raise ValueError("invalid p_value")
            if p_value == 0:
                p_value = P_VALUE_FLOOR
            associations.append(
                BurdenAssociation(
                    source_index=idx,
                    gene=gene,
                    phenotype=phenotype,
                    annotation=annotation,
                    beta=float(beta),
                    p_value=float(p_value),
                    raw=dict(row),
                    pathways=_parse_pathways(_pick(row, ["pathways", "reactome_pathways", "pathway", "mechanisms"])),
                )
            )
        except Exception as exc:
            errors.append(f"row {idx}: {exc}")
    return associations, errors


def normalize_annotation(value: Any) -> str:
    raw = str(value or "").strip()
    lower = raw.lower()
    if "missense" in lower and ("lc" in lower or "low" in lower):
        return "missense|LC"
    if "plof" in lower or "lof" in lower or "loss_of_function" in lower:
        return "pLoF"
    if "missense" in lower:
        return "missense"
    return raw or "unknown"


def _select_phenotype_rows(
    phenotype: str,
    associations: list[BurdenAssociation],
    *,
    prefiltered: bool = False,
) -> tuple[list[BurdenAssociation], list[str], str | None]:
    with_phenotype = [a for a in associations if a.phenotype]
    if not with_phenotype:
        if prefiltered:
            return (
                associations,
                ["No phenotype column was present; caller explicitly declared rows pre-filtered to the requested phenotype."],
                None,
            )
        return (
            [],
            ["No phenotype column was present; set prefiltered=true only if rows were already filtered to the requested phenotype."],
            "NEEDS_PREFILTERED_DECLARATION",
        )
    target = _norm_text(phenotype)
    targets = {target, *(_norm_text(alias) for alias in PHENOTYPE_ALIASES.get(target, []))}
    selected = [
        a for a in associations
        if any(t in _norm_text(a.phenotype) or _norm_text(a.phenotype) in t for t in targets)
    ]
    if selected:
        return selected, [], None
    if prefiltered:
        return (
            associations,
            ["No phenotype-name match found; caller explicitly declared rows pre-filtered to the requested phenotype."],
            None,
        )
    return (
        [],
        ["No phenotype-name match found in a labelled burden table; refusing to rank unrelated phenotypes."],
        "NO_MATCH",
    )


def _tier(
    rows: list[BurdenAssociation],
    q_by_index: dict[int, float],
    bonferroni_threshold: float,
    fdr_alpha: float,
) -> str:
    if any(row.p_value <= bonferroni_threshold for row in rows):
        return "Bonferroni-significant"
    if any(q_by_index.get(row.source_index, 1.0) <= fdr_alpha for row in rows):
        return "BY-FDR-supported"
    return "exploratory"


def _consensus_direction(directions: list[str], fallback: str) -> str:
    usable = [d for d in directions if d in {"activate", "inhibit"}]
    if not usable:
        return fallback
    if len(set(usable)) == 1:
        return usable[0]
    return fallback


def _variant_support(rows: list[BurdenAssociation]) -> str:
    classes = {row.annotation for row in rows}
    directions_by_annotation: dict[str, set[str]] = {}
    for row in rows:
        if row.therapeutic_direction in {"activate", "inhibit"}:
            directions_by_annotation.setdefault(row.annotation, set()).add(row.therapeutic_direction)
    has_plof = "pLoF" in classes
    has_missense_lc = "missense|LC" in classes
    if has_plof and has_missense_lc:
        combined = directions_by_annotation.get("pLoF", set()) | directions_by_annotation.get("missense|LC", set())
        if len(combined) == 1:
            return "pLoF+missense|LC concordant"
        return "pLoF+missense|LC discordant"
    if has_plof:
        return "pLoF only"
    if has_missense_lc:
        return "missense|LC only"
    return ", ".join(sorted(classes)) or "unknown"


def _variant_support_score(label: str) -> float:
    if "concordant" in label:
        return 1.0
    if "discordant" in label:
        return 0.55
    if label == "pLoF only":
        return 0.75
    if label == "missense|LC only":
        return 0.60
    return 0.45


def _orthogonal_bonus(rows: list[BurdenAssociation], direction: str) -> float:
    for row in rows:
        for key in ["open_targets_direction", "independent_direction", "external_direction"]:
            observed = _direction_value(_pick(row.raw, [key]))
            if observed and observed == direction:
                return 0.05
        agrees = _pick(row.raw, ["open_targets_direction_agrees", "independent_direction_agrees"])
        if _truthy(agrees):
            return 0.05
    return 0.0


def _mechanism_clusters(discovery: list[BurdenAssociation]) -> list[dict[str, Any]]:
    clusters: dict[str, set[str]] = {}
    for row in discovery:
        for pathway in row.pathways:
            clusters.setdefault(pathway, set()).add(row.gene)
    result = [
        {"pathway": pathway, "genes": sorted(genes), "n_genes": len(genes)}
        for pathway, genes in clusters.items()
        if len(genes) > 1
    ]
    result.sort(key=lambda row: (-row["n_genes"], row["pathway"]))
    return result[:25]


def _evidence_row(row: BurdenAssociation, q_by_index: dict[int, float]) -> dict[str, Any]:
    return {
        "source_index": row.source_index,
        "gene": row.gene,
        "phenotype": row.phenotype,
        "annotation": row.annotation,
        "beta": row.beta,
        "p_value": row.p_value,
        "q_by": q_by_index.get(row.source_index, 1.0),
        "therapeutic_direction": row.therapeutic_direction,
        "pathways": list(row.pathways),
    }


def _context_value(rows: list[BurdenAssociation], keys: list[str]) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = _pick(row.raw, keys)
        for part in _split_values(value):
            if part and part not in seen:
                seen.add(part)
                values.append(part)
    return "; ".join(values[:8])


def _parse_rows_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, str):
        parsed = json.loads(payload)
    else:
        parsed = payload
    if isinstance(parsed, dict):
        parsed = parsed.get("rows", parsed.get("data", []))
    if not isinstance(parsed, list):
        raise ValueError("burden_rows must be a list of objects or JSON list")
    rows = []
    for row in parsed:
        if not isinstance(row, dict):
            raise ValueError("each burden row must be an object")
        rows.append(dict(row))
    return rows


def _read_rows_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8") as fh:
            return [dict(row) for row in csv.DictReader(fh, delimiter=delimiter)]
    if suffix == ".json":
        return _parse_rows_payload(path.read_text(encoding="utf-8"))
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL burden rows must be objects")
                rows.append(value)
        return rows
    raise ValueError("Unsupported burden file format. Use CSV, TSV, JSON, or JSONL.")


def _pick(row: dict[str, Any], names: list[str]) -> Any:
    normalized = {_norm_key(k): v for k, v in row.items()}
    for name in names:
        key = _norm_key(name)
        if key in normalized:
            return normalized[key]
    return None


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _clamp_float(value: Any, low: float, high: float, *, default: float) -> float:
    number = _as_float(value)
    if number is None:
        return default
    return min(max(float(number), low), high)


def _parse_pathways(value: Any) -> list[str]:
    return _split_values(value)


def _split_values(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        raw_parts = [str(v) for v in value]
    else:
        raw_parts = re.split(r"[;,|]", str(value))
    result = []
    seen = set()
    for part in raw_parts:
        clean = " ".join(part.strip().split())
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _direction_value(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if "inhibit" in text or "inhibition" in text or "decrease" in text:
        return "inhibit"
    if "activate" in text or "activation" in text or "increase" in text or "agon" in text:
        return "activate"
    return None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "agree", "agrees"}


def _priority(score: float) -> str:
    if score > 0.65:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return slug or "phenotype"
