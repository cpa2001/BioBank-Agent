"""Genetic target hypothesis skill.

Design source:
    - GeneBass-like rare-variant burden summary statistics.
    - Concept: agent-callable genetics-first target prioritization.
    - Reference doc: docs/related_works/GENETIC_TARGET_PRIORITIZATION.md
"""

from __future__ import annotations

from typing import Any

from biobank_agent.data.genetic_targets import (
    GENETIC_TARGET_CAVEATS,
    REQUIRED_BURDEN_COLUMNS,
    build_genetic_target_hypotheses,
    load_burden_rows,
    record_genetic_targets_to_action_graph,
    write_genetic_target_report,
)
from biobank_agent.registry import skill


@skill(
    name="genetic_target_hypothesis",
    description=(
        "Rank therapeutic target hypotheses from GeneBass-like rare-variant burden "
        "summary statistics. Genetics drives the ranking; pathway, drug, tissue, "
        "literature, and trial annotations are kept as context for triage."
    ),
    parameters={
        "phenotype": {
            "type": "string",
            "description": "Phenotype or phenotype family to prioritize, e.g. BMI, LDL cholesterol, T2D.",
        },
        "burden_path": {
            "type": "string",
            "description": "Optional local CSV/TSV/JSON/JSONL file with GeneBass-like burden rows.",
            "default": "",
        },
        "burden_rows": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
            },
            "description": (
                "Optional in-memory burden rows. Required columns: gene, phenotype, "
                "annotation, beta, p_value. Optional columns include pathways, drugs, "
                "tissue_context, clinical_trials, and independent_direction."
            ),
            "default": [],
        },
        "discovery_p": {
            "type": "number",
            "description": "Raw-P discovery threshold for candidate genes.",
            "default": 1e-4,
        },
        "fdr_alpha": {
            "type": "number",
            "description": "Family-wise alpha used for Bonferroni and BY-FDR tiers.",
            "default": 0.05,
        },
        "family_size": {
            "type": "integer",
            "description": (
                "Number of tests in the full phenotype family. Use the full GeneBass "
                "gene-annotation test count when rows are pre-filtered."
            ),
            "default": 0,
        },
        "top_n": {
            "type": "integer",
            "description": "Maximum number of ranked targets to return.",
            "default": 20,
        },
        "prefiltered": {
            "type": "boolean",
            "description": (
                "Set true only when supplied rows were already filtered to the requested "
                "phenotype and may lack matching phenotype labels."
            ),
            "default": False,
        },
        "write_report": {
            "type": "boolean",
            "description": "Write companion Markdown and CSV artifacts into the report directory.",
            "default": True,
        },
    },
    required=["phenotype"],
)
def genetic_target_hypothesis(
    phenotype: str,
    burden_path: str = "",
    burden_rows: Any = None,
    discovery_p: float = 1e-4,
    fdr_alpha: float = 0.05,
    family_size: int = 0,
    top_n: int = 20,
    prefiltered: bool = False,
    write_report: bool = True,
    *,
    ctx=None,
) -> dict:
    """Build genetics-first target hypotheses from burden summary statistics."""

    rows = load_burden_rows(burden_rows=burden_rows, burden_path=burden_path or None)
    if not rows:
        return {
            "skill": "genetic_target_hypothesis",
            "phenotype": phenotype,
            "status": "NEEDS_INPUT",
            "message": "Provide GeneBass-like rare-variant burden rows or a local burden_path.",
            "required_columns": list(REQUIRED_BURDEN_COLUMNS),
            "optional_columns": [
                "pathways",
                "known_drugs",
                "tissue_context",
                "clinical_trials",
                "independent_direction",
            ],
            "design_principle": "genetics_drives_ranking_annotations_are_context",
            "caveats": list(GENETIC_TARGET_CAVEATS),
        }

    result = build_genetic_target_hypotheses(
        phenotype=phenotype,
        burden_rows=rows,
        discovery_p=discovery_p,
        fdr_alpha=fdr_alpha,
        family_size=family_size or None,
        top_n=top_n,
        prefiltered=prefiltered,
    )

    if ctx is not None:
        record_genetic_targets_to_action_graph(getattr(ctx, "memory", None), result)
        if write_report and hasattr(ctx, "report_dir"):
            try:
                result["artifacts"] = write_genetic_target_report(result, ctx.report_dir)
            except Exception as exc:
                result.setdefault("warnings", []).append(f"Report artifact write failed: {exc}")

    return result
