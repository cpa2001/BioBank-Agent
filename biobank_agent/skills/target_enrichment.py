"""Biobank target gene-set enrichment skill.

Design source:
    - GSEApy and local GMT gene-set workflows.
    - Concept: enrichment as biobank target-list context, not rank evidence.
    - Reference doc: docs/related_works/TARGET_ANNOTATION_ENRICHMENT.md
"""

from __future__ import annotations

from typing import Any

from biobank_agent.data.target_context import (
    build_target_enrichment,
    record_target_enrichment_to_action_graph,
    write_target_enrichment_report,
)
from biobank_agent.registry import skill


@skill(
    name="target_enrichment",
    description=(
        "Run local gene-set enrichment for biobank target lists. Uses a supplied "
        "GMT file by default; optional gseapy support is used only when available."
    ),
    parameters={
        "gene_list": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Target genes to test for over-representation.",
            "default": [],
        },
        "ranked_genes": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
            "description": "Optional ranked rows with gene and score for prerank/gseapy workflows.",
            "default": [],
        },
        "phenotype": {
            "type": "string",
            "description": "Biobank phenotype or endpoint being interpreted.",
            "default": "",
        },
        "gene_sets_path": {
            "type": "string",
            "description": "Local GMT file. Required for offline ORA and recommended for reproducibility.",
            "default": "",
        },
        "method": {
            "type": "string",
            "description": "ora, prerank, gseapy, or enrichr. Missing gseapy falls back to local ORA.",
            "default": "ora",
        },
        "universe_genes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Explicit test universe. If omitted, the union of GMT genes is used with a warning.",
            "default": [],
        },
        "fdr_alpha": {
            "type": "number",
            "description": "FDR threshold for marking enrichment terms significant.",
            "default": 0.05,
        },
        "top_n": {
            "type": "integer",
            "description": "Maximum enrichment terms to return.",
            "default": 20,
        },
        "write_report": {
            "type": "boolean",
            "description": "Write companion Markdown and CSV artifacts into the report directory.",
            "default": True,
        },
    },
    required=[],
)
def target_enrichment(
    gene_list: Any = None,
    ranked_genes: Any = None,
    phenotype: str = "",
    gene_sets_path: str = "",
    method: str = "ora",
    universe_genes: Any = None,
    fdr_alpha: float = 0.05,
    top_n: int = 20,
    write_report: bool = True,
    *,
    ctx=None,
) -> dict:
    """Run enrichment for a biobank target list."""

    result = build_target_enrichment(
        gene_list=gene_list,
        ranked_genes=ranked_genes,
        phenotype=phenotype,
        gene_sets_path=gene_sets_path or None,
        method=method,
        universe_genes=universe_genes,
        fdr_alpha=fdr_alpha,
        top_n=top_n,
    )

    if ctx is not None:
        record_target_enrichment_to_action_graph(getattr(ctx, "memory", None), result)
        if write_report and hasattr(ctx, "report_dir"):
            try:
                result["artifacts"] = write_target_enrichment_report(result, ctx.report_dir)
            except Exception as exc:
                result.setdefault("warnings", []).append(f"Report artifact write failed: {exc}")

    return result
