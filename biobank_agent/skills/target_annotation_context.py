"""Biobank target annotation context skill.

Design source:
    - Official external target annotation APIs.
    - Concept: source-attributed context for biobank target lists.
    - Reference doc: docs/related_works/TARGET_ANNOTATION_ENRICHMENT.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from biobank_agent.data.target_context import (
    DEFAULT_ANNOTATION_SOURCES,
    build_target_annotation_context,
    record_target_annotations_to_action_graph,
    write_target_annotation_report,
)
from biobank_agent.registry import skill


@skill(
    name="target_annotation_context",
    description=(
        "Annotate biobank target genes with translational context from Open Targets, "
        "UniProt, GTEx, ClinicalTrials.gov and optional CELLxGENE snapshots. "
        "Annotations are context only and do not change genetic-evidence ranking."
    ),
    parameters={
        "targets": {
            "type": "array",
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "object", "additionalProperties": True},
                ]
            },
            "description": "Target genes or rows with at least gene. Optional row keys: ensembl_id, uniprot_id, score, rank.",
            "default": [],
        },
        "phenotype": {
            "type": "string",
            "description": "Biobank phenotype or endpoint being interpreted.",
            "default": "",
        },
        "disease_id": {
            "type": "string",
            "description": "Optional Open Targets disease ID or disease name filter.",
            "default": "",
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Annotation sources to use. Supported: opentargets, uniprot, gtex, "
                "clinicaltrials, cellxgene."
            ),
            "default": DEFAULT_ANNOTATION_SOURCES,
        },
        "source_mode": {
            "type": "string",
            "description": "cache_first, cache_only, or refresh.",
            "default": "cache_first",
        },
        "cellxgene_snapshot_path": {
            "type": "string",
            "description": "Optional local CSV/TSV/JSON snapshot with gene, cell_type, tissue and expression columns.",
            "default": "",
        },
        "top_n": {
            "type": "integer",
            "description": "Maximum targets and per-source rows to return.",
            "default": 5,
        },
        "write_report": {
            "type": "boolean",
            "description": "Write companion Markdown and JSON artifacts into the report directory.",
            "default": True,
        },
    },
    required=["targets"],
)
def target_annotation_context(
    targets: Any,
    phenotype: str = "",
    disease_id: str = "",
    sources: Any = None,
    source_mode: str = "cache_first",
    cellxgene_snapshot_path: str = "",
    top_n: int = 5,
    write_report: bool = True,
    *,
    ctx=None,
) -> dict:
    """Add source-attributed external context to biobank target genes."""

    cache_dir = None
    if ctx is not None and hasattr(ctx, "report_dir"):
        cache_dir = Path(ctx.report_dir) / "cache" / "target_annotations"

    result = build_target_annotation_context(
        targets=targets,
        phenotype=phenotype,
        disease_id=disease_id,
        sources=sources,
        source_mode=source_mode,
        cache_dir=cache_dir,
        cellxgene_snapshot_path=cellxgene_snapshot_path or None,
        top_n=top_n,
    )

    if ctx is not None:
        record_target_annotations_to_action_graph(getattr(ctx, "memory", None), result)
        if write_report and hasattr(ctx, "report_dir"):
            try:
                result["artifacts"] = write_target_annotation_report(result, ctx.report_dir)
            except Exception as exc:
                result.setdefault("warnings", []).append(f"Report artifact write failed: {exc}")

    return result
