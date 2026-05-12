"""Paper replicator + paper-to-StudySpec extractor.

The full v3 ROADMAP target is PDF-to-StudySpec-to-plan-to-figure-diff.
This module implements the half-automatic production path: extract design /
cohort / method hints from a paper, propose a schema-safe UKB replication
plan, write review artifacts, and require user approval before execution.
"""
from .paper_replicator import (
    PaperReplicator,
    PaperReplicationOutcome,
    extract_study_design,
    render_replication_checklist,
)

__all__ = [
    "PaperReplicator",
    "PaperReplicationOutcome",
    "extract_study_design",
    "render_replication_checklist",
]
