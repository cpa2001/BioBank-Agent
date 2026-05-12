"""Auto-link tool outputs into the Evidence Lattice.

The legacy ``biobank_agent/evidence.py`` defines an
``EvidenceLattice`` of (claim ↔ evidence ↔ refutation) edges, but the
v2 agent rarely populates it: ``fetch_paper`` / ``literature_qa`` /
``read_paper`` returned data into the ReAct loop without writing to
the lattice.

This module provides drop-in hooks the v3 runtime calls when a
relevant tool finishes, so:

1. ``fetch_paper`` / ``read_paper`` / ``literature_qa`` results
   register an ``evidence_type="paper"`` node and link it to the
   active ``query`` claim.
2. ``train_model`` / ``survival`` / ``phewas`` outputs are linked as
   ``evidence_type="statistical_result"`` to the current claim.

The hooks are deliberately defensive — any failure logs at debug and
returns. Evidence Lattice is observability, not load-bearing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


_PAPER_SKILLS = {"fetch_paper", "read_paper", "literature_qa"}
_STAT_SKILLS = {"train_model", "survival", "phewas", "prevalence", "gwas_proxy"}


def _has_action_graph(legacy: Any) -> bool:
    return getattr(getattr(legacy, "memory", None), "action_graph", None) is not None


def link_tool_result_to_evidence(
    legacy: Any,
    *,
    skill: str,
    args: dict[str, Any],
    result: dict[str, Any],
    turn_id: Optional[str] = None,
) -> Optional[str]:
    """Auto-link the result into the Action Graph as evidence.

    Returns the evidence node id created, or ``None`` if nothing was
    persisted (skill not relevant, action graph unavailable, etc.).
    """
    if not _has_action_graph(legacy):
        return None
    if skill not in _PAPER_SKILLS and skill not in _STAT_SKILLS:
        return None

    graph = legacy.memory.action_graph
    query_id = getattr(legacy, "_active_query_id", None)
    try:
        if skill in _PAPER_SKILLS:
            doi = (
                result.get("doi")
                or result.get("identifier")
                or args.get("doi")
                or args.get("paper_id")
                or ""
            )
            ev_id = f"paper:{doi or skill}:{turn_id or ''}"
            graph.upsert_node(
                node_type="paper",
                node_id=ev_id,
                payload={
                    "skill": skill,
                    "title": result.get("title", ""),
                    "authors": result.get("authors", []),
                    "doi": doi,
                    "summary": (result.get("summary") or result.get("abstract") or "")[:600],
                },
                score=0.7,
            )
            if query_id:
                graph.link_nodes(
                    src_type="query",
                    src_id=query_id,
                    dst_type="paper",
                    dst_id=ev_id,
                    relation="cited_evidence",
                    weight=0.7,
                )
            return ev_id
        if skill in _STAT_SKILLS:
            kind = "statistical_result"
            ev_id = f"stat:{skill}:{turn_id or ''}:{hash(str(args)) % 1_000_000}"
            payload = {
                "skill": skill,
                "args": dict(args),
                "key_results": {
                    k: result.get(k)
                    for k in (
                        "auc", "mean_auc", "n_cases", "n_total", "p_value",
                        "log_rank_p", "or", "hr", "summary",
                    )
                    if k in result
                },
            }
            graph.upsert_node(
                node_type=kind,
                node_id=ev_id,
                payload=payload,
                score=0.85,
            )
            if query_id:
                graph.link_nodes(
                    src_type="query",
                    src_id=query_id,
                    dst_type=kind,
                    dst_id=ev_id,
                    relation="produced_statistical_evidence",
                    weight=0.85,
                )
            return ev_id
    except Exception as e:
        logger.debug("evidence link failed for %s: %s", skill, e)
    return None


__all__ = ["link_tool_result_to_evidence"]
