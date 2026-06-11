"""Hierarchical skill-tree navigation + auto-classification.

The skill manifest carries a ``tree`` block grouping skills into a shallow hierarchy.
This module reads it to (1) let the model browse one level at a time (``navigate_tree``),
so per-turn context scales with the depth walked rather than the corpus size, and (2) file
a newly created or ingested skill into the right leaf (``classify_skill``). Both degrade
gracefully when the tree is absent: navigation synthesizes a single level from the flat
domain summaries, and classification falls back to deterministic token overlap.

Pure logic with no runtime coupling: ``classify_skill`` takes an optional ``classifier``
callable (e.g. backed by the council CRITIC role) so this module stays offline-testable.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from biobank_agent.skills import manifest as _m

# classify_skill confidence gate: the winning leaf must reach this token-overlap score AND
# beat the runner-up by this margin, else the skill is left for manual review rather than
# mis-filed into a weakly-matching branch.
_MIN_CLASSIFY_SCORE = 2
_MIN_CLASSIFY_MARGIN = 1
UNCLASSIFIED = "pending_classification"


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if len(t) > 1}


def navigate_tree(
    node_id: Optional[str] = None,
    *,
    describe: Optional[Callable[[str], str]] = None,
) -> dict[str, Any]:
    """Return one level of the tree. ``node_id=None`` yields the root level; for a node,
    its child nodes (id + summary) and its leaf skills (name + description). ``describe`` is
    an optional name->description lookup (e.g. the registry). Degrades to a single synthetic
    level built from the flat domain summaries when no tree is present."""
    if not _m.tree_available():
        cats = _m.category_summaries()
        return {
            "node_id": None,
            "summary": "",
            "children": [{"id": d, "summary": s} for d, s in sorted(cats.items())],
            "skills": [],
            "tree_available": False,
        }
    if node_id is None:
        return {
            "node_id": None,
            "summary": "",
            "children": [{"id": r, "summary": _m.node_summary(r)} for r in _m.root_nodes()],
            "skills": [],
            "tree_available": True,
        }
    skills: list[dict[str, str]] = []
    for s in _m.skills_in_node(node_id):
        desc = ""
        if describe is not None:
            try:
                desc = describe(s) or ""
            except Exception:
                desc = ""
        skills.append({"name": s, "description": desc})
    return {
        "node_id": node_id,
        "summary": _m.node_summary(node_id),
        "children": [{"id": c, "summary": _m.node_summary(c)} for c in _m.children_of(node_id)],
        "skills": skills,
        "tree_available": True,
    }


def tree_leaves() -> list[str]:
    """Node ids with no children (the classifiable leaves)."""
    return [nid for nid, n in _m.tree_nodes().items() if not (n.get("children") or [])]


def subtree_skills(node_id: str) -> list[str]:
    """All skills under ``node_id`` (its own + every descendant's)."""
    out: list[str] = []
    seen: set[str] = set()
    stack = [node_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        out.extend(_m.skills_in_node(cur))
        stack.extend(_m.children_of(cur))
    return out


def _leaf_score(name: str, description: str, node_id: str) -> int:
    """Token overlap of the skill's name+description against a leaf's vocabulary
    (node id + summary + the names of skills already filed there)."""
    q = _tokens(name) | _tokens(name.replace("_", " ")) | _tokens(description)
    doc = _tokens(node_id.replace("_", " ")) | _tokens(_m.node_summary(node_id))
    for s in _m.skills_in_node(node_id):
        doc |= _tokens(s.replace("_", " "))
    return len(q & doc)


def classify_skill(
    name: str,
    description: str,
    *,
    classifier: Optional[Callable[[str, str, list[dict[str, str]]], str]] = None,
    default: str = UNCLASSIFIED,
) -> str:
    """Pick the best tree leaf for a new skill. An injected ``classifier`` (e.g. backed by
    the council CRITIC role) decides when provided; otherwise a deterministic token-overlap
    vote over the leaves runs, requiring a minimum score AND a margin over the runner-up,
    else returning ``default`` for manual review. Always offline-safe."""
    leaves = tree_leaves()
    if not leaves:
        return default
    if classifier is not None:
        try:
            candidates = [{"id": nid, "summary": _m.node_summary(nid)} for nid in leaves]
            choice = classifier(name, description, candidates)
            if choice in leaves:
                return choice
        except Exception:
            pass  # fall through to the deterministic vote
    scored = sorted(
        ((_leaf_score(name, description, nid), nid) for nid in leaves),
        key=lambda it: (it[0], it[1]),
        reverse=True,
    )
    best_score, best = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0
    if best_score >= _MIN_CLASSIFY_SCORE and (best_score - runner) >= _MIN_CLASSIFY_MARGIN:
        return best
    return default


__all__ = ["navigate_tree", "tree_leaves", "subtree_skills", "classify_skill", "UNCLASSIFIED"]
