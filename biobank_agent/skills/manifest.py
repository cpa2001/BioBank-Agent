"""Skill-tree manifest: a metadata overlay that tiers skills for lazy LLM exposure.

The registry/engine consult this to inject only the Direct-tier skill schemas into each
turn instead of all ~106, cutting the per-turn token tax and keeping prompt-cache reuse.
This is PURE METADATA — it moves no files and changes no imports. A missing or invalid
``manifest.json`` degrades to "everything Direct" (exactly today's behavior).

Exposure tiers:
  - ``direct``   — always injected into the model's tool list.
  - ``deferred`` — discoverable on demand via the ``skill_search`` tool (default for any
                   unlisted @skill).
  - ``hidden``   — runtime/curator-only; never offered to the model directly.
Native workspace tools (shell/read/write/edit/grep/apply_patch/test/run_job/skill_search/…)
are not @skills and are treated as Direct by the registry, not via this file.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).with_name("manifest.json")

DIRECT = "direct"
DEFERRED = "deferred"
HIDDEN = "hidden"

INTERNAL = "internal"
EXTERNAL = "external"


@functools.lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    try:
        data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("manifest.json is not a JSON object")
        return data
    except FileNotFoundError:
        return {}
    except Exception as exc:  # malformed JSON, etc. — fail safe to "all Direct"
        logger.warning("skill manifest unavailable (%s); treating all skills as Direct", exc)
        return {}


def reload() -> None:
    """Invalidate the cached manifest (call after a create_skill hot-reload)."""
    _load.cache_clear()


def manifest_available() -> bool:
    return bool(_load())


def _direct_set() -> set[str]:
    return set((_load().get("tiers") or {}).get("direct") or [])


def _hidden_set() -> set[str]:
    return set((_load().get("tiers") or {}).get("hidden") or [])


def exposure_of(name: str, *, default: str = DEFERRED) -> str:
    """Tier of a @skill by name. Unlisted -> ``default`` (deferred). No manifest -> direct."""
    if not manifest_available():
        return DIRECT
    if name in _hidden_set():
        return HIDDEN
    if name in _direct_set():
        return DIRECT
    return default


def domain_of(name: str) -> str:
    """Legacy flat domain of a skill. Tree-derived when a tree is present (a skill filed
    into a tree leaf inherits its domain, so the flat view never drifts), falling back to
    the literal ``domains`` map, then ``"other"``."""
    leaf = leaf_of(name)
    if leaf is not None:
        dom = _domain_tag_of(leaf)
        if dom:
            return dom
    for domain, names in (_load().get("domains") or {}).items():
        if name in (names or []):
            return str(domain)
    return "other"


def is_pinned(name: str) -> bool:
    return name in set(_load().get("pinned") or [])


# Runtime overlay for skills ingested this session. Kept separate from the tracked
# manifest.json so a vendored/ingested corpus doesn't dirty version control; rehydrated at
# boot from each corpus's SKILL_PACKAGE.json sidecar.
_external_trust: dict[str, str] = {}


def register_external_trust(names: Iterable[str], value: str = EXTERNAL) -> None:
    """Tag ingested skills as ``external`` at runtime (consulted by ``trust_of`` and thus by
    the curator's auto-promotion exclusion), without writing to the tracked manifest."""
    for n in names:
        if n:
            _external_trust[str(n)] = value


def clear_external_trust() -> None:
    _external_trust.clear()


def trust_of(name: str, *, default: str = INTERNAL) -> str:
    """Trust provenance of a skill: ``internal`` (first-party or locally generated) or
    ``external`` (ingested from a third-party corpus, e.g. a GitHub skill pack). External
    skills never auto-promote into the Direct tier — they stay deferred until a human enables
    the corpus (see ``runtime.curator.recommend_curation``'s ``trust_of`` exclusion). The
    runtime overlay wins over the manifest so an ingested skill is external immediately."""
    if name in _external_trust:
        return _external_trust[name]
    trust = _load().get("trust") or {}
    value = trust.get(name)
    return str(value) if value else default


def category_summaries() -> dict[str, str]:
    """One-line summary per domain, shown to the model so it knows what to skill_search for."""
    return {str(k): str(v) for k, v in (_load().get("summaries") or {}).items()}


def direct_skills() -> set[str]:
    return _direct_set()


def hidden_skills() -> set[str]:
    return _hidden_set()


# ── Hierarchical tree ────────────────────────────────────────
# The ``tree`` block is the source of truth for ``domain_of`` when present; absent or
# malformed, the loader falls back to the flat ``domains`` map (preserving the _load()
# fail-safe). skill_tree.py / the navigate_skill_tree tool read these accessors.

def _tree() -> dict[str, Any]:
    tree = _load().get("tree")
    return tree if isinstance(tree, dict) else {}


def tree_available() -> bool:
    return bool(_tree().get("nodes"))


def tree_nodes() -> dict[str, Any]:
    nodes = _tree().get("nodes")
    return nodes if isinstance(nodes, dict) else {}


def root_nodes() -> list[str]:
    return [str(n) for n in (_tree().get("root") or [])]


def node(node_id: str) -> dict[str, Any]:
    n = tree_nodes().get(node_id)
    return n if isinstance(n, dict) else {}


def children_of(node_id: str) -> list[str]:
    return [str(c) for c in (node(node_id).get("children") or [])]


def node_summary(node_id: str) -> str:
    return str(node(node_id).get("summary") or "")


def skills_in_node(node_id: str) -> list[str]:
    return [str(s) for s in (node(node_id).get("skills") or [])]


def leaf_of(name: str) -> str | None:
    """The tree node whose ``skills`` list contains ``name`` (a skill lives in one node)."""
    for node_id, n in tree_nodes().items():
        if name in (n.get("skills") or []):
            return str(node_id)
    return None


def _domain_tag_of(node_id: str) -> str:
    """The legacy ``domain`` tag of a node, walking up to the nearest tagged ancestor."""
    seen: set[str] = set()
    cur: Any = node_id
    while cur and cur not in seen:
        seen.add(cur)
        n = node(cur)
        dom = n.get("domain")
        if dom:
            return str(dom)
        cur = n.get("parent")
    return ""


def generated_domains() -> dict[str, list[str]]:
    """Reconstruct the flat ``{domain: [skills]}`` map from the tree (back-compat view),
    so a skill filed into a tree leaf appears under its domain with no manual edit."""
    out: dict[str, list[str]] = {}
    for node_id, n in tree_nodes().items():
        dom = _domain_tag_of(node_id) or "other"
        for s in (n.get("skills") or []):
            out.setdefault(dom, []).append(str(s))
    return {k: sorted(v) for k, v in out.items()}


def invalidate_tree_cache() -> None:
    """The tree rides the same ``_load()`` cache; reload() invalidates it."""
    reload()


def validate_manifest(known_skills: Iterable[str] | None = None, *, data: dict[str, Any] | None = None) -> list[str]:
    """Return manifest inconsistencies (empty list = clean):

    - a skill in BOTH the direct and hidden tiers (a contradiction),
    - a skill filed into more than one tree node (``leaf_of`` assumes exactly one),
    - a root or child entry referencing an undefined tree node,
    - and, when ``known_skills`` is supplied, a tier/tree entry for a skill that is not registered.

    Pure; ``data`` overrides the loaded manifest for tests. A missing manifest (the all-Direct
    fail-safe) has nothing to validate. Useful as a /doctor check and a drift regression guard."""
    payload = _load() if data is None else data
    if not payload:
        return []
    issues: list[str] = []
    tiers = payload.get("tiers") or {}
    direct = set(tiers.get("direct") or [])
    hidden = set(tiers.get("hidden") or [])
    for name in sorted(direct & hidden):
        issues.append(f"skill '{name}' is in both the direct and hidden tiers")

    tree = payload.get("tree") or {}
    nodes = tree.get("nodes") if isinstance(tree.get("nodes"), dict) else {}
    filed: dict[str, str] = {}
    for node_id, n in nodes.items():
        for name in ((n or {}).get("skills") or []):
            if name in filed:
                issues.append(f"skill '{name}' is filed into multiple tree nodes ({filed[name]}, {node_id})")
            else:
                filed[name] = node_id
    for root in (tree.get("root") or []):
        if root not in nodes:
            issues.append(f"root node '{root}' is not defined in tree nodes")
    for node_id, n in nodes.items():
        for child in ((n or {}).get("children") or []):
            if child not in nodes:
                issues.append(f"node '{node_id}' references a missing child node '{child}'")

    if known_skills is not None:
        known = set(known_skills)
        for name in sorted((direct | hidden | set(filed)) - known):
            issues.append(f"manifest references unregistered skill '{name}'")
    return issues
