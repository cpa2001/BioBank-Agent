"""v3 tool registry built on ToolHandler.

Wraps the legacy ``biobank_agent.registry.SkillRegistry`` so all
existing skills appear in the v3 registry as
``LegacySkillToolHandler`` instances. New tools (and MCP-discovered
tools) register here directly.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

from biobank_agent.registry import SkillRegistry, get_registry as _get_legacy_registry

from .native import build_native_tools

from .protocol import (
    ActionClass,
    Capability,
    LegacySkillToolHandler,
    SafetyClass,
    ToolContext,
    ToolHandler,
    ToolSpec,
)

logger = logging.getLogger(__name__)


# Capability inference for legacy @skill functions. Until each skill
# is annotated with its real capability set we use
# conservative defaults plus per-name overrides for the obviously
# sensitive ones.
_DEFAULT_LEGACY_CAPS = frozenset({Capability.READ_DATA, Capability.WRITE_REPORTS})

_LEGACY_CAP_OVERRIDES: dict[str, frozenset[Capability]] = {
    # Network-touching skills
    "fetch_paper": frozenset({Capability.READ_DATA, Capability.NETWORK}),
    "read_paper": frozenset({Capability.READ_DATA, Capability.NETWORK, Capability.WRITE_REPORTS}),
    "literature_qa": frozenset({Capability.READ_DATA, Capability.NETWORK, Capability.WRITE_REPORTS}),
    "web_search": frozenset({Capability.NETWORK}),
    "web_fetch": frozenset({Capability.NETWORK}),
    # Memory-mutating skills
    "create_skill": frozenset(
        {Capability.WRITE_REPORTS, Capability.MUTATE_MEMORY}
    ),
    "record_macro": frozenset({Capability.MUTATE_MEMORY}),
    # External coding-agent delegation: spawns a third-party CLI and applies its code edits.
    "delegate_to_coding_agent": frozenset(
        {Capability.SHELL_EXEC, Capability.CALL_REVIEWER, Capability.NETWORK}
    ),
    # Disclosure-sensitive skills (must surface DisclosedResult).
    "phewas": frozenset({Capability.READ_DATA, Capability.EXPORT_AGGREGATE, Capability.WRITE_REPORTS}),
    "prevalence": frozenset({Capability.READ_DATA, Capability.EXPORT_AGGREGATE}),
    "gwas_proxy": frozenset({Capability.READ_DATA, Capability.NETWORK, Capability.EXPORT_AGGREGATE}),
    "survival": frozenset({Capability.READ_DATA, Capability.WRITE_REPORTS, Capability.EXPORT_AGGREGATE}),
    "train_model": frozenset({Capability.READ_DATA, Capability.WRITE_REPORTS, Capability.EXPORT_AGGREGATE}),
    "generate_report": frozenset({Capability.WRITE_REPORTS, Capability.EXPORT_AGGREGATE}),
}

_MUTATING_LEGACY_SKILLS: frozenset[str] = frozenset({
    "create_skill",
    "record_macro",
    "remember_model_config",
    "save_pipeline",
    "track_error",
    "delegate_to_coding_agent",  # applies external-agent code edits to a review branch
})


def infer_legacy_capabilities(name: str) -> frozenset[Capability]:
    """Return the capability set used when wrapping a legacy skill."""
    return _LEGACY_CAP_OVERRIDES.get(name, _DEFAULT_LEGACY_CAPS)


def infer_legacy_is_mutating(name: str) -> bool:
    """Return whether a legacy skill should be treated as mutating."""
    return name in _MUTATING_LEGACY_SKILLS


class ToolRegistry:
    """v3 ToolHandler registry, layered on top of the legacy SkillRegistry."""

    def __init__(self, legacy: Optional[SkillRegistry] = None) -> None:
        self._legacy = legacy or _get_legacy_registry()
        self._handlers: dict[str, ToolHandler] = {}
        self._handlers_loaded_from_legacy: bool = False
        self._native_loaded: bool = False

    # ── Discovery / registration ─────────────────────────────

    def hydrate_from_legacy(self) -> int:
        """Wrap every legacy ``@skill`` as a ``LegacySkillToolHandler``."""
        loaded = 0
        for entry in self._legacy.list_skills():
            name = entry["name"]
            if name in self._handlers:
                continue
            schema = self._legacy._schemas.get(name) or {}
            params_node = schema.get("function", {}).get("parameters", {}) or {}
            spec = ToolSpec(
                name=name,
                description=schema.get("function", {}).get("description", ""),
                parameters=dict(params_node.get("properties", {})),
                required=list(params_node.get("required", [])),
            )
            caps = infer_legacy_capabilities(name)
            # A shell-out / self-modifying skill must not be audit-classified as a benign READ;
            # surface its real risk in the ToolSpec metadata (audit + trajectory records read this).
            if Capability.SHELL_EXEC in caps:
                spec.safety_class = SafetyClass.SELF_MODIFICATION
                spec.action_classes = (ActionClass.SELF_MODIFICATION,)

            def _make_invoker(skill_name=name):
                # Closure that defers to legacy execute() so reload /
                # lazy-load / custom_skills paths keep working.
                def invoke(**kwargs):
                    ctx = kwargs.pop("ctx", None)
                    return self._legacy.execute(skill_name, dict(kwargs), ctx=ctx)
                return invoke

            handler = LegacySkillToolHandler(
                name=name,
                spec=spec,
                callable_=_make_invoker(),
                capabilities=caps,
                is_mutating=infer_legacy_is_mutating(name),
            )
            self._handlers[name] = handler
            loaded += 1
        self._handlers_loaded_from_legacy = True
        if not self._native_loaded:
            for handler in build_native_tools():
                self.register(handler)
            self._native_loaded = True
        return loaded

    def register(self, handler: ToolHandler) -> None:
        if handler.name in self._handlers:
            logger.debug("ToolHandler '%s' replaced", handler.name)
        self._handlers[handler.name] = handler

    def unregister(self, name: str) -> None:
        self._handlers.pop(name, None)

    # ── Access ────────────────────────────────────────────────

    def get(self, name: str) -> Optional[ToolHandler]:
        return self._handlers.get(name)

    def list_handlers(self) -> list[ToolHandler]:
        return list(self._handlers.values())

    def tool_schemas(self) -> list[dict]:
        return [h.spec().to_openai_schema() for h in self._handlers.values()]

    # ── Lazy / tiered exposure ──────────────────────────
    # Native workspace tools (non-legacy handlers) are always Direct; legacy
    # @skills are tiered by skills/manifest.json. Direct skills are always
    # offered to the model; Deferred skills only once activated (e.g. by
    # skill_search); Hidden skills are runtime/curator-only and never offered.

    def _exposure(self, handler: ToolHandler) -> str:
        from biobank_agent.skills import manifest as skill_manifest

        if not isinstance(handler, LegacySkillToolHandler):
            return skill_manifest.DIRECT  # native workspace tools
        return skill_manifest.exposure_of(handler.name)

    def exposed_handlers(self, active: Optional[Iterable[str]] = None) -> list[ToolHandler]:
        """Handlers to offer the model this round: all Direct + any activated Deferred."""
        from biobank_agent.skills import manifest as skill_manifest

        active_set = set(active or ())
        out: list[ToolHandler] = []
        for handler in self._handlers.values():
            exposure = self._exposure(handler)
            if exposure == skill_manifest.DIRECT:
                out.append(handler)
            elif exposure == skill_manifest.DEFERRED and handler.name in active_set:
                out.append(handler)
            # hidden: never offered to the model (runtime/curator call it directly)
        return out

    def exposed_schemas(self, active: Optional[Iterable[str]] = None) -> list[dict]:
        return [h.spec().to_openai_schema() for h in self.exposed_handlers(active)]

    def search(self, query: str, *, k: int = 8, include_hidden: bool = False,
               subtree: Optional[str] = None) -> list[dict]:
        """Rank Deferred (and Direct) skills by token overlap of the query against
        name + description. Returns ``[{name, description, domain, exposure}]`` —
        used by the skill_search tool to surface on-demand skills. ``subtree`` restricts
        candidates to skills under a given skill-tree node (default: the whole corpus)."""
        from biobank_agent.skills import manifest as skill_manifest

        def _tokens(text: str) -> set[str]:
            return {t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if len(t) > 1}

        q = _tokens(query)
        if not q:
            return []
        allowed: Optional[set[str]] = None
        if subtree:
            from biobank_agent.skills import skill_tree as _skill_tree

            allowed = set(_skill_tree.subtree_skills(subtree))
        scored: list[tuple[float, dict]] = []
        for handler in self._handlers.values():
            if allowed is not None and handler.name not in allowed:
                continue
            exposure = self._exposure(handler)
            if exposure == skill_manifest.DIRECT:
                continue  # already in the model's tool list every round
            if exposure == skill_manifest.HIDDEN and not include_hidden:
                continue
            spec = handler.spec()
            doc = _tokens(f"{handler.name} {handler.name.replace('_', ' ')} {spec.description}")
            doc |= _tokens(skill_manifest.domain_of(handler.name))
            if not doc:
                continue
            overlap = len(q & doc)
            if overlap <= 0:
                continue
            score = overlap / (len(q) ** 0.5)
            scored.append((score, {
                "name": handler.name,
                "description": (spec.description or "")[:200],
                "domain": skill_manifest.domain_of(handler.name),
                "exposure": exposure,
            }))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [payload for _score, payload in scored[: max(1, int(k))]]

    def __len__(self) -> int:
        return len(self._handlers)

    def __contains__(self, name: str) -> bool:
        return name in self._handlers

    # ── Helpers ──────────────────────────────────────────────

    def capabilities_for(self, name: str) -> frozenset[Capability]:
        h = self._handlers.get(name)
        return h.required_capabilities() if h else frozenset()


__all__ = ["ToolRegistry", "infer_legacy_capabilities", "infer_legacy_is_mutating"]
