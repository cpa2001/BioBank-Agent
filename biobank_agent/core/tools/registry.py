"""v3 tool registry built on ToolHandler.

Wraps the legacy ``biobank_agent.registry.SkillRegistry`` so all 56
existing skills appear in the v3 registry as
``LegacySkillToolHandler`` instances. New tools (and MCP-discovered
tools in M2.4) register here directly.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from biobank_agent.registry import SkillRegistry, get_registry as _get_legacy_registry

from .protocol import (
    Capability,
    LegacySkillToolHandler,
    ToolContext,
    ToolHandler,
    ToolSpec,
)

logger = logging.getLogger(__name__)


# Capability inference for legacy @skill functions. Until M3
# annotates each skill with its real capability set we use
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
    "external_agents": frozenset({Capability.CALL_REVIEWER, Capability.NETWORK}),
    "codex_plan": frozenset({Capability.CALL_REVIEWER, Capability.NETWORK}),
    "codex_check": frozenset({Capability.CALL_REVIEWER, Capability.NETWORK}),
    "claude_plan": frozenset({Capability.CALL_REVIEWER, Capability.NETWORK}),
    "claude_check": frozenset({Capability.CALL_REVIEWER, Capability.NETWORK}),
    # Memory-mutating skills
    "create_skill": frozenset(
        {Capability.WRITE_REPORTS, Capability.MUTATE_MEMORY}
    ),
    "record_macro": frozenset({Capability.MUTATE_MEMORY}),
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

    def __len__(self) -> int:
        return len(self._handlers)

    def __contains__(self, name: str) -> bool:
        return name in self._handlers

    # ── Helpers ──────────────────────────────────────────────

    def capabilities_for(self, name: str) -> frozenset[Capability]:
        h = self._handlers.get(name)
        return h.required_capabilities() if h else frozenset()


__all__ = ["ToolRegistry", "infer_legacy_capabilities", "infer_legacy_is_mutating"]
