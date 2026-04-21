"""Skill registry — decorator-based registration with lazy loading.

Usage in a skill file::

    from biobank_agent.registry import skill

    @skill(
        name="prevalence",
        description="Calculate disease prevalence for an ICD10 code",
        parameters={
            "icd10_code": {"type": "string", "description": "ICD10 code prefix"},
        },
    )
    def prevalence(icd10_code: str, *, ctx) -> dict:
        ...
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Central registry of all available skills (tools)."""

    def __init__(self) -> None:
        self._schemas: dict[str, dict] = {}          # name → OpenAI tool schema
        self._callables: dict[str, Callable] = {}     # name → function (lazy)
        self._module_paths: dict[str, str] = {}       # name → "biobank_agent.skills.xyz"
        self._descriptions: dict[str, str] = {}       # name → description

    # ── Registration ─────────────────────────────────────

    def register(self, name: str, func: Callable, schema: dict) -> None:
        """Eagerly register a skill (used by the @skill decorator)."""
        self._schemas[name] = schema
        self._callables[name] = func
        self._descriptions[name] = schema["function"]["description"]

    def register_lazy(self, name: str, module_path: str, schema: dict) -> None:
        """Register schema only — implementation loaded on first call."""
        self._schemas[name] = schema
        self._module_paths[name] = module_path
        self._descriptions[name] = schema["function"]["description"]

    # ── Execution ────────────────────────────────────────

    def execute(self, name: str, args: dict, ctx: Any = None) -> Any:
        """Execute a skill by name, injecting ctx if the function accepts it."""
        if name not in self._callables:
            if name in self._module_paths:
                mod = importlib.import_module(self._module_paths[name])
                func = getattr(mod, name, None)
                if func is None:
                    # Search for any decorated function
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if callable(attr) and getattr(attr, "_skill_name", None) == name:
                            func = attr
                            break
                if func is None:
                    raise ValueError(f"Skill '{name}' not found in {self._module_paths[name]}")
                self._callables[name] = func
            else:
                raise ValueError(f"Unknown skill: {name}")

        func = self._callables[name]
        if ctx is not None:
            args["ctx"] = ctx
        return func(**args)

    # ── Schema access ────────────────────────────────────

    def tool_schemas(self) -> list[dict]:
        """Return OpenAI-format tool schemas for all registered skills."""
        return list(self._schemas.values())

    def list_skills(self) -> list[dict]:
        """Return skill name + description pairs."""
        return [
            {"name": n, "description": self._descriptions.get(n, "")}
            for n in self._schemas
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._schemas

    def __len__(self) -> int:
        return len(self._schemas)


# ── Global registry instance ────────────────────────────────

_registry = SkillRegistry()


def get_registry() -> SkillRegistry:
    return _registry


# ── Decorator ───────────────────────────────────────────────

def skill(
    name: str,
    description: str,
    parameters: dict[str, dict],
    required: list[str] | None = None,
):
    """Decorator to register a function as an agent skill (tool).

    Parameters use a simplified dict format::

        parameters={
            "icd10_code": {"type": "string", "description": "ICD10 code"},
            "top_n": {"type": "integer", "description": "Number of results"},
        }
    """
    # Build OpenAI function-calling schema
    props = {}
    for pname, pdef in parameters.items():
        props[pname] = {k: v for k, v in pdef.items()}

    if required is None:
        required = [p for p in parameters if "default" not in parameters[p]]

    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }

    def decorator(func: Callable) -> Callable:
        func._skill_name = name
        func._skill_schema = schema
        _registry.register(name, func, schema)
        return func

    return decorator


# ── Auto-discovery ──────────────────────────────────────────

def autodiscover_skills(package_path: str = "biobank_agent.skills") -> None:
    """Import all modules in the skills package to trigger @skill decorators."""
    try:
        pkg = importlib.import_module(package_path)
    except ImportError:
        logger.warning("Could not import %s", package_path)
        return

    pkg_dir = Path(pkg.__file__).parent
    for finder, module_name, is_pkg in pkgutil.iter_modules([str(pkg_dir)]):
        if module_name.startswith("_"):
            continue
        full_name = f"{package_path}.{module_name}"
        try:
            importlib.import_module(full_name)
            logger.debug("Loaded skill module: %s", full_name)
        except Exception as e:
            logger.warning("Failed to load skill %s: %s", full_name, e)
