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

import ast
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

    def unregister(self, name: str) -> None:
        """Remove a skill by name."""
        self._schemas.pop(name, None)
        self._callables.pop(name, None)
        self._module_paths.pop(name, None)
        self._descriptions.pop(name, None)

    def reload_skill(self, name: str) -> None:
        """Hot-reload a skill: reimport its module and re-register."""
        if name in self._module_paths:
            mod_path = self._module_paths[name]
            mod = importlib.import_module(mod_path)
            importlib.reload(mod)
            # Re-find the decorated function
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if callable(attr) and getattr(attr, "_skill_name", None) == name:
                    schema = getattr(attr, "_skill_schema", None)
                    if schema:
                        self.register(name, attr, schema)
                    break
            logger.info("Hot-reloaded skill: %s", name)
        elif name in self._callables:
            # Direct callable — nothing to reload
            logger.warning("Skill %s is not module-based, cannot reload", name)
        else:
            raise ValueError(f"Unknown skill: {name}")

    # ── Execution ────────────────────────────────────────

    def execute(self, name: str, args: dict | None, ctx: Any = None) -> Any:
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
        call_args = dict(args or {})
        if ctx is not None:
            call_args["ctx"] = ctx
        return func(**call_args)

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

def _parameter_property_schema(pdef: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON Schema property exposed to model providers.

    Some skill decorators use ``required`` as local metadata on an individual
    parameter. OpenAI-compatible function schemas only accept ``required`` as an
    object-level array, so the property-level marker must not be forwarded.
    """
    return {k: v for k, v in pdef.items() if k != "required"}


def _infer_required_parameters(parameters: dict[str, Any]) -> list[str]:
    required: list[str] = []
    for pname, pdef in parameters.items():
        if not isinstance(pdef, dict):
            continue
        marker = pdef.get("required", None)
        if marker is True:
            required.append(pname)
        elif marker is False:
            continue
        elif "default" not in pdef:
            required.append(pname)
    return required


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
        props[pname] = _parameter_property_schema(pdef)

    if required is None:
        required = _infer_required_parameters(parameters)

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
    """Discover skills without importing heavy analysis modules.

    The production skill package is schema-first: we can read literal
    ``@skill(...)`` decorators with ``ast`` and register lazy module paths.
    This keeps CLI startup and `/skills` fast, while preserving the old
    eager-import behavior for ad-hoc test packages and non-literal edge cases.
    """
    try:
        pkg = importlib.import_module(package_path)
    except ImportError:
        logger.warning("Could not import %s", package_path)
        return

    pkg_dir = Path(pkg.__file__).parent
    if package_path == "biobank_agent.skills":
        _autodiscover_builtin_skills(package_path, pkg_dir)
        return

    _autodiscover_eager(package_path, pkg_dir)


def _autodiscover_eager(package_path: str, pkg_dir: Path) -> None:
    """Legacy eager import discovery, kept for third-party/test packages."""
    for finder, module_name, is_pkg in pkgutil.iter_modules([str(pkg_dir)]):
        if module_name.startswith("_"):
            continue
        full_name = f"{package_path}.{module_name}"
        try:
            importlib.import_module(full_name)
            logger.debug("Loaded skill module: %s", full_name)
        except Exception as e:
            logger.warning("Failed to load skill %s: %s", full_name, e)


def _autodiscover_builtin_skills(package_path: str, pkg_dir: Path) -> None:
    """Register built-in skills lazily from decorator schemas."""
    for file_path in sorted(pkg_dir.glob("*.py")):
        module_name = file_path.stem
        if module_name.startswith("_"):
            continue
        module_path = f"{package_path}.{module_name}"
        try:
            loaded = _register_lazy_schemas_from_file(file_path, module_path)
        except Exception as e:
            loaded = False
            logger.debug("AST skill discovery failed for %s: %s", module_path, e)
        if loaded:
            continue
        try:
            importlib.import_module(module_path)
            logger.debug("Loaded skill module: %s", module_path)
        except Exception as e:
            logger.warning("Failed to load skill %s: %s", module_path, e)


def _register_lazy_schemas_from_file(file_path: Path, module_path: str) -> bool:
    """Parse literal @skill decorators and register their schemas lazily.

    Returns True when the file was handled without import. Files with no skill
    decorators are still handled; helper modules should not slow startup.
    """
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            parsed = _schema_from_skill_decorator(decorator)
            if parsed is None:
                continue
            name, schema = parsed
            _registry.register_lazy(name, module_path, schema)
    return True


def _schema_from_skill_decorator(decorator: ast.AST) -> tuple[str, dict] | None:
    if not isinstance(decorator, ast.Call):
        return None
    func_name = getattr(decorator.func, "id", None)
    if func_name != "skill":
        return None

    values: dict[str, Any] = {}
    for keyword in decorator.keywords:
        if keyword.arg is None:
            continue
        values[keyword.arg] = ast.literal_eval(keyword.value)

    name = values.get("name")
    description = values.get("description")
    parameters = values.get("parameters")
    if not isinstance(name, str) or not isinstance(description, str) or not isinstance(parameters, dict):
        raise ValueError("skill decorator must define literal name, description and parameters")

    required = values.get("required")
    if required is None:
        required = _infer_required_parameters(parameters)

    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {pname: _parameter_property_schema(pdef) for pname, pdef in parameters.items()},
                "required": list(required),
            },
        },
    }
    return name, schema


def discover_custom_skills(custom_dir: Path) -> int:
    """Discover and load skills from a custom directory.

    Scans ``custom_dir`` for ``.py`` files containing @skill-decorated functions.
    Returns the number of newly loaded skills.
    """
    import sys

    from biobank_agent.skills.generator import SkillGenerator

    if not custom_dir.exists():
        return 0

    before = len(_registry)
    for py_file in sorted(custom_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        # Load-time safety gate: a custom/ingested module's body and decorators run with full
        # privileges on exec_module, so enforce the same AST whitelist used at generation time
        # BEFORE executing it. An unsafe file is skipped (never exec'd), not fatal — boot proceeds.
        try:
            source = py_file.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("Cannot read custom skill %s: %s", py_file.name, e)
            continue
        is_safe, reason = SkillGenerator.validate_code(source)
        if is_safe:
            # The whole file runs on exec_module — also forbid invoking the shell-out seam
            # at module scope, which would execute at import rather than when the skill is called.
            is_safe, reason = SkillGenerator.validate_load_safety(source)
        if not is_safe:
            logger.warning("Refused unsafe custom skill %s: %s", py_file.name, reason)
            continue
        module_name = f"custom_skills.{py_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec and spec.loader:
            try:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                logger.info("Loaded custom skill: %s", py_file.name)
            except Exception as e:
                logger.warning("Failed to load custom skill %s: %s", py_file.name, e)

    loaded = len(_registry) - before
    if loaded > 0:
        logger.info("Loaded %d custom skill(s) from %s", loaded, custom_dir)
    return loaded
