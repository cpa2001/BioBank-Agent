"""Consume Claude-Code-style plugins from a marketplace.

A marketplace is a git repo with a ``.claude-plugin/marketplace.json`` listing one or more plugins;
each plugin is a directory (``source`` relative to the repo) that may carry ``skills/`` (SKILL.md
folders), ``commands/``, ``agents/``, and ``hooks/hooks.json``. Consumption is GATED, matching the
project's external-trust posture:

* SKILL.md skills load as USABLE deferred KNOWLEDGE skills — their guidance is injected when the agent
  invokes them, the same safe-by-construction model as :mod:`skill_ingest` — tagged ``trust='external'``
  so the curator never auto-promotes them.
* Plugin HOOKS (executable shell commands) are parsed and recorded DISABLED; they never run without an
  explicit per-plugin opt-in. Third-party plugin CODE is never executed on install.

Pure parsing/registration with injected deps so it is unit-testable offline; a live wrapper wires the
pinned clone + the real registry (and the gated hook registry).
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from biobank_agent.runtime.skill_ingest import (
    SkillPackage,
    discover_skill_md,
    ingest_knowledge_corpus,
)

MARKETPLACE_MANIFEST = ".claude-plugin/marketplace.json"
PLUGIN_MANIFEST = ".claude-plugin/plugin.json"


@dataclass
class PluginManifest:
    """One plugin entry from a marketplace's ``plugins`` array."""

    name: str
    description: str = ""
    version: str = ""
    source: str = "./"  # plugin root, relative to the marketplace repo

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginManifest":
        return cls(
            name=str(data.get("name") or "").strip(),
            description=str(data.get("description") or ""),
            version=str(data.get("version") or ""),
            source=str(data.get("source") or "./") or "./",
        )


@dataclass
class Marketplace:
    """A parsed marketplace.json plus the local clone it was read from."""

    name: str
    description: str = ""
    source_url: str = ""
    commit: str = ""
    root: str = ""  # local clone path
    plugins: list[PluginManifest] = field(default_factory=list)

    def plugin(self, name: str) -> PluginManifest | None:
        for p in self.plugins:
            if p.name == name:
                return p
        return None


@dataclass
class PluginHook:
    """A single plugin-supplied hook (an executable shell command). Stored DISABLED — a record of what
    the plugin WOULD run, surfaced for review; it never executes without an explicit opt-in."""

    event: str
    command: str
    matcher: str = ""
    is_async: bool = False
    plugin: str = ""


def parse_marketplace(text: str) -> dict[str, Any]:
    """Parse a marketplace.json (strict JSON). Raises ``ValueError`` on malformed input."""
    try:
        data = json.loads(text or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid marketplace.json: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("plugins"), list):
        raise ValueError("marketplace.json must be an object with a 'plugins' array")
    return data


def discover_marketplace(root: str | Path, *, source_url: str = "", commit: str = "") -> Marketplace:
    """Read ``<root>/.claude-plugin/marketplace.json`` into a :class:`Marketplace`."""
    base = Path(root)
    manifest_path = base / MARKETPLACE_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"no {MARKETPLACE_MANIFEST} under {base}")
    data = parse_marketplace(manifest_path.read_text(encoding="utf-8", errors="replace"))
    plugins = [
        PluginManifest.from_dict(p)
        for p in data.get("plugins", [])
        if isinstance(p, dict) and str(p.get("name") or "").strip()
    ]
    description = str(data.get("description") or (data.get("metadata") or {}).get("description") or "")
    return Marketplace(
        name=str(data.get("name") or base.name),
        description=description,
        source_url=source_url,
        commit=commit,
        root=str(base),
        plugins=plugins,
    )


def plugin_root(marketplace: Marketplace, plugin: PluginManifest) -> Path:
    """Resolve a plugin's root dir inside the cloned marketplace, refusing any path that escapes it."""
    base = Path(marketplace.root).resolve()
    src = (plugin.source or "./").lstrip("/")
    root = (base / src).resolve()
    if root != base and base not in root.parents:
        raise ValueError(f"plugin source escapes the marketplace root: {plugin.source!r}")
    return root


def discover_plugin_hooks(root: str | Path, *, plugin_name: str = "") -> list[PluginHook]:
    """Parse ``<plugin_root>/hooks/hooks.json`` into a flat list of :class:`PluginHook` (DISABLED).

    Claude-Code hooks are keyed by lifecycle event → matcher groups → command hooks. We flatten them so
    the consumer can SHOW exactly what a plugin would run, without ever executing any of it."""
    hooks_path = Path(root) / "hooks" / "hooks.json"
    if not hooks_path.exists():
        return []
    try:
        data = json.loads(hooks_path.read_text(encoding="utf-8", errors="replace") or "{}")
    except json.JSONDecodeError:
        return []
    table = data.get("hooks") if isinstance(data, dict) else None
    if not isinstance(table, dict):
        return []
    out: list[PluginHook] = []
    for event, groups in table.items():
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            matcher = str(group.get("matcher") or "")
            for hook in group.get("hooks") or []:
                if not isinstance(hook, dict) or str(hook.get("type") or "") != "command":
                    continue
                command = str(hook.get("command") or "").strip()
                if not command:
                    continue
                out.append(PluginHook(
                    event=str(event), command=command, matcher=matcher,
                    is_async=bool(hook.get("async")), plugin=plugin_name,
                ))
    return out


def install_plugin_skills(
    marketplace: Marketplace,
    plugin: PluginManifest,
    *,
    register_fn: Callable[[str, Callable[..., dict], dict], None] | None = None,
    classify_fn: Callable[[str, str], str] | None = None,
    trust_fn: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Load a plugin's SKILL.md skills as usable knowledge skills (``trust='external'``). No code runs."""
    root = plugin_root(marketplace, plugin)
    skills_dir = root / "skills"
    discovered = discover_skill_md(skills_dir if skills_dir.exists() else root)
    package = SkillPackage(
        name=f"plugin:{plugin.name}",
        source_url=marketplace.source_url,
        commit=marketplace.commit,
    )
    return ingest_knowledge_corpus(
        discovered, package,
        register_fn=register_fn, classify_fn=classify_fn, trust_fn=trust_fn,
    )


def install_plugin(
    marketplace: Marketplace,
    plugin_name: str,
    *,
    register_fn: Callable[[str, Callable[..., dict], dict], None] | None = None,
    classify_fn: Callable[[str, str], str] | None = None,
    trust_fn: Callable[[Any], None] | None = None,
    allow_hooks: bool = False,
) -> dict[str, Any]:
    """Install a plugin from a parsed marketplace: load its SKILL.md skills (usable) and parse its hooks
    (DISABLED unless ``allow_hooks``). Third-party code is never executed here.

    Returns a status dict: skills installed, hooks discovered, and whether hooks are enabled."""
    plugin = marketplace.plugin(plugin_name)
    if plugin is None:
        return {"status": "not_found", "plugin": plugin_name,
                "available": [p.name for p in marketplace.plugins]}
    skills = install_plugin_skills(
        marketplace, plugin,
        register_fn=register_fn, classify_fn=classify_fn, trust_fn=trust_fn,
    )
    hooks = discover_plugin_hooks(plugin_root(marketplace, plugin), plugin_name=plugin.name)
    return {
        "status": "installed",
        "plugin": plugin.name,
        "version": plugin.version,
        "marketplace": marketplace.name,
        "skills": skills.get("skills", []),
        "skill_count": skills.get("count", 0),
        "hooks_discovered": len(hooks),
        # GATED: third-party hooks never auto-run. They are surfaced for review and execute only after an
        # explicit per-plugin opt-in (allow_hooks), per the project's external-trust posture.
        "hooks_enabled": bool(allow_hooks),
        "hooks": [{"event": h.event, "matcher": h.matcher, "command": h.command} for h in hooks],
    }


# --------------------------------------------------------------------------- live: clone + add

# (argv, cwd) -> (returncode, combined_output). Injectable so the clone path is testable offline.
GitRunner = Callable[[list[str], str], "tuple[int, str]"]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", str(text).strip().lower()).strip("_") or "marketplace"


def resolve_marketplace_spec(spec: str) -> tuple[str, str]:
    """Classify a marketplace spec: ``('local', abspath)`` if it is an existing marketplace directory,
    else ``('git', url)`` — accepting a git URL or an ``owner/repo`` shorthand (→ a github.com URL)."""
    s = str(spec or "").strip()
    if not s:
        raise ValueError("empty marketplace spec")
    p = Path(s).expanduser()
    if (p / MARKETPLACE_MANIFEST).exists():
        return ("local", str(p.resolve()))
    if s.startswith(("http://", "https://", "git@", "ssh://", "file://")):
        return ("git", s)
    if re.match(r"^[\w.-]+/[\w.-]+$", s):  # owner/repo
        return ("git", f"https://github.com/{s}.git")
    return ("git", s)


def _default_git_runner(argv: list[str], cwd: str = "") -> tuple[int, str]:
    proc = subprocess.run(argv, cwd=cwd or None, capture_output=True, text=True, timeout=300, check=False)
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or ""))


def clone_marketplace(url: str, dest: str | Path, *, ref: str = "", runner: GitRunner | None = None) -> str:
    """Shallow-clone a marketplace repo to ``dest`` (checking out ``ref`` if given) and return the
    resolved commit SHA (``''`` if unknown). ``runner`` is injectable for offline tests. Never clobbers
    a pre-existing non-git directory."""
    run = runner or _default_git_runner
    dest = Path(dest)
    if not (dest / ".git").exists():
        if dest.exists() and any(dest.iterdir()):
            raise ValueError(f"destination exists and is not a git clone: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        rc, out = run(["git", "clone", "--depth", "1", str(url), str(dest)], "")
        if rc != 0:
            raise RuntimeError(f"git clone failed: {out[-500:]}")
    if ref:
        run(["git", "fetch", "--depth", "1", "origin", str(ref)], str(dest))
        rc, out = run(["git", "checkout", str(ref)], str(dest))
        if rc != 0:
            raise RuntimeError(f"git checkout {ref} failed: {out[-300:]}")
    rc, sha = run(["git", "rev-parse", "HEAD"], str(dest))
    return sha.strip() if rc == 0 else ""


def add_marketplace(spec: str, plugins_dir: str | Path, *, ref: str = "", runner: GitRunner | None = None) -> Marketplace:
    """Resolve a marketplace spec (an existing local dir, or a git repo cloned under ``plugins_dir``)
    and return the parsed :class:`Marketplace`. Third-party plugin code is never executed here."""
    kind, value = resolve_marketplace_spec(spec)
    if kind == "local":
        return discover_marketplace(value, source_url=value)
    dest = Path(plugins_dir).expanduser() / _slug(spec)
    commit = clone_marketplace(value, dest, ref=ref, runner=runner)
    return discover_marketplace(dest, source_url=value, commit=commit)


__all__ = [
    "MARKETPLACE_MANIFEST",
    "PLUGIN_MANIFEST",
    "PluginManifest",
    "Marketplace",
    "PluginHook",
    "GitRunner",
    "parse_marketplace",
    "discover_marketplace",
    "plugin_root",
    "discover_plugin_hooks",
    "install_plugin_skills",
    "install_plugin",
    "resolve_marketplace_spec",
    "clone_marketplace",
    "add_marketplace",
]
