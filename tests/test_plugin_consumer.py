"""Tests for the gated plugin/marketplace consumer (``runtime/plugins.py``).

Pins the Claude-Code-style consumption contract against a synthetic marketplace that mirrors the shape
of obra/superpowers: SKILL.md skills load as USABLE knowledge skills; plugin hooks (executable shell
commands) are discovered but DISABLED unless explicitly opted in; and a plugin ``source`` can never
escape the marketplace root.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biobank_agent.runtime.plugins import (
    PluginManifest,
    discover_marketplace,
    discover_plugin_hooks,
    install_plugin,
    parse_marketplace,
    plugin_root,
)


def _make_marketplace(tmp_path: Path, *, source: str = "./") -> Path:
    """Build a synthetic Claude-Code marketplace repo (mirrors obra/superpowers' layout)."""
    repo = tmp_path / "market"
    (repo / ".claude-plugin").mkdir(parents=True)
    (repo / ".claude-plugin" / "marketplace.json").write_text(json.dumps({
        "name": "demo-market",
        "description": "demo",
        "plugins": [
            {"name": "superpowers", "description": "core skills", "version": "5.1.0", "source": source}
        ],
    }))
    for sk, desc in (("brainstorming", "Use when exploring ideas"),
                     ("debugging", "Use when stuck on a bug")):
        d = repo / "skills" / sk
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {sk}\ndescription: {desc}\n---\n\n# {sk}\n\nGuidance body for {sk}.\n"
        )
    (repo / "hooks").mkdir()
    (repo / "hooks" / "hooks.json").write_text(json.dumps({
        "hooks": {"SessionStart": [{"matcher": "startup", "hooks": [
            {"type": "command", "command": "${CLAUDE_PLUGIN_ROOT}/hooks/run.sh", "async": False}]}]}
    }))
    return repo


def test_parse_marketplace_valid_and_invalid():
    data = parse_marketplace('{"name":"m","plugins":[{"name":"p"}]}')
    assert data["plugins"][0]["name"] == "p"
    with pytest.raises(ValueError):
        parse_marketplace("not json")
    with pytest.raises(ValueError):
        parse_marketplace('{"name":"m"}')  # missing the plugins array


def test_discover_marketplace(tmp_path):
    repo = _make_marketplace(tmp_path)
    mp = discover_marketplace(repo, source_url="https://example/x", commit="abc123")
    assert mp.name == "demo-market"
    assert [p.name for p in mp.plugins] == ["superpowers"]
    assert mp.plugin("superpowers").version == "5.1.0"
    assert mp.source_url == "https://example/x" and mp.commit == "abc123"


def test_plugin_root_rejects_escape(tmp_path):
    mp = discover_marketplace(_make_marketplace(tmp_path))
    with pytest.raises(ValueError):
        plugin_root(mp, PluginManifest(name="evil", source="../../etc"))


def test_discover_plugin_hooks_flattens_and_records_command(tmp_path):
    repo = _make_marketplace(tmp_path)
    hooks = discover_plugin_hooks(repo, plugin_name="superpowers")
    assert len(hooks) == 1
    assert hooks[0].event == "SessionStart" and hooks[0].matcher == "startup"
    assert "run.sh" in hooks[0].command and hooks[0].plugin == "superpowers"


def test_install_plugin_loads_skills_usable_and_gates_hooks(tmp_path):
    mp = discover_marketplace(_make_marketplace(tmp_path))
    registered: dict[str, tuple] = {}
    result = install_plugin(
        mp, "superpowers",
        register_fn=lambda name, fn, schema: registered.__setitem__(name, (fn, schema)),
        classify_fn=lambda n, d: "external/demo",
        trust_fn=lambda names: None,
        allow_hooks=False,
    )
    assert result["status"] == "installed"
    assert result["skill_count"] == 2
    assert set(registered) == {"brainstorming", "debugging"}
    # the registered skill is USABLE: invoking it returns its SKILL.md guidance (no third-party code runs)
    fn, _schema = registered["brainstorming"]
    out = fn()
    assert out["status"] == "guidance" and "Guidance body" in out["instructions"]
    # hooks are DISCOVERED but DISABLED (gated) — third-party code never auto-runs without opt-in
    assert result["hooks_discovered"] == 1 and result["hooks_enabled"] is False


def test_install_plugin_hooks_enabled_only_on_explicit_optin(tmp_path):
    mp = discover_marketplace(_make_marketplace(tmp_path))
    result = install_plugin(
        mp, "superpowers",
        register_fn=lambda name, fn, schema: None,
        classify_fn=lambda n, d: "external/demo",
        trust_fn=lambda names: None,
        allow_hooks=True,  # explicit per-plugin opt-in
    )
    assert result["hooks_enabled"] is True and result["hooks_discovered"] == 1


def test_install_plugin_not_found(tmp_path):
    mp = discover_marketplace(_make_marketplace(tmp_path))
    result = install_plugin(mp, "nope")
    assert result["status"] == "not_found" and "superpowers" in result["available"]
