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


# --------------------------------------------------------------------------- live: add/clone


def test_resolve_marketplace_spec_local_git_and_shorthand(tmp_path):
    from biobank_agent.runtime.plugins import resolve_marketplace_spec

    repo = _make_marketplace(tmp_path)
    assert resolve_marketplace_spec(str(repo)) == ("local", str(repo.resolve()))
    assert resolve_marketplace_spec("obra/superpowers") == ("git", "https://github.com/obra/superpowers.git")
    assert resolve_marketplace_spec("https://example.com/x.git") == ("git", "https://example.com/x.git")


def test_add_marketplace_local_dir_does_not_clone(tmp_path):
    from biobank_agent.runtime.plugins import add_marketplace

    repo = _make_marketplace(tmp_path)
    mp = add_marketplace(str(repo), tmp_path / "plugins_home")
    assert mp.name == "demo-market" and [p.name for p in mp.plugins] == ["superpowers"]
    assert not (tmp_path / "plugins_home").exists()  # a local dir is used in place, never cloned


def test_clone_marketplace_uses_injected_runner(tmp_path):
    from biobank_agent.runtime.plugins import clone_marketplace

    calls: list[list[str]] = []

    def fake_runner(argv, cwd=""):
        calls.append(argv)
        if argv[:2] == ["git", "clone"]:
            (Path(argv[-1]) / ".git").mkdir(parents=True, exist_ok=True)  # simulate a clone
            return 0, "Cloning into..."
        if argv[:2] == ["git", "rev-parse"]:
            return 0, "abc1234\n"
        return 0, ""

    sha = clone_marketplace("https://example/x.git", tmp_path / "dest", runner=fake_runner)
    assert sha == "abc1234"
    assert any(a[:2] == ["git", "clone"] for a in calls)


def test_cmd_plugin_add_list_and_usage(tmp_path):
    """The /plugin command: add a (local) marketplace, list it, and show usage on no args. Install logic
    is exercised by the unit tests above with injected deps (no global-registry pollution here)."""
    from io import StringIO
    from types import SimpleNamespace

    from rich.console import Console

    from biobank_agent.cli.interactive import InteractiveShell

    repo = _make_marketplace(tmp_path)
    out = StringIO()
    shell = InteractiveShell(
        settings=SimpleNamespace(memory_dir=str(tmp_path / "mem"), plugin_allow_hooks=False),
        console=Console(file=out, force_terminal=False, width=100),
    )
    added = shell._cmd_plugin(f"marketplace add {repo}")
    assert added["status"] == "added" and "superpowers" in added["plugins"]
    listed = shell._cmd_plugin("list")
    assert listed["status"] == "listed" and "demo-market" in listed["marketplaces"]
    assert "superpowers" in out.getvalue()
    assert shell._cmd_plugin("")["status"] == "usage"
    assert shell._cmd_plugin("install nope")["status"] == "not_found"


def test_register_plugin_hooks_binds_gated_external_hooks(tmp_path):
    """`_register_plugin_hooks` binds a plugin's discovered command-hooks into the runtime lifecycle
    registry as gated external hooks: they map to a lifecycle event, are owned by the plugin, and never
    execute (only "skipped") until the plugin is opted in."""
    from io import StringIO
    from types import SimpleNamespace

    from rich.console import Console

    from biobank_agent.cli.interactive import InteractiveShell
    from biobank_agent.runtime.hooks import ON_TURN_START, TRUST_EXTERNAL, default_registry

    default_registry().clear()
    shell = InteractiveShell(
        settings=SimpleNamespace(memory_dir=str(tmp_path / "mem"), plugin_allow_hooks=False),
        console=Console(file=StringIO(), force_terminal=False, width=100),
    )
    # SessionStart is a Claude-Code event name; it must map onto our on_turn_start lifecycle event.
    bound = shell._register_plugin_hooks(
        "superpowers",
        [{"event": "SessionStart", "command": "${CLAUDE_PLUGIN_ROOT}/hooks/run.sh"}],
        allow_hooks=False,
    )
    assert bound == 1

    registry = getattr(getattr(shell, "runtime", None), "hooks", None) or default_registry()
    handles = registry.handles(ON_TURN_START)
    assert handles and handles[0].plugin == "superpowers" and handles[0].trust == TRUST_EXTERNAL

    # Gated: firing the event without opt-in reports "skipped" and never runs the shell command.
    outcomes = registry.emit(ON_TURN_START, allow_external=False, allowed_plugins=set())
    assert outcomes and all(o.status == "skipped" for o in outcomes)
    default_registry().clear()
