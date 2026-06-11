"""Validate repo-local Codex and Claude Code plugin artifacts."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_no_todo(value):
    text = json.dumps(value, ensure_ascii=False)
    assert "[TODO:" not in text


def test_codex_plugin_manifest_and_marketplace_are_concrete():
    manifest = _load_json(ROOT / "plugins/biobank-agent/.codex-plugin/plugin.json")
    marketplace = _load_json(ROOT / ".agents/plugins/marketplace.json")

    assert manifest["name"] == "biobank-agent"
    assert manifest["skills"] == "./skills/"
    assert (ROOT / "plugins/biobank-agent/skills/biobank-agent-runtime/SKILL.md").exists()
    assert marketplace["plugins"][0]["source"]["path"] == "./plugins/biobank-agent"
    _assert_no_todo(manifest)
    _assert_no_todo(marketplace)


def test_claude_plugin_manifest_commands_and_marketplace_exist():
    manifest = _load_json(ROOT / "plugins/biobank-agent-claude/.claude-plugin/plugin.json")
    marketplace = _load_json(ROOT / ".claude-plugin/marketplace.json")

    assert manifest["name"] == "biobank-agent"
    assert marketplace["plugins"][0]["source"] == "./plugins/biobank-agent-claude"
    assert (ROOT / "plugins/biobank-agent-claude/commands/status.md").exists()
    assert (ROOT / "plugins/biobank-agent-claude/commands/plan.md").exists()
    assert (ROOT / "plugins/biobank-agent-claude/commands/check.md").exists()
    assert (ROOT / "plugins/biobank-agent-claude/agents/biobank-reviewer.md").exists()
    _assert_no_todo(manifest)
    _assert_no_todo(marketplace)


def test_plugin_bridge_status_command_runs():
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "plugins/biobank-agent/scripts/biobank_agent_bridge.py",
            "status",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["skills_registered"] >= 40
    assert "prevalence" in payload["skills"]
    assert "Matplotlib" not in proc.stderr
