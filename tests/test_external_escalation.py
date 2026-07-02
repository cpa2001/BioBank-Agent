"""Phase 5: difficulty-triggered escalation to external agents (gated, default-OFF).

When a plan step exhausts its retries and ``external_escalation_enabled`` is on, the shell consults
external coding agents in plan mode and folds their advice into the diagnosis. These tests exercise the
helper directly (the heavy plan loop is covered elsewhere), with the consult function injected.
"""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from biobank_agent.cli.interactive import InteractiveShell
from biobank_agent.runtime import external_orchestration as extorch
from biobank_agent.runtime.external_agents import ExternalAgentResult

_STEP = SimpleNamespace(id="s1", title="Run VCF QC", purpose="quality control before association")


def _shell(tmp_path, *, enabled: bool) -> InteractiveShell:
    settings = SimpleNamespace(
        memory_dir=str(tmp_path / "mem"),
        external_escalation_enabled=enabled,
        external_escalation_agents="codex,claude",
        external_escalation_timeout_s=30,
    )
    return InteractiveShell(settings=settings, console=Console(file=StringIO(), force_terminal=False, width=100))


def test_escalation_disabled_is_a_no_op(tmp_path):
    shell = _shell(tmp_path, enabled=False)
    assert shell._maybe_escalate_step_failure(None, _STEP, "boom") == {}


def test_escalation_enabled_folds_external_advice(tmp_path, monkeypatch):
    calls: dict = {}

    def fake_consult(agents, prompt, *, plan_mode=False, **kw):
        calls["plan_mode"] = plan_mode
        calls["agents"] = agents
        calls["prompt"] = prompt
        return [
            ExternalAgentResult("codex", ok=True, text="1. Check the VCF path is inside the workspace."),
            ExternalAgentResult("claude", ok=False, error="not installed"),  # failures ignored
        ]

    monkeypatch.setattr(extorch, "consult_external_agents", fake_consult)
    shell = _shell(tmp_path, enabled=True)

    out = shell._maybe_escalate_step_failure(None, _STEP, "VCF not found")

    assert calls["plan_mode"] is True  # agents are asked to plan, not execute
    assert "VCF not found" in calls["prompt"] and "Run VCF QC" in calls["prompt"]
    assert "Check the VCF path" in out["external_advice"]
    assert out["external_agents"] == ["codex"]  # only the successful agent
    assert out["repair_options"]


def test_escalation_is_defensive_on_error(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(extorch, "consult_external_agents", boom)
    shell = _shell(tmp_path, enabled=True)
    assert shell._maybe_escalate_step_failure(None, _STEP, "x") == {}  # never raises


def test_no_successful_advice_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(extorch, "consult_external_agents",
                        lambda *a, **k: [ExternalAgentResult("codex", ok=False, error="timeout")])
    shell = _shell(tmp_path, enabled=True)
    assert shell._maybe_escalate_step_failure(None, _STEP, "x") == {}
