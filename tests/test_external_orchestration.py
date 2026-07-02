"""Phase 5: parallel + plan-mode fan-out over external coding-agent CLIs.

Offline and deterministic — the process runner and agent detector are injected, so no real CLI, model,
or network is touched. Pins: results return in spec order; a slow/failing agent never blocks the others;
plan mode wraps the prompt; detection filters to installed agents.
"""

from __future__ import annotations

import subprocess

from biobank_agent.runtime.external_agents import ExternalAgentResult, ExternalAgentSpec
from biobank_agent.runtime.external_orchestration import (
    consult_external_agents,
    plan_mode_prompt,
    run_external_agents_parallel,
)

_SPECS = [
    ExternalAgentSpec("codex", ("codex", "exec", "{prompt}")),
    ExternalAgentSpec("claude", ("claude", "-p", "{prompt}")),
    ExternalAgentSpec("gemini", ("gemini", "-p", "{prompt}")),
]


def test_parallel_fan_out_preserves_order_and_isolates_failures():
    def runner(argv, cwd, timeout_s, env):
        name = argv[0]
        if name == "claude":
            return 3, "", "boom"  # non-zero exit
        if name == "gemini":
            raise subprocess.TimeoutExpired(argv, timeout_s)  # hang
        return 0, f"PLAN from {name}", ""

    results = run_external_agents_parallel(_SPECS, "do X", timeout_s=5, runner=runner)

    assert [r.name for r in results] == ["codex", "claude", "gemini"]  # spec order preserved
    assert results[0].ok and "PLAN from codex" in results[0].text
    assert not results[1].ok and results[1].returncode == 3  # failure isolated, not raised
    assert not results[2].ok and "timed out" in results[2].error


def test_empty_specs_returns_empty():
    assert run_external_agents_parallel([], "x") == []


def test_plan_mode_prompt_wraps_request():
    wrapped = plan_mode_prompt("compare cohorts")
    assert "PLAN MODE" in wrapped and wrapped.endswith("compare cohorts")
    assert "Do NOT modify files" in wrapped


def test_consult_detects_installed_agents_and_can_plan(monkeypatch):
    seen_prompts: list[str] = []

    def runner(argv, cwd, timeout_s, env):
        seen_prompts.append(argv[-1])  # the {prompt} argv element
        return 0, f"ok {argv[0]}", ""

    def detector(names, *, overrides=None):
        # only codex + claude are "installed"
        return [s for s in _SPECS if s.name in names and s.name != "gemini"]

    results = consult_external_agents(
        ["codex", "claude", "gemini"], "analyze Y", plan_mode=True, timeout_s=5,
        detector=detector, runner=runner,
    )

    assert [r.name for r in results] == ["codex", "claude"]  # gemini filtered out (not installed)
    assert all(r.ok for r in results)
    assert all("PLAN MODE" in p and p.endswith("analyze Y") for p in seen_prompts)


def test_custom_invoker_is_honoured():
    calls: list[str] = []

    def invoker(spec, prompt, *, cwd="", timeout_s=0.0, runner=None):
        calls.append(spec.name)
        return ExternalAgentResult(spec.name, ok=True, text="stub")

    results = run_external_agents_parallel(_SPECS[:2], "p", invoker=invoker)
    assert sorted(calls) == ["claude", "codex"]
    assert all(r.text == "stub" for r in results)
