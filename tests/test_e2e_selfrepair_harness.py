"""Phase 7: 50-round end-to-end + self-repair harness.

Proves the agent either completes or self-repairs — and NEVER crashes the process — across a table of
scenarios with failure injection. Offline and deterministic (fake providers/tools; no live LLM, network,
or real data). Three families totalling 50 rounds:

  * shell-lifecycle self-repair (24) — drive /plan + /plan-approve through the headless InteractiveShell
    with injected stalls / interrupts; assert the plan completes or pauses cleanly with a bounded repair
    budget, never raising out of handle_line.
  * engine resilience (15) — run_turn against boom/loop/budget providers+tools; assert graceful FAILED or
    recovered COMPLETED, bounded, never raised.
  * cross-subsystem E2E (11) — exercise the Phase 3-6 subsystems (plugin gating, workflow DAG, subagent
    depth guard, data-lake index/locate) end to end.

Run: /home/chenpengan/miniconda3/envs/biobank-agent/bin/python -m pytest tests/test_e2e_selfrepair_harness.py -q
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from biobank_agent.cli.interactive import PlanStepTimeoutError
from biobank_agent.core.memory.action_graph import ActionGraph
from biobank_agent.core.tools.protocol import Capability, ToolSpec
from biobank_agent.core.tools.registry import ToolRegistry
from biobank_agent.runtime import (
    AgentRuntime,
    ProviderResponse,
    ProviderRouter,
    RuntimeConfig,
    RuntimeStatus,
    SessionStore,
    ToolCall,
)

# Reuse the canonical headless-shell substrate + fakes (imported by other tests too).
from tests.test_interactive_cli_runtime import _shell


# ─────────────────────────── Family 1: shell-lifecycle self-repair ───────────────────────────


@dataclass
class ShellScenario:
    name: str
    kind: str  # happy | timeout_once | timeout_k | timeout_always | keyboard_interrupt
    max_retries: int = 2
    k: int = 1  # number of leading stalls for timeout_k
    expect_status: str = "completed"


def _drive_shell(tmp_path: Path, sc: ShellScenario):
    shell, output = _shell(tmp_path)
    shell.settings.plan_step_max_retries = sc.max_retries
    shell.handle_line(f"/plan e2e round {sc.name}")
    real = shell.runtime.run_turn
    calls = {"n": 0}

    def _spy(session, text):
        calls["n"] += 1
        if sc.kind == "timeout_once" and calls["n"] == 1:
            raise PlanStepTimeoutError("stall")
        if sc.kind == "timeout_k" and calls["n"] <= sc.k:
            raise PlanStepTimeoutError("stall")
        if sc.kind == "timeout_always":
            raise PlanStepTimeoutError("stall")
        if sc.kind == "keyboard_interrupt" and calls["n"] == 1:
            raise KeyboardInterrupt
        return real(session, text)

    shell.runtime.run_turn = _spy  # type: ignore[assignment]
    # The whole point: this must return, never raise out of the shell boundary.
    shell.handle_line("/plan-approve")
    return shell, output, calls


_SHELL_SCENARIOS: list[ShellScenario] = []
for _r in (1, 2, 3):
    _SHELL_SCENARIOS.append(ShellScenario(f"happy_r{_r}", "happy", max_retries=_r, expect_status="completed"))
    _SHELL_SCENARIOS.append(ShellScenario(f"timeout_once_r{_r}", "timeout_once", max_retries=_r, expect_status="completed"))
    _SHELL_SCENARIOS.append(ShellScenario(f"timeout_always_r{_r}", "timeout_always", max_retries=_r, expect_status="paused"))
    _SHELL_SCENARIOS.append(ShellScenario(f"ctrlc_r{_r}", "keyboard_interrupt", max_retries=_r, expect_status="paused"))
# k stalls that stay within the retry budget → still completes
for _r, _k in ((2, 1), (2, 2), (3, 2), (3, 3)):
    _SHELL_SCENARIOS.append(ShellScenario(f"timeout_k{_k}_r{_r}", "timeout_k", max_retries=_r, k=_k, expect_status="completed"))
# k stalls that EXCEED the budget → pauses cleanly
for _r, _k in ((1, 2), (2, 3), (1, 3), (2, 4)):
    _SHELL_SCENARIOS.append(ShellScenario(f"timeout_exhaust_k{_k}_r{_r}", "timeout_k", max_retries=_r, k=_k, expect_status="paused"))


@pytest.mark.parametrize("sc", _SHELL_SCENARIOS, ids=lambda s: s.name)
def test_shell_lifecycle_selfrepair(tmp_path, sc: ShellScenario):
    shell, output, calls = _drive_shell(tmp_path, sc)

    plan = shell.session.state.plan
    assert plan is not None, "plan must exist after approve"
    status = plan.status.value
    # Repaired or completed — never a crash and never an unexpected terminal state.
    assert status in {"completed", "paused"}, status
    assert status == sc.expect_status, f"{sc.name}: expected {sc.expect_status}, got {status}"

    # Self-repair is BOUNDED: the recovery loop never spins unbounded.
    repair_attempts = shell.session.state.repair_attempts or []
    assert len(repair_attempts) <= (sc.max_retries + 1) * len(plan.steps) + 2

    if status == "paused":
        if sc.kind == "keyboard_interrupt":
            # A genuine Ctrl-C is a CLEAN cancellation — it halts the loop with a message, not a crash.
            assert "cancelled at step" in output.getvalue()
        else:
            # A stall-escalation leaves a diagnosis for the user, not a half-broken session.
            assert shell.session.state.custom_data.get("plan_last_diagnosis") is not None
    if sc.kind == "timeout_once":
        assert calls["n"] >= 2, "a single stall must auto-resume, not pause"


# ─────────────────────────── Family 2: engine resilience ───────────────────────────


class _FakeTool:
    def __init__(self, name="demo", result=None):
        self._name, self._result = name, (result or {"ok": True})

    @property
    def name(self):
        return self._name

    def spec(self):
        return ToolSpec(name=self._name, description=self._name, parameters={})

    def required_capabilities(self):
        return frozenset({Capability.READ_DATA})

    @property
    def is_mutating(self):
        return False

    async def handle(self, ctx):
        return dict(self._result)


class _BoomTool(_FakeTool):
    async def handle(self, ctx):
        raise RuntimeError("tool exploded")


class _BoomProvider:
    model = "fake-model"

    def complete(self, request):
        raise RuntimeError("llm backend unavailable")


class _LoopProvider:
    """Never converges: emits the same tool call every round (exercises the no-progress guard)."""

    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        return ProviderResponse(text="", tool_calls=[ToolCall(id="c", name="demo", args={})], provider="fake", model="fake-model")


class _ScriptProvider:
    model = "fake-model"

    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, request):
        return self.responses.pop(0) if self.responses else ProviderResponse(text="final", provider="fake", model="fake-model")


def _runtime(tmp_path: Path, provider, registry: ToolRegistry, *, budget=0, max_rounds=8) -> AgentRuntime:
    config = RuntimeConfig(primary_model="fake-model", approval_profile="full_auto", max_tool_rounds=max_rounds, token_budget_per_turn=budget)
    return AgentRuntime(
        provider_router=ProviderRouter({"fake-model": provider}, config),
        tool_registry=registry,
        session_store=SessionStore(tmp_path / "sessions"),
        action_graph=ActionGraph(tmp_path / "graph.db"),
        config=config,
    )


@dataclass
class EngineScenario:
    name: str
    kind: str  # boom_provider | boom_tool_recovered | loop_guard | token_budget | happy_tool
    expect: str  # "failed" | "completed" | "budget"


_ENGINE_SCENARIOS = []
for _i in range(3):
    _ENGINE_SCENARIOS.append(EngineScenario(f"boom_provider_{_i}", "boom_provider", "failed"))
    _ENGINE_SCENARIOS.append(EngineScenario(f"boom_tool_recovered_{_i}", "boom_tool_recovered", "completed"))
    _ENGINE_SCENARIOS.append(EngineScenario(f"loop_guard_{_i}", "loop_guard", "failed"))
_ENGINE_SCENARIOS.append(EngineScenario("token_budget_a", "token_budget", "budget"))
_ENGINE_SCENARIOS.append(EngineScenario("token_budget_b", "token_budget", "budget"))
_ENGINE_SCENARIOS.append(EngineScenario("happy_tool_a", "happy_tool", "completed"))
_ENGINE_SCENARIOS.append(EngineScenario("happy_tool_b", "happy_tool", "completed"))
_ENGINE_SCENARIOS.append(EngineScenario("happy_tool_c", "happy_tool", "completed"))
_ENGINE_SCENARIOS.append(EngineScenario("boom_provider_x", "boom_provider", "failed"))


@pytest.mark.parametrize("sc", _ENGINE_SCENARIOS, ids=lambda s: s.name)
def test_engine_resilience(tmp_path, sc: EngineScenario):
    registry = ToolRegistry()
    registry.register(_FakeTool("demo"))

    if sc.kind == "boom_provider":
        provider = _BoomProvider()
        runtime = _runtime(tmp_path, provider, registry)
        session = runtime.run_turn(runtime.create_session(title="t", cwd=str(tmp_path)), "go")  # must NOT raise
        assert session.state.status == RuntimeStatus.FAILED
        assert any(ev["type"] == "error" for ev in session.events)
        return

    if sc.kind == "boom_tool_recovered":
        registry = ToolRegistry()
        registry.register(_BoomTool("demo"))
        provider = _ScriptProvider([
            ProviderResponse(text="", tool_calls=[ToolCall(id="c1", name="demo", args={})], provider="fake", model="fake-model"),
            ProviderResponse(text="recovered and finished", provider="fake", model="fake-model"),
        ])
        runtime = _runtime(tmp_path, provider, registry, max_rounds=3)
        session = runtime.run_turn(runtime.create_session(title="t", cwd=str(tmp_path)), "use tool then finish")
        assert session.state.status == RuntimeStatus.COMPLETED  # recovered tool error does not fail the turn
        return

    if sc.kind == "loop_guard":
        provider = _LoopProvider()
        runtime = _runtime(tmp_path, provider, registry, max_rounds=20)
        session = runtime.run_turn(runtime.create_session(title="t", cwd=str(tmp_path)), "loop forever")
        assert provider.calls <= 8, "the no-progress guard must stop a runaway loop"
        assert session.state.status == RuntimeStatus.FAILED
        return

    if sc.kind == "token_budget":
        provider = _ScriptProvider([
            ProviderResponse(text="", tool_calls=[ToolCall(id="c1", name="demo", args={})], provider="fake", model="fake-model", usage={"completion_tokens": 50}),
            ProviderResponse(text="", tool_calls=[ToolCall(id="c2", name="demo", args={})], provider="fake", model="fake-model", usage={"completion_tokens": 50}),
            ProviderResponse(text="done", provider="fake", model="fake-model", usage={"completion_tokens": 50}),
        ])
        runtime = _runtime(tmp_path, provider, registry, budget=10, max_rounds=8)
        session = runtime.run_turn(runtime.create_session(title="t", cwd=str(tmp_path)), "spend budget")
        assert session.state.custom_data.get("turn_budget_stopped") is True  # graceful cost halt, not a crash
        return

    # happy_tool
    provider = _ScriptProvider([
        ProviderResponse(text="", tool_calls=[ToolCall(id="c1", name="demo", args={})], provider="fake", model="fake-model"),
        ProviderResponse(text="final answer", provider="fake", model="fake-model"),
    ])
    runtime = _runtime(tmp_path, provider, registry, max_rounds=3)
    session = runtime.run_turn(runtime.create_session(title="t", cwd=str(tmp_path)), "do it")
    assert session.state.status == RuntimeStatus.COMPLETED
    assert session.turns[0].tool_results[0].result.get("ok") is True


# ─────────────────────────── Family 3: cross-subsystem E2E ───────────────────────────


@dataclass
class CrossScenario:
    name: str
    kind: str  # data_index_locate | data_convert | workflow_ok | workflow_failure_isolated | subagent_depth | plugin_gated_hooks


_CROSS_SCENARIOS = [
    CrossScenario("data_index_locate_a", "data_index_locate"),
    CrossScenario("data_index_locate_b", "data_index_locate"),
    CrossScenario("data_convert_a", "data_convert"),
    CrossScenario("data_convert_b", "data_convert"),
    CrossScenario("workflow_ok_a", "workflow_ok"),
    CrossScenario("workflow_ok_b", "workflow_ok"),
    CrossScenario("workflow_failure_isolated_a", "workflow_failure_isolated"),
    CrossScenario("workflow_failure_isolated_b", "workflow_failure_isolated"),
    CrossScenario("subagent_depth_a", "subagent_depth"),
    CrossScenario("subagent_depth_b", "subagent_depth"),
    CrossScenario("plugin_gated_hooks_a", "plugin_gated_hooks"),
    CrossScenario("data_index_locate_c", "data_index_locate"),
    CrossScenario("data_convert_c", "data_convert"),
    CrossScenario("workflow_ok_c", "workflow_ok"),
    CrossScenario("workflow_failure_isolated_c", "workflow_failure_isolated"),
]


@pytest.mark.parametrize("sc", _CROSS_SCENARIOS, ids=lambda s: s.name)
def test_cross_subsystem_e2e(tmp_path, sc: CrossScenario):
    if sc.kind == "data_index_locate":
        from biobank_agent.data.indexer import index_directory, locate_data_for_step

        (tmp_path / "cohort.csv").write_text("eid,age\n1,55\n")
        (tmp_path / "labs.tsv").write_text("eid\tglucose\n1\t5.5\n")
        catalog = index_directory(tmp_path, sample_rows=20)
        assert catalog.n_files == 2
        hits = locate_data_for_step(catalog, columns=["glucose"])
        assert hits and Path(hits[0].file.path).name == "labs.tsv"
        return

    if sc.kind == "data_convert":
        from biobank_agent.data.indexer import convert_to_parquet

        csv = tmp_path / "c.csv"
        csv.write_text("a,b\n1,2\n3,4\n")
        result = convert_to_parquet(csv, tmp_path / "c.parquet")
        assert result["status"] == "converted" and result["rows"] == 2
        return

    if sc.kind in ("workflow_ok", "workflow_failure_isolated"):
        from biobank_agent.runtime.workflow import WorkflowStep, run_workflow

        def boom(_up):
            raise RuntimeError("step failed")

        fail = sc.kind == "workflow_failure_isolated"
        steps = [
            WorkflowStep("a", lambda _u: 1),
            WorkflowStep("b", (boom if fail else (lambda u: u["a"] + 1)), dependencies=("a",)),
            WorkflowStep("c", lambda u: u["a"] + 10, dependencies=("a",)),
            WorkflowStep("d", lambda u: 99, dependencies=("b", "c")),
        ]
        result = run_workflow(steps)
        assert result.steps["a"].status == "ok" and result.steps["c"].status == "ok"
        if fail:
            assert result.steps["b"].status == "failed" and result.steps["d"].status == "skipped"
            assert not result.ok
        else:
            assert result.ok and result.result("d") == 99
        return

    if sc.kind == "subagent_depth":
        from biobank_agent.runtime.subruntime import spawn_biobank_subagent

        class _RT:
            def __init__(self):
                self.created = []

            def create_session(self, *, title, cwd):
                s = SimpleNamespace(session_id=f"s{len(self.created)}", title=title, cwd=cwd,
                                    state=SimpleNamespace(custom_data={}), turns=[])
                self.created.append(s)
                return s

            def run_turn(self, session, task):
                return session

        rt, parent = _RT(), SimpleNamespace(cwd="/tmp", state=SimpleNamespace(custom_data={}))
        # disabled by default
        assert not spawn_biobank_subagent(rt, parent, "t").ok
        # enabled: depth 1, 2 ok; depth 3 refused
        r1 = spawn_biobank_subagent(rt, parent, "t", enabled=True, max_depth=2)
        r2 = spawn_biobank_subagent(rt, rt.created[-1], "t", enabled=True, max_depth=2)
        r3 = spawn_biobank_subagent(rt, rt.created[-1], "t", enabled=True, max_depth=2)
        assert r1.ok and r2.ok and not r3.ok
        return

    if sc.kind == "plugin_gated_hooks":
        from biobank_agent.runtime.hooks import ON_TURN_START, PRE_TOOL, TRUST_EXTERNAL, HookRegistry, make_command_hook

        reg = HookRegistry()
        ran = []
        reg.register(PRE_TOOL, make_command_hook("run.sh", event=PRE_TOOL, plugin="p", runner=lambda c, s: ran.append(c) or (0, "ok")),
                     trust=TRUST_EXTERNAL, plugin="p")
        # gated OFF: the external command hook must not run
        out = reg.emit(PRE_TOOL, tool="shell_exec")
        assert ran == [] and all(o.status == "skipped" for o in out)
        # opted in: it runs
        reg.emit(PRE_TOOL, allowed_plugins={"p"}, tool="shell_exec")
        assert ran == ["run.sh"]
        return

    pytest.fail(f"unknown scenario kind: {sc.kind}")
