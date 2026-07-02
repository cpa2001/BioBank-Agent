"""Tests for the general runtime hook registry and its gated third-party execution.

The registry generalises the evidence-hook pattern: built-in hooks always run, but hooks
contributed by installed plugins (``trust="external"``) never execute until the owning plugin is
opted in. These tests pin the gating and the defensive semantics, then prove the runtime emits the
lifecycle events around a real ``run_turn``.
"""

from __future__ import annotations

from pathlib import Path

from biobank_agent.core.memory.action_graph import ActionGraph
from biobank_agent.core.tools.protocol import Capability, ToolSpec
from biobank_agent.core.tools.registry import ToolRegistry
from biobank_agent.runtime import (
    AgentRuntime,
    FakeProvider,
    ProviderResponse,
    ProviderRouter,
    RuntimeConfig,
    SessionStore,
    ToolCall,
)
from biobank_agent.runtime.hooks import (
    ON_COMPLETE,
    ON_TURN_START,
    POST_TOOL,
    PRE_TOOL,
    TRUST_BUILTIN,
    TRUST_EXTERNAL,
    HookRegistry,
    default_registry,
    emit_hook,
    hook,
    make_command_hook,
    map_plugin_event,
)


# --- registry unit semantics -------------------------------------------------------------------


def test_builtin_hook_fires_and_external_is_gated_by_default():
    reg = HookRegistry()
    seen: list[str] = []
    reg.register(PRE_TOOL, lambda **kw: seen.append("builtin"), trust=TRUST_BUILTIN)
    reg.register(PRE_TOOL, lambda **kw: seen.append("external"), trust=TRUST_EXTERNAL, plugin="sp")

    outcomes = reg.emit(PRE_TOOL, tool="shell_exec")

    assert seen == ["builtin"], "external hook must not run without opt-in"
    statuses = {o.status for o in outcomes}
    assert {"fired", "skipped"} <= statuses


def test_external_hook_runs_when_plugin_opted_in():
    reg = HookRegistry()
    seen: list[str] = []
    reg.register(PRE_TOOL, lambda **kw: seen.append("external"), trust=TRUST_EXTERNAL, plugin="sp")

    reg.emit(PRE_TOOL, allowed_plugins={"sp"}, tool="x")
    assert seen == ["external"]

    seen.clear()
    reg.emit(PRE_TOOL, allowed_plugins={"other"}, tool="x")
    assert seen == [], "opting in a different plugin must not enable this one"


def test_global_allow_external_enables_all_external_hooks():
    reg = HookRegistry()
    seen: list[str] = []
    reg.register(PRE_TOOL, lambda **kw: seen.append("external"), trust=TRUST_EXTERNAL, plugin="sp")
    reg.emit(PRE_TOOL, allow_external=True, tool="x")
    assert seen == ["external"]


def test_a_raising_hook_never_breaks_emit():
    reg = HookRegistry()
    seen: list[str] = []
    reg.register(PRE_TOOL, lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")), name="raiser")
    reg.register(PRE_TOOL, lambda **kw: seen.append("after"), name="after")

    outcomes = reg.emit(PRE_TOOL, tool="x")  # must not raise

    assert seen == ["after"], "a later hook still runs after an earlier one raises"
    assert {o.name: o.status for o in outcomes}["raiser"] == "error"


def test_decorator_registration_and_unregister_plugin():
    reg = HookRegistry()

    @reg.hook(ON_COMPLETE, trust=TRUST_EXTERNAL, plugin="sp")
    def _done(**kw):
        return "done"

    assert len(reg.handles(ON_COMPLETE)) == 1
    removed = reg.unregister_plugin("sp")
    assert removed == 1 and reg.handles(ON_COMPLETE) == []


def test_command_hook_serialises_payload_and_drops_session_object():
    captured: dict[str, str] = {}

    def fake_runner(cmd: str, stdin: str):
        captured["cmd"] = cmd
        captured["stdin"] = stdin
        return 0, "ran"

    class _Sess:
        session_id = "sess-123"

    ch = make_command_hook("run.sh", event=PRE_TOOL, plugin="sp", runner=fake_runner)
    result = ch(tool="shell_exec", session=_Sess(), args={"cmd": "ls"})

    assert result == {"command": "run.sh", "returncode": 0, "output": "ran"}
    assert captured["cmd"] == "run.sh"
    assert '"session_id": "sess-123"' in captured["stdin"]
    assert '"tool": "shell_exec"' in captured["stdin"]


def test_map_plugin_event_normalises_claude_code_names():
    assert map_plugin_event("PreToolUse") == PRE_TOOL
    assert map_plugin_event("PostToolUse") == POST_TOOL
    assert map_plugin_event("Stop") == ON_COMPLETE
    assert map_plugin_event("pre_tool") == PRE_TOOL  # already a lifecycle name
    assert map_plugin_event("SomethingCustom") == "SomethingCustom"


def test_default_registry_backs_module_level_api():
    reg = default_registry()
    reg.clear()
    seen: list[str] = []

    @hook(ON_TURN_START)
    def _on_start(**kw):
        seen.append("start")

    emit_hook(ON_TURN_START, text="hi")
    assert seen == ["start"]
    reg.clear()


# --- runtime integration -----------------------------------------------------------------------


class _FakeTool:
    def __init__(self, name: str, result: dict) -> None:
        self._name = name
        self._result = result

    @property
    def name(self) -> str:
        return self._name

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self._name, description=self._name, parameters={})

    def required_capabilities(self):
        return frozenset({Capability.READ_DATA})

    @property
    def is_mutating(self) -> bool:
        return False

    async def handle(self, ctx):
        return dict(self._result)


def _runtime(tmp_path: Path, registry: ToolRegistry, hooks: HookRegistry, *, allow_hooks: bool = False):
    config = RuntimeConfig(primary_model="fake-model", approval_profile="full_auto", max_tool_rounds=2, plugin_allow_hooks=allow_hooks)
    fake = FakeProvider(
        scripted_responses=[
            ProviderResponse(text="", tool_calls=[ToolCall(id="c1", name="prevalence", args={"icd10_code": "E11"})], provider="fake", model="fake-model"),
            ProviderResponse(text="final answer", provider="fake", model="fake-model"),
        ],
        model="fake-model",
    )
    return AgentRuntime(
        provider_router=ProviderRouter({"fake-model": fake}, config),
        tool_registry=registry,
        session_store=SessionStore(tmp_path / "sessions"),
        action_graph=ActionGraph(tmp_path / "graph.db"),
        config=config,
        hooks=hooks,
    )


def test_run_turn_emits_lifecycle_hooks(tmp_path):
    registry = ToolRegistry()
    registry.register(_FakeTool("prevalence", {"summary": "ok"}))
    hooks = HookRegistry()
    events: list[tuple[str, str]] = []
    for ev in (ON_TURN_START, PRE_TOOL, POST_TOOL, ON_COMPLETE):
        hooks.register(ev, (lambda e: (lambda **kw: events.append((e, kw.get("tool", "")))))(ev))

    runtime = _runtime(tmp_path, registry, hooks)
    runtime.run_turn(runtime.create_session(title="demo", cwd=str(tmp_path)), "compute prevalence")

    kinds = [e for e, _ in events]
    assert ON_TURN_START in kinds and ON_COMPLETE in kinds
    assert ("pre_tool", "prevalence") in events
    assert ("post_tool", "prevalence") in events


def test_runtime_gates_external_plugin_hooks_until_opted_in(tmp_path):
    registry = ToolRegistry()
    registry.register(_FakeTool("prevalence", {"summary": "ok"}))
    hooks = HookRegistry()
    ran: list[str] = []
    hooks.register(PRE_TOOL, lambda **kw: ran.append("ext"), trust=TRUST_EXTERNAL, plugin="sp")

    # Default: gated — the external hook must not run during a real turn.
    runtime = _runtime(tmp_path, registry, hooks)
    runtime.run_turn(runtime.create_session(title="d", cwd=str(tmp_path)), "go")
    assert ran == [], "plugin hook ran without opt-in"

    # Per-plugin opt-in enables it. A fresh runtime shares the same registry but has its own
    # provider (the scripted FakeProvider is single-use per turn).
    runtime2 = _runtime(tmp_path, registry, hooks)
    runtime2.allow_plugin_hooks("sp")
    runtime2.run_turn(runtime2.create_session(title="d2", cwd=str(tmp_path)), "go again")
    assert ran == ["ext"]


def test_runtime_config_gate_opts_all_external_hooks_in(tmp_path):
    registry = ToolRegistry()
    registry.register(_FakeTool("prevalence", {"summary": "ok"}))
    hooks = HookRegistry()
    ran: list[str] = []
    hooks.register(PRE_TOOL, lambda **kw: ran.append("ext"), trust=TRUST_EXTERNAL, plugin="sp")

    runtime = _runtime(tmp_path, registry, hooks, allow_hooks=True)
    runtime.run_turn(runtime.create_session(title="d", cwd=str(tmp_path)), "go")
    assert ran == ["ext"], "plugin_allow_hooks=True should enable external hooks"
