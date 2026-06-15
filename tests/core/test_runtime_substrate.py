"""Tests for the runtime substrate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biobank_agent.core.events import AgentEvent, AgentEventType
from biobank_agent.core.memory.action_graph import ActionGraph
from biobank_agent.core.tools.protocol import Capability, ToolSpec
from biobank_agent.core.tools.registry import ToolRegistry
from biobank_agent.runtime import (
    AgentEvent,
    AgentEventType,
    AgentRuntime,
    AgentSession,
    FakeProvider,
    ProviderRequest,
    ProviderResponse,
    ProviderRole,
    ProviderRouter,
    RuntimeConfig,
    RuntimeState,
    RuntimeStatus,
    SessionStore,
    ToolCall,
    ToolResult,
)
from biobank_agent.runtime.types import ApprovalRequest, AssistantMessage, UserTurn


def _runtime_with_registry(tmp_path: Path, registry: ToolRegistry, *, profile: str = "full_auto") -> AgentRuntime:
    config = RuntimeConfig(primary_model="fake-model", approval_profile=profile, max_tool_rounds=2)
    return AgentRuntime(
        provider_router=ProviderRouter({"fake-model": FakeProvider(model="fake-model")}, config),
        tool_registry=registry,
        session_store=SessionStore(tmp_path / "sessions"),
        action_graph=ActionGraph(tmp_path / "graph.db"),
        config=config,
    )


class _FakeTool:
    def __init__(self, name: str, result: dict[str, object], caps=frozenset({Capability.READ_DATA})) -> None:
        self._name = name
        self._result = result
        self._caps = caps

    @property
    def name(self) -> str:
        return self._name

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self._name, description=self._name, parameters={})

    def required_capabilities(self):
        return self._caps

    @property
    def is_mutating(self) -> bool:
        return False

    async def handle(self, ctx):
        return dict(self._result)


def test_runtime_types_roundtrip():
    msg = AssistantMessage(
        id="m1",
        text="hello",
        tool_calls=[ToolCall(id="c1", name="prevalence", args={"icd10_code": "E11"})],
        model="deepseek/deepseek-v4-pro",
        provider="fake",
        usage={"prompt_tokens": 10},
    )
    turn = UserTurn(
        id="t1",
        content="task",
        assistant_messages=[msg],
        tool_results=[ToolResult(call_id="c1", name="prevalence", result={"summary": "ok"})],
        approvals=[ApprovalRequest(call_id="c1", tool_name="prevalence", decision="allow")],
        status=RuntimeStatus.COMPLETED.value,
    )
    session = AgentSession(
        session_id="s1",
        title="title",
        cwd="/tmp",
        config=RuntimeConfig(),
        state=RuntimeState(status=RuntimeStatus.RUNNING, summary="summary"),
        turns=[turn],
        events=[AgentEvent.make(AgentEventType.SESSION_STARTED, session_id="s1").to_dict()],
    )

    payload = session.to_dict()
    restored = AgentSession.from_dict(payload)
    assert restored.session_id == "s1"
    assert restored.turns[0].assistant_messages[0].tool_calls[0].name == "prevalence"
    assert restored.state.summary == "summary"


def test_session_store_save_and_load(tmp_path):
    store = AgentRuntime(
        provider_router=ProviderRouter({"fake-model": FakeProvider()}),
        tool_registry=ToolRegistry(),
        session_store=SessionStore(tmp_path),
        action_graph=ActionGraph(tmp_path / "graph.db"),
        config=RuntimeConfig(),
    ).session_store

    session = AgentSession(
        session_id="s1",
        title="title",
        cwd="/work",
        config=RuntimeConfig(),
    )
    session.events.append(AgentEvent.make(AgentEventType.SESSION_STARTED, session_id="s1").to_dict())
    path = store.save(session, events=[AgentEvent.make(AgentEventType.SESSION_STARTED, session_id="s1")])
    assert path.exists()
    restored = store.load("s1")
    assert restored.session_id == "s1"
    assert restored.events[0]["type"] == "session_started"


def test_runtime_save_session_does_not_duplicate_trajectory_rows(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    runtime = AgentRuntime(
        provider_router=ProviderRouter({"fake-model": FakeProvider()}),
        tool_registry=ToolRegistry(),
        session_store=store,
        action_graph=ActionGraph(tmp_path / "graph.db"),
        config=RuntimeConfig(),
    )

    session = runtime.create_session(title="demo", cwd=str(tmp_path))
    session.events.append(AgentEvent.make(AgentEventType.SESSION_STARTED, session_id=session.session_id).to_dict())
    runtime.save_session(session)
    session.events.append(AgentEvent.make(AgentEventType.USER_TURN_STARTED, session_id=session.session_id, turn_id="t1", text="hello").to_dict())
    runtime.save_session(session)

    trajectory = store.rollout_file(session.session_id).read_text(encoding="utf-8").strip().splitlines()
    assert len(trajectory) == len(session.events)
    assert json.loads(trajectory[-1])["event"]["type"] == "session_saved"
    assert any(json.loads(row)["event"]["type"] == "checkpoint_created" for row in trajectory)


def test_provider_router_and_fake_provider():
    fake = FakeProvider(scripted_responses=[ProviderResponse(text="hello", provider="fake")], model="fake-model")
    router = ProviderRouter({"fake-model": fake}, RuntimeConfig(primary_model="fake-model"))
    req = ProviderRequest(session_id="s1", turn_id="t1", role=ProviderRole.PRIMARY_EXECUTOR, messages=[{"role": "user", "content": "x"}], model="fake-model")
    provider = router.resolve(ProviderRole.PRIMARY_EXECUTOR)
    resp = provider.complete(req)
    assert resp.text == "hello"
    assert fake.requests[0].session_id == "s1"


def test_runtime_config_resolves_role_models():
    config = RuntimeConfig(
        primary_model="primary",
        planner_model="planner",
        critic_model="critic",
        summarizer_model="summary",
        safety_reviewer_model="safety",
    )
    assert config.model_for_role(ProviderRole.PRIMARY_EXECUTOR) == "primary"
    assert config.model_for_role(ProviderRole.PLANNER) == "planner"
    assert config.model_for_role(ProviderRole.CRITIC) == "critic"
    assert config.model_for_role(ProviderRole.SUMMARIZER) == "summary"
    assert config.model_for_role(ProviderRole.SAFETY_REVIEWER) == "safety"


def test_runtime_can_run_fake_tool_loop(tmp_path):
    registry = ToolRegistry()
    registry.register(_FakeTool("prevalence", {"summary": "ok", "n_cases": 10}))
    fake = FakeProvider(
        scripted_responses=[
            ProviderResponse(
                text="",
                tool_calls=[ToolCall(id="c1", name="prevalence", args={"icd10_code": "E11"})],
                provider="fake",
                model="fake-model",
            ),
            ProviderResponse(text="final answer", provider="fake", model="fake-model"),
        ],
        model="fake-model",
    )
    router = ProviderRouter({"fake-model": fake}, RuntimeConfig(primary_model="fake-model", max_tool_rounds=2))
    runtime = AgentRuntime(
        provider_router=router,
        tool_registry=registry,
        session_store=SessionStore(tmp_path / "sessions"),
        action_graph=ActionGraph(tmp_path / "graph.db"),
        config=RuntimeConfig(primary_model="fake-model", max_tool_rounds=2),
    )
    session = runtime.create_session(title="demo", cwd=str(tmp_path))
    out = runtime.run_turn(session, "compute prevalence")
    assert out.turns[0].tool_results[0].result["summary"] == "ok"
    assert out.state.status in {RuntimeStatus.COMPLETED, RuntimeStatus.FAILED}
    assert any(ev["type"] == "session_started" for ev in out.events)
    assert any(ev["type"] == "checkpoint_created" for ev in out.events)
    assert any(ev["type"] == "tool_call_completed" for ev in out.events)
    assert (tmp_path / "sessions" / out.session_id / "session.json").exists()
    assert (tmp_path / "sessions" / out.session_id / "trajectory.jsonl").exists()
    assert out.action_graph_refs[0].node_type == "tool"
    graph_hits = runtime.action_graph.search_nodes("prevalence", node_types=["tool"], limit=10)
    assert graph_hits
    assert graph_hits[0]["node_id"] == "c1"


# --- Robustness: turn status, provider failure, and the no-progress loop guard ---

class _BoomTool(_FakeTool):
    async def handle(self, ctx):
        raise RuntimeError("tool exploded")


class _BoomProvider:
    def __init__(self) -> None:
        self.model = "fake-model"
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        raise RuntimeError("llm backend unavailable")


class _LoopProvider:
    """Never converges: returns the same tool call every round."""

    def __init__(self) -> None:
        self.model = "fake-model"
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        return ProviderResponse(
            text="",
            tool_calls=[ToolCall(id=f"c{self.calls}", name="boom", args={"x": 1})],
            provider="fake",
            model="fake-model",
        )


def _runtime_with(tmp_path: Path, registry: ToolRegistry, provider, *, max_rounds: int = 8) -> AgentRuntime:
    config = RuntimeConfig(primary_model="fake-model", approval_profile="full_auto", max_tool_rounds=max_rounds)
    return AgentRuntime(
        provider_router=ProviderRouter({"fake-model": provider}, config),
        tool_registry=registry,
        session_store=SessionStore(tmp_path / "sessions"),
        action_graph=ActionGraph(tmp_path / "graph.db"),
        config=config,
    )


def test_turn_completed_despite_recovered_tool_error(tmp_path):
    """A turn that CONVERGES to a final answer is COMPLETED even if an intermediate
    tool call errored (the model recovered). The old rule marked it FAILED on any
    tool error, conflating a recovered error with turn failure."""
    registry = ToolRegistry()
    registry.register(_BoomTool("boom", {}))
    provider = FakeProvider(
        scripted_responses=[
            ProviderResponse(text="", tool_calls=[ToolCall(id="c1", name="boom", args={})], provider="fake", model="fake-model"),
            ProviderResponse(text="the tool failed but here is my answer", provider="fake", model="fake-model"),
        ],
        model="fake-model",
    )
    runtime = _runtime_with(tmp_path, registry, provider, max_rounds=8)
    session = runtime.create_session(title="recover", cwd=str(tmp_path))
    runtime.run_turn(session, "use the tool")
    turn = session.turns[-1]
    assert any(r.error for r in turn.tool_results), "the tool should have errored"
    assert turn.status == RuntimeStatus.COMPLETED.value, "a converged turn must be COMPLETED despite a recovered tool error"


def test_empty_final_response_does_not_complete_turn(tmp_path):
    """A degenerate empty response (no text, no tool calls, no prior tool work) must NOT be treated
    as a converged final answer — that would mark a plan step COMPLETED with zero output (field
    problem #7: a finished run with no result). It fails honestly instead."""
    provider = FakeProvider(
        scripted_responses=[ProviderResponse(text="", tool_calls=[], provider="fake", model="fake-model")],
        model="fake-model",
    )
    runtime = _runtime_with(tmp_path, ToolRegistry(), provider, max_rounds=1)
    session = runtime.create_session(title="empty", cwd=str(tmp_path))
    runtime.run_turn(session, "answer the question")
    turn = session.turns[-1]
    assert turn.status == RuntimeStatus.FAILED.value, "an empty, output-less turn must not be COMPLETED"


def test_empty_response_is_nudged_then_recovers(tmp_path):
    """An empty first response is nudged once for a real answer; if the model then answers, the turn
    COMPLETES — graceful recovery rather than a silent, output-less success."""
    provider = FakeProvider(
        scripted_responses=[
            ProviderResponse(text="", tool_calls=[], provider="fake", model="fake-model"),
            ProviderResponse(text="here is the actual answer", provider="fake", model="fake-model"),
        ],
        model="fake-model",
    )
    runtime = _runtime_with(tmp_path, ToolRegistry(), provider, max_rounds=4)
    session = runtime.create_session(title="nudge", cwd=str(tmp_path))
    runtime.run_turn(session, "answer the question")
    turn = session.turns[-1]
    assert turn.status == RuntimeStatus.COMPLETED.value
    assert any("empty model response" in str((e.get("payload") or {}).get("message", "")) for e in session.events), "a nudge must be recorded"


def test_empty_final_after_tool_work_still_completes(tmp_path):
    """A turn that did real tool work and then returns an empty final message still COMPLETES — the
    tool results are real output, so it is not the output-less degenerate case."""
    registry = ToolRegistry()
    registry.register(_BoomTool("boom", {}))
    provider = FakeProvider(
        scripted_responses=[
            ProviderResponse(text="", tool_calls=[ToolCall(id="c1", name="boom", args={})], provider="fake", model="fake-model"),
            ProviderResponse(text="", tool_calls=[], provider="fake", model="fake-model"),
        ],
        model="fake-model",
    )
    runtime = _runtime_with(tmp_path, registry, provider, max_rounds=8)
    session = runtime.create_session(title="toolwork", cwd=str(tmp_path))
    runtime.run_turn(session, "use the tool then finish")
    turn = session.turns[-1]
    assert turn.tool_results, "the tool should have produced a result"
    assert turn.status == RuntimeStatus.COMPLETED.value, "empty final after real tool work is still completed"


def test_per_turn_token_budget_stops_turn_gracefully(tmp_path):
    """When completion tokens cross token_budget_per_turn, run_turn stops the round loop CLEANLY
    (records a budget event + sets turn_budget_stopped) instead of crashing or burning the whole
    round budget — the cost guard that lets the inactivity watchdog stay lenient toward a model that
    is genuinely producing output (a timeout must not stop a producing task; cost is what bounds it)."""
    registry = ToolRegistry()
    registry.register(_BoomTool("boom", {}))
    # Each round calls a tool (so the loop would otherwise keep going) and reports 100 completion tokens.
    provider = FakeProvider(
        scripted_responses=[
            ProviderResponse(text="", tool_calls=[ToolCall(id=f"c{i}", name="boom", args={"i": i})],
                             usage={"completion_tokens": 100}, provider="fake", model="fake-model")
            for i in range(6)
        ],
        model="fake-model",
    )
    config = RuntimeConfig(primary_model="fake-model", approval_profile="full_auto",
                           max_tool_rounds=8, token_budget_per_turn=150)
    runtime = AgentRuntime(
        provider_router=ProviderRouter({"fake-model": provider}, config),
        tool_registry=registry,
        session_store=SessionStore(tmp_path / "sessions"),
        action_graph=ActionGraph(tmp_path / "graph.db"),
        config=config,
    )
    session = runtime.create_session(title="budget", cwd=str(tmp_path))
    runtime.run_turn(session, "loop forever")  # must NOT raise
    # 100 tokens/round, budget 150 -> stops after round 2 (200 >= 150), not all 8 rounds.
    assert session.state.custom_data.get("turn_budget_stopped") is True
    assert any("token budget reached" in str((e.get("payload") or {}).get("message", ""))
               for e in session.events), "a clear budget-stop event must be recorded"
    assert len(provider.scripted_responses) == 4, "stopped after 2 rounds; 4 scripted responses remain"


def test_turn_fails_gracefully_on_provider_error(tmp_path):
    """A provider/LLM exception must NOT escape run_turn; the turn is FAILED and an
    ERROR event is recorded (previously the exception propagated uncaught)."""
    runtime = _runtime_with(tmp_path, ToolRegistry(), _BoomProvider(), max_rounds=4)
    session = runtime.create_session(title="provider-down", cwd=str(tmp_path))
    runtime.run_turn(session, "hello")  # must not raise
    turn = session.turns[-1]
    assert turn.status == RuntimeStatus.FAILED.value
    assert any("error" in str(e.get("type", "")).lower() for e in session.events)


class _RecordingProvider:
    def __init__(self) -> None:
        self.model = "fake-model"
        self.last_messages = None
        self.calls = 0

    def complete(self, request):
        self.last_messages = list(request.messages)
        self.calls += 1
        return ProviderResponse(text="ok done", provider="fake", model="fake-model")


def test_executor_turn_has_system_prompt_and_history(tmp_path):
    """The executor must send a system prompt (identity + workspace + tool-use
    discipline) and prior-turn history — previously it sent only the raw user
    message, which caused flailing and no cross-turn memory."""
    provider = _RecordingProvider()
    runtime = _runtime_with(tmp_path, ToolRegistry(), provider, max_rounds=2)
    session = runtime.create_session(title="sys", cwd=str(tmp_path))

    runtime.run_turn(session, "first question")
    msgs = provider.last_messages
    assert msgs[0]["role"] == "system"
    sys = msgs[0]["content"]
    assert "BioBank Agent" in sys
    assert "RUN it" in sys and "NEVER repeat" in sys  # tool-use discipline present
    assert str(tmp_path) in sys                        # workspace sandbox stated
    assert msgs[-1] == {"role": "user", "content": "first question"}

    runtime.run_turn(session, "second question")
    msgs2 = provider.last_messages
    assert msgs2[0]["role"] == "system"
    assert any(m["role"] == "user" and m["content"] == "first question" for m in msgs2), "prior turn missing"
    assert any(m["role"] == "assistant" and "ok done" in str(m["content"]) for m in msgs2), "prior answer missing"
    assert msgs2[-1] == {"role": "user", "content": "second question"}


class _DistinctProvider:
    """Returns N DISTINCT tool calls (different args), then a final answer."""

    def __init__(self, n: int = 6) -> None:
        self.model = "fake-model"
        self.n = n
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        if self.calls <= self.n:
            return ProviderResponse(
                text="",
                tool_calls=[ToolCall(id=f"c{self.calls}", name="noop", args={"i": self.calls})],
                provider="fake",
                model="fake-model",
            )
        return ProviderResponse(text="all done", provider="fake", model="fake-model")


def test_loop_guard_does_not_kill_legitimate_distinct_progress(tmp_path):
    """Six DISTINCT successful tool calls (different args) must NOT trip the guard
    — only CONSECUTIVE identical calls/errors do. Guards against false positives
    from arg-truncation or cumulative counting."""
    registry = ToolRegistry()
    registry.register(_FakeTool("noop", {"ok": True}))
    provider = _DistinctProvider(n=6)
    runtime = _runtime_with(tmp_path, registry, provider, max_rounds=10)
    session = runtime.create_session(title="distinct", cwd=str(tmp_path))
    runtime.run_turn(session, "do six distinct steps then finish")
    assert provider.calls == 7, f"should run all 6 steps + final answer, got {provider.calls}"
    assert session.turns[-1].status == RuntimeStatus.COMPLETED.value


def test_no_progress_loop_guard_stops_runaway(tmp_path):
    """The model repeating the same failing tool call must be stopped well before
    burning all max_tool_rounds (observed 33 calls retrying a rejected path)."""
    registry = ToolRegistry()
    registry.register(_BoomTool("boom", {}))
    provider = _LoopProvider()
    runtime = _runtime_with(tmp_path, registry, provider, max_rounds=30)
    session = runtime.create_session(title="loop", cwd=str(tmp_path))
    runtime.run_turn(session, "loop forever")
    assert provider.calls <= 8, f"loop guard should stop early, got {provider.calls} rounds"
    assert session.turns[-1].status == RuntimeStatus.FAILED.value


def test_tool_registry_loads_native_tools_with_legacy(tmp_path):
    from biobank_agent.registry import SkillRegistry

    registry = ToolRegistry(legacy=SkillRegistry())
    registry.hydrate_from_legacy()
    names = {handler.name for handler in registry.list_handlers()}
    assert {"shell", "file_read", "file_write", "file_edit", "apply_patch", "search", "git_status", "git_diff", "test_runner"} <= names


def test_event_serialization_roundtrip():
    event = AgentEvent.make(AgentEventType.MODEL_DELTA, turn_id="t1", text="chunk")
    restored = AgentEvent.from_dict(event.to_dict())
    assert restored.type == AgentEventType.MODEL_DELTA
    assert restored.payload["text"] == "chunk"


def test_runtime_public_api_exports_event_model():
    from biobank_agent.runtime import AgentEvent as ExportedAgentEvent, AgentEventType as ExportedAgentEventType

    event = ExportedAgentEvent.make(ExportedAgentEventType.SESSION_STARTED, session_id="s-public")
    assert event.type == ExportedAgentEventType.SESSION_STARTED
    assert event.payload["session_id"] == "s-public"


def test_runtime_plan_goal_checkpoint_compact_roundtrip(tmp_path):
    runtime = AgentRuntime(
        provider_router=ProviderRouter({"fake-model": FakeProvider()}),
        tool_registry=ToolRegistry(),
        session_store=SessionStore(tmp_path / "sessions"),
        action_graph=ActionGraph(tmp_path / "graph.db"),
        config=RuntimeConfig(),
    )
    session = runtime.create_session(title="demo", cwd=str(tmp_path))

    plan = runtime.update_plan(session, "Build a robust long-running workflow")
    goal = runtime.set_goal(session, "Finish long-running workflow intelligence")
    runtime.approve_plan(session)
    runtime.record_verification(session, command="pytest tests/core", status="failed", summary="one check failed", returncode=1)
    runtime.record_repair_attempt(session, reason="fix failed check", attempt=1, max_attempts=2, status="attempted")
    runtime.compact_session(session)
    runtime.save_session(session)

    restored = runtime.load_session(session.session_id)
    assert restored.state.plan is not None
    assert restored.state.plan.objective == plan.objective
    assert restored.state.goal is not None
    assert restored.state.goal.objective == goal.objective
    compacted = restored.state.compacted_summary
    assert compacted["objective"] == goal.objective
    assert compacted["plan"]["objective"] == plan.objective
    assert compacted["verification_results"][0]["status"] == "failed"
    assert compacted["failed_attempts"][0]["attempt"] == 1
    assert restored.state.checkpoints
    event_types = [event["type"] for event in restored.events]
    assert "plan_updated" in event_types
    assert "goal_updated" in event_types
    assert "verification_completed" in event_types
    assert "repair_attempted" in event_types
    assert "compact_completed" in event_types
    assert runtime.session_store.checkpoint_file(session.session_id, restored.state.checkpoints[-1].id).exists()


def test_runtime_records_file_change_checkpoint_for_mutating_tool(tmp_path):
    from biobank_agent.core.tools.native import WriteFileTool

    registry = ToolRegistry()
    registry.register(WriteFileTool())
    fake = FakeProvider(
        scripted_responses=[
            ProviderResponse(
                text="",
                tool_calls=[ToolCall(id="w1", name="file_write", args={"path": "written.txt", "content": "hello"})],
                provider="fake",
                model="fake-model",
            ),
            ProviderResponse(text="done", provider="fake", model="fake-model"),
        ],
        model="fake-model",
    )
    runtime = AgentRuntime(
        provider_router=ProviderRouter({"fake-model": fake}, RuntimeConfig(primary_model="fake-model", approval_profile="full_auto", max_tool_rounds=2)),
        tool_registry=registry,
        session_store=SessionStore(tmp_path / "sessions"),
        action_graph=ActionGraph(tmp_path / "graph.db"),
        config=RuntimeConfig(primary_model="fake-model", approval_profile="full_auto", max_tool_rounds=2),
    )
    session = runtime.create_session(title="demo", cwd=str(tmp_path))

    runtime.run_turn(session, "write a file")

    assert (tmp_path / "written.txt").read_text(encoding="utf-8") == "hello"
    assert any(checkpoint.reason == "file_change" for checkpoint in session.state.checkpoints)
    file_change = next(checkpoint for checkpoint in session.state.checkpoints if checkpoint.reason == "file_change")
    assert runtime.session_store.checkpoint_file(session.session_id, file_change.id).exists()


def test_runtime_invoke_tool_records_approval_allow_and_deny(tmp_path):
    from biobank_agent.core.tools.native import ReadFileTool, WriteFileTool
    from biobank_agent.runtime.audit import audit_session

    (tmp_path / "input.txt").write_text("hello", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    runtime = _runtime_with_registry(tmp_path, registry, profile="read_only")
    session = runtime.create_session(title="approval", cwd=str(tmp_path))

    allowed = runtime.invoke_tool(session, "file_read", {"path": "input.txt"}, call_id="read-1")
    denied = runtime.invoke_tool(session, "file_write", {"path": "blocked.txt", "content": "no"}, call_id="write-denied")
    runtime.save_session(session)

    assert allowed.error == ""
    assert allowed.result["status"] == "ok"
    assert denied.error
    assert not (tmp_path / "blocked.txt").exists()
    approvals = [
        event["payload"]["approval"]
        for event in session.events
        if event["type"] == "approval_requested"
    ]
    assert [approval["decision"] for approval in approvals] == ["allow", "deny"]
    assert any(event["type"] == "tool_call_completed" and event["payload"]["state"] == "failed" for event in session.events)
    report = audit_session(runtime, session)
    assert any(approval["decision"] == "deny" for approval in report.approvals)
    assert any("Approval denied" in warning for warning in report.safety_warnings)
    assert all(ref.node_id in {"read-1", "write-denied"} for ref in session.action_graph_refs)


def test_runtime_invoke_tool_executes_file_edit_and_patch_paths(tmp_path):
    from biobank_agent.core.tools.native import ApplyPatchTool, EditFileTool, WriteFileTool

    registry = ToolRegistry()
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(ApplyPatchTool())
    runtime = _runtime_with_registry(tmp_path, registry, profile="full_auto")
    session = runtime.create_session(title="file tools", cwd=str(tmp_path))

    write = runtime.invoke_tool(session, "file_write", {"path": "notes.txt", "content": "alpha\n"}, call_id="write-1")
    edit = runtime.invoke_tool(
        session,
        "file_edit",
        {"path": "notes.txt", "replacements": [{"find": "alpha", "replace": "beta"}]},
        call_id="edit-1",
    )
    patch = runtime.invoke_tool(
        session,
        "apply_patch",
        {"path": "notes.txt", "diff": "--- a/notes.txt\n+++ b/notes.txt\n@@\n-beta\n+gamma\n", "dry_run": True},
        call_id="patch-dry-run",
    )
    runtime.save_session(session)

    assert write.result["status"] == "ok"
    assert edit.result["status"] == "ok"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "beta\n"
    assert patch.result["status"] == "ok"
    assert patch.result["dry_run"] is True
    assert [ref.node_id for ref in session.action_graph_refs] == ["write-1", "edit-1", "patch-dry-run"]
    assert sum(1 for checkpoint in session.state.checkpoints if checkpoint.reason == "file_change") == 3
    assert runtime.session_store.rollout_file(session.session_id).exists()


def test_plan_mode_profile_denies_edits_and_shell(tmp_path):
    from biobank_agent.core.tools.approval import ApprovalPolicy, Decision, builtin_profile
    from biobank_agent.core.tools.native import ShellTool, WriteFileTool

    policy = ApprovalPolicy(builtin_profile("plan"))
    assert policy.decide(WriteFileTool(), {"path": "x", "content": "y"}).decision == Decision.DENY
    assert policy.decide(ShellTool(), {"command": "echo hi"}).decision == Decision.DENY


def test_auto_compact_threshold_preserves_required_state(tmp_path):
    runtime = AgentRuntime(
        provider_router=ProviderRouter({"fake-model": FakeProvider()}),
        tool_registry=ToolRegistry(),
        session_store=SessionStore(tmp_path / "sessions"),
        action_graph=ActionGraph(tmp_path / "graph.db"),
        config=RuntimeConfig(),
    )
    session = runtime.create_session(title="demo", cwd=str(tmp_path))
    runtime.set_goal(session, "Auto compact goal")
    runtime.update_plan(session, "Auto compact plan")

    compacted = runtime.maybe_auto_compact(session, event_threshold=1, turn_threshold=99)

    assert compacted is True
    assert session.state.compacted_summary["objective"] == "Auto compact goal"
    assert session.state.compacted_summary["plan"]["objective"] == "Auto compact plan"
    assert session.state.compacted_summary["trajectory_ids"]


def test_failed_verification_triggers_bounded_repair(tmp_path):
    runtime = AgentRuntime(
        provider_router=ProviderRouter({"fake-model": FakeProvider()}),
        tool_registry=ToolRegistry(),
        session_store=SessionStore(tmp_path / "sessions"),
        action_graph=ActionGraph(tmp_path / "graph.db"),
        config=RuntimeConfig(),
    )
    session = runtime.create_session(title="demo", cwd=str(tmp_path))
    runtime.update_plan(session, "Repair bounded workflow")

    runtime.verification_repair_loop(
        session,
        checks=[
            {"command": "pytest failing", "status": "failed", "summary": "first failure", "returncode": 1, "attempt": 1},
            {"command": "pytest failing", "status": "failed", "summary": "second failure", "returncode": 1, "attempt": 2},
            {"command": "pytest failing", "status": "failed", "summary": "third failure", "returncode": 1, "attempt": 3},
        ],
        max_attempts=2,
    )

    assert len(session.state.verification_results) == 3
    assert len(session.state.repair_attempts) == 2
    assert session.state.repair_attempts[-1]["status"] == "bounded_stop"
    event_types = [event["type"] for event in session.events]
    assert event_types.count("repair_attempted") == 2
    assert event_types.count("verification_completed") == 3


def test_run_verification_commands_executes_real_subprocess_and_bounded_stops(tmp_path):
    runtime = AgentRuntime(
        provider_router=ProviderRouter({"fake-model": FakeProvider()}),
        tool_registry=ToolRegistry(),
        session_store=SessionStore(tmp_path / "sessions"),
        action_graph=ActionGraph(tmp_path / "graph.db"),
        config=RuntimeConfig(),
    )
    session = runtime.create_session(title="demo", cwd=str(tmp_path))

    results = runtime.run_verification_commands(
        session,
        ["python -c 'import sys; sys.exit(1)'"],
        max_attempts=2,
        timeout_s=10,
    )

    assert len(results) == 2
    assert results[-1].status == "failed"
    assert len(session.state.repair_attempts) == 2
    assert session.state.repair_attempts[-1]["status"] == "bounded_stop"
    assert session.state.verification_results[-1].attempt == 2
    assert session.state.plan is None or session.state.plan.status.value == "failed"


def test_run_verification_commands_repair_callback_can_make_retry_pass(tmp_path):
    marker = tmp_path / "marker.txt"
    marker.write_text("fail", encoding="utf-8")
    runtime = AgentRuntime(
        provider_router=ProviderRouter({"fake-model": FakeProvider()}),
        tool_registry=ToolRegistry(),
        session_store=SessionStore(tmp_path / "sessions"),
        action_graph=ActionGraph(tmp_path / "graph.db"),
        config=RuntimeConfig(),
    )
    session = runtime.create_session(title="demo", cwd=str(tmp_path))
    seen: list[tuple[str, int]] = []
    command = (
        "python -c 'from pathlib import Path; import sys; "
        "sys.exit(0 if Path(\"marker.txt\").read_text().strip() == \"pass\" else 1)'"
    )

    def repair_fn(_session, command, attempt, result):
        seen.append((command, attempt))
        marker.write_text("pass", encoding="utf-8")

    results = runtime.run_verification_commands(
        session,
        [command],
        max_attempts=2,
        timeout_s=10,
        repair_fn=repair_fn,
    )

    assert seen == [(command, 1)]
    assert len(results) == 2
    assert results[-1].status == "passed"
    assert session.state.verification_results[-1].status == "passed"
    assert len(session.state.repair_attempts) == 1
    assert session.state.plan is None or session.state.plan.status.value == "completed"
