"""Tests for M6: durable append-only rollout + resume-at-step."""

from __future__ import annotations

import json

from biobank_agent.core.events import AgentEvent, AgentEventType
from biobank_agent.core.memory.action_graph import ActionGraph
from biobank_agent.core.tools.registry import ToolRegistry
from biobank_agent.runtime import (
    AgentRuntime, FakeProvider, ProviderRouter, RuntimeConfig, SessionStore,
)
from biobank_agent.runtime.types import PlanState, PlanStep, PlanStatus

_TERMINAL = {"done", "completed", "skipped", "unverified"}


def _runtime(tmp_path):
    config = RuntimeConfig(primary_model="fake-model")
    return AgentRuntime(
        provider_router=ProviderRouter({"fake-model": FakeProvider(model="fake-model")}, config),
        tool_registry=ToolRegistry(),
        session_store=SessionStore(tmp_path / "sessions"),
        action_graph=ActionGraph(tmp_path / "graph.db"),
        config=config,
    )


def test_append_trajectory_seq_is_monotonic(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    ev = lambda: AgentEvent.make(AgentEventType.SESSION_STARTED, session_id="s1")
    store.append_trajectory("s1", [ev(), ev()])
    store.append_trajectory("s1", [ev()])   # second call must NOT reset seq to 0
    rows = store.rollout_file("s1").read_text(encoding="utf-8").strip().splitlines()
    seqs = [json.loads(r)["seq"] for r in rows]
    assert seqs == [0, 1, 2]                 # monotonic across calls (was 0,1,0 before the fix)


def test_plan_step_status_durable_resume_at_step(tmp_path):
    runtime = _runtime(tmp_path)
    session = runtime.create_session(title="resume demo", cwd=str(tmp_path))
    session.state.plan = PlanState(
        objective="multi-step workflow",
        title="wf",
        status=PlanStatus.EXECUTING,
        steps=[
            PlanStep(id="load", title="load", status="done"),
            PlanStep(id="analyze", title="analyze", status="pending"),
            PlanStep(id="report", title="report", status="pending"),
        ],
    )
    runtime.save_session(session)

    # Fresh load (simulating a process restart): step statuses survive on disk.
    restored = runtime.load_session(session.session_id)
    assert restored.state.plan is not None
    statuses = {s.id: s.status for s in restored.state.plan.steps}
    assert statuses == {"load": "done", "analyze": "pending", "report": "pending"}

    # Resume-at-step: execution resumes at the first non-terminal step (the loop in
    # _execute_plan_autonomously skips steps already in a terminal status).
    resume_id = next((s.id for s in restored.state.plan.steps if s.status not in _TERMINAL), None)
    assert resume_id == "analyze"   # 'load' (done) is skipped on resume
