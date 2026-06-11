"""里程碑6: surface the dormant M15 run-tree online-eval substrate.

Covers the render helper, the audit-side fold (`_run_tree_eval`) and the two new
domain evaluators. Pure and offline — synthetic AgentEvent streams, no runtime.
"""

from __future__ import annotations

from types import SimpleNamespace

from rich.tree import Tree

from biobank_agent.cli.render import render_run_tree
from biobank_agent.core.events import AgentEvent, AgentEventType
from biobank_agent.runtime.audit import _run_tree_eval
from biobank_agent.runtime.run_eval import no_repeated_tool_failures, phase_errors_free
from biobank_agent.runtime.run_tree import build_run_tree, summarize_run_tree


def _tool(call_id: str, name: str, closer: AgentEventType) -> list[AgentEvent]:
    """Bus taxonomy (live scheduler)."""
    return [
        AgentEvent.make(AgentEventType.TOOL_STARTED, turn_id="t1", tool_call_id=call_id, name=name),
        AgentEvent.make(closer, turn_id="t1", tool_call_id=call_id),
    ]


def _persisted_tool(call_id: str, name: str, state: str = "done") -> list[AgentEvent]:
    """The taxonomy the engine actually persists into ``session.events``."""
    return [
        AgentEvent.make(AgentEventType.TOOL_CALL_STARTED, turn_id="t1", tool_call_id=call_id, tool=name),
        AgentEvent.make(AgentEventType.TOOL_CALL_COMPLETED, turn_id="t1", tool_call_id=call_id, tool=name, state=state),
    ]


def test_build_run_tree_folds_persisted_taxonomy() -> None:
    # Real sessions persist TOOL_CALL_STARTED/COMPLETED, not the bus TOOL_STARTED/RESULT — the
    # online eval is meaningless if the fold misses them (regression guard for the M15 wiring).
    events = _persisted_tool("c1", "vcf_qc", "done") + _persisted_tool("c2", "plink", "failed")
    summary = summarize_run_tree(build_run_tree(events))
    assert summary["tool_calls"] == 2
    assert summary["tool_errors"] == 1
    assert summary["tool_success_rate"] == 0.5


def test_render_run_tree_returns_rich_tree() -> None:
    tree = build_run_tree(_tool("c1", "vcf_qc", AgentEventType.TOOL_RESULT), session_id="s1")
    rendered = render_run_tree(tree)
    assert isinstance(rendered, Tree)
    assert rendered.children  # the turn node hangs off the session root


def test_run_tree_eval_helper_grades_a_clean_session() -> None:
    # Grade through the REAL persisted taxonomy (what audit_session feeds the helper).
    events = [ev.to_dict() for ev in _persisted_tool("c1", "vcf_qc", "done")]
    session = SimpleNamespace(events=events, session_id="s1")
    summary, online = _run_tree_eval(session)
    assert summary["tool_calls"] == 1
    assert online["passed"] is True


def test_run_tree_eval_helper_is_defensive_on_bad_events() -> None:
    session = SimpleNamespace(events=[{"type": "not-a-real-event"}], session_id="s1")
    summary, online = _run_tree_eval(session)
    # degrades rather than raising: either an empty summary or a vacuous pass
    assert isinstance(summary, dict) and isinstance(online, dict)


def test_no_repeated_tool_failures_flags_stuck_loop() -> None:
    events: list[AgentEvent] = []
    for i in range(3):
        events += _tool(f"c{i}", "plink", AgentEventType.TOOL_ERROR)
    tree = build_run_tree(events)
    (result,) = no_repeated_tool_failures()(tree)
    assert not result.passed and "plink" in result.note


def test_no_repeated_tool_failures_passes_on_distinct_tools() -> None:
    events = _tool("c1", "plink", AgentEventType.TOOL_ERROR) + _tool(
        "c2", "bcftools", AgentEventType.TOOL_RESULT
    )
    tree = build_run_tree(events)
    (result,) = no_repeated_tool_failures()(tree)
    assert result.passed


def test_phase_errors_free_flags_failed_phase() -> None:
    events = [AgentEvent.make(AgentEventType.PLAN_PHASE, turn_id="t1", phase="QC", status="failed")]
    tree = build_run_tree(events)
    (result,) = phase_errors_free()(tree)
    assert not result.passed and "QC" in result.note
