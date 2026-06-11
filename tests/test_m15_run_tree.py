"""Tests for M15: run-tree fold over session.events + online evaluators (pure)."""

from __future__ import annotations

from biobank_agent.core.events import AgentEvent, AgentEventType as ET
from biobank_agent.runtime.run_tree import (
    ERROR, OK, RUNNING, TOOL, build_run_tree, summarize_run_tree,
)
from biobank_agent.runtime.run_eval import (
    error_free, evaluate_run_tree, latency_budget, tool_success_rate,
)


def _ev(t, ts, *, turn_id=None, tool_call_id=None, model_id=None, **payload):
    return AgentEvent(type=t, payload=dict(payload), ts=ts,
                      turn_id=turn_id, tool_call_id=tool_call_id, model_id=model_id)


def test_tool_span_pairs_and_times():
    events = [
        _ev(ET.TURN_STARTED, 100.0, turn_id="t1"),
        _ev(ET.TOOL_STARTED, 100.5, turn_id="t1", tool_call_id="c1", name="vcf_pca"),
        _ev(ET.TOOL_RESULT, 102.5, turn_id="t1", tool_call_id="c1"),
        _ev(ET.TURN_FINISHED, 103.0, turn_id="t1"),
    ]
    root = build_run_tree(events)
    turn = root.children[0]
    assert turn.kind == "turn" and turn.id == "t1"
    tool = [c for c in turn.children if c.kind == TOOL][0]
    assert tool.name == "vcf_pca"
    assert tool.status == OK
    assert tool.duration == 2.0


def test_tool_error_and_unclosed_running():
    events = [
        _ev(ET.TOOL_STARTED, 1.0, turn_id="t1", tool_call_id="a", name="boom"),
        _ev(ET.TOOL_ERROR, 1.4, turn_id="t1", tool_call_id="a"),
        _ev(ET.TOOL_STARTED, 2.0, turn_id="t1", tool_call_id="b", name="hang"),
        # no closer for "b" -> stays running
    ]
    root = build_run_tree(events)
    spans = {c.name: c for c in root.children[0].children if c.kind == TOOL}
    assert spans["boom"].status == ERROR
    assert spans["hang"].status == RUNNING and spans["hang"].duration is None


def test_phases_and_multiple_turns_nest():
    events = [
        _ev(ET.PLAN_PHASE, 1.0, turn_id="t1", phase="Planning", status="success"),
        _ev(ET.TOOL_STARTED, 1.2, turn_id="t1", tool_call_id="c1", name="a"),
        _ev(ET.TOOL_RESULT, 1.3, turn_id="t1", tool_call_id="c1"),
        _ev(ET.PLAN_PHASE, 2.0, turn_id="t2", phase="Review", status="error"),
    ]
    root = build_run_tree(events)
    assert len(root.children) == 2
    t1 = root.children[0]
    phases = [c for c in t1.children if c.kind == "phase"]
    assert phases[0].name == "Planning" and phases[0].status == OK
    t2 = root.children[1]
    assert [c for c in t2.children if c.kind == "phase"][0].status == ERROR


def test_closer_without_opener_is_ignored_and_no_raise():
    # A dangling TOOL_RESULT (id never opened) must not raise or create a phantom span.
    events = [_ev(ET.TOOL_RESULT, 5.0, turn_id="t1", tool_call_id="ghost")]
    root = build_run_tree(events)
    assert [c for c in root.children[0].children if c.kind == TOOL] == []


def test_missing_tool_call_id_falls_back_to_recent_open():
    events = [
        _ev(ET.TOOL_STARTED, 1.0, turn_id="t1", name="noid"),   # no tool_call_id
        _ev(ET.TOOL_RESULT, 1.5, turn_id="t1"),                 # no id -> close most-recent running
    ]
    root = build_run_tree(events)
    tool = [c for c in root.children[0].children if c.kind == TOOL][0]
    assert tool.status == OK and tool.duration == 0.5


def test_summarize_counts_latency_and_success_rate():
    events = [
        _ev(ET.TOOL_STARTED, 0.0, turn_id="t1", tool_call_id="c1", name="ok1"),
        _ev(ET.TOOL_RESULT, 1.0, turn_id="t1", tool_call_id="c1"),
        _ev(ET.TOOL_STARTED, 1.0, turn_id="t1", tool_call_id="c2", name="bad"),
        _ev(ET.TOOL_ERROR, 4.0, turn_id="t1", tool_call_id="c2"),
    ]
    s = summarize_run_tree(build_run_tree(events))
    assert s["tool_calls"] == 2
    assert s["tool_errors"] == 1
    assert s["tool_success_rate"] == 0.5
    assert s["total_tool_latency"] == 4.0
    assert s["slowest"][0] == ("bad", 3.0)


def test_evaluators_score_deterministically():
    events = [
        _ev(ET.TOOL_STARTED, 0.0, turn_id="t1", tool_call_id="c1", name="ok1"),
        _ev(ET.TOOL_RESULT, 0.2, turn_id="t1", tool_call_id="c1"),
        _ev(ET.TOOL_STARTED, 1.0, turn_id="t1", tool_call_id="c2", name="bad"),
        _ev(ET.TOOL_ERROR, 9.0, turn_id="t1", tool_call_id="c2"),
    ]
    root = build_run_tree(events)
    report = evaluate_run_tree(root, [tool_success_rate(min_rate=0.8), error_free(), latency_budget(5.0)])
    by_name = {r.name: r for r in report.results}
    assert by_name["tool_success_rate"].passed is False     # 50% < 80%
    assert by_name["error_free"].passed is False            # one error span
    assert by_name["latency_budget"].passed is False        # 8s > 5s budget
    assert report.passed is False
    assert 0.0 <= report.score <= 1.0
