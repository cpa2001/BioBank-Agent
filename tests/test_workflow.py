"""Phase 5: lightweight workflow DAG runner (steps + deps, fan-out / fan-in).

Offline and deterministic — step work is injected. Pins: topological levels, concurrent batches, fan-in
of upstream results, failure isolation (dependents skipped), and rejection of cycles / bad edges.
"""

from __future__ import annotations

import pytest

from biobank_agent.runtime.workflow import WorkflowStep, run_workflow


def _const(value):
    return lambda _upstream: value


def test_diamond_runs_levels_concurrently_and_fans_in():
    steps = [
        WorkflowStep("a", _const(1)),
        WorkflowStep("b", _const(2), dependencies=("a",)),
        WorkflowStep("c", _const(3), dependencies=("a",)),
        WorkflowStep("d", lambda up: up["b"] + up["c"], dependencies=("b", "c")),
    ]
    result = run_workflow(steps, max_workers=4)

    assert result.ok
    assert result.result("d") == 5  # fan-in of b and c
    assert result.order[0] == ["a"]
    assert set(result.order[1]) == {"b", "c"}  # same topological level → one batch
    assert result.order[2] == ["d"]


def test_upstream_results_are_passed_to_dependents():
    seen = {}

    def capture(name):
        def _run(upstream):
            seen[name] = dict(upstream)
            return name
        return _run

    steps = [
        WorkflowStep("root", capture("root")),
        WorkflowStep("leaf", capture("leaf"), dependencies=("root",)),
    ]
    run_workflow(steps)
    assert seen["root"] == {}
    assert seen["leaf"] == {"root": "root"}


def test_failed_step_skips_only_its_dependents():
    def boom(_up):
        raise RuntimeError("nope")

    steps = [
        WorkflowStep("a", _const(1)),
        WorkflowStep("b", boom, dependencies=("a",)),
        WorkflowStep("c", _const(3), dependencies=("a",)),
        WorkflowStep("d", _const(9), dependencies=("b", "c")),
    ]
    result = run_workflow(steps)

    assert result.steps["a"].status == "ok"
    assert result.steps["b"].status == "failed"
    assert result.steps["c"].status == "ok"  # independent of b
    assert result.steps["d"].status == "skipped"  # depends on failed b
    assert not result.ok


def test_cycle_is_rejected():
    with pytest.raises(ValueError, match="cycle"):
        run_workflow([
            WorkflowStep("x", _const(0), dependencies=("y",)),
            WorkflowStep("y", _const(0), dependencies=("x",)),
        ])


def test_unknown_dependency_and_duplicate_id_are_rejected():
    with pytest.raises(ValueError, match="unknown step"):
        run_workflow([WorkflowStep("a", _const(0), dependencies=("z",))])
    with pytest.raises(ValueError, match="duplicate"):
        run_workflow([WorkflowStep("a", _const(0)), WorkflowStep("a", _const(1))])


def test_duplicate_dependency_edge_is_not_a_false_cycle():
    # ("a", "a") is a redundant edge, not a cycle — it must run, not raise.
    result = run_workflow([
        WorkflowStep("a", _const(1)),
        WorkflowStep("b", lambda up: up["a"] + 1, dependencies=("a", "a")),
    ])
    assert result.ok and result.result("b") == 2


def test_empty_workflow_is_not_ok():
    result = run_workflow([])
    assert result.steps == {} and not result.ok
