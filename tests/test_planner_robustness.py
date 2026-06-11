"""Tests for council planner robustness + tabular-analysis fallback (issue #5)."""

from __future__ import annotations

import pytest

import biobank_agent.runtime.planner as planner_mod
from biobank_agent.runtime.council import CouncilError
from biobank_agent.runtime.planner import RuntimePlanner, is_tabular_analysis_objective


@pytest.mark.parametrize("objective,expected", [
    ("classify the 243 traits in /d/all_traits_5e-11_gwas_results.csv and find the characteristic genes of each class", True),
    ("cluster samples in cohort.tsv into subtypes", True),
    ("find marker genes per group in expr.parquet", True),
    ("Read the first 5 rows of /d/x.csv and show the column names", False),  # inspection, not analysis
    ("what is the prevalence of diabetes", False),                           # no file
    ("run a GWAS on cohort.vcf", False),                                     # not a tabular ext
])
def test_is_tabular_analysis_objective(objective, expected):
    assert is_tabular_analysis_objective(objective) is expected


def _planner_with_failing_council(monkeypatch):
    # run_parallel returns no candidate results => _parse_candidates yields [] =>
    # build_plan raises CouncilError, exercising the fallback path.
    monkeypatch.setattr(planner_mod, "run_parallel", lambda ctx, jobs: [])
    return RuntimePlanner(object(), enable_clarification=False, num_candidates=1)


def test_council_failure_falls_back_to_tabular_plan(monkeypatch):
    planner = _planner_with_failing_council(monkeypatch)
    plan = planner.build_plan(
        "classify the 243 traits in /d/all_traits.csv and find characteristic genes of each class",
        tool_names=["python_exec", "generate_report"],
    )
    ids = [s.id for s in plan.steps]
    assert ids[:3] == ["load", "analyze", "characterize"]
    assert "report" in ids  # generate_report available -> report step added
    # every step uses only available tools
    for step in plan.steps:
        assert all(t in {"python_exec", "generate_report"} for t in step.tool_scope)
    assert "fallback" in plan.audit_summary.lower()


def test_tabular_fallback_without_report_tool(monkeypatch):
    planner = _planner_with_failing_council(monkeypatch)
    plan = planner.build_plan("cluster genes in /d/expr.csv and describe each cluster's markers",
                              tool_names=["python_exec"])
    assert "report" not in [s.id for s in plan.steps]  # no generate_report tool -> no report step


def test_council_failure_falls_back_to_scaffold_for_non_tabular(monkeypatch):
    # Planning never hard-fails — a non-tabular goal degrades to a labeled
    # gather→execute→report scaffold instead of raising CouncilError.
    planner = _planner_with_failing_council(monkeypatch)
    plan = planner.build_plan("what is the prevalence of diabetes",
                              tool_names=["python_exec", "generate_report"])
    ids = [s.id for s in plan.steps]
    assert ids[:2] == ["gather", "execute"]
    assert "report" in ids  # generate_report available -> report step
    assert "fallback" in plan.audit_summary.lower()  # degradation is surfaced
    for step in plan.steps:
        assert all(t in {"python_exec", "generate_report"} for t in step.tool_scope)


def test_empty_objective_still_hard_fails(monkeypatch):
    # An empty objective is a true error, not a degradable plan — it must still raise.
    planner = _planner_with_failing_council(monkeypatch)
    with pytest.raises(CouncilError):
        planner.build_plan("   ", tool_names=["python_exec"])


def test_repair_json_object_salvages_malformed_candidates():
    # A candidate whose JSON is truncated / comma-trailed / prose-wrapped is salvaged,
    # so a single sloppy model response does not drop the whole candidate.
    r = RuntimePlanner._repair_json_object
    assert r('{"steps": [{"id": "a"}]') == {"steps": [{"id": "a"}]}        # truncated tail
    assert r('Here:\n{"title": "t", "steps": [1, 2,],}\nThanks') == {"title": "t", "steps": [1, 2]}
    assert r("no json here at all") is None
    assert r("") is None


def test_plan_state_filters_unavailable_tools():
    planner = RuntimePlanner(object())
    data = {
        "title": "t",
        "steps": [{"id": "s1", "title": "x", "purpose": "p", "tool_scope": ["python_exec", "bogus_tool"]}],
    }
    plan = planner._plan_state_from_json(data, "goal", previous=None, refinement="",
                                         available_tools=["python_exec"])
    assert plan.steps[0].tool_scope == ["python_exec"]
    assert any("bogus_tool" in r for r in plan.risks)
    assert any("bogus_tool" in q for q in plan.open_questions)


def test_plan_state_keeps_tools_when_no_allowlist():
    planner = RuntimePlanner(object())
    data = {"title": "t", "steps": [{"id": "s1", "title": "x", "purpose": "p", "tool_scope": ["anything"]}]}
    plan = planner._plan_state_from_json(data, "goal", previous=None, refinement="")
    assert plan.steps[0].tool_scope == ["anything"]  # no available_tools -> no filtering
