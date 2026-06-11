"""Minimal behavioral eval runner for v3 always-pass expectations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from biobank_agent.domain.reproducibility import PaperReplicator
from biobank_agent.eval.behavioral import _score_published_paper_gold_case
from biobank_agent.planner import LongHorizonPlan, LongHorizonPlanner, PlanSchemaValidator, PlanStep


CASE_DIR = Path(__file__).parent

AVAILABLE_SKILLS = [
    "replicate_paper",
    "fetch_paper",
    "read_paper",
    "ukb_data_inventory",
    "ukb_field_resolve",
    "ukb_materialize_fields",
    "deep_research",
    "field_search",
    "bank_data_probe",
    "cohort_summary",
    "cohort_card",
    "trajectory_tokenize",
    "missing_data",
    "train_model",
    "evaluate_model",
    "calibration",
    "feature_importance",
    "paper_replication_compare",
    "smart_plot",
    "statistical_review",
    "safety_check",
    "world_model_audit",
    "generate_report",
    "think",
    "critical_thinking",
]


def _load_cases(policy: str = "always") -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(CASE_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("policy", "always") != policy:
            continue
        for case in data.get("cases", []) or []:
            case = dict(case)
            case["_source"] = path.name
            cases.append(case)
    return cases


def _steps_by_id(plan: LongHorizonPlan) -> dict[str, PlanStep]:
    return {step.id: step for step in plan.steps}


def _depends_on_skill(plan: LongHorizonPlan, step: PlanStep, skill: str) -> bool:
    steps = _steps_by_id(plan)
    seen: set[str] = set()
    stack = list(step.depends_on or [])
    while stack:
        sid = stack.pop()
        if sid in seen:
            continue
        seen.add(sid)
        dep = steps.get(sid)
        if dep is None:
            continue
        if dep.skill == skill:
            return True
        stack.extend(dep.depends_on or [])
    return False


def _assert_planner_case(case: dict[str, Any]) -> None:
    if case["kind"] == "planner":
        plan = LongHorizonPlanner().decompose(case["query"], AVAILABLE_SKILLS)
        issues = PlanSchemaValidator(AVAILABLE_SKILLS).validate(plan)
        assert not issues, f"{case['id']} generated invalid plan: {[issue.format() for issue in issues]}"

        skills = [step.skill for step in plan.steps]
        for skill in case.get("must_include_skills", []):
            assert skill in skills, f"{case['id']} missing skill {skill}"

        by_skill = {step.skill: step for step in plan.steps}
        for skill, args in (case.get("nonpositive_or_absent_args") or {}).items():
            step = by_skill.get(skill)
            assert step is not None, f"{case['id']} missing {skill}"
            for arg in args:
                value = step.args.get(arg, 0)
                assert value in (0, None, ""), f"{case['id']} expected {skill}.{arg} to be full-data default, got {value!r}"

        report = by_skill.get("generate_report")
        assert report is not None
        for skill in case.get("report_must_follow", []):
            assert _depends_on_skill(plan, report, skill), f"{case['id']} report does not depend on {skill}"
        for key, expected in (case.get("required_report_args") or {}).items():
            assert report.args.get(key) == expected
        return

    raise AssertionError(f"Unsupported planner case kind: {case['kind']}")


def _score_planner_case(case: dict[str, Any]) -> tuple[int, int, list[str]]:
    plan = LongHorizonPlanner().decompose(case["query"], AVAILABLE_SKILLS)
    issues = PlanSchemaValidator(AVAILABLE_SKILLS).validate(plan)
    skills = {step.skill for step in plan.steps}
    by_skill = {step.skill: step for step in plan.steps}
    passed = 0
    total = 0
    messages: list[str] = []

    total += 1
    if not issues:
        passed += 1
    else:
        messages.append(f"invalid plan: {[issue.format() for issue in issues]}")

    for group, options in (case.get("skill_groups") or {}).items():
        total += 1
        if any(skill in skills for skill in options):
            passed += 1
        else:
            messages.append(f"missing skill group {group}: {options}")

    min_required_skills = int(case.get("min_required_skills", 0) or 0)
    if min_required_skills:
        total += 1
        if len(skills) >= min_required_skills:
            passed += 1
        else:
            messages.append(f"only {len(skills)} unique skills, expected >= {min_required_skills}")

    report = by_skill.get("generate_report")
    for key, expected in (case.get("required_report_args") or {}).items():
        total += 1
        if report is not None and report.args.get(key) == expected:
            passed += 1
        else:
            observed = None if report is None else report.args.get(key)
            messages.append(f"generate_report.{key}={observed!r}, expected {expected!r}")

    return passed, total, messages


@pytest.mark.parametrize("case", _load_cases("always"), ids=lambda c: c["id"])
def test_behavioral_always_passes(case: dict[str, Any]):
    if case["kind"] == "planner":
        _assert_planner_case(case)
        return

    if case["kind"] == "paper_replication":
        outcome = PaperReplicator().from_text(case["source"], paper_path="10.1038/example")
        skills = [step["skill"] for step in outcome.plan]
        for skill in case.get("expected_inner_plan_skills", []):
            assert skill in skills, f"{case['id']} missing plan skill {skill}"
        top_level = LongHorizonPlanner().decompose(
            f"replicate this paper: {case['source']}",
            AVAILABLE_SKILLS,
        )
        top_skills = [step.skill for step in top_level.steps]
        for skill in case.get("expected_top_level_plan_skills", []):
            assert skill in top_skills, f"{case['id']} missing top-level plan skill {skill}"
        assert [row["target_type"] for row in outcome.table_figure_diff] == case["expected_diff_target_types"]
        assert all(row["status"] == "awaiting_execution" for row in outcome.table_figure_diff)
        return

    if case["kind"] == "published_paper_gold":
        passed, total, messages = _score_published_paper_gold_case(case)
        assert passed == total, f"{case['id']} failed: {messages}"
        return

    raise AssertionError(f"Unknown behavioral case kind: {case['kind']}")


@pytest.mark.usually
@pytest.mark.skipif(
    os.getenv("BIOBANK_RUN_USUALLY_EVALS") != "1",
    reason="Set BIOBANK_RUN_USUALLY_EVALS=1 to run weekly/soft behavioral cases.",
)
def test_behavioral_usually_passes_threshold():
    cases = _load_cases("usually")
    assert cases, "usually-pass suite is empty"

    passed = 0
    total = 0
    failures: list[str] = []
    threshold = 0.7
    for case in cases:
        threshold = float(case.get("pass_threshold", threshold) or threshold)
        if case["kind"] == "planner_score":
            case_passed, case_total, messages = _score_planner_case(case)
            passed += case_passed
            total += case_total
            failures.extend(f"{case['id']}: {msg}" for msg in messages)
            continue
        raise AssertionError(f"Unknown usually-pass case kind: {case['kind']}")

    ratio = passed / total if total else 0.0
    assert ratio >= threshold, f"usually-pass score {ratio:.2%} below {threshold:.0%}: {failures}"
