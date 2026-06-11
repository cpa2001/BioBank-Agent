"""Behavioral eval runner with local pass-rate history artifacts."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from biobank_agent.domain.reproducibility import PaperReplicator
from biobank_agent.planner import LongHorizonPlan, LongHorizonPlanner, PlanSchemaValidator, PlanStep


AVAILABLE_SKILLS = [
    "replicate_paper",
    "fetch_paper",
    "read_paper",
    "project_doc",
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


@dataclass
class BehavioralCaseResult:
    """One behavioral case result."""

    case_id: str
    policy: str
    kind: str
    source: str
    passed: bool
    score: float
    passed_checks: int
    total_checks: int
    messages: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


@dataclass
class BehavioralRunResult:
    """Aggregate behavioral eval result."""

    suite: str
    status: str
    timestamp: str
    case_dir: str
    output_dir: str
    policies: list[str]
    cases: list[BehavioralCaseResult]
    total_elapsed_s: float
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return len(self.cases)

    @property
    def n_passed(self) -> int:
        return sum(1 for case in self.cases if case.passed)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_total if self.n_total else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["n_total"] = self.n_total
        data["n_passed"] = self.n_passed
        data["pass_rate"] = self.pass_rate
        return data


def default_case_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "eval" / "behavioral"


def load_behavioral_cases(case_dir: Path, policies: Iterable[str] = ("always", "usually")) -> list[dict[str, Any]]:
    selected = {str(policy) for policy in policies}
    if "all" in selected:
        selected = {"always", "usually"}
    cases: list[dict[str, Any]] = []
    for path in sorted(case_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        policy = str(data.get("policy", "always"))
        if policy not in selected:
            continue
        for case in data.get("cases", []) or []:
            if not isinstance(case, dict):
                continue
            item = dict(case)
            item["_source"] = path.name
            item["_policy"] = policy
            cases.append(item)
    return cases


def run_behavioral_eval(
    *,
    case_dir: str | Path | None = None,
    output_dir: str | Path = "reports/eval/behavioral",
    policies: Iterable[str] = ("always", "usually"),
    write_artifacts: bool = True,
) -> BehavioralRunResult:
    t0 = time.time()
    case_root = Path(case_dir) if case_dir is not None else default_case_dir()
    selected_policies = list(policies)
    cases = load_behavioral_cases(case_root, selected_policies)
    results = [_evaluate_case(case) for case in cases]
    status = "PASS" if all(case.passed for case in results) else "FAIL"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = BehavioralRunResult(
        suite="behavioral",
        status=status,
        timestamp=timestamp,
        case_dir=str(case_root),
        output_dir=str(output_dir),
        policies=selected_policies,
        cases=results,
        total_elapsed_s=time.time() - t0,
    )
    if write_artifacts:
        result.artifacts.update(_write_artifacts(result, Path(output_dir)))
    return result


def _write_artifacts(result: BehavioralRunResult, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / f"behavioral_{result.timestamp}.json"
    latest_path = output_dir / "latest.json"
    history_path = output_dir / "behavioral_history.jsonl"
    artifacts = {
        "run_json": str(run_path),
        "latest_json": str(latest_path),
        "history_jsonl": str(history_path),
    }
    result.artifacts.update(artifacts)

    data = result.to_dict()
    run_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    latest_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    history = {
        "timestamp": result.timestamp,
        "status": result.status,
        "n_passed": result.n_passed,
        "n_total": result.n_total,
        "pass_rate": result.pass_rate,
        "policies": result.policies,
        "run_path": str(run_path),
    }
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(history, sort_keys=True) + "\n")
    return artifacts


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


def _evaluate_case(case: dict[str, Any]) -> BehavioralCaseResult:
    t0 = time.time()
    kind = str(case.get("kind", ""))
    policy = str(case.get("_policy", "always"))
    source = str(case.get("_source", ""))
    case_id = str(case.get("id", "unknown"))
    try:
        if kind == "planner":
            passed, total, messages = _score_always_planner_case(case)
        elif kind == "paper_replication":
            passed, total, messages = _score_paper_replication_case(case)
        elif kind == "published_paper_gold":
            passed, total, messages = _score_published_paper_gold_case(case)
        elif kind == "planner_score":
            passed, total, messages = _score_soft_planner_case(case)
        else:
            passed, total, messages = 0, 1, [f"Unknown behavioral case kind: {kind}"]
    except Exception as exc:
        passed, total, messages = 0, 1, [f"{type(exc).__name__}: {exc}"]

    score = passed / total if total else 0.0
    threshold = float(case.get("pass_threshold", 1.0 if policy == "always" else 0.7) or 0.0)
    is_passed = score >= threshold and not messages if policy == "always" else score >= threshold
    return BehavioralCaseResult(
        case_id=case_id,
        policy=policy,
        kind=kind,
        source=source,
        passed=is_passed,
        score=score,
        passed_checks=passed,
        total_checks=total,
        messages=messages,
        elapsed_s=time.time() - t0,
    )


def _score_always_planner_case(case: dict[str, Any]) -> tuple[int, int, list[str]]:
    plan = LongHorizonPlanner().decompose(str(case["query"]), AVAILABLE_SKILLS)
    issues = PlanSchemaValidator(AVAILABLE_SKILLS).validate(plan)
    skills = [step.skill for step in plan.steps]
    by_skill = {step.skill: step for step in plan.steps}
    passed = 0
    total = 1
    messages: list[str] = []
    if not issues:
        passed += 1
    else:
        messages.append(f"invalid plan: {[issue.format() for issue in issues]}")

    for skill in case.get("must_include_skills", []) or []:
        total += 1
        if skill in skills:
            passed += 1
        else:
            messages.append(f"missing skill {skill}")

    for skill, args in (case.get("nonpositive_or_absent_args") or {}).items():
        step = by_skill.get(skill)
        total += len(args or [])
        if step is None:
            messages.append(f"missing {skill}")
            continue
        for arg in args or []:
            value = step.args.get(arg, 0)
            if value in (0, None, ""):
                passed += 1
            else:
                messages.append(f"expected {skill}.{arg} to be full-data default, got {value!r}")

    report = by_skill.get("generate_report")
    total += 1
    if report is not None:
        passed += 1
    else:
        messages.append("missing generate_report")

    if report is not None:
        for skill in case.get("report_must_follow", []) or []:
            total += 1
            if _depends_on_skill(plan, report, skill):
                passed += 1
            else:
                messages.append(f"report does not depend on {skill}")
        for key, expected in (case.get("required_report_args") or {}).items():
            total += 1
            if report.args.get(key) == expected:
                passed += 1
            else:
                messages.append(f"generate_report.{key}={report.args.get(key)!r}, expected {expected!r}")
    return passed, total, messages


def _score_soft_planner_case(case: dict[str, Any]) -> tuple[int, int, list[str]]:
    plan = LongHorizonPlanner().decompose(str(case["query"]), AVAILABLE_SKILLS)
    issues = PlanSchemaValidator(AVAILABLE_SKILLS).validate(plan)
    skills = {step.skill for step in plan.steps}
    by_skill = {step.skill: step for step in plan.steps}
    passed = 0
    total = 1
    messages: list[str] = []
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


def _score_paper_replication_case(case: dict[str, Any]) -> tuple[int, int, list[str]]:
    outcome = PaperReplicator().from_text(str(case["source"]), paper_path="10.1038/example")
    skills = [step["skill"] for step in outcome.plan]
    top_level = LongHorizonPlanner().decompose(
        f"replicate this paper: {case['source']}",
        AVAILABLE_SKILLS,
    )
    top_skills = [step.skill for step in top_level.steps]
    passed = 0
    total = 0
    messages: list[str] = []

    for skill in case.get("expected_inner_plan_skills", []) or []:
        total += 1
        if skill in skills:
            passed += 1
        else:
            messages.append(f"missing inner plan skill {skill}")
    for skill in case.get("expected_top_level_plan_skills", []) or []:
        total += 1
        if skill in top_skills:
            passed += 1
        else:
            messages.append(f"missing top-level plan skill {skill}")
    total += 1
    diff_types = [row["target_type"] for row in outcome.table_figure_diff]
    if diff_types == case.get("expected_diff_target_types", []):
        passed += 1
    else:
        messages.append(f"diff target types {diff_types!r}")
    total += 1
    if all(row["status"] == "awaiting_execution" for row in outcome.table_figure_diff):
        passed += 1
    else:
        messages.append("paper diff rows are not awaiting execution")
    return passed, total, messages


def _score_published_paper_gold_case(case: dict[str, Any]) -> tuple[int, int, list[str]]:
    doi = str(case["doi"])
    min_text_length = int(case.get("min_text_length", 10000) or 0)
    passed = 0
    total = 0
    messages: list[str] = []

    from biobank_agent.skills.fetch_paper import _find_local_pdf_for_identifier, fetch_paper
    from biobank_agent.skills.read_paper import read_paper

    total += 1
    local_pdf = _find_local_pdf_for_identifier(doi)
    if local_pdf is not None and local_pdf.exists():
        passed += 1
    else:
        messages.append(f"missing local PDF fixture for {doi}")

    total += 1
    paper = fetch_paper(doi)
    if paper.get("source") == "doi_local_cache" and paper.get("pdf_path"):
        passed += 1
    else:
        messages.append(f"fetch_paper did not use local cache: {paper.get('source')}")

    total += 1
    if len(str(paper.get("full_text") or "")) >= min_text_length:
        passed += 1
    else:
        messages.append(f"fetch_paper extracted only {len(str(paper.get('full_text') or ''))} chars")

    total += 1
    read = read_paper(doi, focus="ukb-relevance")
    if int(read.get("full_text_length", 0) or 0) >= min_text_length:
        passed += 1
    else:
        messages.append(f"read_paper full_text_length={read.get('full_text_length')}")

    plan = LongHorizonPlanner().decompose(str(case["query"]), AVAILABLE_SKILLS)
    issues = PlanSchemaValidator(AVAILABLE_SKILLS).validate(plan)
    skills = [step.skill for step in plan.steps]
    by_skill = {step.skill: step for step in plan.steps}
    total += 1
    if not issues:
        passed += 1
    else:
        messages.append(f"invalid plan: {[issue.format() for issue in issues]}")

    for skill in case.get("must_include_skills", []) or []:
        total += 1
        if skill in skills:
            passed += 1
        else:
            messages.append(f"missing skill {skill}")

    compare = by_skill.get("paper_replication_compare")
    for skill in case.get("comparison_must_follow", []) or []:
        total += 1
        if compare is not None and _depends_on_skill(plan, compare, skill):
            passed += 1
        else:
            messages.append(f"paper_replication_compare does not depend on {skill}")

    report = by_skill.get("generate_report")
    for key, expected in (case.get("required_report_args") or {}).items():
        total += 1
        if report is not None and report.args.get(key) == expected:
            passed += 1
        else:
            observed = None if report is None else report.args.get(key)
            messages.append(f"generate_report.{key}={observed!r}, expected {expected!r}")
    for skill in case.get("report_must_follow", []) or []:
        total += 1
        if report is not None and _depends_on_skill(plan, report, skill):
            passed += 1
        else:
            messages.append(f"report does not depend on {skill}")
    return passed, total, messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Biobank Agent behavioral evals and write pass-rate history.")
    parser.add_argument("--case-dir", default=str(default_case_dir()))
    parser.add_argument("--output-dir", default="reports/eval/behavioral")
    parser.add_argument("--policy", choices=["always", "usually", "all"], default="all")
    args = parser.parse_args(argv)

    policies = ("always", "usually") if args.policy == "all" else (args.policy,)
    result = run_behavioral_eval(case_dir=args.case_dir, output_dir=args.output_dir, policies=policies)
    print(f"BEHAVIORAL_STATUS={result.status}")
    print(f"BEHAVIORAL_CASES={result.n_passed}/{result.n_total}")
    for name, path in result.artifacts.items():
        print(f"{name.upper()}={path}")
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AVAILABLE_SKILLS",
    "BehavioralCaseResult",
    "BehavioralRunResult",
    "load_behavioral_cases",
    "run_behavioral_eval",
]
