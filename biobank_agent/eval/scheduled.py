"""Scheduled quality gates for CI or operator cron.

This runner intentionally uses the same production evaluators exposed by the
CLI instead of a special test harness. It writes a small manifest that links
behavioral pass-rate history and self-evolution pattern-mining artifacts.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from biobank_agent.core.evolution.pattern_mining import run_pattern_mining
from biobank_agent.eval.behavioral import run_behavioral_eval
from biobank_agent.tool_learner import ToolLearner


@dataclass
class ScheduledEvalResult:
    """Aggregate scheduled gate result."""

    status: str
    timestamp: str
    behavioral_status: str
    evolution_status: str
    output_dir: str
    behavioral_artifacts: dict[str, str]
    evolution_artifacts: dict[str, str]
    behavioral_trend: dict[str, Any]
    evolution_trend: dict[str, Any]
    total_elapsed_s: float
    action_required: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_scheduled_quality_gates(
    *,
    output_dir: str | Path = "reports/eval/scheduled",
    behavioral_output_dir: str | Path = "reports/eval/behavioral",
    evolution_output_dir: str | Path = "reports/eval/evolution",
    behavioral_policies: Iterable[str] = ("always", "usually"),
    evolution_history: str | Path | None = None,
    evolution_min_count: int = 3,
    fail_on_evolution_patterns: bool = False,
    write_artifacts: bool = True,
) -> ScheduledEvalResult:
    """Run scheduled behavioral/evolution gates and write a manifest."""
    t0 = time.time()
    behavioral = run_behavioral_eval(
        output_dir=behavioral_output_dir,
        policies=list(behavioral_policies),
        write_artifacts=write_artifacts,
    )
    learner = _learner_from_history(evolution_history)
    evolution = run_pattern_mining(
        learner,
        output_dir=evolution_output_dir,
        min_count=evolution_min_count,
        write_artifacts=write_artifacts,
    )

    action_required: list[str] = []
    if behavioral.status != "PASS":
        action_required.append("behavioral eval failed")
    if evolution.status == "NEEDS_REVIEW":
        action_required.append("repeated tool failures need review")

    status = "PASS" if behavioral.status == "PASS" else "FAIL"
    if fail_on_evolution_patterns and evolution.status == "NEEDS_REVIEW":
        status = "FAIL"
    elif evolution.status == "NEEDS_REVIEW" and status == "PASS":
        status = "NEEDS_REVIEW"

    result = ScheduledEvalResult(
        status=status,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        behavioral_status=behavioral.status,
        evolution_status=evolution.status,
        output_dir=str(output_dir),
        behavioral_artifacts=dict(behavioral.artifacts),
        evolution_artifacts=dict(evolution.artifacts),
        behavioral_trend=_summarize_history(behavioral.artifacts.get("history_jsonl", "")),
        evolution_trend=_summarize_history(evolution.artifacts.get("history_jsonl", "")),
        action_required=action_required,
        total_elapsed_s=time.time() - t0,
    )
    if write_artifacts:
        _write_artifacts(result, Path(output_dir))
    return result


def _learner_from_history(history_path: str | Path | None) -> ToolLearner:
    learner = ToolLearner()
    if history_path is None:
        return learner
    path = Path(history_path)
    if not path.exists():
        return learner
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        skill = str(row.get("skill") or row.get("skill_name") or "")
        if not skill:
            continue
        args = row.get("args") if isinstance(row.get("args"), dict) else {}
        elapsed_s = float(row.get("elapsed_s", 0.0) or 0.0)
        if row.get("success") is True:
            result = row.get("result") if isinstance(row.get("result"), dict) else {}
        else:
            result = row.get("result") if isinstance(row.get("result"), dict) else {}
            if "error" not in result:
                result = {"error": str(row.get("error") or row.get("error_signature") or "scheduled failure")}
        learner.record(skill, args, result, elapsed_s=elapsed_s)
    return learner


def _write_artifacts(result: ScheduledEvalResult, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_json = output_dir / f"scheduled_quality_{result.timestamp}.json"
    latest_json = output_dir / "latest.json"
    history_jsonl = output_dir / "scheduled_quality_history.jsonl"
    run_md = output_dir / f"scheduled_quality_{result.timestamp}.md"
    artifacts = {
        "run_json": str(run_json),
        "latest_json": str(latest_json),
        "history_jsonl": str(history_jsonl),
        "run_md": str(run_md),
    }
    result.artifacts.update(artifacts)
    payload = result.to_dict()
    run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    run_md.write_text(_render_markdown(result), encoding="utf-8")
    with history_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "timestamp": result.timestamp,
            "status": result.status,
            "behavioral_status": result.behavioral_status,
            "evolution_status": result.evolution_status,
            "run_json": str(run_json),
            "run_md": str(run_md),
        }, sort_keys=True) + "\n")
    return artifacts


def _summarize_history(path_value: str, *, max_rows: int = 50) -> dict[str, Any]:
    path = Path(path_value) if path_value else Path()
    if not path_value or not path.exists():
        return {"n_runs": 0, "latest_status": "", "status_counts": {}, "mean_pass_rate": None}
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-max_rows:]:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    status_counts: dict[str, int] = {}
    pass_rates: list[float] = []
    for row in rows:
        status = str(row.get("status", ""))
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
        if isinstance(row.get("pass_rate"), (int, float)):
            pass_rates.append(float(row["pass_rate"]))
    return {
        "n_runs": len(rows),
        "latest_status": str(rows[-1].get("status", "")) if rows else "",
        "status_counts": status_counts,
        "mean_pass_rate": sum(pass_rates) / len(pass_rates) if pass_rates else None,
    }


def _render_markdown(result: ScheduledEvalResult) -> str:
    lines = [
        "# Biobank Agent Scheduled Quality Gates",
        "",
        f"- Status: {result.status}",
        f"- Behavioral: {result.behavioral_status}",
        f"- Evolution: {result.evolution_status}",
        f"- Elapsed seconds: {result.total_elapsed_s:.3f}",
        f"- Behavioral recent runs: {result.behavioral_trend.get('n_runs', 0)}",
        f"- Behavioral recent mean pass rate: {result.behavioral_trend.get('mean_pass_rate')}",
        f"- Evolution recent runs: {result.evolution_trend.get('n_runs', 0)}",
        "",
        "## Action Required",
        "",
    ]
    if result.action_required:
        lines.extend(f"- {item}" for item in result.action_required)
    else:
        lines.append("- None")
    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- Behavioral latest: {result.behavioral_artifacts.get('latest_json', '')}",
        f"- Behavioral history: {result.behavioral_artifacts.get('history_jsonl', '')}",
        f"- Evolution latest: {result.evolution_artifacts.get('latest_json', '')}",
        f"- Evolution history: {result.evolution_artifacts.get('history_jsonl', '')}",
    ])
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run scheduled Biobank Agent quality gates.")
    parser.add_argument("--output-dir", default="reports/eval/scheduled")
    parser.add_argument("--behavioral-output-dir", default="reports/eval/behavioral")
    parser.add_argument("--evolution-output-dir", default="reports/eval/evolution")
    parser.add_argument("--behavioral-policy", choices=["always", "usually", "all"], default="all")
    parser.add_argument("--evolution-history", default="")
    parser.add_argument("--evolution-min-count", type=int, default=3)
    parser.add_argument("--fail-on-evolution-patterns", action="store_true")
    args = parser.parse_args(argv)

    policies = ("always", "usually") if args.behavioral_policy == "all" else (args.behavioral_policy,)
    result = run_scheduled_quality_gates(
        output_dir=args.output_dir,
        behavioral_output_dir=args.behavioral_output_dir,
        evolution_output_dir=args.evolution_output_dir,
        behavioral_policies=policies,
        evolution_history=args.evolution_history or None,
        evolution_min_count=args.evolution_min_count,
        fail_on_evolution_patterns=args.fail_on_evolution_patterns,
    )
    print(f"SCHEDULED_STATUS={result.status}")
    print(f"BEHAVIORAL_STATUS={result.behavioral_status}")
    print(f"EVOLUTION_STATUS={result.evolution_status}")
    for name, path in result.artifacts.items():
        print(f"{name.upper()}={path}")
    return 0 if result.status in {"PASS", "NEEDS_REVIEW"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["ScheduledEvalResult", "run_scheduled_quality_gates"]
