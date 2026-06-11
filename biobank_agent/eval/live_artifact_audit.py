"""Strict artifact checks for human-style live Biobank Agent runs.

The live-test runner exercises the CLI like a human operator. This module adds
case-specific acceptance checks on top of the runner's generic report checks so
paper-replication and trajectory demos cannot pass merely because a report
directory exists.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


_DUAL_REPORT_ARTIFACTS = (
    "report.md",
    "report_technical.md",
    "report_nature.md",
    "_report_with_css.md",
    "_report_nature_with_css.md",
    "report.html",
    "report_nature.html",
)


@dataclass
class ArtifactCheck:
    name: str
    status: str
    message: str
    path: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveArtifactAudit:
    status: str
    run_dir: str
    generated_at: str
    checks: list[ArtifactCheck]
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def n_failed(self) -> int:
        return sum(1 for check in self.checks if check.status == "FAIL")

    @property
    def n_warning(self) -> int:
        return sum(1 for check in self.checks if check.status == "WARN")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _worker_dirs(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.glob("worker-*") if path.is_dir())


def _worker_bundle(worker_dir: Path) -> dict[str, Any]:
    index_path = worker_dir / "artifact_index.json"
    index = _read_json(index_path)
    plan_text = "\n".join(_read_text(path) for path in (worker_dir / "workspace" / "plans").glob("*.md"))
    transcript = _read_text(worker_dir / "transcript.clean.log")
    audit_report_dirs = (index.get("audit") or {}).get("report_dirs")
    if isinstance(audit_report_dirs, list):
        report_dirs = [Path(path) for path in audit_report_dirs if str(path).strip()]
    else:
        report_dirs = sorted((worker_dir / "workspace" / "reports").glob("*"))
    report_dirs = [path for path in report_dirs if path.is_dir()]
    report_text = "\n".join(
        _read_text(report_dir / name)
        for report_dir in report_dirs
        for name in ("report_technical.md", "report.md", "report_nature.md")
    )
    return {
        "index": index,
        "plan_text": plan_text,
        "transcript": transcript,
        "report_dirs": report_dirs,
        "report_text": report_text,
        "all_text": "\n".join([plan_text, transcript, report_text]),
    }


def _ok(name: str, message: str, path: Path | str = "", **details: Any) -> ArtifactCheck:
    return ArtifactCheck(name=name, status="PASS", message=message, path=str(path), details=details)


def _fail(name: str, message: str, path: Path | str = "", **details: Any) -> ArtifactCheck:
    return ArtifactCheck(name=name, status="FAIL", message=message, path=str(path), details=details)


def _warn(name: str, message: str, path: Path | str = "", **details: Any) -> ArtifactCheck:
    return ArtifactCheck(name=name, status="WARN", message=message, path=str(path), details=details)


def _has_all(text: str, required: tuple[str, ...]) -> tuple[bool, list[str]]:
    missing = [item for item in required if item not in text]
    return not missing, missing


def _has_dual_format(plan_text: str) -> bool:
    normalized = re.sub(r"\s+", "", plan_text)
    return (
        "format='dual'" in normalized
        or "'format':'dual'" in normalized
        or '"format":"dual"' in normalized
        or "format=dual" in normalized
    )


def _report_dir_checks(run_dir: Path, worker_bundles: list[tuple[Path, dict[str, Any]]]) -> list[ArtifactCheck]:
    checks: list[ArtifactCheck] = []
    report_dirs = [report_dir for _, bundle in worker_bundles for report_dir in bundle["report_dirs"]]
    if not report_dirs:
        return [_fail("dual_reports_present", "No report directories were found in the live run.", run_dir)]
    for report_dir in report_dirs:
        missing = [name for name in _DUAL_REPORT_ARTIFACTS if not (report_dir / name).exists()]
        if missing:
            checks.append(_fail("dual_report_artifacts", "Dual report artifacts missing.", report_dir, missing=missing))
        else:
            checks.append(_ok("dual_report_artifacts", "Dual report artifacts are complete.", report_dir))
    return checks


def _find_worker(worker_bundles: list[tuple[Path, dict[str, Any]]], *needles: str) -> tuple[Path, dict[str, Any]] | None:
    lowered = [needle.lower() for needle in needles]
    for worker_dir, bundle in worker_bundles:
        text = str(bundle["all_text"]).lower()
        if all(needle in text for needle in lowered):
            return worker_dir, bundle
    return None


def _find_paper_worker(worker_bundles: list[tuple[Path, dict[str, Any]]]) -> tuple[Path, dict[str, Any]] | None:
    for worker_dir, bundle in worker_bundles:
        plan_text = str(bundle["plan_text"]).lower()
        if "10.1038/s41588-024-01898-1" in plan_text and (
            "replicate_paper" in plan_text or "paper_replication_compare" in plan_text
        ):
            return worker_dir, bundle
        for report_dir in bundle["report_dirs"]:
            if (report_dir / "paper_replication_comparison.md").exists():
                return worker_dir, bundle
    return None


def _find_trajectory_worker(worker_bundles: list[tuple[Path, dict[str, Any]]]) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for worker_dir, bundle in worker_bundles:
        plan_text = str(bundle["plan_text"]).lower()
        if "trajectory_tokenize" in plan_text and ("healthformer" in plan_text or "trajectory" in plan_text):
            candidates.append((worker_dir, bundle))
    if not candidates:
        return None
    # Prefer the dedicated trajectory case over the grand-challenge case.
    dedicated = [
        item for item in candidates
        if "build a longitudinal healthformer-style trajectory forecast over time" in str(item[1]["plan_text"]).lower()
    ]
    return dedicated[0] if dedicated else candidates[0]


def _paper_replication_checks(worker_bundles: list[tuple[Path, dict[str, Any]]]) -> list[ArtifactCheck]:
    worker = _find_paper_worker(worker_bundles)
    if worker is None:
        return [_fail("paper_replication_worker", "No live worker referenced the frozen MILTON DOI fixture.")]

    worker_dir, bundle = worker
    checks: list[ArtifactCheck] = [_ok("paper_replication_worker", "Found MILTON paper replication worker.", worker_dir)]
    plan_text = str(bundle["plan_text"])
    required_plan = (
        "replicate_paper",
        "read_paper",
        "paper_replication_compare",
        "statistical_review",
        "safety_check",
        "world_model_audit",
        "generate_report",
    )
    ok, missing = _has_all(plan_text, required_plan)
    if ok and _has_dual_format(plan_text):
        checks.append(_ok("paper_plan_dependencies", "Paper replication plan contains required gated steps.", worker_dir))
    else:
        if not _has_dual_format(plan_text):
            missing.append("format=dual")
        checks.append(_fail("paper_plan_dependencies", "Paper replication plan is missing required gated steps.", worker_dir, missing=missing))

    comparison_paths = [
        report_dir / "paper_replication_comparison.md"
        for report_dir in bundle["report_dirs"]
        if (report_dir / "paper_replication_comparison.md").exists()
    ]
    if not comparison_paths:
        checks.append(_fail("paper_comparison_artifact", "No paper_replication_comparison.md artifact found.", worker_dir))
        return checks

    comparison = comparison_paths[0]
    text = _read_text(comparison)
    checks.append(_ok("paper_comparison_artifact", "Paper replication comparison artifact exists.", comparison))
    expected = (
        "Acceptance verdict: PASS_WITH_LIMITATIONS",
        "paper_access",
        "cohort_count",
        "model_auc",
        "calibration_ece",
        "feature_importance",
        "figure_artifacts",
    )
    ok, missing = _has_all(text, expected)
    if ok and re.search(r"\|\s*paper_access\s*\|\s*PASS\s*\|", text):
        checks.append(_ok("paper_acceptance_gates", "Paper comparison renders all required acceptance gates.", comparison))
    else:
        checks.append(_fail("paper_acceptance_gates", "Paper comparison does not render the required acceptance gates.", comparison, missing=missing))
    return checks


def _trajectory_checks(worker_bundles: list[tuple[Path, dict[str, Any]]]) -> list[ArtifactCheck]:
    worker = _find_trajectory_worker(worker_bundles)
    if worker is None:
        return [_fail("trajectory_worker", "No live worker exercised the HealthFormer-style trajectory case.")]

    worker_dir, bundle = worker
    checks: list[ArtifactCheck] = [_ok("trajectory_worker", "Found HealthFormer-style trajectory worker.", worker_dir)]
    plan_text = str(bundle["plan_text"])
    required_plan = ("trajectory_tokenize", "statistical_review", "safety_check", "world_model_audit", "generate_report")
    ok, missing = _has_all(plan_text, required_plan)
    if ok and _has_dual_format(plan_text):
        checks.append(_ok("trajectory_plan_dependencies", "Trajectory plan contains tokenization, guardrails, audit, and dual report.", worker_dir))
    else:
        if not _has_dual_format(plan_text):
            missing.append("format=dual")
        checks.append(_fail("trajectory_plan_dependencies", "Trajectory plan is missing required gated steps.", worker_dir, missing=missing))

    report_text = str(bundle["report_text"])
    token_matches = [
        int(match.replace(",", ""))
        for match in re.findall(r"(?:n_tokens\s*\|\s*|Trajectory layer:\s*)([0-9][0-9,]+)", report_text)
    ]
    max_tokens = max(token_matches) if token_matches else 0
    if max_tokens > 0:
        checks.append(_ok("trajectory_tokens_positive", "Trajectory tokenization produced nonzero tokens.", worker_dir, n_tokens=max_tokens))
    else:
        checks.append(_fail("trajectory_tokens_positive", "Trajectory report did not show a positive token count.", worker_dir))

    required_report = (
        "available_tokens",
        "training_distribution_coverage",
        "association_conditioned_forecast",
        "trajectory_time_source",
        "world_model_audit",
    )
    ok, missing = _has_all(report_text, required_report)
    if ok:
        checks.append(_ok("trajectory_claim_boundary", "World-model audit preserves association-only claim boundary.", worker_dir))
    else:
        checks.append(_fail("trajectory_claim_boundary", "Trajectory report is missing claim-boundary evidence.", worker_dir, missing=missing))
    return checks


def run_live_artifact_audit(
    run_dir: str | Path,
    *,
    require_paper_replication: bool = True,
) -> LiveArtifactAudit:
    run_path = Path(run_dir).expanduser().resolve()
    checks: list[ArtifactCheck] = []
    if not run_path.exists():
        return LiveArtifactAudit(
            status="FAIL",
            run_dir=str(run_path),
            generated_at=datetime.now().isoformat(timespec="seconds"),
            checks=[_fail("run_dir_exists", "Run directory does not exist.", run_path)],
        )

    live_json = _read_json(run_path / "live_test_audit.json")
    live_md = _read_text(run_path / "LIVE_TEST_AUDIT.md")
    if live_json.get("status") == "PASS" or "Status: **PASS**" in live_md:
        checks.append(_ok("live_runner_status", "Base live runner audit passed.", run_path))
    else:
        checks.append(_fail("live_runner_status", "Base live runner audit was not PASS.", run_path))

    workers = _worker_dirs(run_path)
    if workers:
        checks.append(_ok("worker_dirs", f"Found {len(workers)} worker directories.", run_path, n_workers=len(workers)))
    else:
        checks.append(_fail("worker_dirs", "No worker directories found.", run_path))

    bundles = [(worker_dir, _worker_bundle(worker_dir)) for worker_dir in workers]
    checks.extend(_report_dir_checks(run_path, bundles))
    if require_paper_replication:
        checks.extend(_paper_replication_checks(bundles))
    else:
        paper_worker = _find_paper_worker(bundles)
        if paper_worker is None:
            checks.append(_ok(
                "paper_replication_not_requested",
                "No paper-replication worker was required for this live run.",
                run_path,
            ))
        else:
            checks.extend(_paper_replication_checks(bundles))
    checks.extend(_trajectory_checks(bundles))

    status = "FAIL" if any(check.status == "FAIL" for check in checks) else (
        "WARN" if any(check.status == "WARN" for check in checks) else "PASS"
    )
    return LiveArtifactAudit(
        status=status,
        run_dir=str(run_path),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        checks=checks,
    )


def write_audit_artifacts(audit: LiveArtifactAudit, output_dir: str | Path | None = None) -> LiveArtifactAudit:
    out_dir = Path(output_dir).expanduser() if output_dir else Path(audit.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"strict_live_artifact_audit_{stamp}.json"
    md_path = out_dir / f"strict_live_artifact_audit_{stamp}.md"
    payload = asdict(audit)
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Strict Live Artifact Audit",
        "",
        f"- Status: **{audit.status}**",
        f"- Run directory: `{audit.run_dir}`",
        f"- Generated: {audit.generated_at}",
        "",
        "| Check | Status | Message | Path |",
        "|---|---:|---|---|",
    ]
    for check in audit.checks:
        path = check.path.replace("|", "/")
        lines.append(f"| {check.name} | {check.status} | {check.message.replace('|', '/')} | `{path}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit.artifacts = {"json": str(json_path), "markdown": str(md_path)}
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit live-test artifacts for strict v3 acceptance evidence.")
    parser.add_argument("run_dir", help="Live-test run directory containing worker-* subdirectories.")
    parser.add_argument("--output-dir", default="", help="Optional directory for audit JSON/Markdown artifacts.")
    parser.add_argument("--write", action="store_true", help="Write strict audit artifacts.")
    args = parser.parse_args(argv)

    audit = run_live_artifact_audit(args.run_dir)
    if args.write:
        audit = write_audit_artifacts(audit, args.output_dir or None)
    print(json.dumps(asdict(audit), indent=2, default=str))
    return 0 if audit.status == "PASS" else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
