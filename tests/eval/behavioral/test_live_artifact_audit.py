"""Strict artifact audit tests for human-style live benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path

from biobank_agent.eval.live_artifact_audit import run_live_artifact_audit, write_audit_artifacts


def _write_dual_report_dir(path: Path, technical: str = "") -> None:
    path.mkdir(parents=True)
    for name in (
        "report.md",
        "report_technical.md",
        "report_nature.md",
        "_report_with_css.md",
        "_report_nature_with_css.md",
        "report.html",
        "report_nature.html",
    ):
        payload = technical if name == "report_technical.md" else "# Report\n"
        path.joinpath(name).write_text(payload, encoding="utf-8")


def _write_worker(run_dir: Path, worker: str, plan: str, report_name: str, technical: str) -> Path:
    worker_dir = run_dir / worker
    report_dir = worker_dir / "workspace" / "reports" / report_name
    plan_dir = worker_dir / "workspace" / "plans"
    plan_dir.mkdir(parents=True)
    plan_dir.joinpath("plan.md").write_text(plan, encoding="utf-8")
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_dir.joinpath("transcript.clean.log").write_text(plan, encoding="utf-8")
    _write_dual_report_dir(report_dir, technical)
    index = {
        "status": "PASS",
        "audit": {
            "report_dirs": [str(report_dir)],
            "report_audits": [{"path": str(report_dir), "status": "PASS"}],
        },
    }
    worker_dir.joinpath("artifact_index.json").write_text(json.dumps(index), encoding="utf-8")
    return report_dir


def test_strict_live_artifact_audit_passes_for_paper_and_trajectory(tmp_path):
    run_dir = tmp_path / "live"
    run_dir.mkdir()
    run_dir.joinpath("LIVE_TEST_AUDIT.md").write_text("- Status: **PASS**\n", encoding="utf-8")

    paper_plan = (
        "10.1038/s41588-024-01898-1 replicate_paper read_paper "
        "paper_replication_compare statistical_review safety_check world_model_audit "
        "generate_report format='dual'"
    )
    paper_report = _write_worker(run_dir, "worker-02", paper_plan, "paper_report", "# Paper report\n")
    paper_report.joinpath("paper_replication_comparison.md").write_text(
        "\n".join([
            "# Paper Replication Comparison",
            "Acceptance verdict: PASS_WITH_LIMITATIONS",
            "| Gate | Status | Observed |",
            "|---|---:|---|",
            "| paper_access | PASS | full text |",
            "| cohort_count | PASS | n=100 |",
            "| model_auc | PASS | 0.75 |",
            "| calibration_ece | PASS | 0.1 |",
            "| feature_importance | PASS | 5 |",
            "| figure_artifacts | PASS | matched |",
        ]),
        encoding="utf-8",
    )

    trajectory_plan = (
        "HealthFormer trajectory trajectory_tokenize statistical_review safety_check "
        "world_model_audit generate_report format='dual'"
    )
    trajectory_text = "\n".join([
        "# Trajectory report",
        "Trajectory layer: 3,536,009 HealthFormer-style tokens prepared",
        "| n_tokens | 3536009 |",
        "| available_tokens | 3536009 |",
        "| training_distribution_coverage | 0.0000 |",
        "| allowed_claim_type | association_conditioned_forecast |",
        "| trajectory_time_source | synthetic_assessment_instance_dates |",
        "| world_model_audit | PARTIAL |",
    ])
    _write_worker(run_dir, "worker-06", trajectory_plan, "trajectory_report", trajectory_text)

    audit = run_live_artifact_audit(run_dir)

    assert audit.status == "PASS"
    assert audit.n_failed == 0
    names = {check.name for check in audit.checks if check.status == "PASS"}
    assert {"paper_acceptance_gates", "trajectory_tokens_positive", "trajectory_claim_boundary"} <= names

    written = write_audit_artifacts(audit, tmp_path / "audit")
    assert Path(written.artifacts["json"]).exists()
    assert Path(written.artifacts["markdown"]).exists()


def test_strict_live_artifact_audit_fails_when_trajectory_boundary_is_missing(tmp_path):
    run_dir = tmp_path / "live"
    run_dir.mkdir()
    run_dir.joinpath("LIVE_TEST_AUDIT.md").write_text("- Status: **PASS**\n", encoding="utf-8")
    _write_worker(
        run_dir,
        "worker-06",
        "HealthFormer trajectory trajectory_tokenize statistical_review safety_check world_model_audit generate_report format='dual'",
        "trajectory_report",
        "| n_tokens | 120 |",
    )

    audit = run_live_artifact_audit(run_dir)

    assert audit.status == "FAIL"
    failures = {check.name: check for check in audit.checks if check.status == "FAIL"}
    assert "paper_replication_worker" in failures
    assert "trajectory_claim_boundary" in failures


def test_strict_live_artifact_audit_can_scope_to_current_trajectory_run(tmp_path):
    run_dir = tmp_path / "live"
    run_dir.mkdir()
    run_dir.joinpath("LIVE_TEST_AUDIT.md").write_text("- Status: **PASS**\n", encoding="utf-8")
    _write_worker(
        run_dir,
        "worker-01",
        "grand challenge trajectory_tokenize statistical_review safety_check world_model_audit generate_report format='dual'",
        "trajectory_report",
        "\n".join([
            "Trajectory layer: 3,536,009 tokens",
            "| available_tokens | 3536009 |",
            "| training_distribution_coverage | 0 |",
            "| association_conditioned_forecast | yes |",
            "| trajectory_time_source | synthetic_assessment_instance_dates |",
            "| world_model_audit | PARTIAL |",
        ]),
    )

    audit = run_live_artifact_audit(run_dir, require_paper_replication=False)

    assert audit.status == "PASS"
    names = {check.name for check in audit.checks}
    assert "paper_replication_not_requested" in names
    assert "trajectory_claim_boundary" in names
