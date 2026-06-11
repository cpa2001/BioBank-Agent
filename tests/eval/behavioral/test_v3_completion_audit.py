"""Completion-audit tests for the v3 objective."""

from __future__ import annotations

import json
from pathlib import Path

from biobank_agent.eval.v3_completion import collect_external_evidence_for_completion, run_v3_completion_audit


def _write_required_repo_files(root: Path) -> None:
    for rel in (
        "biobank_agent/core/runtime.py",
        "biobank_agent/core/events.py",
        "biobank_agent/core/tools/scheduler.py",
        "biobank_agent/core/evolution/patch_classifier.py",
        "biobank_agent/core/evolution/pattern_mining.py",
        "biobank_agent/core/telemetry.py",
        "biobank_agent/plan_executor.py",
        "biobank_agent/plan_state.py",
        "biobank_agent/skills/report.py",
        "biobank_agent/skills/trajectory_tokenize.py",
        "biobank_agent/skills/world_model_audit.py",
        "biobank_agent/skills/bank_data_readiness.py",
        "biobank_agent/skills/replicate_paper.py",
        "biobank_agent/eval/live_artifact_audit.py",
        "biobank_agent/eval/behavioral.py",
        "biobank_agent/eval/scheduled.py",
        "biobank_agent/eval/v3_completion.py",
        "tests/eval/behavioral/always_passes.yaml",
        "tests/eval/behavioral/usually_passes.yaml",
        "docs/architecture/OVERVIEW.md",
        "docs/guides/CLI_COMMAND_PLUGINS.md",
        "docs/guides/OBSERVABILITY.md",
        "docs/examples/README.md",
        "docs/examples/mcp_http_stub_demo.py",
        "docs/examples/paper_replication_fixture.md",
        "docs/examples/sdk_streaming_client.py",
        ".github/workflows/biobank-scheduled-eval.yml",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")
    (root / "benchmarks" / "biobank_agent_manual_cases" / "cases").mkdir(parents=True)
    (root / "biobank_agent" / "config.py").write_text(
        "data_deidentified: bool = True\n"
        "max_train_rows_default: int = 0\n"
        "default_analysis_sample_size: int = 0\n"
        "disclosure_control_mode = 'internal'\n",
        encoding="utf-8",
    )
    (root / "biobank_agent" / "data" / "cohort.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "biobank_agent" / "data" / "cohort.py").write_text("Using all controls\n", encoding="utf-8")
    (root / "biobank_agent" / "skills" / "train_model.py").write_text("full_dataset_default\n", encoding="utf-8")
    (root / "biobank_agent" / "core" / "tools" / "scheduler.py").write_text("disclosure_control_mode\n", encoding="utf-8")
    (root / "biobank_agent" / "skills" / "safety_check.py").write_text("disclosure_control_mode\n", encoding="utf-8")
    token_files = {
        "biobank_agent/cli/tui/main_screen.py": "(\"f5\", \"approve_plan\" tui_event_callback tui_confirm_fn\n",
        "biobank_agent/cli/tui/progress_panel.py": "ok\n",
        "biobank_agent/cli/tui/run_detail.py": "ok\n",
        "biobank_agent/cli/tui/worker_review.py": "Worker/review queue F5 approve Ctrl+J/K select\n",
        "biobank_agent/cli/tui/confirm_modal.py": "ok\n",
        "biobank_agent/cli/commands/registry.py": "ok\n",
        "biobank_agent/cli/commands/mcp.py": "/mcp-start /mcp-health /mcp-call\n",
        "biobank_agent/cli/live/streaming_renderer.py": "ok\n",
        "tests/test_cli_v3_modules.py": "test_tui_textual_runtime_exports_nonblank_large_worker_screenshot\n",
        "biobank_agent/domain/banks/base.py": "ok\n",
        "biobank_agent/domain/banks/ukb_adapter.py": "ok\n",
        "biobank_agent/domain/banks/hpp_adapter.py": "ok\n",
        "biobank_agent/domain/banks/ckb_adapter.py": "ok\n",
        "biobank_agent/domain/banks/rap_adapter.py": "RAPAdapter register_adapter\n",
        "biobank_agent/skills/bank_data_probe.py": "bank_data_probe n_subjects\n",
        "tests/core/test_bank_data_path.py": "bank_data_readiness_runs_configured_hpp_ckb_fixture_probes\n",
        "tests/core/test_domain_layer.py": "ok\n",
        "biobank_agent/extensions/mcp_manager.py": "ok\n",
        "tests/core/test_subagent_mcp.py": (
            "test_mcp_manager_registers_http_sse_tools "
            "test_mcp_manager_parses_text_event_stream_json_rpc_responses "
            "test_mcp_tool_call_reconnects_once_after_http_failure\n"
        ),
        "biobank_agent/domain/reproducibility/paper_replicator.py": "paper_replication_compare generate_report format\n",
        "biobank_agent/skills/paper_replication_compare.py": "PASS_WITH_LIMITATIONS paper_replication_comparison\n",
        "tests/eval/behavioral/always_passes.yaml": "published_milton_gold_fixture_route 10.1038/s41588-024-01898-1\n",
        "biobank_agent/core/evolution/reflexion.py": "RetryWithCorrectedArgs SwapSkill InsertPrerequisiteStep SkipStep\n",
        "biobank_agent/core/evolution/auto_merger.py": "pr_ready AutoMerger allow-list\n",
        "tests/core/test_evolution_layer.py": "test_auto_merger_routes_high_risk_allowed_patch_to_pr\n",
        "biobank_agent/core/telemetry.py": "OpenTelemetryBridge otel_policy=production low-cardinality\n",
        "tests/core/test_telemetry.py": "test_opentelemetry_production_policy_requires_otlp_collector\n",
        "biobank_agent/sdk/client.py": "AsyncBiobankClient stream\n",
    }
    for rel, content in token_files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (root / "docs" / "related_works").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "related_works" / "s41588-024-01898-1.pdf").write_bytes(b"%PDF-1.4\n")
    for rel in ("reports/eval/behavioral/latest.json", "reports/eval/scheduled/latest.json"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")


def _write_dual_report(report_dir: Path, technical: str) -> None:
    report_dir.mkdir(parents=True)
    for name in (
        "report.md",
        "report_technical.md",
        "report_nature.md",
        "_report_with_css.md",
        "_report_nature_with_css.md",
        "report.html",
        "report_nature.html",
    ):
        report_dir.joinpath(name).write_text(technical if name == "report_technical.md" else "# Report\n", encoding="utf-8")


def _write_live_run(root: Path) -> Path:
    run_dir = root / "reports" / "biobank_live_tests" / "manual" / "20260511_100909"
    run_dir.mkdir(parents=True)
    rows = "\n".join(f"| {idx:02d} | PASS | 1s | t | 1 |" for idx in range(1, 13))
    run_dir.joinpath("LIVE_TEST_AUDIT.md").write_text(f"- Status: **PASS**\n\n{rows}\n", encoding="utf-8")

    paper_worker = run_dir / "worker-02"
    paper_plan = paper_worker / "workspace" / "plans"
    paper_report = paper_worker / "workspace" / "reports" / "paper"
    paper_plan.mkdir(parents=True)
    paper_plan.joinpath("plan.md").write_text(
        "10.1038/s41588-024-01898-1 replicate_paper read_paper paper_replication_compare "
        "statistical_review safety_check world_model_audit generate_report {'format': 'dual'}",
        encoding="utf-8",
    )
    _write_dual_report(paper_report, "# Report\n")
    paper_report.joinpath("paper_replication_comparison.md").write_text(
        "Acceptance verdict: PASS_WITH_LIMITATIONS\n"
        "| Gate | Status | Observed |\n|---|---:|---|\n"
        "| paper_access | PASS | full |\n| cohort_count | PASS | ok |\n"
        "| model_auc | PASS | ok |\n| calibration_ece | PASS | ok |\n"
        "| feature_importance | PASS | ok |\n| figure_artifacts | PASS | ok |\n",
        encoding="utf-8",
    )
    paper_worker.joinpath("artifact_index.json").write_text(json.dumps({"audit": {"report_dirs": [str(paper_report)]}}), encoding="utf-8")

    trajectory_worker = run_dir / "worker-06"
    trajectory_plan = trajectory_worker / "workspace" / "plans"
    trajectory_report = trajectory_worker / "workspace" / "reports" / "trajectory"
    trajectory_plan.mkdir(parents=True)
    trajectory_plan.joinpath("plan.md").write_text(
        "Build a longitudinal HealthFormer-style trajectory forecast over time "
        "trajectory_tokenize statistical_review safety_check world_model_audit generate_report {'format': 'dual'}",
        encoding="utf-8",
    )
    _write_dual_report(
        trajectory_report,
        "Trajectory layer: 3,536,009 tokens\n"
        "| available_tokens | 3536009 |\n"
        "| training_distribution_coverage | 0 |\n"
        "| association_conditioned_forecast | yes |\n"
        "| trajectory_time_source | synthetic_assessment_instance_dates |\n"
        "| world_model_audit | PARTIAL |\n",
    )
    trajectory_worker.joinpath("artifact_index.json").write_text(json.dumps({"audit": {"report_dirs": [str(trajectory_report)]}}), encoding="utf-8")
    return run_dir


def test_v3_completion_audit_marks_external_evidence_as_blocked(tmp_path: Path):
    _write_required_repo_files(tmp_path)
    live_run = _write_live_run(tmp_path)

    audit = run_v3_completion_audit(
        repo_root=tmp_path,
        live_run_dir=live_run,
        output_dir=tmp_path / "reports" / "eval" / "v3_completion",
    )

    assert audit.status == "BLOCKED_EXTERNAL"
    blocked = {item.id for item in audit.criteria if item.status == "BLOCKED_EXTERNAL"}
    assert {
        "credentialed_hpp_ckb_rap_data",
        "real_mcp_compatibility_matrix",
        "remote_ci_scheduled_eval",
        "credentialed_high_risk_pr_path",
    } <= blocked
    assert (tmp_path / "reports" / "eval" / "v3_completion" / "latest.json").exists()


def test_v3_completion_audit_can_reach_complete_with_external_evidence(tmp_path: Path):
    _write_required_repo_files(tmp_path)
    live_run = _write_live_run(tmp_path)
    readiness = tmp_path / "bank_readiness.json"
    readiness.write_text(json.dumps({
        "artifact_type": "bank_data_readiness",
        "generated_by": "bank_data_readiness",
        "banks": [
            {
                "bank_id": "ukb",
                "status": "READY",
                "n_subjects": 500_000,
                "source_env": "BIOBANK_UKB_DATA_DIR",
                "diagnosis_probe": {"n_case_subjects": 4_321},
            },
            {
                "bank_id": "hpp",
                "status": "READY",
                "n_subjects": 100,
                "source_env": "BIOBANK_HPP_DATA_DIR",
                "diagnosis_probe": {"n_case_subjects": 10},
            },
            {
                "bank_id": "ckb",
                "status": "READY",
                "n_subjects": 100,
                "source_env": "BIOBANK_CKB_DATA_DIR",
                "diagnosis_probe": {"n_case_subjects": 10},
            },
            {
                "bank_id": "ukb_rap",
                "status": "REMOTE_READY",
                "credential_envs_present": ["DX_PROJECT_CONTEXT_ID"],
            },
        ]
    }), encoding="utf-8")
    mcp = tmp_path / "mcp_compat.json"
    mcp.write_text(json.dumps({
        "artifact_type": "mcp_compatibility_matrix",
        "servers": [
            {"name": "github", "transport": "stdio", "status": "PASS", "health_checked": True, "tool_called": True},
            {"name": "postgres", "transport": "http_sse", "status": "READY", "health_checked": True, "tool_called": True},
        ],
    }), encoding="utf-8")
    ci = tmp_path / "ci.json"
    ci.write_text(json.dumps({
        "artifact_type": "remote_ci_scheduled_eval",
        "status": "SUCCESS",
        "url": "https://github.com/example/repo/actions/runs/1",
        "run_id": "1",
    }), encoding="utf-8")
    high_pr = tmp_path / "high_pr.json"
    high_pr.write_text(json.dumps({
        "artifact_type": "high_risk_pr_evidence",
        "generated_by": "external_evidence.write_high_pr_evidence",
        "status": "PR_OPENED",
        "pr_url": "https://github.com/example/repo/pull/1",
        "branch": "auto-improve/demo-00001",
        "verified_by_gh": True,
        "gh": {
            "url": "https://github.com/example/repo/pull/1",
            "headRefName": "auto-improve/demo-00001",
            "state": "OPEN",
        },
    }), encoding="utf-8")

    audit = run_v3_completion_audit(
        repo_root=tmp_path,
        live_run_dir=live_run,
        hpp_ckb_rap_readiness=readiness,
        mcp_compat_evidence=mcp,
        remote_ci_evidence=ci,
        high_pr_evidence=str(high_pr),
        write_artifacts=False,
    )

    assert audit.status == "COMPLETE"
    assert all(item.status == "PASS" for item in audit.criteria)


def test_v3_completion_audit_auto_discovers_latest_external_evidence(tmp_path: Path):
    _write_required_repo_files(tmp_path)
    live_run = _write_live_run(tmp_path)
    evidence_dir = tmp_path / "reports" / "eval" / "external_evidence"
    evidence_dir.mkdir(parents=True)
    evidence_dir.joinpath("bank_readiness_latest.json").write_text(json.dumps({
        "artifact_type": "bank_data_readiness",
        "generated_by": "bank_data_readiness",
        "banks": [
            {"bank_id": "ukb", "status": "READY", "n_subjects": 500_000, "source_env": "BIOBANK_UKB_DATA_DIR", "diagnosis_probe": {"n_case_subjects": 4_321}},
            {"bank_id": "hpp", "status": "READY", "n_subjects": 100, "source_env": "BIOBANK_HPP_DATA_DIR", "diagnosis_probe": {"n_case_subjects": 10}},
            {"bank_id": "ckb", "status": "READY", "n_subjects": 100, "source_env": "BIOBANK_CKB_DATA_DIR", "diagnosis_probe": {"n_case_subjects": 10}},
            {"bank_id": "ukb_rap", "status": "REMOTE_READY", "credential_envs_present": ["DX_PROJECT_CONTEXT_ID"]},
        ],
    }), encoding="utf-8")
    evidence_dir.joinpath("mcp_compat_latest.json").write_text(json.dumps({
        "artifact_type": "mcp_compatibility_matrix",
        "servers": [
            {"name": "github", "transport": "stdio", "status": "PASS", "health_checked": True, "tool_called": True},
            {"name": "postgres", "transport": "http_sse", "status": "READY", "health_checked": True, "tool_called": True},
        ],
    }), encoding="utf-8")
    evidence_dir.joinpath("remote_ci_latest.json").write_text(json.dumps({
        "artifact_type": "remote_ci_scheduled_eval",
        "status": "SUCCESS",
        "url": "https://github.com/example/repo/actions/runs/1",
        "run_id": "1",
    }), encoding="utf-8")
    evidence_dir.joinpath("high_pr_latest.json").write_text(json.dumps({
        "artifact_type": "high_risk_pr_evidence",
        "generated_by": "external_evidence.write_high_pr_evidence",
        "status": "PR_OPENED",
        "pr_url": "https://github.com/example/repo/pull/1",
        "branch": "auto-improve/demo-00001",
        "verified_by_gh": True,
        "gh": {
            "url": "https://github.com/example/repo/pull/1",
            "headRefName": "auto-improve/demo-00001",
            "state": "OPEN",
        },
    }), encoding="utf-8")

    audit = run_v3_completion_audit(
        repo_root=tmp_path,
        live_run_dir=live_run,
        write_artifacts=False,
    )

    assert audit.status == "COMPLETE"


def test_v3_completion_audit_rejects_under_specified_external_evidence(tmp_path: Path):
    _write_required_repo_files(tmp_path)
    live_run = _write_live_run(tmp_path)
    readiness = tmp_path / "bank_readiness_minimal.json"
    readiness.write_text(json.dumps({
        "banks": [
            {"bank_id": "hpp", "status": "READY"},
            {"bank_id": "ckb", "status": "READY"},
            {"bank_id": "ukb_rap", "status": "REMOTE_READY"},
        ]
    }), encoding="utf-8")
    mcp = tmp_path / "mcp_minimal.json"
    mcp.write_text(json.dumps({"servers": [{"status": "PASS"}, {"status": "READY"}]}), encoding="utf-8")
    ci = tmp_path / "ci_minimal.json"
    ci.write_text(json.dumps({"status": "SUCCESS", "url": "https://example.test/run/1"}), encoding="utf-8")

    audit = run_v3_completion_audit(
        repo_root=tmp_path,
        live_run_dir=live_run,
        hpp_ckb_rap_readiness=readiness,
        mcp_compat_evidence=mcp,
        remote_ci_evidence=ci,
        high_pr_evidence="https://example.test/pull/1",
        write_artifacts=False,
    )

    blocked = {item.id: item for item in audit.criteria if item.status == "BLOCKED_EXTERNAL"}
    assert audit.status == "BLOCKED_EXTERNAL"
    assert "credentialed_hpp_ckb_rap_data" in blocked
    assert "real_mcp_compatibility_matrix" in blocked
    assert "remote_ci_scheduled_eval" in blocked
    assert "credentialed_high_risk_pr_path" in blocked


def test_collect_external_evidence_for_completion_returns_all_artifact_paths(tmp_path: Path, monkeypatch):
    def artifact(kind: str, path: str):
        return type("Artifact", (), {"kind": kind, "path": path})()

    monkeypatch.setattr(
        "biobank_agent.eval.external_evidence.collect_bank_readiness_evidence",
        lambda **kwargs: artifact("bank_data_readiness", str(tmp_path / "bank.json")),
    )
    monkeypatch.setattr(
        "biobank_agent.eval.external_evidence.collect_mcp_compatibility_evidence",
        lambda **kwargs: artifact("mcp_compatibility_matrix", str(tmp_path / "mcp.json")),
    )
    monkeypatch.setattr(
        "biobank_agent.eval.external_evidence.collect_remote_ci_evidence",
        lambda **kwargs: artifact("remote_ci_scheduled_eval", str(tmp_path / "ci.json")),
    )
    monkeypatch.setattr(
        "biobank_agent.eval.external_evidence.collect_high_pr_evidence",
        lambda **kwargs: artifact("high_risk_pr_evidence", str(tmp_path / "pr.json")),
    )

    paths = collect_external_evidence_for_completion(output_dir=tmp_path, pr_url="https://github.com/x/y/pull/1", branch="b")

    assert paths == {
        "bank_data_readiness": str(tmp_path / "bank.json"),
        "mcp_compatibility_matrix": str(tmp_path / "mcp.json"),
        "remote_ci_scheduled_eval": str(tmp_path / "ci.json"),
        "high_risk_pr_evidence": str(tmp_path / "pr.json"),
    }
