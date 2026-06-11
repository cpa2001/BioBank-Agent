"""Completion audit for the Biobank Agent v3 objective.

This is deliberately stricter than normal regression tests: it maps the
user-facing v3 objective to concrete artifacts and marks externally blocked
items explicitly instead of treating local scaffolds as completion.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from biobank_agent.eval.live_artifact_audit import run_live_artifact_audit


@dataclass
class CompletionCriterion:
    id: str
    requirement: str
    status: str
    evidence: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class V3CompletionAudit:
    status: str
    generated_at: str
    objective: str
    criteria: list[CompletionCriterion]
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def n_passed(self) -> int:
        return sum(1 for item in self.criteria if item.status == "PASS")

    @property
    def n_blocked_external(self) -> int:
        return sum(1 for item in self.criteria if item.status == "BLOCKED_EXTERNAL")

    @property
    def n_failed(self) -> int:
        return sum(1 for item in self.criteria if item.status == "FAIL")


_OBJECTIVE = (
    "Implement the full Biobank Agent v3 plan, verify it with the real "
    "Biobank Agent rather than injected tests, pass all manual benchmark cases, "
    "and ensure trajectory workflows operate according to design."
)


def collect_external_evidence_for_completion(
    *,
    output_dir: str | Path = "reports/eval/external_evidence",
    banks: str = "ukb,hpp,ckb,ukb_rap",
    icd10_code: str = "E11",
    probe_fields: str = "hba1c,bmi,glucose",
    mcp_config: str | Path | None = None,
    mcp_call_args: str | Path | None = None,
    mcp_min_servers: int = 2,
    workflow: str = "biobank-scheduled-eval.yml",
    repo_root: str | Path = ".",
    pr_url: str = "",
    branch: str = "",
) -> dict[str, str]:
    """Collect all external evidence artifacts before running completion audit."""
    from biobank_agent.eval.external_evidence import (
        collect_bank_readiness_evidence,
        collect_high_pr_evidence,
        collect_mcp_compatibility_evidence,
        collect_remote_ci_evidence,
    )

    artifacts = [
        collect_bank_readiness_evidence(
            output_dir=output_dir,
            banks=banks,
            icd10_code=icd10_code,
            probe_fields=probe_fields,
        ),
        collect_mcp_compatibility_evidence(
            output_dir=output_dir,
            config_path=mcp_config,
            call_args_path=mcp_call_args,
            min_servers=mcp_min_servers,
        ),
        collect_remote_ci_evidence(
            output_dir=output_dir,
            workflow=workflow,
            repo_root=repo_root,
        ),
        collect_high_pr_evidence(
            output_dir=output_dir,
            pr_url=pr_url,
            branch=branch,
            repo_root=repo_root,
        ),
    ]
    return {item.kind: item.path for item in artifacts}


def run_v3_completion_audit(
    *,
    repo_root: str | Path = ".",
    live_run_dir: str | Path | None = None,
    hpp_ckb_rap_readiness: str | Path | None = None,
    mcp_compat_evidence: str | Path | None = None,
    remote_ci_evidence: str | Path | None = None,
    high_pr_evidence: str = "",
    external_evidence_dir: str | Path | None = None,
    write_artifacts: bool = True,
    output_dir: str | Path = "reports/eval/v3_completion",
) -> V3CompletionAudit:
    root = Path(repo_root).expanduser().resolve()
    external_paths = _discover_external_evidence(root, external_evidence_dir)
    hpp_ckb_rap_readiness = hpp_ckb_rap_readiness or external_paths.get("bank_data_readiness")
    mcp_compat_evidence = mcp_compat_evidence or external_paths.get("mcp_compatibility_matrix")
    remote_ci_evidence = remote_ci_evidence or external_paths.get("remote_ci_scheduled_eval")
    high_pr_evidence = high_pr_evidence or str(external_paths.get("high_risk_pr_evidence") or "")
    criteria: list[CompletionCriterion] = []

    criteria.append(_check_core_files(root))
    criteria.append(_check_eval_and_docs(root))
    criteria.append(_check_cli_tui_foundation(root))
    criteria.append(_check_bank_adapter_foundation(root))
    criteria.append(_check_mcp_foundation(root))
    criteria.append(_check_paper_replication_foundation(root))
    criteria.append(_check_self_evolution_foundation(root))
    criteria.append(_check_telemetry_sdk_docs(root))
    criteria.append(_check_local_eval_artifacts(root))
    live_path = _resolve_live_run(root, live_run_dir)
    criteria.append(_check_live_run(live_path))
    criteria.append(_check_strict_live_artifacts(live_path))
    criteria.append(_check_full_data_policy(root))
    criteria.append(_check_credentialed_ukb_data(hpp_ckb_rap_readiness))
    criteria.append(_check_hpp_ckb_rap_readiness(hpp_ckb_rap_readiness))
    criteria.append(_check_mcp_compat(mcp_compat_evidence))
    criteria.append(_check_remote_ci(remote_ci_evidence))
    criteria.append(_check_high_pr(high_pr_evidence))

    status = _overall_status(criteria)
    audit = V3CompletionAudit(
        status=status,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        objective=_OBJECTIVE,
        criteria=criteria,
    )
    if write_artifacts:
        _write_artifacts(audit, Path(output_dir))
    return audit


def _discover_external_evidence(
    root: Path,
    external_evidence_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Return latest external evidence artifacts by artifact_type.

    Explicit ``run_v3_completion_audit`` arguments still win. This helper only
    removes operator friction after `biobank eval --suite external_evidence`
    writes its collector outputs.
    """
    evidence_dir = Path(external_evidence_dir).expanduser() if external_evidence_dir else root / "reports" / "eval" / "external_evidence"
    if not evidence_dir.is_absolute():
        evidence_dir = root / evidence_dir
    candidates = [
        evidence_dir / "bank_readiness_latest.json",
        evidence_dir / "mcp_compat_latest.json",
        evidence_dir / "remote_ci_latest.json",
        evidence_dir / "high_pr_latest.json",
    ]
    # Fall back to timestamped files if latest aliases are absent.
    for pattern in ("bank_readiness_*.json", "mcp_compat_*.json", "remote_ci_*.json", "high_pr_*.json"):
        candidates.extend(sorted(evidence_dir.glob(pattern), reverse=True))
    out: dict[str, Path] = {}
    for path in candidates:
        if not path.exists() or path.name.endswith("_manifest_latest.json"):
            continue
        payload = _read_json(path)
        artifact_type = str(payload.get("artifact_type") or "")
        if artifact_type and artifact_type not in out:
            out[artifact_type] = path
    return out


def _overall_status(criteria: list[CompletionCriterion]) -> str:
    if any(item.status == "FAIL" for item in criteria):
        return "FAIL"
    if any(item.status == "BLOCKED_EXTERNAL" for item in criteria):
        return "BLOCKED_EXTERNAL"
    if all(item.status == "PASS" for item in criteria):
        return "COMPLETE"
    return "PARTIAL"


def _check_core_files(root: Path) -> CompletionCriterion:
    required = [
        "biobank_agent/core/runtime.py",
        "biobank_agent/core/events.py",
        "biobank_agent/core/tools/scheduler.py",
        "biobank_agent/plan_executor.py",
        "biobank_agent/plan_state.py",
        "biobank_agent/skills/report.py",
        "biobank_agent/skills/trajectory_tokenize.py",
        "biobank_agent/skills/world_model_audit.py",
        "biobank_agent/skills/bank_data_readiness.py",
        "biobank_agent/eval/live_artifact_audit.py",
    ]
    missing = [item for item in required if not (root / item).exists()]
    return CompletionCriterion(
        id="core_runtime_plan_report",
        requirement="Core async/runtime, schema-gated plan execution, dual-report artifact gates, trajectory/audit skills, and readiness tooling exist in the repo.",
        status="PASS" if not missing else "FAIL",
        evidence=[item for item in required if (root / item).exists()],
        missing=missing,
    )


def _check_eval_and_docs(root: Path) -> CompletionCriterion:
    required = [
        "benchmarks/biobank_agent_manual_cases/cases",
        "tests/eval/behavioral/always_passes.yaml",
        "tests/eval/behavioral/usually_passes.yaml",
        "biobank_agent/eval/behavioral.py",
        "biobank_agent/eval/scheduled.py",
        "biobank_agent/eval/v3_completion.py",
        "docs/architecture/OVERVIEW.md",
    ]
    missing = [item for item in required if not (root / item).exists()]
    return CompletionCriterion(
        id="benchmarks_eval_docs",
        requirement="Manual benchmark cases, behavioral/scheduled gates, and evidence-based v3 documentation are present.",
        status="PASS" if not missing else "FAIL",
        evidence=[item for item in required if (root / item).exists()],
        missing=missing,
    )


def _missing_files(root: Path, required: list[str]) -> list[str]:
    return [item for item in required if not (root / item).exists()]


def _missing_tokens(root: Path, token_checks: dict[str, list[str]]) -> list[str]:
    missing: list[str] = []
    for rel, tokens in token_checks.items():
        text = _read_text(root / rel)
        missing.extend(f"{rel}: {token}" for token in tokens if token not in text)
    return missing


def _check_cli_tui_foundation(root: Path) -> CompletionCriterion:
    required = [
        "biobank_agent/cli/tui/main_screen.py",
        "biobank_agent/cli/tui/progress_panel.py",
        "biobank_agent/cli/tui/run_detail.py",
        "biobank_agent/cli/tui/worker_review.py",
        "biobank_agent/cli/tui/confirm_modal.py",
        "biobank_agent/cli/commands/registry.py",
        "biobank_agent/cli/live/streaming_renderer.py",
        "tests/test_cli_v3_modules.py",
        "docs/guides/CLI_COMMAND_PLUGINS.md",
    ]
    token_checks = {
        "biobank_agent/cli/tui/main_screen.py": ["(\"f5\", \"approve_plan\"", "tui_event_callback", "tui_confirm_fn"],
        "biobank_agent/cli/tui/worker_review.py": ["Worker/review queue", "F5 approve", "Ctrl+J/K select"],
        "tests/test_cli_v3_modules.py": ["test_tui_textual_runtime_exports_nonblank_large_worker_screenshot"],
    }
    missing = _missing_files(root, required) + _missing_tokens(root, token_checks)
    return CompletionCriterion(
        id="cli_tui_productization_foundation",
        requirement="Textual/rich CLI foundation, shared slash-command registry, worker/review queue, repair shortcuts, and screenshot-backed TUI QA are wired.",
        status="PASS" if not missing else "FAIL",
        evidence=[item for item in required if (root / item).exists()],
        missing=missing,
    )


def _check_bank_adapter_foundation(root: Path) -> CompletionCriterion:
    required = [
        "biobank_agent/domain/banks/base.py",
        "biobank_agent/domain/banks/ukb_adapter.py",
        "biobank_agent/domain/banks/hpp_adapter.py",
        "biobank_agent/domain/banks/ckb_adapter.py",
        "biobank_agent/domain/banks/rap_adapter.py",
        "biobank_agent/skills/bank_data_probe.py",
        "biobank_agent/skills/bank_data_readiness.py",
        "tests/core/test_bank_data_path.py",
        "tests/core/test_domain_layer.py",
    ]
    token_checks = {
        "biobank_agent/domain/banks/rap_adapter.py": ["RAPAdapter", "register_adapter"],
        "biobank_agent/skills/bank_data_probe.py": ["bank_data_probe", "n_subjects"],
        "tests/core/test_bank_data_path.py": ["bank_data_readiness_runs_configured_hpp_ckb_fixture_probes"],
    }
    missing = _missing_files(root, required) + _missing_tokens(root, token_checks)
    return CompletionCriterion(
        id="bank_adapter_data_path_foundation",
        requirement="UKB/HPP/CKB/RAP adapters, bank data probe/readiness skills, and fixture-level DataManager/cohort/model tests are present.",
        status="PASS" if not missing else "FAIL",
        evidence=[item for item in required if (root / item).exists()],
        missing=missing,
    )


def _check_mcp_foundation(root: Path) -> CompletionCriterion:
    required = [
        "biobank_agent/extensions/mcp_manager.py",
        "biobank_agent/cli/commands/mcp.py",
        "docs/examples/mcp_http_stub_demo.py",
        "tests/core/test_subagent_mcp.py",
        "tests/test_cli_v3_modules.py",
    ]
    token_checks = {
        "tests/core/test_subagent_mcp.py": [
            "test_mcp_manager_registers_http_sse_tools",
            "test_mcp_manager_parses_text_event_stream_json_rpc_responses",
            "test_mcp_tool_call_reconnects_once_after_http_failure",
        ],
        "biobank_agent/cli/commands/mcp.py": ["/mcp-start", "/mcp-health", "/mcp-call"],
    }
    missing = _missing_files(root, required) + _missing_tokens(root, token_checks)
    return CompletionCriterion(
        id="mcp_local_foundation",
        requirement="MCP manager and CLI support STDIO/HTTP-SSE discovery, tool calls, health checks, and reconnect behavior under local tests.",
        status="PASS" if not missing else "FAIL",
        evidence=[item for item in required if (root / item).exists()],
        missing=missing,
    )


def _check_paper_replication_foundation(root: Path) -> CompletionCriterion:
    required = [
        "biobank_agent/domain/reproducibility/paper_replicator.py",
        "biobank_agent/skills/replicate_paper.py",
        "biobank_agent/skills/paper_replication_compare.py",
        "docs/related_works/s41588-024-01898-1.pdf",
        "docs/examples/paper_replication_fixture.md",
        "tests/core/test_domain_layer.py",
        "tests/eval/behavioral/always_passes.yaml",
    ]
    token_checks = {
        "biobank_agent/domain/reproducibility/paper_replicator.py": ["paper_replication_compare", "generate_report", "format"],
        "biobank_agent/skills/paper_replication_compare.py": ["PASS_WITH_LIMITATIONS", "paper_replication_comparison"],
        "tests/eval/behavioral/always_passes.yaml": ["published_milton_gold_fixture_route", "10.1038/s41588-024-01898-1"],
    }
    missing = _missing_files(root, required) + _missing_tokens(root, token_checks)
    return CompletionCriterion(
        id="paper_replication_foundation",
        requirement="Published-paper replication route has deterministic MILTON fixture, review-to-plan artifacts, comparison gates, and report integration.",
        status="PASS" if not missing else "FAIL",
        evidence=[item for item in required if (root / item).exists()],
        missing=missing,
    )


def _check_self_evolution_foundation(root: Path) -> CompletionCriterion:
    required = [
        "biobank_agent/core/evolution/reflexion.py",
        "biobank_agent/core/evolution/patch_classifier.py",
        "biobank_agent/core/evolution/auto_merger.py",
        "biobank_agent/core/evolution/pattern_mining.py",
        "tests/core/test_evolution_layer.py",
        "biobank_agent/eval/scheduled.py",
        ".github/workflows/biobank-scheduled-eval.yml",
    ]
    token_checks = {
        "biobank_agent/core/evolution/reflexion.py": ["RetryWithCorrectedArgs", "SwapSkill", "InsertPrerequisiteStep", "SkipStep"],
        "biobank_agent/core/evolution/auto_merger.py": ["pr_ready", "AutoMerger", "allow-list"],
        "tests/core/test_evolution_layer.py": ["test_auto_merger_routes_high_risk_allowed_patch_to_pr"],
    }
    missing = _missing_files(root, required) + _missing_tokens(root, token_checks)
    return CompletionCriterion(
        id="self_evolution_foundation",
        requirement="Reflexion actions, LOW/MEDIUM/HIGH patch routing, generated-skill allow-listing, scheduled pattern mining, and CI wrapper are implemented locally.",
        status="PASS" if not missing else "FAIL",
        evidence=[item for item in required if (root / item).exists()],
        missing=missing,
    )


def _check_telemetry_sdk_docs(root: Path) -> CompletionCriterion:
    required = [
        "biobank_agent/core/telemetry.py",
        "tests/core/test_telemetry.py",
        "biobank_agent/sdk/client.py",
        "docs/examples/sdk_streaming_client.py",
        "docs/guides/OBSERVABILITY.md",
        "docs/examples/README.md",
    ]
    token_checks = {
        "biobank_agent/core/telemetry.py": ["OpenTelemetryBridge", "otel_policy=production", "low-cardinality"],
        "tests/core/test_telemetry.py": ["test_opentelemetry_production_policy_requires_otlp_collector"],
        "biobank_agent/sdk/client.py": ["AsyncBiobankClient", "stream"],
    }
    missing = _missing_files(root, required) + _missing_tokens(root, token_checks)
    return CompletionCriterion(
        id="telemetry_sdk_docs_foundation",
        requirement="OpenTelemetry policy/bridge, SDK streaming client, and operator documentation/examples are present.",
        status="PASS" if not missing else "FAIL",
        evidence=[item for item in required if (root / item).exists()],
        missing=missing,
    )


def _check_local_eval_artifacts(root: Path) -> CompletionCriterion:
    required = [
        "reports/eval/behavioral/latest.json",
        "reports/eval/scheduled/latest.json",
    ]
    missing = _missing_files(root, required)
    status_notes: list[str] = []
    for rel in required:
        payload = _read_json(root / rel)
        status = str(payload.get("status") or payload.get("behavioral_status") or "").upper()
        if status != "PASS":
            missing.append(f"{rel}: status PASS")
        status_notes.append(f"{rel}={status or 'UNKNOWN'}")
    return CompletionCriterion(
        id="local_eval_artifacts_green",
        requirement="Behavioral and scheduled eval artifacts from the real project CLI are present and green.",
        status="PASS" if not missing else "FAIL",
        evidence=[item for item in required if (root / item).exists()],
        missing=missing,
        notes=", ".join(status_notes),
    )


def _resolve_live_run(root: Path, live_run_dir: str | Path | None) -> Path | None:
    if live_run_dir:
        path = Path(live_run_dir).expanduser()
        return path if path.is_absolute() else root / path
    candidates = sorted((root / "reports" / "biobank_live_tests").glob("*/*/LIVE_TEST_AUDIT.md"))
    return candidates[-1].parent if candidates else None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _worker_has_content(worker_dir: Path) -> bool:
    """Audit hardening: when a worker directory counts as having real content.

    A worker directory counts as "content_ok" when:

    1. ``artifact_index.json`` exists and reports ``audit.status == "PASS"``
       (the per-worker auditor already made a content-aware judgement —
       a query that legitimately produces no report, e.g.
       ``bank_data_readiness`` probes, still passes its own audit).
    2. ``transcript.clean.log`` (or ``transcript.raw.log`` fallback) has
       at least 10 non-empty lines, i.e. the agent actually executed
       something.
    3. *If* ``audit.report_dirs`` is non-empty, every listed report dir
       must contain at least one non-empty ``report*.md`` (rendering must
       not silently fail).

    This replaces the previous ``len(worker_dirs) >= 12`` rule which
    accepted empty directories as evidence of live benchmark coverage.
    """
    transcript = worker_dir / "transcript.clean.log"
    if not transcript.exists():
        transcript = worker_dir / "transcript.raw.log"
    transcript_lines = [
        line
        for line in _read_text(transcript).splitlines()
        if line.strip()
    ]

    index_path = worker_dir / "artifact_index.json"
    index = _read_json(index_path)
    audit = index.get("audit") if isinstance(index, dict) else None

    # Transcript ≥10 lines is the primary evidence of live execution.
    # We tolerate a missing/short transcript only when the per-worker
    # audit explicitly declares report_dirs containing rendered reports
    # (synthetic fixtures + future report-only runners).
    has_transcript = len(transcript_lines) >= 10
    has_declared_reports = (
        isinstance(audit, dict)
        and isinstance(audit.get("report_dirs"), list)
        and bool(audit.get("report_dirs"))
    )
    if not has_transcript and not has_declared_reports:
        return False
    if not isinstance(audit, dict):
        # No structured audit — fall back to the legacy "is there a
        # report markdown?" heuristic.
        reports_root = worker_dir / "workspace" / "reports"
        if not reports_root.exists():
            return False
        return any(
            md.stat().st_size > 0 and _read_text(md).strip()
            for md in reports_root.glob("*/report*.md")
        )

    audit_status = str(audit.get("status", "")).upper()
    if audit_status and audit_status != "PASS":
        # Explicit non-PASS verdict from the worker auditor.
        return False

    declared_dirs = audit.get("report_dirs") or []
    if not declared_dirs:
        # Query type legitimately did not require a report. Trust the
        # per-worker audit's empty report_dirs only when it explicitly
        # marked PASS — when status is absent we instead fall back to
        # the report-presence heuristic below.
        if audit_status == "PASS":
            return True
        reports_root = worker_dir / "workspace" / "reports"
        if not reports_root.exists():
            return False
        return any(
            md.stat().st_size > 0 and _read_text(md).strip()
            for md in reports_root.glob("*/report*.md")
        )

    for report_dir in declared_dirs:
        report_root = Path(str(report_dir))
        if not report_root.exists():
            return False
        rendered = list(report_root.glob("report*.md"))
        if not any(
            md.stat().st_size > 0 and _read_text(md).strip()
            for md in rendered
        ):
            return False
    return True


def _check_live_run(live_path: Path | None) -> CompletionCriterion:
    if live_path is None:
        return CompletionCriterion(
            id="manual_live_benchmark_all_cases",
            requirement="All manual benchmark cases under benchmarks/biobank_agent_manual_cases/cases pass through the real CLI live runner.",
            status="FAIL",
            missing=["LIVE_TEST_AUDIT.md"],
            notes="No live-test run directory was found.",
        )
    md = live_path / "LIVE_TEST_AUDIT.md"
    text = _read_text(md)
    live_json = _read_json(live_path / "live_test_audit.json")
    worker_dirs = [path for path in live_path.glob("worker-*") if path.is_dir()]
    content_ok = sum(1 for path in worker_dirs if _worker_has_content(path))
    audit_marks_pass = (
        live_json.get("status") == "PASS"
        or "Status: **PASS**" in text
        or "AUDIT_STATUS=PASS" in text
    )
    # Audit hardening:
    # The markdown PASS rows alone are not enough — every worker
    # directory that *does* exist must demonstrate content_ok (real
    # transcript + per-worker audit PASS). Empty worker dirs are no
    # longer accepted as evidence.
    content_complete = (
        bool(worker_dirs) and content_ok == len(worker_dirs)
    )
    pass_status = audit_marks_pass and content_complete
    return CompletionCriterion(
            id="manual_live_benchmark_all_cases",
            requirement="The provided manual live benchmark run passes through the live CLI runner with deterministic transcript + per-worker audit PASS (rejecting empty worker directories).",
        status="PASS" if pass_status else "FAIL",
        evidence=[str(md)] if md.exists() else [],
        missing=[] if md.exists() and pass_status else (
            [str(md)] if not md.exists() else [
                f"only {content_ok} of {len(worker_dirs)} worker dirs have ≥10 transcript lines + per-worker audit PASS"
            ]
        ),
        notes=f"worker_dirs={len(worker_dirs)}, content_ok={content_ok}, audit_marks_pass={audit_marks_pass}",
    )


def _check_strict_live_artifacts(live_path: Path | None) -> CompletionCriterion:
    if live_path is None:
        return CompletionCriterion(
            id="strict_trajectory_and_paper_artifacts",
            requirement="Strict live artifact audit proves dual reports, MILTON paper acceptance gates, and trajectory token/world-model claim boundaries.",
            status="FAIL",
            missing=["live run directory"],
        )
    strict = run_live_artifact_audit(live_path, require_paper_replication=False)
    failed = [check.name for check in strict.checks if check.status == "FAIL"]
    evidence = [str(live_path / "LIVE_TEST_AUDIT.md")]
    evidence.extend(str(path) for path in sorted(live_path.glob("strict_live_artifact_audit_*.json"))[-2:])
    return CompletionCriterion(
        id="strict_trajectory_and_paper_artifacts",
        requirement="Strict live artifact audit proves dual reports and trajectory token/world-model claim boundaries for the provided live run.",
        status="PASS" if strict.status == "PASS" else "FAIL",
        evidence=evidence,
        missing=failed,
        notes=f"strict_status={strict.status}",
    )


def _check_full_data_policy(root: Path) -> CompletionCriterion:
    files = {
        "config": root / "biobank_agent" / "config.py",
        "cohort": root / "biobank_agent" / "data" / "cohort.py",
        "train_model": root / "biobank_agent" / "skills" / "train_model.py",
        "scheduler": root / "biobank_agent" / "core" / "tools" / "scheduler.py",
        "safety_check": root / "biobank_agent" / "skills" / "safety_check.py",
    }
    text = "\n".join(_read_text(path) for path in files.values())
    required_tokens = (
        "data_deidentified: bool = True",
        "max_train_rows_default: int = 0",
        "default_analysis_sample_size: int = 0",
        "Using all",
        "full_dataset_default",
        "disclosure_control_mode",
    )
    missing = [token for token in required_tokens if token not in text]
    return CompletionCriterion(
        id="full_data_no_second_pii_drop",
        requirement="The agent preserves already de-identified local data by default and does not silently re-sample or re-redact analytical data.",
        status="PASS" if not missing else "FAIL",
        evidence=[str(path) for path in files.values()],
        missing=missing,
    )


def _check_credentialed_ukb_data(path_value: str | Path | None) -> CompletionCriterion:
    """rc1 criterion 13a (split from credentialed bank data).

    Required for v3.0 final: UKB is the canonical biobank for the agent;
    every release must demonstrate a real ``bank_data_readiness`` artifact
    where the ``ukb`` row reports READY (not SKIPPED_*) with a positive
    case-subject probe. HPP/CKB/RAP credentials live in the separate
    criterion 13b and are scoped to v3.1.
    """
    payload = _read_json(path_value)
    if not payload:
        return CompletionCriterion(
            id="credentialed_ukb_data",
            requirement="A real bank_data_readiness artifact reports UKB status=READY with positive case-subject probe.",
            status="BLOCKED_EXTERNAL",
            missing=["credentialed bank_data_readiness JSON artifact"],
            notes=(
                "v3.0 final requires UKB readiness evidence. Run "
                "`biobank-agent skill bank_data_readiness --banks ukb` against the configured "
                "local UKB data dir and point completion audit at the produced JSON."
            ),
        )
    missing: list[str] = []
    if payload.get("artifact_type") != "bank_data_readiness":
        missing.append("artifact_type=bank_data_readiness")
    if str(payload.get("generated_by") or "") != "bank_data_readiness":
        missing.append("generated_by=bank_data_readiness")
    banks = {str(item.get("bank_id")): str(item.get("status", "")).upper() for item in payload.get("banks", [])}
    bank_rows = {str(item.get("bank_id")): item for item in payload.get("banks", []) if isinstance(item, dict)}
    if banks.get("ukb") != "READY":
        missing.append("ukb: status=READY")
    ukb_row = bank_rows.get("ukb") or {}
    if not isinstance(ukb_row.get("n_subjects"), int) or int(ukb_row.get("n_subjects") or 0) <= 0:
        missing.append("ukb: n_subjects>0")
    diag = ukb_row.get("diagnosis_probe") if isinstance(ukb_row.get("diagnosis_probe"), dict) else {}
    if diag.get("n_case_subjects") is None:
        missing.append("ukb: diagnosis_probe.n_case_subjects present")
    return CompletionCriterion(
        id="credentialed_ukb_data",
        requirement="A real bank_data_readiness artifact reports UKB status=READY with positive case-subject probe.",
        status="PASS" if not missing else "BLOCKED_EXTERNAL",
        evidence=[str(path_value)] if path_value else [],
        missing=missing,
    )


def _check_hpp_ckb_rap_readiness(path_value: str | Path | None) -> CompletionCriterion:
    payload = _read_json(path_value)
    if not payload:
        return CompletionCriterion(
            id="credentialed_hpp_ckb_rap_data",
            requirement="Real HPP/CKB/RAP data paths and RAP credentials are validated with aggregate readiness artifacts. (v3.1 deliverable; v3.0 may release with this BLOCKED_EXTERNAL.)",
            status="BLOCKED_EXTERNAL",
            missing=["credentialed bank_data_readiness JSON artifact"],
            notes="v3.1 milestone. Provide --hpp-ckb-rap-readiness pointing to a bank_readiness_*.json generated against real configured datasets when institutional access is available.",
        )
    missing: list[str] = []
    if payload.get("artifact_type") != "bank_data_readiness":
        missing.append("artifact_type=bank_data_readiness")
    if str(payload.get("generated_by") or "") != "bank_data_readiness":
        missing.append("generated_by=bank_data_readiness")
    banks = {str(item.get("bank_id")): str(item.get("status", "")).upper() for item in payload.get("banks", [])}
    bank_rows = {str(item.get("bank_id")): item for item in payload.get("banks", []) if isinstance(item, dict)}
    required = {"hpp", "ckb", "ukb_rap"}
    missing.extend(bank for bank in sorted(required) if banks.get(bank) not in {"READY", "REMOTE_READY"})
    for bank in ("hpp", "ckb"):
        row = bank_rows.get(bank) or {}
        if not isinstance(row.get("n_subjects"), int) or int(row.get("n_subjects") or 0) <= 0:
            missing.append(f"{bank}: n_subjects>0")
        if not (row.get("source") or row.get("source_env") or row.get("data_dir")):
            missing.append(f"{bank}: configured data source")
        diag = row.get("diagnosis_probe") if isinstance(row.get("diagnosis_probe"), dict) else {}
        if diag.get("n_case_subjects") is None:
            missing.append(f"{bank}: diagnosis_probe.n_case_subjects")
    rap = bank_rows.get("ukb_rap") or {}
    if banks.get("ukb_rap") == "REMOTE_READY" and not rap.get("credential_envs_present"):
        missing.append("ukb_rap: credential_envs_present")
    return CompletionCriterion(
        id="credentialed_hpp_ckb_rap_data",
        requirement="Real HPP/CKB/RAP data paths and RAP credentials are validated with aggregate readiness artifacts. (v3.1 deliverable; v3.0 may release with this BLOCKED_EXTERNAL.)",
        status="PASS" if not missing else "BLOCKED_EXTERNAL",
        evidence=[str(path_value)],
        missing=missing,
        notes=f"statuses={banks}",
    )


def _check_mcp_compat(path_value: str | Path | None) -> CompletionCriterion:
    payload = _read_json(path_value)
    if not payload:
        return CompletionCriterion(
            id="real_mcp_compatibility_matrix",
            requirement="At least two real external MCP servers are started, health-checked, called, and recorded as compatibility evidence.",
            status="BLOCKED_EXTERNAL",
            missing=["real MCP compatibility evidence JSON"],
            notes="Provide --mcp-compat-evidence when GitHub/Postgres-style MCP servers are configured.",
        )
    missing: list[str] = []
    if payload.get("artifact_type") != "mcp_compatibility_matrix":
        missing.append("artifact_type=mcp_compatibility_matrix")
    servers = payload.get("servers") if isinstance(payload.get("servers"), list) else []
    ok = [
        srv for srv in servers
        if isinstance(srv, dict)
        and srv.get("status") in {"PASS", "READY"}
        and bool(srv.get("name"))
        and bool(srv.get("tool_called"))
        and bool(srv.get("health_checked"))
        and str(srv.get("transport") or "") in {"stdio", "http_sse", "sse", "http"}
    ]
    if len(ok) < 2:
        missing.append("two real server rows with name/transport/health_checked/tool_called")
    return CompletionCriterion(
        id="real_mcp_compatibility_matrix",
        requirement="At least two real external MCP servers are started, health-checked, called, and recorded as compatibility evidence.",
        status="PASS" if not missing else "BLOCKED_EXTERNAL",
        evidence=[str(path_value)],
        missing=missing,
        notes=f"pass_servers={len(ok)}",
    )


def _check_remote_ci(path_value: str | Path | None) -> CompletionCriterion:
    payload = _read_json(path_value)
    if not payload:
        return CompletionCriterion(
            id="remote_ci_scheduled_eval",
            requirement="The scheduled eval GitHub Actions wrapper has a real remote run artifact proving CI execution.",
            status="BLOCKED_EXTERNAL",
            evidence=[".github/workflows/biobank-scheduled-eval.yml"],
            missing=["remote CI run artifact JSON"],
            notes="Provide --remote-ci-evidence after the workflow runs on GitHub.",
        )
    status = str(payload.get("status", "")).upper()
    url = str(payload.get("url") or payload.get("artifact_url") or "")
    missing = []
    if payload.get("artifact_type") != "remote_ci_scheduled_eval":
        missing.append("artifact_type=remote_ci_scheduled_eval")
    if status not in {"PASS", "SUCCESS"}:
        missing.append("status PASS/SUCCESS")
    if "github.com/" not in url or "/actions/runs/" not in url:
        missing.append("GitHub Actions run URL")
    if not (payload.get("run_id") or payload.get("sha") or payload.get("workflow")):
        missing.append("run_id/sha/workflow")
    return CompletionCriterion(
        id="remote_ci_scheduled_eval",
        requirement="The scheduled eval GitHub Actions wrapper has a real remote run artifact proving CI execution.",
        status="PASS" if not missing else "BLOCKED_EXTERNAL",
        evidence=[str(path_value)],
        missing=missing,
    )


def _check_high_pr(high_pr_evidence: str) -> CompletionCriterion:
    value = str(high_pr_evidence or "").strip()
    if not value:
        return CompletionCriterion(
            id="credentialed_high_risk_pr_path",
            requirement="HIGH self-evolution patches can create a credentialed remote PR instead of only a local branch fallback.",
            status="BLOCKED_EXTERNAL",
            missing=["credentialed PR URL or evidence id"],
            notes="Provide --high-pr-evidence with a PR URL after gh/GitHub credentials are configured.",
        )
    payload = _read_json(value) if value and Path(value).expanduser().exists() else {}
    if value and not payload:
        return CompletionCriterion(
            id="credentialed_high_risk_pr_path",
            requirement="HIGH self-evolution patches can create a credentialed remote PR instead of only a local branch fallback.",
            status="BLOCKED_EXTERNAL",
            missing=["high_risk_pr_evidence JSON artifact verified by gh"],
            notes="Run `biobank eval --suite external_evidence --collect high-pr ...` and pass the generated JSON path.",
        )
    if payload:
        pr_url = str(payload.get("pr_url") or payload.get("url") or "")
        status = str(payload.get("status") or "").upper()
        missing = []
        if payload.get("artifact_type") != "high_risk_pr_evidence":
            missing.append("artifact_type=high_risk_pr_evidence")
        if payload.get("generated_by") != "external_evidence.write_high_pr_evidence":
            missing.append("generated_by=external_evidence.write_high_pr_evidence")
        if status not in {"PASS", "SUCCESS", "PR_OPENED"}:
            missing.append("status PASS/SUCCESS/PR_OPENED")
        if "github.com/" not in pr_url or "/pull/" not in pr_url:
            missing.append("GitHub PR URL")
        if not payload.get("branch"):
            missing.append("branch")
        if payload.get("verified_by_gh") is not True:
            missing.append("verified_by_gh=true")
        gh = payload.get("gh") if isinstance(payload.get("gh"), dict) else {}
        if str(gh.get("url") or "") != pr_url:
            missing.append("gh.url matches pr_url")
        if str(gh.get("headRefName") or "") != str(payload.get("branch") or ""):
            missing.append("gh.headRefName matches branch")
        if str(gh.get("state") or "").upper() not in {"OPEN", "MERGED"}:
            missing.append("gh.state OPEN/MERGED")
        return CompletionCriterion(
            id="credentialed_high_risk_pr_path",
            requirement="HIGH self-evolution patches can create a credentialed remote PR instead of only a local branch fallback.",
            status="PASS" if not missing else "BLOCKED_EXTERNAL",
            evidence=[value],
            missing=missing,
        )
    return CompletionCriterion(
        id="credentialed_high_risk_pr_path",
        requirement="HIGH self-evolution patches can create a credentialed remote PR instead of only a local branch fallback.",
        status="BLOCKED_EXTERNAL",
        missing=["high_risk_pr_evidence JSON artifact verified by gh"],
    )


def _write_artifacts(audit: V3CompletionAudit, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_json = output_dir / f"v3_completion_{stamp}.json"
    latest_json = output_dir / "latest.json"
    run_md = output_dir / f"v3_completion_{stamp}.md"
    history_jsonl = output_dir / "v3_completion_history.jsonl"
    artifacts = {
        "run_json": str(run_json),
        "latest_json": str(latest_json),
        "run_md": str(run_md),
        "history_jsonl": str(history_jsonl),
    }
    audit.artifacts.update(artifacts)
    payload = asdict(audit)
    run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    run_md.write_text(_render_markdown(audit), encoding="utf-8")
    with history_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "generated_at": audit.generated_at,
            "status": audit.status,
            "passed": audit.n_passed,
            "blocked_external": audit.n_blocked_external,
            "failed": audit.n_failed,
            "run_json": str(run_json),
        }, sort_keys=True) + "\n")


def _render_markdown(audit: V3CompletionAudit) -> str:
    lines = [
        "# Biobank Agent v3 Completion Audit",
        "",
        f"- Status: **{audit.status}**",
        f"- Generated: {audit.generated_at}",
        f"- Objective: {audit.objective}",
        f"- Passed: {audit.n_passed}",
        f"- Blocked external: {audit.n_blocked_external}",
        f"- Failed: {audit.n_failed}",
        "",
        "| ID | Status | Requirement | Evidence | Missing |",
        "|---|---:|---|---|---|",
    ]
    for item in audit.criteria:
        evidence = "<br>".join(item.evidence)
        missing = "<br>".join(item.missing)
        lines.append(
            f"| {item.id} | {item.status} | {item.requirement} | {evidence} | {missing} |"
        )
    lines.extend(["", "## Notes", ""])
    for item in audit.criteria:
        if item.notes:
            lines.append(f"- **{item.id}**: {item.notes}")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Biobank Agent v3 completion against concrete evidence.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--live-run-dir", default="")
    parser.add_argument("--hpp-ckb-rap-readiness", default="")
    parser.add_argument("--mcp-compat-evidence", default="")
    parser.add_argument("--remote-ci-evidence", default="")
    parser.add_argument("--high-pr-evidence", default="")
    parser.add_argument("--external-evidence-dir", default="")
    parser.add_argument("--collect-external", action="store_true")
    parser.add_argument("--banks", default="ukb,hpp,ckb,ukb_rap")
    parser.add_argument("--icd10-code", default="E11")
    parser.add_argument("--probe-fields", default="hba1c,bmi,glucose")
    parser.add_argument("--mcp-config", default="")
    parser.add_argument("--mcp-call-args", default="")
    parser.add_argument("--mcp-min-servers", type=int, default=2)
    parser.add_argument("--workflow", default="biobank-scheduled-eval.yml")
    parser.add_argument("--pr-url", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--output-dir", default="reports/eval/v3_completion")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    external_evidence_dir = args.external_evidence_dir or "reports/eval/external_evidence"
    if args.collect_external:
        collected = collect_external_evidence_for_completion(
            output_dir=external_evidence_dir,
            banks=args.banks,
            icd10_code=args.icd10_code,
            probe_fields=args.probe_fields,
            mcp_config=args.mcp_config or None,
            mcp_call_args=args.mcp_call_args or None,
            mcp_min_servers=args.mcp_min_servers,
            workflow=args.workflow,
            repo_root=args.repo_root,
            pr_url=args.pr_url,
            branch=args.branch,
        )
        for name, path in collected.items():
            print(f"COLLECTED_{name.upper()}={path}")

    audit = run_v3_completion_audit(
        repo_root=args.repo_root,
        live_run_dir=args.live_run_dir or None,
        hpp_ckb_rap_readiness=args.hpp_ckb_rap_readiness or None,
        mcp_compat_evidence=args.mcp_compat_evidence or None,
        remote_ci_evidence=args.remote_ci_evidence or None,
        high_pr_evidence=args.high_pr_evidence,
        external_evidence_dir=external_evidence_dir,
        output_dir=args.output_dir,
        write_artifacts=not args.no_write,
    )
    print(f"V3_COMPLETION_STATUS={audit.status}")
    print(f"PASSED={audit.n_passed}")
    print(f"BLOCKED_EXTERNAL={audit.n_blocked_external}")
    print(f"FAILED={audit.n_failed}")
    for name, path in audit.artifacts.items():
        print(f"{name.upper()}={path}")
    return 0 if audit.status in {"COMPLETE", "BLOCKED_EXTERNAL", "PARTIAL"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))


__all__ = [
    "CompletionCriterion",
    "V3CompletionAudit",
    "collect_external_evidence_for_completion",
    "run_v3_completion_audit",
]
