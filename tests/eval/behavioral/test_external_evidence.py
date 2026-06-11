"""Tests for credentialed external-evidence collection helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from biobank_agent.eval.external_evidence import (
    collect_bank_readiness_evidence,
    collect_high_pr_evidence,
    collect_remote_ci_evidence,
    write_high_pr_evidence,
)


def test_bank_readiness_evidence_invokes_readiness_skill(tmp_path: Path, monkeypatch):
    def fake_readiness(**kwargs):
        out = Path(kwargs["output_dir"]) / "bank_readiness_fake.json"
        payload = {
            "artifact_type": "bank_data_readiness",
            "generated_by": "bank_data_readiness",
            "status": "READY",
            "banks": [
                {
                    "bank_id": "ukb",
                    "status": "READY",
                    "n_subjects": 100,
                    "source": "env:settings.data_dir",
                    "diagnosis_probe": {"n_case_subjects": 10},
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
            ],
            "artifact_json": str(out),
        }
        out.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("biobank_agent.skills.bank_data_readiness.bank_data_readiness", fake_readiness)

    artifact = collect_bank_readiness_evidence(output_dir=tmp_path)

    assert artifact.status == "PASS"
    assert artifact.kind == "bank_data_readiness"
    assert Path(artifact.path).exists()


def test_remote_ci_evidence_parses_successful_gh_run(tmp_path: Path, monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{
                "databaseId": 123,
                "status": "completed",
                "conclusion": "success",
                "url": "https://github.com/example/repo/actions/runs/123",
                "headSha": "abc123",
                "workflowName": "Biobank Scheduled Eval",
                "createdAt": "2026-05-11T00:00:00Z",
            }]),
            stderr="",
        )

    monkeypatch.setattr("biobank_agent.eval.external_evidence.subprocess.run", fake_run)

    artifact = collect_remote_ci_evidence(output_dir=tmp_path, repo_root=tmp_path)
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))

    assert artifact.status == "SUCCESS"
    assert payload["artifact_type"] == "remote_ci_scheduled_eval"
    assert payload["run_id"] == "123"
    assert "/actions/runs/123" in payload["url"]


def test_remote_ci_evidence_blocks_without_gh_success(tmp_path: Path, monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="not authenticated")

    monkeypatch.setattr("biobank_agent.eval.external_evidence.subprocess.run", fake_run)

    artifact = collect_remote_ci_evidence(output_dir=tmp_path, repo_root=tmp_path)
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))

    assert artifact.status == "BLOCKED_EXTERNAL"
    assert payload["artifact_type"] == "remote_ci_scheduled_eval"
    assert "not authenticated" in payload["missing"][0]


def test_high_pr_evidence_requires_github_pull_url_and_branch(tmp_path: Path):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "url": "https://github.com/example/repo/pull/7",
                "headRefName": "auto-improve/demo-00007",
                "state": "OPEN",
                "number": 7,
            }),
            stderr="",
        )

    from pytest import MonkeyPatch
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("biobank_agent.eval.external_evidence.subprocess.run", fake_run)
    ok = write_high_pr_evidence(
        output_dir=tmp_path,
        pr_url="https://github.com/example/repo/pull/7",
        branch="auto-improve/demo-00007",
    )
    monkeypatch.undo()
    ok_payload = json.loads(Path(ok.path).read_text(encoding="utf-8"))
    assert ok.status == "PR_OPENED"
    assert ok_payload["artifact_type"] == "high_risk_pr_evidence"
    assert ok_payload["verified_by_gh"] is True

    blocked = write_high_pr_evidence(output_dir=tmp_path, pr_url="https://example.test/pull/7", branch="")
    blocked_payload = json.loads(Path(blocked.path).read_text(encoding="utf-8"))
    assert blocked.status == "BLOCKED_EXTERNAL"
    assert "GitHub PR URL" in blocked_payload["missing"]
    assert "branch" in blocked_payload["missing"]


def test_high_pr_collector_preserves_existing_latest_when_args_missing(tmp_path: Path, monkeypatch):
    latest = tmp_path / "high_pr_latest.json"
    latest.write_text(json.dumps({
        "artifact_type": "high_risk_pr_evidence",
        "status": "PR_OPENED",
        "pr_url": "https://github.com/example/repo/pull/7",
        "branch": "auto-improve/demo-00007",
        "verified_by_gh": True,
        "missing": [],
    }), encoding="utf-8")

    def fail_run(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("gh should not be called when preserving existing high_pr_latest.json")

    monkeypatch.setattr("biobank_agent.eval.external_evidence.subprocess.run", fail_run)

    artifact = collect_high_pr_evidence(output_dir=tmp_path, pr_url="", branch="")

    assert artifact.status == "PR_OPENED"
    assert artifact.path == str(latest)
    assert artifact.notes == "preserved_existing_latest"
    assert json.loads(latest.read_text(encoding="utf-8"))["verified_by_gh"] is True
