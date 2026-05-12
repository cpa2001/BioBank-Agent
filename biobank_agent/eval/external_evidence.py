"""Collect credentialed external evidence for the v3 completion audit.

The v3 completion gate intentionally refuses to treat local scaffolds as proof
of production readiness. This module writes the four external artifact shapes
that ``v3_completion`` accepts when real credentials and services are present.
It also writes blocked artifacts when credentials are absent so operators get a
concrete next step instead of a silent no-op.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from biobank_agent.core.tools.protocol import ToolContext
from biobank_agent.core.tools.registry import ToolRegistry
from biobank_agent.extensions.mcp_manager import McpManager


@dataclass
class EvidenceArtifact:
    """One written external-evidence artifact."""

    kind: str
    status: str
    path: str
    missing: list[str] = field(default_factory=list)
    notes: str = ""


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(output_dir: str | Path, prefix: str, payload: dict[str, Any]) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{prefix}_{_stamp()}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    latest = out / f"{prefix}_latest.json"
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(path)


def _load_call_args(path_value: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path_value:
        return {}
    try:
        data = json.loads(Path(path_value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        out[str(key)] = dict(value) if isinstance(value, dict) else {}
    return out


def collect_bank_readiness_evidence(
    *,
    output_dir: str | Path = "reports/eval/external_evidence",
    banks: str = "ukb,hpp,ckb,ukb_rap",
    icd10_code: str = "E11",
    probe_fields: str = "hba1c,bmi,glucose",
) -> EvidenceArtifact:
    """Run the credential-gated bank readiness skill and return its JSON artifact."""
    from biobank_agent.config import get_settings
    from biobank_agent.domain.banks import canonical_bank_id
    from biobank_agent.skills.bank_data_readiness import bank_data_readiness

    settings = get_settings()
    report_dir = Path(output_dir)
    ctx = SimpleNamespace(
        settings=settings,
        report_dir=report_dir,
        dm=None,
        emit_progress=lambda *args, **kwargs: None,
    )
    result = bank_data_readiness(
        banks=banks,
        icd10_code=icd10_code,
        probe_fields=probe_fields,
        output_dir=str(report_dir),
        use_current_context=False,
        ctx=ctx,
    )
    artifact_path = str(result.get("artifact_json") or "")
    statuses = {
        str(item.get("bank_id")): str(item.get("status", "")).upper()
        for item in result.get("banks", [])
        if isinstance(item, dict)
    }
    required = {canonical_bank_id(item.strip()) for item in str(banks or "").split(",") if item.strip()}
    missing = [
        bank for bank in sorted(required)
        if statuses.get(bank) not in {"READY", "REMOTE_READY"}
    ]
    status = "PASS" if not missing and result.get("status") == "READY" else "BLOCKED_EXTERNAL"
    return EvidenceArtifact(
        "bank_data_readiness",
        status,
        artifact_path,
        missing,
        notes=f"readiness_status={result.get('status', '')}",
    )


def collect_mcp_compatibility_evidence(
    *,
    output_dir: str | Path = "reports/eval/external_evidence",
    config_path: str | Path | None = None,
    call_args_path: str | Path | None = None,
    min_servers: int = 2,
) -> EvidenceArtifact:
    """Start configured MCP servers, health-check them, call one tool each, and write evidence."""
    return asyncio.run(_collect_mcp_compatibility_evidence_async(
        output_dir=output_dir,
        config_path=config_path,
        call_args_path=call_args_path,
        min_servers=min_servers,
    ))


async def _collect_mcp_compatibility_evidence_async(
    *,
    output_dir: str | Path,
    config_path: str | Path | None,
    call_args_path: str | Path | None,
    min_servers: int,
) -> EvidenceArtifact:
    registry = ToolRegistry()
    manager = McpManager(registry, config_path=Path(config_path).expanduser() if config_path else None)
    call_args = _load_call_args(call_args_path)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    try:
        await manager.start()
        health_rows = {row.get("name"): row for row in await manager.health_check(repair=True)}
        for name, cfg in manager.configs.items():
            handlers = list(manager.handlers_by_server.get(name, []))
            health = health_rows.get(name, {})
            row: dict[str, Any] = {
                "name": name,
                "transport": cfg.transport,
                "status": "BLOCKED_EXTERNAL",
                "health_checked": bool(health),
                "health_ok": bool(health.get("ok")),
                "tool_called": False,
                "loaded_tools": len(handlers),
                "error": str(health.get("error") or manager.errors.get(name, "")),
            }
            if not handlers:
                row["error"] = row["error"] or "no tools loaded"
                rows.append(row)
                continue
            handler = handlers[0]
            args = (
                call_args.get(handler.name)
                or call_args.get(handler.remote_name)
                or call_args.get(name)
                or {}
            )
            row["tool_name"] = handler.name
            row["remote_tool_name"] = handler.remote_name
            try:
                result = await handler.handle(ToolContext(
                    name=handler.name,
                    args=args,
                    capabilities=handler.required_capabilities(),
                ))
                row["tool_called"] = True
                row["call_result_keys"] = sorted(result.keys())[:20] if isinstance(result, dict) else []
                row["status"] = "PASS" if row["health_ok"] else "READY"
                row["error"] = ""
            except Exception as exc:  # pragma: no cover - depends on external server
                row["error"] = str(exc)
            rows.append(row)
    finally:
        await manager.stop()

    passing = [
        row for row in rows
        if row.get("status") in {"PASS", "READY"}
        and row.get("health_checked")
        and row.get("tool_called")
    ]
    if len(passing) < min_servers:
        missing.append(f"{min_servers} MCP servers with health_checked=true and tool_called=true")
    payload = {
        "artifact_type": "mcp_compatibility_matrix",
        "schema_version": 1,
        "generated_by": "external_evidence.collect_mcp_compatibility_evidence",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASS" if not missing else "BLOCKED_EXTERNAL",
        "config_path": str(manager.config_path),
        "servers": rows,
        "missing": missing,
    }
    path = _write_json(output_dir, "mcp_compat", payload)
    return EvidenceArtifact("mcp_compatibility_matrix", payload["status"], path, missing)


def collect_remote_ci_evidence(
    *,
    output_dir: str | Path = "reports/eval/external_evidence",
    workflow: str = "biobank-scheduled-eval.yml",
    repo_root: str | Path = ".",
) -> EvidenceArtifact:
    """Read the latest GitHub Actions workflow run via ``gh`` and write evidence."""
    missing: list[str] = []
    row: dict[str, Any] = {}
    try:
        proc = subprocess.run(
            [
                "gh",
                "run",
                "list",
                "--workflow",
                workflow,
                "--limit",
                "1",
                "--json",
                "databaseId,status,conclusion,url,headSha,workflowName,createdAt",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        proc = None
        missing.append(f"gh unavailable: {exc}")
    if proc is not None and proc.returncode != 0:
        missing.append(proc.stderr.strip() or proc.stdout.strip() or "gh run list failed")
    if proc is not None and proc.returncode == 0:
        try:
            runs = json.loads(proc.stdout or "[]")
        except json.JSONDecodeError:
            runs = []
        if not runs:
            missing.append(f"no GitHub Actions runs found for {workflow}")
        else:
            row = dict(runs[0])
            if str(row.get("conclusion", "")).lower() != "success":
                missing.append("latest workflow conclusion is not success")
            if "github.com/" not in str(row.get("url", "")) or "/actions/runs/" not in str(row.get("url", "")):
                missing.append("latest workflow run has no GitHub Actions URL")

    payload = {
        "artifact_type": "remote_ci_scheduled_eval",
        "schema_version": 1,
        "generated_by": "external_evidence.collect_remote_ci_evidence",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "SUCCESS" if not missing else "BLOCKED_EXTERNAL",
        "workflow": workflow,
        "url": str(row.get("url", "")),
        "run_id": str(row.get("databaseId", "")),
        "sha": str(row.get("headSha", "")),
        "raw": row,
        "missing": missing,
    }
    path = _write_json(output_dir, "remote_ci", payload)
    return EvidenceArtifact("remote_ci_scheduled_eval", payload["status"], path, missing)


def _verify_pr_with_gh(pr_url: str, *, repo_root: str | Path = ".") -> tuple[dict[str, Any], str]:
    try:
        proc = subprocess.run(
            ["gh", "pr", "view", pr_url, "--json", "url,headRefName,state,number"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {}, f"gh unavailable: {exc}"
    if proc.returncode != 0:
        return {}, proc.stderr.strip() or proc.stdout.strip() or "gh pr view failed"
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {}, "gh pr view returned non-JSON output"
    return dict(payload) if isinstance(payload, dict) else {}, ""


def write_high_pr_evidence(
    *,
    pr_url: str,
    branch: str,
    output_dir: str | Path = "reports/eval/external_evidence",
    status: str = "PR_OPENED",
    repo_root: str | Path = ".",
    verify: bool = True,
) -> EvidenceArtifact:
    """Write a credentialed high-risk PR evidence artifact."""
    missing: list[str] = []
    if "github.com/" not in str(pr_url) or "/pull/" not in str(pr_url):
        missing.append("GitHub PR URL")
    if not str(branch).strip():
        missing.append("branch")
    gh_payload: dict[str, Any] = {}
    gh_error = ""
    if verify and not missing:
        gh_payload, gh_error = _verify_pr_with_gh(pr_url, repo_root=repo_root)
        if gh_error:
            missing.append(f"gh pr view verification: {gh_error}")
        elif str(gh_payload.get("url") or "") != str(pr_url):
            missing.append("gh pr view URL mismatch")
        elif str(gh_payload.get("headRefName") or "") != str(branch):
            missing.append("gh pr view branch mismatch")
        elif str(gh_payload.get("state") or "").upper() not in {"OPEN", "MERGED"}:
            missing.append("PR state OPEN/MERGED")
    normalized_status = status if not missing else "BLOCKED_EXTERNAL"
    payload = {
        "artifact_type": "high_risk_pr_evidence",
        "schema_version": 1,
        "generated_by": "external_evidence.write_high_pr_evidence",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": normalized_status,
        "pr_url": str(pr_url),
        "branch": str(branch),
        "verified_by_gh": bool(gh_payload) and not gh_error,
        "gh": gh_payload,
        "missing": missing,
    }
    path = _write_json(output_dir, "high_pr", payload)
    return EvidenceArtifact("high_risk_pr_evidence", normalized_status, path, missing)


def collect_high_pr_evidence(
    *,
    pr_url: str,
    branch: str,
    output_dir: str | Path = "reports/eval/external_evidence",
    repo_root: str | Path = ".",
    preserve_existing_without_args: bool = True,
) -> EvidenceArtifact:
    """Collect HIGH-risk PR evidence without clobbering an existing verified artifact.

    Operators often run ``--collect all`` repeatedly while GitHub credentials or
    PR URLs are not available yet. If a verified ``high_pr_latest.json`` already
    exists, missing ``pr_url``/``branch`` should not overwrite it with a blocked
    placeholder. Passing both values always refreshes the artifact.
    """
    latest = Path(output_dir) / "high_pr_latest.json"
    if preserve_existing_without_args and (not pr_url or not branch) and latest.exists():
        payload = _read_json_file(latest)
        status = str(payload.get("status") or "BLOCKED_EXTERNAL")
        missing = list(payload.get("missing") or [])
        return EvidenceArtifact("high_risk_pr_evidence", status, str(latest), missing, notes="preserved_existing_latest")
    return write_high_pr_evidence(
        output_dir=output_dir,
        pr_url=pr_url,
        branch=branch,
        repo_root=repo_root,
    )


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect external evidence for v3_completion.")
    parser.add_argument("--collect", choices=["bank-readiness", "mcp", "remote-ci", "high-pr", "all"], default="all")
    parser.add_argument("--output-dir", default="reports/eval/external_evidence")
    parser.add_argument("--banks", default="ukb,hpp,ckb,ukb_rap")
    parser.add_argument("--icd10-code", default="E11")
    parser.add_argument("--probe-fields", default="hba1c,bmi,glucose")
    parser.add_argument("--mcp-config", default="")
    parser.add_argument("--mcp-call-args", default="")
    parser.add_argument("--mcp-min-servers", type=int, default=2)
    parser.add_argument("--workflow", default="biobank-scheduled-eval.yml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--pr-url", default="")
    parser.add_argument("--branch", default="")
    args = parser.parse_args(argv)

    artifacts: list[EvidenceArtifact] = []
    if args.collect in {"bank-readiness", "all"}:
        artifacts.append(collect_bank_readiness_evidence(
            output_dir=args.output_dir,
            banks=args.banks,
            icd10_code=args.icd10_code,
            probe_fields=args.probe_fields,
        ))
    if args.collect in {"mcp", "all"}:
        artifacts.append(collect_mcp_compatibility_evidence(
            output_dir=args.output_dir,
            config_path=args.mcp_config or None,
            call_args_path=args.mcp_call_args or None,
            min_servers=args.mcp_min_servers,
        ))
    if args.collect in {"remote-ci", "all"}:
        artifacts.append(collect_remote_ci_evidence(
            output_dir=args.output_dir,
            workflow=args.workflow,
            repo_root=args.repo_root,
        ))
    if args.collect in {"high-pr", "all"}:
        artifacts.append(collect_high_pr_evidence(
            output_dir=args.output_dir,
            pr_url=args.pr_url,
            branch=args.branch,
            repo_root=args.repo_root,
        ))
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": [asdict(item) for item in artifacts],
    }
    manifest_path = _write_json(args.output_dir, "external_evidence_manifest", manifest)
    print(f"MANIFEST={manifest_path}")
    for item in artifacts:
        print(f"{item.kind}={item.status} {item.path}")
    return 0 if all(item.status in {"PASS", "SUCCESS", "PR_OPENED"} for item in artifacts) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EvidenceArtifact",
    "collect_bank_readiness_evidence",
    "collect_high_pr_evidence",
    "collect_mcp_compatibility_evidence",
    "collect_remote_ci_evidence",
    "write_high_pr_evidence",
]
