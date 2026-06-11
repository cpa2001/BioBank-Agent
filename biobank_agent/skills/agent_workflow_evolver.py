"""Agent workflow evolution audit skill.

This skill keeps brittle workflow-routing knowledge visible and reviewable.
It does not mutate code by default; trusted harnesses can consume the emitted
artifact to decide which skills, MCP resources or tests should be generated.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from biobank_agent.registry import skill


DEFAULT_SCAN_ROOTS = ("biobank_agent", "scripts", "tests")
RULE_SURFACE_HINTS = (
    "goal",
    "intent",
    "keyword",
    "token",
    "startswith",
    "endswith",
    " in lower",
    " in text",
    "required_skills",
    "required_artifact",
    "missing_report_terms",
    "clarification",
)
HARNESS_HINTS = ("required_skills", "required_artifact", "missing_report_terms", "short_tasks")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _out_dir(ctx: Any) -> Path:
    report_dir = getattr(ctx, "report_dir", None)
    if report_dir is None:
        settings = getattr(ctx, "settings", None)
        report_dir = getattr(settings, "reports_dir", Path("./reports")) if settings else Path("./reports")
    out = Path(report_dir) / "results" / "07_WorkflowEvolution"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _scan_python_file(path: Path, root: Path, *, max_lines: int = 5000, max_findings: int = 40) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return [{
            "file": str(path.relative_to(root)),
            "line": 0,
            "kind": "read_error",
            "snippet": str(exc)[:300],
            "migration_target": "manual_review",
        }]

    findings: list[dict[str, Any]] = []
    rel = str(path.relative_to(root))
    for idx, line in enumerate(lines[:max_lines], 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        folded = stripped.lower()
        if not any(hint in folded for hint in RULE_SURFACE_HINTS):
            continue
        if not any(marker in folded for marker in ("if ", "elif ", "for ", "required", "missing", "clarification", "task", "goal", "intent")):
            continue
        snippet = stripped[:700]
        migration_target = "goal_intent_classifier"
        if any(hint in folded for hint in HARNESS_HINTS) or "scripts/" in rel:
            migration_target = "harness_profile"
        if "clarification" in folded:
            migration_target = "clarification_policy_skill"
        findings.append({
            "file": rel,
            "line": idx,
            "kind": "line_rule_surface",
            "snippet": snippet,
            "migration_target": migration_target,
        })
        if len(findings) >= max_findings:
            break
    return findings


def _summarize_findings(findings: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, int] = {}
    by_file: dict[str, int] = {}
    for item in findings:
        by_target[str(item.get("migration_target", "unknown"))] = by_target.get(str(item.get("migration_target", "unknown")), 0) + 1
        by_file[str(item.get("file", ""))] = by_file.get(str(item.get("file", "")), 0) + 1
    return {
        "n_rule_surfaces": len(findings),
        "by_migration_target": dict(sorted(by_target.items())),
        "top_files": [
            {"file": file, "n": n}
            for file, n in sorted(by_file.items(), key=lambda item: item[1], reverse=True)[:20]
        ],
    }


def _proposal_payload(goal_profile: str, findings: list[dict[str, Any]], trusted_harness_mode: bool) -> list[dict[str, Any]]:
    top_targets = _summarize_findings(findings)["by_migration_target"]
    return [
        {
            "name": "clarification_policy_skill",
            "purpose": "Generate interactive plan questions from an intent profile, available skills and data readiness instead of planner-local prompt rules.",
            "priority": "high" if top_targets.get("clarification_policy_skill", 0) else "medium",
            "activation_policy": "auto_allowed" if trusted_harness_mode else "review_required",
        },
        {
            "name": "harness_profile_loader",
            "purpose": "Load expected skills, artifacts, report terms and interaction keys from trajectory profiles rather than Python constants.",
            "priority": "high" if top_targets.get("harness_profile", 0) else "medium",
            "activation_policy": "auto_allowed" if trusted_harness_mode else "review_required",
        },
        {
            "name": "agentic_mcp_resource_broker",
            "purpose": "Choose configured MCP/web/local resources for motif databases, regulatory annotations, enrichment and literature based on the current workflow gaps.",
            "priority": "medium",
            "activation_policy": "auto_allowed" if trusted_harness_mode else "review_required",
        },
        {
            "name": "trajectory_regression_generator",
            "purpose": f"Generate compact CLI/harness cases for the active goal profile: {goal_profile or 'unknown'}.",
            "priority": "medium",
            "activation_policy": "auto_allowed" if trusted_harness_mode else "review_required",
        },
    ]


@skill(
    name="agent_workflow_evolver",
    description=(
        "Audit BioBank Agent routing rules, harness gates and workflow gaps, then emit structured "
        "skill/harness/MCP evolution proposals without mutating code unless a trusted harness approves."
    ),
    parameters={
        "goal_profile": {"type": "string", "description": "Current workflow or intent profile name", "default": ""},
        "scan_roots": {"type": "string", "description": "Comma-separated repo-relative roots to scan", "default": "biobank_agent,scripts,tests"},
        "trusted_harness_mode": {"type": "boolean", "description": "Whether automatic activation is allowed by the harness", "default": False},
        "max_findings": {"type": "integer", "description": "Maximum detailed findings to include", "default": 120},
    },
    required=[],
)
def agent_workflow_evolver(
    goal_profile: str = "",
    scan_roots: str = "biobank_agent,scripts,tests",
    trusted_harness_mode: bool = False,
    max_findings: int = 120,
    *,
    ctx=None,
) -> dict:
    root = _repo_root()
    roots = [item.strip() for item in str(scan_roots or "").split(",") if item.strip()] or list(DEFAULT_SCAN_ROOTS)
    findings: list[dict[str, Any]] = []
    scanned_files = 0
    skipped_files = 0
    max_files = 120
    max_file_bytes = 180_000
    max_seconds = 4.0
    deadline = time.monotonic() + max_seconds
    for rel_root in roots:
        base = (root / rel_root).resolve()
        if not base.exists() or root not in base.parents and base != root:
            continue
        for path in sorted(base.rglob("*.py")):
            if scanned_files >= max_files or time.monotonic() >= deadline:
                skipped_files += 1
                continue
            if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            try:
                if path.stat().st_size > max_file_bytes:
                    skipped_files += 1
                    continue
            except OSError:
                skipped_files += 1
                continue
            scanned_files += 1
            remaining = max(1, int(max_findings or 120) * 3 - len(findings))
            findings.extend(_scan_python_file(path, root, max_findings=min(40, remaining)))
            if len(findings) >= int(max_findings or 120) * 3:
                break
        if scanned_files >= max_files or time.monotonic() >= deadline or len(findings) >= int(max_findings or 120) * 3:
            break

    findings = sorted(findings, key=lambda item: (item.get("migration_target", ""), item.get("file", ""), item.get("line", 0)))
    limit = max(1, int(max_findings or 120))
    summary = _summarize_findings(findings)
    proposals = _proposal_payload(goal_profile, findings, bool(trusted_harness_mode))
    payload = {
        "goal_profile": goal_profile,
        "scanned_files": scanned_files,
        "skipped_files": skipped_files,
        "scan_budget": {
            "max_files": max_files,
            "max_file_bytes": max_file_bytes,
            "max_seconds": max_seconds,
            "mode": "bounded_line_audit",
        },
        "summary": summary,
        "findings": findings[:limit],
        "truncated_findings": max(0, len(findings) - limit),
        "evolution_proposals": proposals,
        "policy": "This audit proposes skills/harness/MCP evolution. Code mutation remains review-gated unless trusted_harness_mode is enabled by the harness.",
    }

    out = _out_dir(ctx)
    json_path = out / "agent_workflow_evolution_audit.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = out / "agent_workflow_evolution_audit.md"
    lines = [
        "# Agent Workflow Evolution Audit",
        "",
        f"- Goal profile: {goal_profile or 'unspecified'}",
        f"- Scanned Python files: {scanned_files}",
        f"- Rule surfaces found: {summary['n_rule_surfaces']}",
        "",
        "## Migration Targets",
    ]
    for target, count in summary["by_migration_target"].items():
        lines.append(f"- {target}: {count}")
    lines.extend(["", "## Proposed Skills/Harness Changes"])
    for item in proposals:
        lines.append(f"- {item['name']}: {item['purpose']} ({item['activation_policy']})")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        custom = getattr(getattr(ctx, "state", None), "custom_data", None)
        if isinstance(custom, dict):
            custom["agent_workflow_evolver"] = payload
    except Exception:
        pass
    payload["audit_json"] = str(json_path)
    payload["audit_markdown"] = str(md_path)
    return payload
