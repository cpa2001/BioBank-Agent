"""Controlled runtime self-evolution proposals.

This module turns trajectory evidence into review-only improvement proposals.
It never edits code, prompts, routing, or memory unless a caller explicitly
routes a proposal through a separate approval and merge path.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from biobank_agent.core.events import AgentEventType

from .audit import RuntimeAuditReport, audit_session, redact_secrets
from .engine import AgentRuntime, AgentSession


@dataclass
class EvolutionProposal:
    """Review-only improvement proposal linked to trajectory evidence."""

    proposal_id: str
    category: str
    summary: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    trajectory_ids: list[str] = field(default_factory=list)
    risk: str = "review_required"
    approval_required: bool = True
    tests_required: list[str] = field(default_factory=list)
    apply_mode: str = "review_only"
    status: str = "proposed"
    # Concrete patch payload for the transactional self-evolution apply loop
    # (see runtime/self_evolve.py). A proposal is only APPLICABLE when it carries
    # a unified diff, a target path, and the test command(s) that must pass.
    target_path: str = ""
    diff: str = ""
    test_commands: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return redact_secrets(asdict(self))


@dataclass
class LearningReport:
    """Structured output for /learn and /evolve."""

    session_id: str
    status: str
    proposals: list[EvolutionProposal] = field(default_factory=list)
    feedback: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    policy: str = (
        "Review-only by default. Persistent changes require explicit approval, "
        "tests, and a separate recorded merge path."
    )

    def to_dict(self) -> dict[str, Any]:
        return redact_secrets(
            {
                "session_id": self.session_id,
                "status": self.status,
                "proposals": [proposal.to_dict() for proposal in self.proposals],
                "feedback": self.feedback,
                "artifacts": dict(self.artifacts),
                "policy": self.policy,
            }
        )


def learn_from_session(runtime: AgentRuntime, session: AgentSession) -> LearningReport:
    """Collect trajectory feedback and propose review-only improvements."""

    audit = audit_session(runtime, session)
    feedback = collect_feedback(audit)
    proposals = propose_improvements(session, feedback)
    return LearningReport(
        session_id=session.session_id,
        status="no_feedback" if not feedback else "proposed",
        proposals=proposals,
        feedback=feedback,
    )


def collect_feedback(audit: RuntimeAuditReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, warning in enumerate(audit.safety_warnings):
        rows.append(
            {
                "kind": "safety_warning",
                "summary": warning,
                "trajectory_id": str(idx),
                "severity": "warning",
            }
        )
    for item in audit.tool_calls:
        state = str(item.get("state") or "").lower()
        result = item.get("result") or {}
        error = str(result.get("error") or item.get("error") or "")
        if state in {"failed", "error"} or error:
            rows.append(
                {
                    "kind": "tool_failure",
                    "tool": item.get("tool", ""),
                    "summary": error or f"{item.get('tool', 'tool')} returned {state}",
                    "trajectory_id": str(item.get("seq") or item.get("call_id") or ""),
                    "severity": "error",
                }
            )
    for item in audit.verification:
        if str(item.get("status") or "").lower() not in {"passed", "pass", "ok", "success"}:
            rows.append(
                {
                    "kind": "verification_failure",
                    "command": item.get("command", ""),
                    "summary": item.get("summary", ""),
                    "trajectory_id": str(item.get("id") or ""),
                    "severity": "error",
                }
            )
    for item in audit.approvals:
        decision = str(item.get("decision") or "").lower()
        if decision in {"deny", "denied", "blocked"}:
            rows.append(
                {
                    "kind": "approval_denied",
                    "tool": item.get("tool_name", ""),
                    "summary": item.get("rationale", ""),
                    "trajectory_id": str(item.get("call_id") or ""),
                    "severity": "warning",
                }
            )
    return rows


def propose_improvements(session: AgentSession, feedback: list[dict[str, Any]]) -> list[EvolutionProposal]:
    proposals: list[EvolutionProposal] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in feedback:
        key = str(row.get("kind") or "feedback")
        grouped.setdefault(key, []).append(row)

    for kind, rows in sorted(grouped.items()):
        trajectory_ids = [str(row.get("trajectory_id") or "") for row in rows if row.get("trajectory_id")]
        category, summary, tests = _proposal_shape(kind, rows)
        proposals.append(
            EvolutionProposal(
                proposal_id=_proposal_id(session.session_id, kind, len(proposals)),
                category=category,
                summary=summary,
                evidence=rows,
                trajectory_ids=trajectory_ids,
                tests_required=tests,
            )
        )
    if not proposals and session.turns:
        proposals.append(
            EvolutionProposal(
                proposal_id=_proposal_id(session.session_id, "harness_case", 0),
                category="harness",
                summary="Convert this successful trajectory into a reusable harness case after review.",
                evidence=[{"kind": "successful_session", "turns": len(session.turns), "events": len(session.events)}],
                trajectory_ids=[str(i) for i in range(min(len(session.events), 20))],
                tests_required=["harness replay must pass before activation"],
            )
        )
    return proposals


def write_learning_report(report: LearningReport, output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    base = _safe_filename(report.session_id)
    json_path = root / f"{base}.learning.json"
    md_path = root / f"{base}.learning.md"
    payload = report.to_dict()
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        f"# Runtime Learning: {report.session_id}",
        "",
        f"- Status: {report.status}",
        f"- Proposals: {len(report.proposals)}",
        "",
        report.policy,
        "",
    ]
    for idx, proposal in enumerate(report.proposals, start=1):
        lines.extend(
            [
                f"## {idx}. {proposal.category}",
                "",
                proposal.summary,
                "",
                f"- Risk: {proposal.risk}",
                f"- Approval required: {proposal.approval_required}",
                f"- Trajectory ids: {', '.join(proposal.trajectory_ids) or '(none)'}",
                f"- Tests required: {', '.join(proposal.tests_required) or '(to be defined before apply)'}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def reject_persistent_apply(reason: str = "") -> dict[str, Any]:
    """Return a structured refusal for unsafe or unapproved evolution apply."""

    return {
        "status": "rejected",
        "reason": reason
        or "Persistent self-evolution requires explicit approval, tests, and a recorded merge path.",
        "approval_required": True,
    }


def _proposal_shape(kind: str, rows: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    if kind == "tool_failure":
        tools = sorted({str(row.get("tool") or "tool") for row in rows})
        return (
            "skill",
            f"Review repeated tool failure pattern for: {', '.join(tools)}.",
            ["add or update a harness case that reproduces the tool failure"],
        )
    if kind == "verification_failure":
        return (
            "harness",
            "Add a regression harness for the failed verification path before attempting repair.",
            ["failed verification must fail before a proposed fix and pass after review"],
        )
    if kind == "approval_denied":
        return (
            "permissions",
            "Review permission policy or planner tool scope for denied actions.",
            ["approval matrix must prove the action remains gated"],
        )
    return (
        "audit",
        "Review safety warning and decide whether it should become a harness invariant.",
        ["audit/replay test must preserve the warning until fixed"],
    )


def _proposal_id(session_id: str, kind: str, idx: int) -> str:
    return f"{_safe_filename(session_id)[:12]}-{_safe_filename(kind)}-{idx}-{int(time.time())}"


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "runtime")).strip("_") or "runtime"


__all__ = [
    "EvolutionProposal",
    "LearningReport",
    "learn_from_session",
    "collect_feedback",
    "propose_improvements",
    "write_learning_report",
    "reject_persistent_apply",
]
