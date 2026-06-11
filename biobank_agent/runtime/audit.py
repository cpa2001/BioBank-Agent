"""Runtime session audit summaries.

The audit layer is intentionally derived from persisted runtime state:
``AgentSession.events``, ``trajectory.jsonl``, checkpoints, and action-graph
references. It does not call providers or rerun tools.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from biobank_agent.core.events import AgentEventType

from .engine import AgentRuntime, AgentSession
from .replay import replay_trajectory


_SECRET_KEY_RE = re.compile(r"(api[_-]?key|token|password|passwd|secret|credential)", re.I)
_SECRET_VALUE_RE = re.compile(r"\b(?:sk|rk|pk|ak)-[A-Za-z0-9_\-]{12,}\b")


@dataclass
class RuntimeAuditReport:
    """Serializable audit report for one runtime session."""

    session_id: str
    title: str
    cwd: str
    task_summary: dict[str, Any] = field(default_factory=dict)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] | None = None
    goal: dict[str, Any] | None = None
    provider_roles: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    shell_commands: list[dict[str, Any]] = field(default_factory=list)
    file_changes: list[dict[str, Any]] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    verification: list[dict[str, Any]] = field(default_factory=list)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    compactions: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    action_graph: dict[str, Any] = field(default_factory=dict)
    replay_status: dict[str, Any] = field(default_factory=dict)
    safety_warnings: list[str] = field(default_factory=list)
    event_type_counts: dict[str, int] = field(default_factory=dict)
    run_tree_summary: dict[str, Any] = field(default_factory=dict)
    online_eval: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "warning" if self.safety_warnings else "ok"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status
        return redact_secrets(payload)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    def to_markdown(self) -> str:
        body = self.to_dict()
        lines = [
            f"# Runtime Audit: {body['title'] or body['session_id']}",
            "",
            f"- Session: `{body['session_id']}`",
            f"- Workspace: `{body['cwd']}`",
            f"- Status: **{body['status']}**",
            f"- Turns: {body['task_summary'].get('turns', 0)}",
            f"- Events: {body['task_summary'].get('events', 0)}",
            f"- Trajectory: `{body['task_summary'].get('trajectory_path', '')}`",
            "",
            "## Task Summary",
            "",
            f"- Last user input: {body['task_summary'].get('last_user_input', '') or '(none)'}",
            f"- Plan status: {(body.get('plan') or {}).get('status', 'none')}",
            f"- Goal status: {(body.get('goal') or {}).get('status', 'none')}",
            "",
            "## Evidence Counts",
            "",
            f"- Tool calls: {len(body.get('tool_calls') or [])}",
            f"- Approvals: {len(body.get('approvals') or [])}",
            f"- Verifications: {len(body.get('verification') or [])}",
            f"- Repairs: {len(body.get('repairs') or [])}",
            f"- Checkpoints: {len(body.get('checkpoints') or [])}",
            f"- Action graph refs: {body.get('action_graph', {}).get('ref_count', 0)}",
            "",
            "## Replay",
            "",
            f"- Status: {body.get('replay_status', {}).get('status', 'unknown')}",
            f"- Violations: {len(body.get('replay_status', {}).get('violations') or [])}",
            "",
        ]
        warnings = body.get("safety_warnings") or []
        if warnings:
            lines.extend(["## Safety Warnings", ""])
            lines.extend(f"- {warning}" for warning in warnings)
            lines.append("")
        lines.extend(["## Timeline", ""])
        for item in (body.get("timeline") or [])[:40]:
            lines.append(f"- `{item.get('type')}` {item.get('summary', '')}")
        return "\n".join(lines).rstrip() + "\n"


def _run_tree_eval(session: AgentSession) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fold ``session.events`` into a run tree (M15) and grade it with the default online
    evaluators. Read-side and defensive: a malformed event stream degrades to empty dicts
    rather than breaking the audit."""
    try:
        from biobank_agent.core.events import AgentEvent
        from biobank_agent.runtime.run_eval import default_evaluators, evaluate_run_tree
        from biobank_agent.runtime.run_tree import build_run_tree, summarize_run_tree

        events = [AgentEvent.from_dict(ev) for ev in session.events if isinstance(ev, dict)]
        tree = build_run_tree(events, session_id=session.session_id)
        return summarize_run_tree(tree), evaluate_run_tree(tree, default_evaluators()).to_dict()
    except Exception as exc:  # pragma: no cover - audit must never break on bad data
        return {}, {"error": str(exc)[:200]}


def audit_session(
    runtime: AgentRuntime,
    session: AgentSession,
    *,
    include_timeline_limit: int = 120,
) -> RuntimeAuditReport:
    """Build an audit report from current session state and trajectory."""

    event_counts: dict[str, int] = {}
    timeline: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    approvals: list[dict[str, Any]] = []
    compactions: list[dict[str, Any]] = []
    shell_commands: list[dict[str, Any]] = []
    safety_warnings: list[str] = []
    requested_by_id: dict[str, dict[str, Any]] = {}

    for idx, raw_event in enumerate(session.events):
        event_type = str(raw_event.get("type") or "unknown")
        payload = dict(raw_event.get("payload") or {})
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        if idx < include_timeline_limit:
            timeline.append(
                {
                    "index": idx,
                    "type": event_type,
                    "turn_id": raw_event.get("turn_id"),
                    "tool_call_id": raw_event.get("tool_call_id"),
                    "summary": _event_summary(event_type, payload),
                    "ts": raw_event.get("ts"),
                }
            )

        if event_type in {AgentEventType.COMMAND_STARTED.value, AgentEventType.COMMAND_FINISHED.value}:
            shell_commands.append(
                {
                    "type": event_type,
                    "command": payload.get("command", ""),
                    "arg": payload.get("arg", ""),
                    "turn_id": raw_event.get("turn_id"),
                    "summary": _event_summary(event_type, payload),
                }
            )

        if event_type == AgentEventType.TOOL_CALL_REQUESTED.value:
            for call in payload.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_id = str(call.get("id") or "")
                row = {
                    "call_id": call_id,
                    "tool": str(call.get("name") or ""),
                    "args": dict(call.get("args") or {}),
                    "turn_id": raw_event.get("turn_id"),
                    "state": "requested",
                }
                requested_by_id[call_id] = row
                tool_calls.append(row)
                if row["tool"] in {"shell", "test_runner"}:
                    shell_commands.append(row)

        if event_type == AgentEventType.TOOL_CALL_COMPLETED.value:
            call_id = str(raw_event.get("tool_call_id") or payload.get("tool_call_id") or "")
            row = dict(requested_by_id.get(call_id) or {})
            row.update(
                {
                    "call_id": call_id,
                    "tool": payload.get("tool") or row.get("tool", ""),
                    "state": payload.get("state", "completed"),
                    "result": payload.get("result") or {},
                    "turn_id": raw_event.get("turn_id"),
                }
            )
            tool_calls.append(row)
            if row.get("tool") in {"shell", "test_runner"} and row not in shell_commands:
                shell_commands.append(row)

        if event_type == AgentEventType.APPROVAL_REQUESTED.value:
            approval = dict(payload.get("approval") or {})
            approvals.append(approval)
            if str(approval.get("decision") or "").lower() in {"deny", "denied", "blocked"}:
                safety_warnings.append(f"Approval denied for {approval.get('tool_name', 'unknown tool')}")

        if event_type in {AgentEventType.ERROR.value, AgentEventType.COMMAND_FAILED.value}:
            message = payload.get("message") or payload.get("error") or "runtime error"
            safety_warnings.append(str(message)[:240])

        if event_type in {AgentEventType.COMPACT_STARTED.value, AgentEventType.COMPACT_COMPLETED.value}:
            compactions.append({"type": event_type, "payload": payload})

    checkpoints = [checkpoint.to_dict() for checkpoint in session.state.checkpoints]
    file_changes = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.get("reason") == "file_change"
        or "file" in str(checkpoint.get("summary") or "").lower()
    ]
    verification = [item.to_dict() for item in session.state.verification_results]
    repairs = [dict(item) for item in session.state.repair_attempts]
    for result in verification:
        if str(result.get("status") or "").lower() not in {"passed", "pass", "ok", "success"}:
            safety_warnings.append(f"Verification failed: {result.get('command', '')}")

    trajectory_path = runtime.session_store.rollout_file(session.session_id)
    replay = replay_trajectory(trajectory_path).to_dict() if trajectory_path.exists() else {
        "status": "missing",
        "violations": [f"trajectory not found: {trajectory_path}"],
    }
    if replay.get("status") != "ok":
        for violation in replay.get("violations") or []:
            safety_warnings.append(f"Replay: {violation}")

    if tool_calls and not session.action_graph_refs:
        safety_warnings.append("Tool calls exist but no action graph refs were recorded.")

    last_user_input = session.turns[-1].content if session.turns else ""
    provider_roles = {
        "active_role": session.config.active_role,
        "primary": session.config.primary_model,
        "planner": session.config.planner_model,
        "critic": session.config.critic_model,
        "summarizer": session.config.summarizer_model,
        "safety": session.config.safety_reviewer_model,
        "model_requests": [
            {
                "role": event.get("payload", {}).get("role"),
                "model": event.get("payload", {}).get("model"),
                "round": event.get("payload", {}).get("round"),
                "turn_id": event.get("turn_id"),
            }
            for event in session.events
            if event.get("type") == AgentEventType.MODEL_REQUEST_STARTED.value
        ],
    }

    run_tree_summary, online_eval = _run_tree_eval(session)
    action_graph_refs = [ref.to_dict() for ref in session.action_graph_refs]
    return RuntimeAuditReport(
        session_id=session.session_id,
        title=session.title,
        cwd=session.cwd,
        task_summary={
            "turns": len(session.turns),
            "events": len(session.events),
            "last_user_input": last_user_input,
            "trajectory_path": str(trajectory_path),
            "session_file": str(runtime.session_store.session_file(session.session_id)),
        },
        timeline=timeline,
        plan=session.state.plan.to_dict() if session.state.plan else None,
        goal=session.state.goal.to_dict() if session.state.goal else None,
        provider_roles=provider_roles,
        tool_calls=tool_calls,
        shell_commands=shell_commands,
        file_changes=file_changes,
        approvals=approvals,
        verification=verification,
        repairs=repairs,
        compactions=compactions,
        checkpoints=checkpoints,
        action_graph={
            "ref_count": len(action_graph_refs),
            "refs": action_graph_refs,
            "node_types": _count_values(ref.get("node_type") for ref in action_graph_refs),
        },
        replay_status=replay,
        safety_warnings=list(dict.fromkeys(safety_warnings)),
        event_type_counts=event_counts,
        run_tree_summary=run_tree_summary,
        online_eval=online_eval,
    )


def write_audit_report(report: RuntimeAuditReport, output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    base = _safe_filename(report.session_id)
    json_path = root / f"{base}.audit.json"
    md_path = root / f"{base}.audit.md"
    json_path.write_text(report.to_json(indent=2), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def redact_secrets(value: Any) -> Any:
    """Redact secrets from audit/harness/replay payloads."""

    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if _SECRET_KEY_RE.search(str(key)):
                out[key] = "<REDACTED>"
            else:
                out[key] = redact_secrets(item)
        return out
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    if isinstance(value, str):
        return _SECRET_VALUE_RE.sub("<REDACTED>", value)
    return value


def _event_summary(event_type: str, payload: dict[str, Any]) -> str:
    if "command" in payload:
        return str(payload.get("command") or "")
    if "text" in payload:
        return str(payload.get("text") or "")[:160]
    if "message" in payload:
        return str(payload.get("message") or "")[:160]
    if "tool" in payload:
        return str(payload.get("tool") or "")
    if "checkpoint" in payload and isinstance(payload["checkpoint"], dict):
        return str(payload["checkpoint"].get("reason") or "")
    return ""


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "session")).strip("_") or "session"


__all__ = ["RuntimeAuditReport", "audit_session", "write_audit_report", "redact_secrets"]
