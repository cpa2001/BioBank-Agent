"""Trajectory replay and invariant checking.

Replay here means deterministic validation of persisted runtime trajectory
records. It never calls a model provider or executes tools.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from biobank_agent.core.events import AgentEventType

from .types import TrajectoryRecord


@dataclass
class ReplayReport:
    """Result of replaying a runtime trajectory JSONL file."""

    status: str
    trajectory_path: str
    session_id: str = ""
    event_count: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    file_changes: list[dict[str, Any]] = field(default_factory=list)
    compactions: list[dict[str, Any]] = field(default_factory=list)
    provider_roles: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    model_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def replay_trajectory(path: str | Path) -> ReplayReport:
    """Validate a trajectory JSONL file without executing any side effects."""

    trajectory_path = Path(path)
    if not trajectory_path.exists():
        return ReplayReport(
            status="missing",
            trajectory_path=str(trajectory_path),
            violations=[f"trajectory not found: {trajectory_path}"],
        )

    records: list[TrajectoryRecord] = []
    violations: list[str] = []
    with trajectory_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(TrajectoryRecord.from_dict(json.loads(text)))
            except Exception as exc:
                violations.append(f"line {line_no}: invalid trajectory record: {exc}")

    if not records:
        violations.append("trajectory contains no events")
        return ReplayReport(status="failed", trajectory_path=str(trajectory_path), violations=violations)

    session_ids = {record.session_id for record in records if record.session_id}
    if len(session_ids) > 1:
        violations.append(f"multiple session ids in one trajectory: {sorted(session_ids)}")
    session_id = next(iter(session_ids), "")

    for expected_seq, record in enumerate(records):
        if record.seq != expected_seq:
            violations.append(f"record {expected_seq}: expected seq {expected_seq}, found {record.seq}")

    open_turns: set[str] = set()
    requested_tools: set[str] = set()
    started_tools: set[str] = set()
    approval_by_call: set[str] = set()
    compact_started = 0
    compact_completed = 0
    tool_calls: list[dict[str, Any]] = []
    approvals: list[dict[str, Any]] = []
    file_changes: list[dict[str, Any]] = []
    compactions: list[dict[str, Any]] = []
    provider_roles: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    for record in records:
        event = record.event
        payload = dict(event.payload or {})
        event_type = event.type.value
        if event_type == AgentEventType.USER_TURN_STARTED.value:
            turn_id = str(event.turn_id or payload.get("turn_id") or "")
            if not turn_id:
                violations.append(f"record {record.seq}: user turn started without turn_id")
            open_turns.add(turn_id)

        elif event_type == AgentEventType.USER_TURN_COMPLETED.value:
            turn_id = str(event.turn_id or payload.get("turn_id") or "")
            if turn_id not in open_turns:
                violations.append(f"record {record.seq}: completed unknown turn {turn_id}")
            open_turns.discard(turn_id)

        elif event_type == AgentEventType.MODEL_REQUEST_STARTED.value:
            provider_roles.append(
                {
                    "seq": record.seq,
                    "turn_id": event.turn_id,
                    "role": payload.get("role"),
                    "model": payload.get("model"),
                    "round": payload.get("round"),
                }
            )

        elif event_type == AgentEventType.TOOL_CALL_REQUESTED.value:
            for call in payload.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_id = str(call.get("id") or "")
                requested_tools.add(call_id)
                tool_calls.append(
                    {
                        "seq": record.seq,
                        "call_id": call_id,
                        "tool": str(call.get("name") or ""),
                        "args": dict(call.get("args") or {}),
                        "state": "requested",
                    }
                )

        elif event_type == AgentEventType.APPROVAL_REQUESTED.value:
            approval = dict(payload.get("approval") or {})
            call_id = str(event.tool_call_id or payload.get("tool_call_id") or approval.get("call_id") or "")
            approval_by_call.add(call_id)
            approvals.append({"seq": record.seq, **approval})

        elif event_type == AgentEventType.TOOL_CALL_STARTED.value:
            call_id = str(event.tool_call_id or payload.get("tool_call_id") or "")
            if call_id not in requested_tools:
                violations.append(f"record {record.seq}: tool started before request: {call_id}")
            if call_id not in approval_by_call:
                violations.append(f"record {record.seq}: tool started before approval event: {call_id}")
            started_tools.add(call_id)

        elif event_type == AgentEventType.TOOL_CALL_COMPLETED.value:
            call_id = str(event.tool_call_id or payload.get("tool_call_id") or "")
            if call_id not in started_tools:
                violations.append(f"record {record.seq}: tool completed before start: {call_id}")
            tool_calls.append(
                {
                    "seq": record.seq,
                    "call_id": call_id,
                    "tool": payload.get("tool", ""),
                    "state": payload.get("state", "completed"),
                    "result": payload.get("result") or {},
                }
            )

        elif event_type == AgentEventType.CHECKPOINT_CREATED.value:
            checkpoint = dict(payload.get("checkpoint") or {})
            checkpoints.append({"seq": record.seq, **checkpoint})
            reason = str(checkpoint.get("reason") or "")
            if reason == "file_change":
                file_changes.append({"seq": record.seq, **checkpoint})

        elif event_type == AgentEventType.COMPACT_STARTED.value:
            compact_started += 1
            compactions.append({"seq": record.seq, "type": event_type, "payload": payload})

        elif event_type == AgentEventType.COMPACT_COMPLETED.value:
            compact_completed += 1
            compactions.append({"seq": record.seq, "type": event_type, "payload": payload})
            if compact_completed > compact_started:
                violations.append(f"record {record.seq}: compaction completed before start")

    if open_turns:
        violations.append(f"unclosed user turn(s): {sorted(open_turns)}")
    if compact_started != compact_completed:
        violations.append(f"compaction start/complete mismatch: {compact_started}/{compact_completed}")

    return ReplayReport(
        status="ok" if not violations else "failed",
        trajectory_path=str(trajectory_path),
        session_id=session_id,
        event_count=len(records),
        tool_calls=tool_calls,
        approvals=approvals,
        file_changes=file_changes,
        compactions=compactions,
        provider_roles=provider_roles,
        checkpoints=checkpoints,
        violations=violations,
        model_calls=0,
    )


__all__ = ["ReplayReport", "replay_trajectory"]
