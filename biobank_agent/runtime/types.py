"""Typed runtime substrate for Codex-like BioBank-Agent sessions.

These dataclasses are the stable public objects for the next runtime layer.
They intentionally reuse existing lower-level concepts where possible:

- ``AgentEvent`` comes from ``biobank_agent.core.events``.
- tool execution uses ``core.tools.ToolHandler`` and ``ToolScheduler``.
- action graph persistence adapts ``memory.ActionGraph``.

All objects provide explicit ``to_dict`` / ``from_dict`` methods so session
state can be written as plain JSON without pickle or arbitrary object loading.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from biobank_agent.core.events import AgentEvent


SCHEMA_VERSION = 1


class RuntimeStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"
    PAUSED = "paused"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class ProviderRole(str, Enum):
    PRIMARY_EXECUTOR = "primary_executor"
    PLANNER = "planner"
    CRITIC = "critic"
    SUMMARIZER = "summarizer"
    SAFETY_REVIEWER = "safety_reviewer"


@dataclass
class RuntimeConfig:
    """Runtime execution settings independent of the CLI/TUI."""

    primary_model: str = "deepseek/deepseek-v4-pro"
    planner_model: str = "moonshotai/kimi-k2.6"
    critic_model: str = "z-ai/glm-5.1"
    summarizer_model: str = "deepseek/deepseek-v4-pro"
    safety_reviewer_model: str = "z-ai/glm-5.1"
    active_role: str = ProviderRole.PRIMARY_EXECUTOR.value
    approval_profile: str = "yolo"
    max_tool_rounds: int = 8
    # Multi-model plan debate (CONCAT/EVOCHAMBER-style). When >=2 distinct models
    # are configured, candidate drafts debate over bounded rounds with
    # confidence-based consensus pruning before the orchestrator synthesizes.
    # All additive + optional, so old checkpoints load with these defaults.
    enable_debate: bool = True
    debate_rounds: int = 2
    consensus_threshold: float = 0.85
    debate_confidence_floor: float = 0.45
    # M16: route /plan through the asymmetric proposer/red-team/referee council instead of
    # the symmetric debate. Default off; flipped on once it wins the council_ab A/B gate.
    adversarial_council_enabled: bool = False
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_settings(cls, settings: Any) -> "RuntimeConfig":
        preferred = [
            item.strip()
            for item in str(getattr(settings, "preferred_multi_models", "") or "").split(",")
            if item.strip()
        ]
        primary = str(getattr(settings, "llm_model", "") or "").strip() or cls.primary_model
        return cls(
            primary_model=primary,
            planner_model=preferred[1] if len(preferred) > 1 else cls.planner_model,
            critic_model=preferred[2] if len(preferred) > 2 else cls.critic_model,
            summarizer_model=primary,
            safety_reviewer_model=preferred[2] if len(preferred) > 2 else cls.safety_reviewer_model,
            active_role=ProviderRole.PRIMARY_EXECUTOR.value,
            max_tool_rounds=int(getattr(settings, "max_tool_rounds", cls.max_tool_rounds)),
            enable_debate=bool(getattr(settings, "plan_debate_enabled", cls.enable_debate)),
            debate_rounds=int(getattr(settings, "debate_rounds", cls.debate_rounds)),
            consensus_threshold=float(getattr(settings, "plan_consensus_threshold", cls.consensus_threshold)),
            debate_confidence_floor=float(getattr(settings, "plan_debate_confidence_floor", cls.debate_confidence_floor)),
            adversarial_council_enabled=bool(getattr(settings, "adversarial_council_enabled", cls.adversarial_council_enabled)),
        )

    def model_for_role(self, role: ProviderRole | str) -> str:
        role_value = role.value if isinstance(role, ProviderRole) else str(role)
        if role_value == ProviderRole.PLANNER.value:
            return self.planner_model
        if role_value == ProviderRole.CRITIC.value:
            return self.critic_model
        if role_value == ProviderRole.SUMMARIZER.value:
            return self.summarizer_model
        if role_value == ProviderRole.SAFETY_REVIEWER.value:
            return self.safety_reviewer_model
        return self.primary_model

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        values = dict(data or {})
        values.pop("schema_version", None)
        return cls(**{k: v for k, v in values.items() if k in cls.__dataclass_fields__})


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "args": dict(self.args)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        return cls(id=str(data["id"]), name=str(data["name"]), args=dict(data.get("args") or {}))


@dataclass
class ToolResult:
    call_id: str
    name: str
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResult":
        return cls(
            call_id=str(data["call_id"]),
            name=str(data["name"]),
            result=dict(data.get("result") or {}),
            error=str(data.get("error") or ""),
            elapsed_seconds=float(data.get("elapsed_seconds") or 0.0),
        )


@dataclass
class ApprovalRequest:
    call_id: str
    tool_name: str
    capability: str = ""
    rationale: str = ""
    decision: str = ""
    profile: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovalRequest":
        return cls(
            call_id=str(data["call_id"]),
            tool_name=str(data["tool_name"]),
            capability=str(data.get("capability") or ""),
            rationale=str(data.get("rationale") or ""),
            decision=str(data.get("decision") or ""),
            profile=str(data.get("profile") or ""),
            ts=float(data.get("ts") or time.time()),
        )


@dataclass
class AssistantMessage:
    id: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "model": self.model,
            "provider": self.provider,
            "usage": dict(self.usage),
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssistantMessage":
        return cls(
            id=str(data["id"]),
            text=str(data.get("text") or ""),
            tool_calls=[ToolCall.from_dict(x) for x in data.get("tool_calls") or []],
            model=str(data.get("model") or ""),
            provider=str(data.get("provider") or ""),
            usage=dict(data.get("usage") or {}),
            ts=float(data.get("ts") or time.time()),
        )


@dataclass
class UserTurn:
    id: str
    content: str
    assistant_messages: list[AssistantMessage] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    approvals: list[ApprovalRequest] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    status: str = RuntimeStatus.RUNNING.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "assistant_messages": [m.to_dict() for m in self.assistant_messages],
            "tool_results": [r.to_dict() for r in self.tool_results],
            "approvals": [a.to_dict() for a in self.approvals],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserTurn":
        return cls(
            id=str(data["id"]),
            content=str(data.get("content") or ""),
            assistant_messages=[AssistantMessage.from_dict(x) for x in data.get("assistant_messages") or []],
            tool_results=[ToolResult.from_dict(x) for x in data.get("tool_results") or []],
            approvals=[ApprovalRequest.from_dict(x) for x in data.get("approvals") or []],
            started_at=float(data.get("started_at") or time.time()),
            completed_at=(float(data["completed_at"]) if data.get("completed_at") is not None else None),
            status=str(data.get("status") or RuntimeStatus.RUNNING.value),
        )


@dataclass
class PlanStep:
    id: str
    title: str
    purpose: str = ""
    status: str = "pending"
    dependencies: list[str] = field(default_factory=list)
    tool_scope: list[str] = field(default_factory=list)
    file_scope: list[str] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanStep":
        return cls(
            id=str(data.get("id") or ""),
            title=str(data.get("title") or ""),
            purpose=str(data.get("purpose") or ""),
            status=str(data.get("status") or "pending"),
            dependencies=[str(x) for x in data.get("dependencies") or []],
            tool_scope=[str(x) for x in data.get("tool_scope") or []],
            file_scope=[str(x) for x in data.get("file_scope") or []],
            verification=[str(x) for x in data.get("verification") or []],
            risks=[str(x) for x in data.get("risks") or []],
        )


@dataclass
class PlanState:
    objective: str = ""
    title: str = ""
    status: PlanStatus = PlanStatus.DRAFT
    summary: str = ""
    context_gathering: list[str] = field(default_factory=list)
    steps: list[PlanStep] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    verification_plan: list[str] = field(default_factory=list)
    required_approvals: list[str] = field(default_factory=list)
    proposed_tool_scope: list[str] = field(default_factory=list)
    proposed_file_scope: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    audit_summary: str = ""
    revision: int = 0
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "title": self.title,
            "status": self.status.value if isinstance(self.status, PlanStatus) else str(self.status),
            "summary": self.summary,
            "context_gathering": list(self.context_gathering),
            "steps": [step.to_dict() for step in self.steps],
            "risks": list(self.risks),
            "verification_plan": list(self.verification_plan),
            "required_approvals": list(self.required_approvals),
            "proposed_tool_scope": list(self.proposed_tool_scope),
            "proposed_file_scope": list(self.proposed_file_scope),
            "open_questions": list(self.open_questions),
            "audit_summary": self.audit_summary,
            "revision": int(self.revision),
            "updated_at": float(self.updated_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanState":
        status = PlanStatus(str(data.get("status") or PlanStatus.DRAFT.value))
        return cls(
            objective=str(data.get("objective") or ""),
            title=str(data.get("title") or ""),
            status=status,
            summary=str(data.get("summary") or ""),
            context_gathering=[str(x) for x in data.get("context_gathering") or []],
            steps=[PlanStep.from_dict(dict(x)) for x in data.get("steps") or []],
            risks=[str(x) for x in data.get("risks") or []],
            verification_plan=[str(x) for x in data.get("verification_plan") or []],
            required_approvals=[str(x) for x in data.get("required_approvals") or []],
            proposed_tool_scope=[str(x) for x in data.get("proposed_tool_scope") or []],
            proposed_file_scope=[str(x) for x in data.get("proposed_file_scope") or []],
            open_questions=[str(x) for x in data.get("open_questions") or []],
            audit_summary=str(data.get("audit_summary") or ""),
            revision=int(data.get("revision") or 0),
            updated_at=float(data.get("updated_at") or time.time()),
        )


@dataclass
class GoalState:
    objective: str = ""
    status: GoalStatus = GoalStatus.ACTIVE
    milestones: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    completion_state: str = "not_started"
    audit_summary: str = ""
    revision: int = 0
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "status": self.status.value if isinstance(self.status, GoalStatus) else str(self.status),
            "milestones": [dict(x) for x in self.milestones],
            "tasks": [dict(x) for x in self.tasks],
            "dependencies": [dict(x) for x in self.dependencies],
            "risks": list(self.risks),
            "verification_commands": list(self.verification_commands),
            "open_questions": list(self.open_questions),
            "completion_state": self.completion_state,
            "audit_summary": self.audit_summary,
            "revision": int(self.revision),
            "updated_at": float(self.updated_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoalState":
        status = GoalStatus(str(data.get("status") or GoalStatus.ACTIVE.value))
        return cls(
            objective=str(data.get("objective") or ""),
            status=status,
            milestones=[dict(x) for x in data.get("milestones") or []],
            tasks=[dict(x) for x in data.get("tasks") or []],
            dependencies=[dict(x) for x in data.get("dependencies") or []],
            risks=[str(x) for x in data.get("risks") or []],
            verification_commands=[str(x) for x in data.get("verification_commands") or []],
            open_questions=[str(x) for x in data.get("open_questions") or []],
            completion_state=str(data.get("completion_state") or "not_started"),
            audit_summary=str(data.get("audit_summary") or ""),
            revision=int(data.get("revision") or 0),
            updated_at=float(data.get("updated_at") or time.time()),
        )


@dataclass
class VerificationResult:
    id: str
    command: str
    status: str
    summary: str = ""
    returncode: int | None = None
    attempt: int = 1
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationResult":
        return cls(
            id=str(data.get("id") or ""),
            command=str(data.get("command") or ""),
            status=str(data.get("status") or ""),
            summary=str(data.get("summary") or ""),
            returncode=(int(data["returncode"]) if data.get("returncode") is not None else None),
            attempt=int(data.get("attempt") or 1),
            ts=float(data.get("ts") or time.time()),
        )


@dataclass
class RuntimeCheckpoint:
    id: str
    reason: str
    summary: str
    ts: float = field(default_factory=time.time)
    event_count: int = 0
    turn_count: int = 0
    plan_revision: int = 0
    goal_revision: int = 0
    workspace_fingerprint: str = ""
    trajectory_ids: list[str] = field(default_factory=list)
    action_graph_refs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeCheckpoint":
        return cls(
            id=str(data.get("id") or ""),
            reason=str(data.get("reason") or ""),
            summary=str(data.get("summary") or ""),
            ts=float(data.get("ts") or time.time()),
            event_count=int(data.get("event_count") or 0),
            turn_count=int(data.get("turn_count") or 0),
            plan_revision=int(data.get("plan_revision") or 0),
            goal_revision=int(data.get("goal_revision") or 0),
            workspace_fingerprint=str(data.get("workspace_fingerprint") or ""),
            trajectory_ids=[str(x) for x in data.get("trajectory_ids") or []],
            action_graph_refs=[dict(x) for x in data.get("action_graph_refs") or []],
        )


@dataclass
class RuntimeState:
    status: RuntimeStatus = RuntimeStatus.CREATED
    active_turn_id: str | None = None
    summary: str = ""
    errors: list[str] = field(default_factory=list)
    custom_data: dict[str, Any] = field(default_factory=dict)
    last_orchestration: dict[str, Any] = field(default_factory=dict)
    plan: PlanState | None = None
    goal: GoalState | None = None
    checkpoints: list[RuntimeCheckpoint] = field(default_factory=list)
    compacted_summary: dict[str, Any] = field(default_factory=dict)
    verification_results: list[VerificationResult] = field(default_factory=list)
    repair_attempts: list[dict[str, Any]] = field(default_factory=list)
    pending_work: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "active_turn_id": self.active_turn_id,
            "summary": self.summary,
            "errors": list(self.errors),
            "custom_data": dict(self.custom_data),
            "last_orchestration": dict(self.last_orchestration),
            "plan": self.plan.to_dict() if self.plan else None,
            "goal": self.goal.to_dict() if self.goal else None,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "compacted_summary": dict(self.compacted_summary),
            "verification_results": [result.to_dict() for result in self.verification_results],
            "repair_attempts": [dict(x) for x in self.repair_attempts],
            "pending_work": [dict(x) for x in self.pending_work],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeState":
        status = RuntimeStatus(str(data.get("status") or RuntimeStatus.CREATED.value))
        return cls(
            status=status,
            active_turn_id=data.get("active_turn_id"),
            summary=str(data.get("summary") or ""),
            errors=[str(x) for x in data.get("errors") or []],
            custom_data=dict(data.get("custom_data") or {}),
            last_orchestration=dict(data.get("last_orchestration") or {}),
            plan=(PlanState.from_dict(dict(data["plan"])) if data.get("plan") else None),
            goal=(GoalState.from_dict(dict(data["goal"])) if data.get("goal") else None),
            checkpoints=[RuntimeCheckpoint.from_dict(dict(x)) for x in data.get("checkpoints") or []],
            compacted_summary=dict(data.get("compacted_summary") or {}),
            verification_results=[VerificationResult.from_dict(dict(x)) for x in data.get("verification_results") or []],
            repair_attempts=[dict(x) for x in data.get("repair_attempts") or []],
            pending_work=[dict(x) for x in data.get("pending_work") or []],
        )


@dataclass
class ProviderRequest:
    session_id: str
    turn_id: str
    role: ProviderRole = ProviderRole.PRIMARY_EXECUTOR
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional per-call streaming hook: ``stream_cb(delta_text)`` is invoked for
    # each token/chunk as the model generates. Runtime-only (NOT serialised:
    # excluded from to_dict/from_dict so checkpoints stay plain data). When set,
    # the provider routes through the streaming API; when None, behaviour is
    # exactly as before (plain, non-streaming completion).
    stream_cb: Any = field(default=None, compare=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "role": self.role.value,
            "messages": list(self.messages),
            "tools": list(self.tools),
            "model": self.model,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderRequest":
        return cls(
            session_id=str(data["session_id"]),
            turn_id=str(data["turn_id"]),
            role=ProviderRole(str(data.get("role") or ProviderRole.PRIMARY_EXECUTOR.value)),
            messages=[dict(x) for x in data.get("messages") or []],
            tools=[dict(x) for x in data.get("tools") or []],
            model=str(data.get("model") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ProviderResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    deltas: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    provider: str = "fake"
    model: str = ""
    finish_reason: str = "stop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "deltas": list(self.deltas),
            "usage": dict(self.usage),
            "provider": self.provider,
            "model": self.model,
            "finish_reason": self.finish_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderResponse":
        return cls(
            text=str(data.get("text") or ""),
            tool_calls=[ToolCall.from_dict(x) for x in data.get("tool_calls") or []],
            deltas=[str(x) for x in data.get("deltas") or []],
            usage=dict(data.get("usage") or {}),
            provider=str(data.get("provider") or "fake"),
            model=str(data.get("model") or ""),
            finish_reason=str(data.get("finish_reason") or "stop"),
        )


@dataclass
class StreamEvent:
    event: AgentEvent

    def to_dict(self) -> dict[str, Any]:
        return self.event.to_dict()


@dataclass
class ActionGraphNode:
    node_type: str
    node_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    score: float = 1.0
    node_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionGraphNode":
        return cls(
            node_type=str(data["node_type"]),
            node_id=str(data["node_id"]),
            payload=dict(data.get("payload") or {}),
            score=float(data.get("score") or 1.0),
            node_key=str(data.get("node_key") or ""),
        )


@dataclass
class TrajectoryRecord:
    session_id: str
    event: AgentEvent
    seq: int
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "seq": self.seq,
            "ts": self.ts,
            "event": self.event.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryRecord":
        return cls(
            session_id=str(data["session_id"]),
            event=AgentEvent.from_dict(dict(data["event"])),
            seq=int(data.get("seq") if data.get("seq") is not None else 0),
            ts=float(data.get("ts") or time.time()),
        )


@dataclass
class TrajectoryRecorder:
    """Append-only trajectory helper for runtime and tool audits."""

    session_id: str
    records: list[TrajectoryRecord] = field(default_factory=list)

    def add(self, event: AgentEvent, *, seq: int | None = None) -> TrajectoryRecord:
        record = TrajectoryRecord(
            session_id=self.session_id,
            event=event,
            seq=int(seq if seq is not None else len(self.records)),
        )
        self.records.append(record)
        return record

    def extend(self, events: list[AgentEvent]) -> list[TrajectoryRecord]:
        return [self.add(event) for event in events]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "records": [record.to_dict() for record in self.records],
        }


__all__ = [
    "SCHEMA_VERSION",
    "RuntimeStatus",
    "PlanStatus",
    "GoalStatus",
    "ProviderRole",
    "RuntimeConfig",
    "ToolCall",
    "ToolResult",
    "ApprovalRequest",
    "AssistantMessage",
    "UserTurn",
    "PlanStep",
    "PlanState",
    "GoalState",
    "VerificationResult",
    "RuntimeCheckpoint",
    "RuntimeState",
    "ProviderRequest",
    "ProviderResponse",
    "StreamEvent",
    "ActionGraphNode",
    "TrajectoryRecord",
    "TrajectoryRecorder",
]
