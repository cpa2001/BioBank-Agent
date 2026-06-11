"""Core runtime loop and session persistence for BioBank-Agent 2.0."""

from __future__ import annotations

import asyncio
import json
import hashlib
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from biobank_agent.core.events import AgentEvent, AgentEventBus, AgentEventType
from biobank_agent.core.tools.approval import ApprovalPolicy, builtin_profile
from biobank_agent.core.tools.protocol import ToolContext
from biobank_agent.core.tools.scheduler import ToolOutcome, ToolRequest, ToolScheduler, ToolState
from biobank_agent.core.memory.action_graph import ActionGraph as GraphStore

from .types import (
    ActionGraphNode,
    ApprovalRequest,
    GoalState,
    AssistantMessage,
    PlanState,
    PlanStatus,
    PlanStep,
    ProviderRequest,
    ProviderResponse,
    ProviderRole,
    RuntimeConfig,
    RuntimeCheckpoint,
    RuntimeState,
    RuntimeStatus,
    ToolCall,
    ToolResult,
    TrajectoryRecorder,
    TrajectoryRecord as RuntimeTrajectoryRecord,
    UserTurn,
    VerificationResult,
)


@dataclass
class AgentSession:
    """Serializable session snapshot for runtime replay."""

    session_id: str
    title: str
    cwd: str
    config: RuntimeConfig
    state: RuntimeState = field(default_factory=RuntimeState)
    turns: list[UserTurn] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    action_graph_refs: list[ActionGraphNode] = field(default_factory=list)
    schema_version: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "title": self.title,
            "cwd": self.cwd,
            "config": self.config.to_dict(),
            "state": self.state.to_dict(),
            "turns": [turn.to_dict() for turn in self.turns],
            "events": [dict(e) for e in self.events],
            "action_graph_refs": [ref.to_dict() for ref in self.action_graph_refs],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentSession":
        return cls(
            session_id=str(data["session_id"]),
            title=str(data.get("title") or ""),
            cwd=str(data.get("cwd") or ""),
            config=RuntimeConfig.from_dict(dict(data.get("config") or {})),
            state=RuntimeState.from_dict(dict(data.get("state") or {})),
            turns=[UserTurn.from_dict(x) for x in data.get("turns") or []],
            events=[dict(x) for x in data.get("events") or []],
            action_graph_refs=[ActionGraphNode.from_dict(x) for x in data.get("action_graph_refs") or []],
            schema_version=int(data.get("schema_version", 1)),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
        )

    def touch(self) -> None:
        self.updated_at = time.time()


class SessionStore:
    """JSON session persistence with simple listing/reload semantics."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def session_file(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session.json"

    def rollout_file(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "trajectory.jsonl"

    def checkpoint_dir(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "checkpoints"

    def checkpoint_file(self, session_id: str, checkpoint_id: str) -> Path:
        return self.checkpoint_dir(session_id) / f"{checkpoint_id}.json"

    def save(self, session: AgentSession, *, events: Iterable[AgentEvent] | None = None) -> Path:
        sdir = self.session_dir(session.session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        session.updated_at = time.time()
        self.session_file(session.session_id).write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if events is not None:
            recorder = TrajectoryRecorder(session.session_id)
            recorder.extend([AgentEvent.from_dict(event.to_dict()) for event in events])
            rollout = self.rollout_file(session.session_id)
            with rollout.open("w", encoding="utf-8") as fh:
                for record in recorder.records:
                    fh.write(json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n")
        return self.session_file(session.session_id)

    def save_checkpoint(self, session: AgentSession, checkpoint: RuntimeCheckpoint) -> Path:
        path = self.checkpoint_file(session.session_id, checkpoint.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "session_id": session.session_id,
                    "checkpoint": checkpoint.to_dict(),
                    "state": session.state.to_dict(),
                    "config": session.config.to_dict(),
                    "action_graph_refs": [ref.to_dict() for ref in session.action_graph_refs],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return path

    def load(self, session_id: str) -> AgentSession:
        data = json.loads(self.session_file(session_id).read_text(encoding="utf-8"))
        return AgentSession.from_dict(data)

    def list(self) -> list[AgentSession]:
        sessions: list[AgentSession] = []
        for child in sorted(self.root.iterdir(), reverse=True):
            if not child.is_dir():
                continue
            session_file = child / "session.json"
            if not session_file.exists():
                continue
            try:
                sessions.append(AgentSession.from_dict(json.loads(session_file.read_text(encoding="utf-8"))))
            except Exception:
                continue
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def append_trajectory(self, session_id: str, events: Iterable[AgentEvent]) -> Path:
        rollout = self.rollout_file(session_id)
        rollout.parent.mkdir(parents=True, exist_ok=True)
        # Continue the sequence from the existing rollout so the append-only log stays
        # monotonic across calls and process restarts (M6 durable rollout). A reset-to-0
        # seq would collide with earlier records and corrupt replay/resume ordering.
        base = 0
        if rollout.exists():
            with rollout.open("r", encoding="utf-8") as fh:
                base = sum(1 for line in fh if line.strip())
        with rollout.open("a", encoding="utf-8") as fh:
            for offset, event in enumerate(events):
                record = RuntimeTrajectoryRecord(session_id, event, base + offset)
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n")
        return rollout


class ProviderRouter:
    """Role-based provider selection for the runtime."""

    def __init__(self, providers: dict[str, Any] | None = None, config: RuntimeConfig | None = None) -> None:
        self.providers = dict(providers or {})
        self.config = config or RuntimeConfig()

    def model_for_role(self, role: ProviderRole | str) -> str:
        return self.config.model_for_role(role)

    def resolve(self, role: ProviderRole | str) -> Any:
        model = self.model_for_role(role)
        provider = self.providers.get(model)
        if provider is not None:
            return provider
        # Degrade gracefully instead of crashing a turn or council job on a misconfigured
        # role->model: prefer the primary model's provider, then any registered provider.
        # Raise only when nothing is registered at all (a genuine setup error).
        primary = self.providers.get(self.config.primary_model)
        if primary is not None:
            return primary
        if self.providers:
            return next(iter(self.providers.values()))
        raise KeyError(f"no provider registered for model {model!r} and no fallback is available")


class FakeProvider:
    """Deterministic provider used by tests."""

    def __init__(self, *, scripted_responses: list[ProviderResponse] | None = None, model: str = "fake-model") -> None:
        self.model = model
        self.scripted_responses = list(scripted_responses or [])
        self.requests: list[ProviderRequest] = []

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if self.scripted_responses:
            response = self.scripted_responses.pop(0)
            if not response.model:
                response.model = request.model or self.model
            return response
        return ProviderResponse(text="done", provider="fake", model=request.model or self.model)


class AgentRuntime:
    """Codex-like runtime substrate built from existing BioBank primitives."""

    def __init__(
        self,
        *,
        provider_router: ProviderRouter,
        tool_registry: Any,
        session_store: SessionStore,
        action_graph: GraphStore,
        config: RuntimeConfig | None = None,
        bus: AgentEventBus | None = None,
        approval_policy: ApprovalPolicy | None = None,
        tool_context_factory: Any | None = None,
        planner: Any | None = None,
    ) -> None:
        self.provider_router = provider_router
        self.tool_registry = tool_registry
        self.session_store = session_store
        self.action_graph = action_graph
        self.config = config or RuntimeConfig()
        self.bus = bus or AgentEventBus()
        self.approval_policy = approval_policy or ApprovalPolicy(builtin_profile(self.config.approval_profile))
        self._scheduler = ToolScheduler(policy=self.approval_policy, bus=self.bus)
        self._tool_context_factory = tool_context_factory
        # Optional council-backed planner. When injected, build_plan delegates to
        # it and FAILS LOUDLY on error instead of returning the static template.
        self.planner = planner

    def set_approval_profile(self, profile_name: str) -> None:
        profile = builtin_profile(profile_name)
        self.config.approval_profile = profile.name
        self.provider_router.config.approval_profile = profile.name
        self.approval_policy = ApprovalPolicy(profile)
        self._scheduler.policy = self.approval_policy

    def invoke_tool(
        self,
        session: AgentSession,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        turn_id: str | None = None,
        call_id: str | None = None,
        turn: UserTurn | None = None,
        context_factory: Any | None = None,
        record_request_event: bool = True,
    ) -> ToolOutcome:
        """Run one tool through the shared scheduler, approval policy, and audit path."""
        handler = self.tool_registry.get(tool_name)
        if handler is None:
            raise KeyError(f"unknown tool {tool_name}")

        call_id = str(call_id or uuid.uuid4().hex)
        call_args = dict(args or {})
        resolved_turn_id = str(
            turn_id
            or (turn.id if turn is not None else "")
            or session.state.active_turn_id
            or uuid.uuid4().hex
        )

        if record_request_event:
            self._record_event(
                session,
                AgentEvent.make(
                    AgentEventType.TOOL_CALL_REQUESTED,
                    session_id=session.session_id,
                    turn_id=resolved_turn_id,
                    tool_calls=[ToolCall(id=call_id, name=tool_name, args=dict(call_args)).to_dict()],
                ),
            )

        approval = self.approval_policy.decide(handler, call_args)
        approval_request = ApprovalRequest(
            call_id=call_id,
            tool_name=tool_name,
            capability=approval.capability.value if approval.capability else "",
            rationale=approval.rationale,
            decision=approval.decision.value,
            profile=approval.profile,
        )
        if turn is not None:
            turn.approvals.append(approval_request)
        self._record_event(
            session,
            AgentEvent.make(
                AgentEventType.APPROVAL_REQUESTED,
                session_id=session.session_id,
                turn_id=resolved_turn_id,
                tool_call_id=call_id,
                approval=approval_request.to_dict(),
            ),
        )
        self.create_checkpoint(session, "approval", summary=f"{tool_name}: {approval.decision.value}")
        self._record_event(
            session,
            AgentEvent.make(
                AgentEventType.TOOL_CALL_STARTED,
                session_id=session.session_id,
                turn_id=resolved_turn_id,
                tool_call_id=call_id,
                tool=tool_name,
            ),
        )

        runtime_context_factory = context_factory
        if runtime_context_factory is None:
            def _default_context_factory(req: ToolRequest) -> ToolContext:
                return ToolContext(
                    name=req.handler.name,
                    args=dict(req.args or {}),
                    capabilities=req.handler.required_capabilities(),
                    workspace_root=session.cwd,
                    permission_mode=self.config.approval_profile,
                    turn_id=resolved_turn_id,
                    tool_call_id=call_id,
                    tool_registry=self.tool_registry,
                    activate_skills=self._make_activate_skills(session),
                )

            runtime_context_factory = _default_context_factory

        outcome = asyncio.run(
            self._scheduler.run(
                ToolRequest(
                    call_id=call_id,
                    handler=handler,
                    args=call_args,
                    turn_id=resolved_turn_id,
                    context_factory=runtime_context_factory,
                )
            )
        )

        result = ToolResult(
            call_id=call_id,
            name=tool_name,
            result=dict(outcome.result or {}),
            error=outcome.error or "",
            elapsed_seconds=outcome.elapsed_seconds,
        )
        if turn is not None:
            turn.tool_results.append(result)
        self._record_event(
            session,
            AgentEvent.make(
                AgentEventType.TOOL_CALL_COMPLETED,
                session_id=session.session_id,
                turn_id=resolved_turn_id,
                tool_call_id=call_id,
                tool=tool_name,
                state=outcome.state.value,
                result=result.to_dict(),
            ),
        )
        self.create_checkpoint(session, "tool_call", summary=f"{tool_name}: {outcome.state.value}")
        if _tool_may_change_files(handler, tool_name):
            self.create_checkpoint(session, "file_change", summary=f"{tool_name}: {outcome.state.value}")
        node_key = self.action_graph.upsert_node(
            "tool",
            call_id,
            payload={
                "name": tool_name,
                "args": call_args,
                "session_id": session.session_id,
                "turn_id": resolved_turn_id,
            },
        )
        session.action_graph_refs.append(
            ActionGraphNode(
                node_type="tool",
                node_id=call_id,
                payload={"name": tool_name, "session_id": session.session_id, "turn_id": resolved_turn_id},
                score=1.0,
                node_key=node_key,
            )
        )
        self._record_event(
            session,
            AgentEvent.make(
                AgentEventType.ACTION_GRAPH_NODE_CREATED,
                session_id=session.session_id,
                turn_id=resolved_turn_id,
                node_key=node_key,
                node_type="tool",
                node_id=call_id,
            ),
        )
        return outcome

    def create_session(self, *, title: str, cwd: str) -> AgentSession:
        return AgentSession(
            session_id=uuid.uuid4().hex,
            title=title,
            cwd=cwd,
            config=self.config,
        )

    def create_checkpoint(self, session: AgentSession, reason: str, *, summary: str = "") -> RuntimeCheckpoint:
        checkpoint = RuntimeCheckpoint(
            id=uuid.uuid4().hex[:12],
            reason=reason,
            summary=summary or self._checkpoint_summary(session, reason),
            event_count=len(session.events),
            turn_count=len(session.turns),
            plan_revision=session.state.plan.revision if session.state.plan else 0,
            goal_revision=session.state.goal.revision if session.state.goal else 0,
            workspace_fingerprint=self.workspace_fingerprint(session.cwd),
            trajectory_ids=[str(i) for i in range(len(session.events))],
            action_graph_refs=[ref.to_dict() for ref in session.action_graph_refs],
        )
        session.state.checkpoints.append(checkpoint)
        self._record_event(
            session,
            AgentEvent.make(
                AgentEventType.CHECKPOINT_CREATED,
                session_id=session.session_id,
                checkpoint=checkpoint.to_dict(),
            ),
        )
        self.session_store.save_checkpoint(session, checkpoint)
        return checkpoint

    def workspace_fingerprint(self, cwd: str) -> str:
        root = Path(cwd)
        try:
            proc = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            payload = proc.stdout if proc.returncode == 0 else f"nogit:{root.resolve()}"
        except Exception:
            payload = f"nogit:{root.resolve() if root.exists() else root}"
        return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:16]

    def build_plan(self, objective: str, *, previous: PlanState | None = None, refinement: str = "", **plan_kwargs: Any) -> PlanState:
        # When a council-backed planner is injected, delegate to it and let any
        # failure propagate — we must never silently mask planning failure with
        # the generic static template below.
        if self.planner is not None:
            return self.planner.build_plan(objective, previous=previous, refinement=refinement, **plan_kwargs)
        clean_objective = " ".join(str(objective or "").split())
        title = _title_from_text(clean_objective)
        revision = (previous.revision + 1) if previous else 1
        context_gathering = [
            "Inspect existing repository, session state, available tools, and relevant files before making changes.",
            "Prefer read-only discovery until the user approves the plan.",
        ]
        risks = [
            "The task may require file edits, shell execution, or external tools that need approval.",
            "Verification can fail; failed checks should update the plan before repair attempts continue.",
        ]
        verification_plan = [
            "Run targeted tests or diagnostics that directly cover the changed behavior.",
            "Inspect generated artifacts, session state, and trajectory records before claiming completion.",
        ]
        required_approvals = [
            "User approval before mutating files or running shell commands outside read-only diagnostics.",
            "Explicit permission mode change before workspace writes if current policy denies edits.",
        ]
        tool_scope = ["file_read", "search", "git_status", "git_diff"]
        file_scope = ["workspace files discovered during read-only context gathering"]
        if previous and previous.proposed_tool_scope:
            tool_scope = list(dict.fromkeys([*tool_scope, *previous.proposed_tool_scope]))
        if previous and previous.proposed_file_scope:
            file_scope = list(dict.fromkeys([*file_scope, *previous.proposed_file_scope]))
        steps = [
            PlanStep(
                id="context",
                title="Gather context",
                purpose="Read relevant files and current session state without modifying the workspace.",
                tool_scope=["file_read", "search", "git_status"],
                file_scope=file_scope,
                verification=["Confirm the gathered evidence is cited in the plan summary."],
            ),
            PlanStep(
                id="design",
                title="Design approach",
                purpose="Break the objective into auditable work items, dependencies, and risk gates.",
                dependencies=["context"],
                tool_scope=[],
                verification=["Plan includes success criteria, risks, approvals, and proposed scope."],
            ),
            PlanStep(
                id="execute",
                title="Execute approved work",
                purpose="Run tools and edit files only after approval and within the proposed scope.",
                dependencies=["design"],
                tool_scope=["file_write", "file_edit", "apply_patch", "shell", "test_runner"],
                verification=["Every tool call is recorded in session events, trajectory, and action graph."],
                risks=["Blocked by denied approvals or failed checks."],
            ),
            PlanStep(
                id="verify",
                title="Verify and repair",
                purpose="Run checks, summarize failures, attempt bounded repairs, and rerun checks.",
                dependencies=["execute"],
                tool_scope=["test_runner", "shell", "git_diff"],
                verification=verification_plan,
            ),
        ]
        if refinement:
            risks.append(f"Refinement requested by user: {refinement[:240]}")
        return PlanState(
            objective=clean_objective,
            title=title,
            status=PlanStatus.DRAFT,
            summary=f"Draft plan for: {clean_objective}",
            context_gathering=context_gathering,
            steps=steps,
            risks=risks,
            verification_plan=verification_plan,
            required_approvals=required_approvals,
            proposed_tool_scope=tool_scope,
            proposed_file_scope=file_scope,
            open_questions=[],
            audit_summary="Plan generated from user-visible objective and runtime state; hidden chain-of-thought is not stored.",
            revision=revision,
        )

    def update_plan(self, session: AgentSession, objective: str, *, refinement: str = "", **plan_kwargs: Any) -> PlanState:
        plan = self.build_plan(objective, previous=session.state.plan, refinement=refinement, **plan_kwargs)
        session.state.plan = plan
        session.state.pending_work = [step.to_dict() for step in plan.steps if step.status not in {"done", "completed"}]
        self._record_event(
            session,
            AgentEvent.make(
                AgentEventType.PLAN_UPDATED,
                session_id=session.session_id,
                plan=plan.to_dict(),
            ),
        )
        self.create_checkpoint(session, "plan_update", summary=plan.summary)
        return plan

    def approve_plan(self, session: AgentSession) -> PlanState | None:
        if session.state.plan is None:
            return None
        session.state.plan.status = PlanStatus.APPROVED
        session.state.plan.updated_at = time.time()
        self._record_event(
            session,
            AgentEvent.make(
                AgentEventType.PLAN_UPDATED,
                session_id=session.session_id,
                plan=session.state.plan.to_dict(),
                approval="approved",
            ),
        )
        self.create_checkpoint(session, "plan_approved", summary=session.state.plan.summary)
        return session.state.plan

    def reject_plan(self, session: AgentSession, reason: str = "") -> PlanState | None:
        if session.state.plan is None:
            return None
        session.state.plan.status = PlanStatus.REJECTED
        if reason:
            session.state.plan.risks.append(f"Rejected by user: {reason[:240]}")
        session.state.plan.updated_at = time.time()
        self._record_event(
            session,
            AgentEvent.make(
                AgentEventType.PLAN_UPDATED,
                session_id=session.session_id,
                plan=session.state.plan.to_dict(),
                approval="rejected",
            ),
        )
        self.create_checkpoint(session, "plan_rejected", summary=reason or session.state.plan.summary)
        return session.state.plan

    def set_goal(self, session: AgentSession, objective: str) -> GoalState:
        clean_objective = " ".join(str(objective or "").split())
        previous = session.state.goal
        revision = (previous.revision + 1) if previous else 1
        if previous is None:
            milestones = [
                {"id": "plan", "title": "Plan approved", "status": "pending"},
                {"id": "execute", "title": "Approved work executed", "status": "pending"},
                {"id": "verify", "title": "Verification passed or limitations recorded", "status": "pending"},
            ]
            tasks = [
                {"id": "context", "title": "Gather context", "status": "pending"},
                {"id": "plan", "title": "Create/refine plan", "status": "pending"},
                {"id": "verify", "title": "Run verification checks", "status": "pending"},
            ]
            dependencies = [
                {"from": "context", "to": "plan"},
                {"from": "plan", "to": "verify"},
            ]
            risks = [
                "Goal may span multiple turns and must be resumed from checkpoints if interrupted.",
                "Completion requires explicit verification evidence, not only an assistant summary.",
            ]
            verification_commands: list[str] = []
            open_questions: list[str] = []
            completion_state = "active"
            status = GoalState().status
        else:
            milestones = [dict(x) for x in previous.milestones]
            tasks = [dict(x) for x in previous.tasks]
            dependencies = [dict(x) for x in previous.dependencies]
            risks = list(previous.risks)
            verification_commands = list(previous.verification_commands)
            open_questions = list(previous.open_questions)
            completion_state = previous.completion_state
            status = previous.status
        goal = GoalState(
            objective=clean_objective,
            status=status,
            milestones=milestones,
            tasks=tasks,
            dependencies=dependencies,
            risks=risks,
            verification_commands=verification_commands,
            open_questions=open_questions,
            completion_state=completion_state,
            audit_summary="Goal state stores user-visible objective, milestones, risks, and verification boundaries.",
            revision=revision,
        )
        session.state.goal = goal
        session.title = _title_from_text(clean_objective)
        self._record_event(
            session,
            AgentEvent.make(
                AgentEventType.GOAL_UPDATED,
                session_id=session.session_id,
                goal=goal.to_dict(),
            ),
        )
        self.create_checkpoint(session, "goal_update", summary=clean_objective)
        return goal

    def compact_session(self, session: AgentSession, summary: str = "") -> AgentSession:
        compacted = self.build_compact_summary(session, summary=summary)
        self._record_event(session, AgentEvent.make(AgentEventType.COMPACT_STARTED, session_id=session.session_id, summary_before=session.state.summary))
        session.state.summary = compacted["summary"]
        session.state.compacted_summary = compacted
        self._record_event(session, AgentEvent.make(AgentEventType.COMPACT_COMPLETED, session_id=session.session_id, summary_after=session.state.summary, compacted=compacted))
        self.create_checkpoint(session, "compaction", summary=session.state.summary)
        session.touch()
        return session

    def maybe_auto_compact(self, session: AgentSession, *, event_threshold: int = 120, turn_threshold: int = 20) -> bool:
        if len(session.events) < event_threshold and len(session.turns) < turn_threshold:
            return False
        self.compact_session(session, summary="Auto compacted runtime context while preserving audit-critical state.")
        return True

    def build_compact_summary(self, session: AgentSession, *, summary: str = "") -> dict[str, Any]:
        tool_results = [
            result.to_dict()
            for turn in session.turns
            for result in turn.tool_results
        ]
        approvals = [
            approval.to_dict()
            for turn in session.turns
            for approval in turn.approvals
        ]
        verification_results = [result.to_dict() for result in session.state.verification_results]
        failed_attempts = [dict(item) for item in session.state.repair_attempts]
        return {
            "summary": summary or session.state.summary or f"Compacted {len(session.turns)} turn(s) and {len(session.events)} event(s).",
            "objective": session.state.goal.objective if session.state.goal else "",
            "plan": session.state.plan.to_dict() if session.state.plan else None,
            "goal": session.state.goal.to_dict() if session.state.goal else None,
            "completed_tasks": [
                step.to_dict() for step in (session.state.plan.steps if session.state.plan else [])
                if step.status in {"done", "completed"}
            ],
            "pending_tasks": [dict(x) for x in session.state.pending_work],
            "tool_results": tool_results,
            "approvals": approvals,
            "verification_results": verification_results,
            "failed_attempts": failed_attempts,
            "risks": list(session.state.plan.risks if session.state.plan else []) + list(session.state.goal.risks if session.state.goal else []),
            "trajectory_ids": [str(i) for i in range(len(session.events))],
            "action_graph_refs": [ref.to_dict() for ref in session.action_graph_refs],
            "checkpoints": [checkpoint.to_dict() for checkpoint in session.state.checkpoints],
        }

    def record_verification(
        self,
        session: AgentSession,
        *,
        command: str,
        status: str,
        summary: str = "",
        returncode: int | None = None,
        attempt: int = 1,
    ) -> VerificationResult:
        result = VerificationResult(
            id=uuid.uuid4().hex[:12],
            command=command,
            status=status,
            summary=summary,
            returncode=returncode,
            attempt=attempt,
        )
        session.state.verification_results.append(result)
        self._record_event(
            session,
            AgentEvent.make(
                AgentEventType.VERIFICATION_COMPLETED,
                session_id=session.session_id,
                verification=result.to_dict(),
            ),
        )
        self.create_checkpoint(session, "verification", summary=summary or command)
        return result

    def record_repair_attempt(self, session: AgentSession, *, reason: str, attempt: int, max_attempts: int, status: str) -> dict[str, Any]:
        payload = {
            "id": uuid.uuid4().hex[:12],
            "reason": reason,
            "attempt": int(attempt),
            "max_attempts": int(max_attempts),
            "status": status,
            "ts": time.time(),
        }
        session.state.repair_attempts.append(payload)
        if session.state.plan:
            session.state.plan.risks.append(f"Repair attempt {attempt}/{max_attempts}: {reason[:240]}")
            session.state.plan.updated_at = time.time()
        self._record_event(
            session,
            AgentEvent.make(
                AgentEventType.REPAIR_ATTEMPTED,
                session_id=session.session_id,
                repair=payload,
            ),
        )
        self.create_checkpoint(session, "repair_attempt", summary=reason)
        return payload

    def verification_repair_loop(
        self,
        session: AgentSession,
        *,
        checks: Iterable[dict[str, Any]],
        max_attempts: int = 2,
    ) -> list[VerificationResult]:
        """Record a bounded verify/repair loop from externally run checks.

        This method does not fake passing checks. Callers provide observed check
        results; failed results trigger recorded repair attempts up to
        ``max_attempts`` and update the plan risks for later execution.
        """
        observed: list[VerificationResult] = []
        for raw in checks:
            command = str(raw.get("command") or "")
            status = str(raw.get("status") or "")
            summary = str(raw.get("summary") or "")
            returncode = raw.get("returncode")
            attempt = int(raw.get("attempt") or 1)
            self._record_event(
                session,
                AgentEvent.make(
                    AgentEventType.VERIFICATION_STARTED,
                    session_id=session.session_id,
                    command=command,
                    attempt=attempt,
                ),
            )
            result = self.record_verification(
                session,
                command=command,
                status=status,
                summary=summary,
                returncode=(int(returncode) if returncode is not None else None),
                attempt=attempt,
            )
            observed.append(result)
            if status.lower() in {"passed", "pass", "ok", "success"}:
                continue
            if attempt <= max_attempts:
                self.record_repair_attempt(
                    session,
                    reason=summary or f"verification failed: {command}",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    status="queued" if attempt < max_attempts else "bounded_stop",
                )
        return observed

    def run_verification_commands(
        self,
        session: AgentSession,
        commands: Iterable[str],
        *,
        max_attempts: int = 2,
        timeout_s: int = 120,
        repair_fn: Any | None = None,
    ) -> list[VerificationResult]:
        """Run real verification commands and record observed results.

        Passing status is derived only from the subprocess return code. Failed
        commands may trigger a caller-provided repair callback before bounded
        retry; if no callback is provided, failures are still recorded and the
        loop stops at the configured attempt limit.
        """
        observed: list[VerificationResult] = []
        command_list = [str(raw or "").strip() for raw in commands if str(raw or "").strip()]
        if not command_list:
            return observed
        attempts = max(1, int(max_attempts or 1))
        timeout = max(1, int(timeout_s or 120))
        overall_success = True
        if session.state.plan and session.state.plan.status != PlanStatus.EXECUTING:
            session.state.plan.status = PlanStatus.EXECUTING
            session.state.plan.updated_at = time.time()
            self._record_event(
                session,
                AgentEvent.make(
                    AgentEventType.PLAN_UPDATED,
                    session_id=session.session_id,
                    plan=session.state.plan.to_dict(),
                    verification_state="executing",
                ),
            )
            self.create_checkpoint(session, "plan_update", summary="verification executing")
        for command in command_list:
            command_passed = False
            for attempt in range(1, attempts + 1):
                self._record_event(
                    session,
                    AgentEvent.make(
                        AgentEventType.VERIFICATION_STARTED,
                        session_id=session.session_id,
                        command=command,
                        attempt=attempt,
                    ),
                )
                try:
                    proc = subprocess.run(
                        shlex.split(command),
                        cwd=session.cwd,
                        text=True,
                        capture_output=True,
                        timeout=timeout,
                        check=False,
                    )
                    status = "passed" if proc.returncode == 0 else "failed"
                    output = (proc.stdout or proc.stderr or "").strip()
                    summary = output[:2000]
                    returncode: int | None = int(proc.returncode)
                except Exception as exc:
                    status = "failed"
                    summary = str(exc)
                    returncode = None
                result = self.record_verification(
                    session,
                    command=command,
                    status=status,
                    summary=summary,
                    returncode=returncode,
                    attempt=attempt,
                )
                observed.append(result)
                if status == "passed":
                    command_passed = True
                    break
                can_retry = attempt < attempts
                repair = self.record_repair_attempt(
                    session,
                    reason=summary or f"verification failed: {command}",
                    attempt=attempt,
                    max_attempts=attempts,
                    status="attempted" if can_retry else "bounded_stop",
                )
                if can_retry and callable(repair_fn):
                    repair_fn(session, command, attempt, result)
                    repair["callback"] = "completed"
            if not command_passed:
                overall_success = False
        if session.state.plan is not None:
            session.state.plan.status = PlanStatus.COMPLETED if overall_success else PlanStatus.FAILED
            session.state.plan.updated_at = time.time()
            self._record_event(
                session,
                AgentEvent.make(
                    AgentEventType.PLAN_UPDATED,
                    session_id=session.session_id,
                    plan=session.state.plan.to_dict(),
                    verification_state="completed" if overall_success else "failed",
                ),
            )
            self.create_checkpoint(
                session,
                "plan_update",
                summary=f"verification {'completed' if overall_success else 'failed'}",
            )
        return observed

    def save_session(self, session: AgentSession) -> Path:
        self.create_checkpoint(session, "session_save", summary="Session saved.")
        self._record_event(
            session,
            AgentEvent.make(
                AgentEventType.SESSION_SAVED,
                session_id=session.session_id,
                title=session.title,
            ),
        )
        return self.session_store.save(
            session,
            events=[AgentEvent.from_dict(e) for e in session.events],
        )

    def load_session(self, session_id: str) -> AgentSession:
        return self.session_store.load(session_id)

    def list_sessions(self) -> list[AgentSession]:
        return self.session_store.list()

    def _executor_system_prompt(self, session: AgentSession) -> str:
        """System prompt for the primary tool-using executor.

        The executor previously ran with NO system prompt (only the raw user
        message), which made it flail — repeating a tool call that already
        succeeded, speculatively profiling unclear input, and never converging.
        This establishes identity, the workspace sandbox, and the tool-use
        discipline a coding/research agent needs (write→run→observe→fix; never
        repeat a succeeded action; finish with a concrete answer)."""
        lazy_tools = (
            getattr(self.config, "lazy_tools_enabled", True)
            and hasattr(self.tool_registry, "exposed_handlers")
        )
        try:
            if lazy_tools:
                tool_names = [h.spec().name for h in self.tool_registry.exposed_handlers()]
            else:
                tool_names = [h.spec().name for h in self.tool_registry.list_handlers()]
        except Exception:
            tool_names = []
        tools_hint = ", ".join(sorted(tool_names)[:60]) if tool_names else "(none registered)"
        more_tools = ""
        if lazy_tools:
            use_tree = getattr(self.config, "skill_tree_enabled", True)
            try:
                from biobank_agent.skills import manifest as _skill_manifest

                if use_tree and _skill_manifest.tree_available():
                    cat_lines = "\n".join(
                        f"  - {r}: {_skill_manifest.node_summary(r)}" for r in _skill_manifest.root_nodes()
                    )
                    more_tools = (
                        "\nMORE TOOLS ON DEMAND: only high-frequency tools are listed above. The full "
                        "skill corpus is organized as a tree — call navigate_skill_tree('<node>') to "
                        "browse a category, or skill_search('<intent>') to load by intent (matched skills "
                        "become callable on your next step). Top-level categories:\n" + cat_lines + "\n"
                    )
                else:
                    cats = _skill_manifest.category_summaries()
                    if cats:
                        cat_lines = "\n".join(f"  - {dom}: {summary}" for dom, summary in cats.items())
                        more_tools = (
                            "\nMORE TOOLS ON DEMAND: only high-frequency tools are listed above. Many more "
                            "domain skills exist, grouped by category — call skill_search('<intent>') to load "
                            "the ones you need (they become callable on your next step):\n" + cat_lines + "\n"
                        )
            except Exception:
                more_tools = ""
        return (
            "You are BioBank Agent — an autonomous AI agent for biobank research and software "
            "engineering, operating in a command-line session with REAL tools that take REAL "
            "actions (read/write/edit files, run shell commands, run tests, search, git, and "
            "domain analysis skills).\n\n"
            f"WORKSPACE: your working directory is {session.cwd}. Use relative paths, or paths under "
            "it (or another allowed root). If a path is rejected, the error message lists the allowed "
            "roots — pick one of those instead of retrying the rejected path. For scratch space, use a "
            "subdirectory of the workspace.\n"
            f"PERMISSIONS: approval profile is '{self.config.approval_profile}'. Act autonomously; "
            "do not ask the user to confirm steps you can perform yourself.\n"
            f"TOOLS AVAILABLE: {tools_hint}.\n{more_tools}\n"
            "HOW TO WORK (important):\n"
            "1. Take real actions with tools, then USE the result. After a tool call SUCCEEDS, move "
            "forward — NEVER repeat a call that already succeeded.\n"
            "2. For coding tasks: write the file ONCE, then RUN it (shell or test_runner), read the "
            "ACTUAL output, and edit only if it failed. Loop write→run→observe→fix→re-run until it "
            "works.\n"
            "3. If a tool returns an error, read it and CHANGE your approach. Do not call the same "
            "tool with the same arguments again. If a path was rejected, switch to a path inside the "
            "workspace.\n"
            "4. When the task is complete, STOP calling tools and give a concise final answer that "
            "states the concrete result you observed (e.g. the exact program output).\n"
            "5. If the request is empty, gibberish, or unclear, briefly say so and ask for specifics "
            "— do NOT call tools speculatively.\n"
            "6. Report only what actually happened. Never fabricate tool output or results.\n"
            "7. For commands that take many minutes (plink/gatk/bcftools, a WGS pipeline), use the "
            "run_job tool to run them in the BACKGROUND, then job_wait/job_status — do not block on a "
            "long foreground command.\n"
            "8. If the input data does NOT match what a standard tool expects (wrong format, a missing "
            "FORMAT field, a missing file or uninstalled tool), call pause_and_ask to surface the "
            "mismatch with options. Do NOT silently write a simplified replacement script that only "
            "partially does the job — pause and let the user fix the data or redirect."
        )

    def _history_messages(self, session: AgentSession, current: UserTurn, *, max_turns: int = 6) -> list[dict[str, Any]]:
        """Compact prior-turn history (user + final assistant text) so the executor
        has conversational memory across turns. Intermediate tool chatter is
        omitted to keep context lean."""
        def _cap(text: str, limit: int = 2000) -> str:
            text = str(text or "")
            suffix = " …[truncated]"
            return text if len(text) <= limit else text[: limit - len(suffix)] + suffix

        msgs: list[dict[str, Any]] = []
        prior = [t for t in session.turns if t.id != current.id][-max_turns:]
        for t in prior:
            if t.content:
                msgs.append({"role": "user", "content": _cap(str(t.content))})
            # Only replay the assistant conclusion of turns that actually completed;
            # a failed turn's final text is not authoritative and would mislead.
            if str(getattr(t, "status", "")) != RuntimeStatus.COMPLETED.value:
                continue
            final = ""
            for m in reversed(getattr(t, "assistant_messages", []) or []):
                if getattr(m, "text", ""):
                    final = m.text
                    break
            if final:
                msgs.append({"role": "assistant", "content": _cap(final)})
        return msgs

    @staticmethod
    def _make_activate_skills(session: "AgentSession"):
        """Closure for skill_search to append activated skill names to the session's
        active set, so their schemas inject on the next round (M0 lazy exposure).
        Wired into every ToolContext so lazy exposure works for bare AgentRuntime
        callers (harness/headless), not only the interactive CLI."""

        def _activate(names: list[str]) -> None:
            active = session.state.custom_data.setdefault("active_skills", [])
            for name in names:
                if name and name not in active:
                    active.append(name)

        return _activate

    def run_turn(self, session: AgentSession, user_text: str) -> AgentSession:
        turn = UserTurn(id=uuid.uuid4().hex, content=user_text)
        session.turns.append(turn)
        session.state.status = RuntimeStatus.RUNNING
        session.state.active_turn_id = turn.id
        self.create_checkpoint(session, "user_turn", summary=user_text[:240])
        self._record_event(session, AgentEvent.make(AgentEventType.SESSION_STARTED, session_id=session.session_id, title=session.title, cwd=session.cwd))
        self._record_event(session, AgentEvent.make(AgentEventType.USER_TURN_STARTED, session_id=session.session_id, turn_id=turn.id, text=user_text))
        try:
            active_role = ProviderRole(str(self.config.active_role or ProviderRole.PRIMARY_EXECUTOR.value))
        except Exception:
            active_role = ProviderRole.PRIMARY_EXECUTOR
        try:
            provider = self.provider_router.resolve(active_role)
        except KeyError:
            provider = self.provider_router.resolve(ProviderRole.PRIMARY_EXECUTOR)
            active_role = ProviderRole.PRIMARY_EXECUTOR
        lazy_tools = (
            getattr(self.config, "lazy_tools_enabled", True)
            and hasattr(self.tool_registry, "exposed_schemas")
        )

        def _round_tool_schemas() -> list[dict]:
            # Recomputed each round so a skill_search activation surfaces the matched
            # skill on the NEXT round of the same turn (M0 lazy tool exposure).
            if not lazy_tools:
                return self.tool_registry.tool_schemas()
            active = session.state.custom_data.get("active_skills") or []
            return self.tool_registry.exposed_schemas(active)

        tool_schemas = _round_tool_schemas()
        messages = [{"role": "system", "content": self._executor_system_prompt(session)}]
        messages.extend(self._history_messages(session, turn))
        messages.append({"role": "user", "content": user_text})
        max_rounds = max(1, int(self.config.max_tool_rounds))
        runtime_tool_context_factory = self._tool_context_factory

        import asyncio

        # A turn SUCCEEDS when the model converges to a final answer (a round with
        # no tool calls). Intermediate tool errors are normal — they are fed back
        # to the model, which recovers — so they must NOT by themselves fail the
        # turn (the old `any(tool_result.error)` rule marked a fully-answered turn
        # FAILED if a single recovered tool errored). A turn FAILS only if it never
        # converged (exhausted rounds) or the provider call itself failed.
        converged = False
        provider_error = ""
        # No-progress / anti-loop guard. We track the ORDERED sequence of tool-call
        # signatures and errors and look at the trailing CONSECUTIVE run — so a
        # genuinely-progressing turn that re-uses an idempotent read tool (git_status,
        # search, re-running tests around a fix) is NOT killed, only one that repeats
        # the SAME call (or hits the SAME error) many times in a row. Signatures use
        # the FULL args (no truncation) so distinct calls never collapse, and unknown
        # tool calls are counted too. (Observed 33 calls retrying a rejected path.)
        tool_seq: list[str] = []
        err_seq: list[str] = []
        nudged = False

        def _trailing_run(seq: list) -> int:
            if not seq:
                return 0
            last = seq[-1]
            n = 0
            for item in reversed(seq):
                if item == last:
                    n += 1
                else:
                    break
            return n

        def _sig(name: str, args: dict) -> str:
            try:
                return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
            except Exception:
                return f"{name}:{args!r}"

        for round_index in range(max_rounds):
            tool_schemas = _round_tool_schemas()
            request = ProviderRequest(
                session_id=session.session_id,
                turn_id=turn.id,
                role=active_role,
                messages=list(messages),
                tools=tool_schemas,
                model=self.provider_router.model_for_role(active_role),
                metadata={"round": round_index},
            )
            self._record_event(
                session,
                AgentEvent.make(
                    AgentEventType.MODEL_REQUEST_STARTED,
                    session_id=session.session_id,
                    turn_id=turn.id,
                    model=request.model,
                    role=request.role.value,
                    round=round_index,
                ),
            )
            try:
                response = provider.complete(request)
            except Exception as exc:  # provider/LLM failure (retries already exhausted in the client)
                provider_error = str(exc)
                self._record_event(
                    session,
                    AgentEvent.make(
                        AgentEventType.ERROR,
                        session_id=session.session_id,
                        turn_id=turn.id,
                        message=f"provider error: {provider_error}",
                        round=round_index,
                    ),
                )
                break
            for chunk in response.deltas:
                self._record_event(session, AgentEvent.make(AgentEventType.MODEL_DELTA, session_id=session.session_id, turn_id=turn.id, text=chunk, model=response.model, provider=response.provider, round=round_index))

            assistant = AssistantMessage(
                id=uuid.uuid4().hex,
                text=response.text,
                tool_calls=[ToolCall(id=tc.id, name=tc.name, args=dict(tc.args)) for tc in response.tool_calls],
                model=response.model,
                provider=response.provider,
                usage=dict(response.usage),
            )
            turn.assistant_messages.append(assistant)
            messages.append({"role": "assistant", "content": response.text, "tool_calls": [tc.to_dict() for tc in response.tool_calls]})

            if not response.tool_calls:
                converged = True
                break

            self._record_event(
                session,
                AgentEvent.make(
                    AgentEventType.TOOL_CALL_REQUESTED,
                    session_id=session.session_id,
                    turn_id=turn.id,
                    round=round_index,
                    tool_calls=[tc.to_dict() for tc in response.tool_calls],
                ),
            )

            for tc in response.tool_calls:
                handler = self.tool_registry.get(tc.name)
                if handler is None:
                    result = ToolResult(call_id=tc.id, name=tc.name, error=f"unknown tool {tc.name}")
                    turn.tool_results.append(result)
                    self._record_event(session, AgentEvent.make(AgentEventType.ERROR, session_id=session.session_id, turn_id=turn.id, tool_call_id=tc.id, message=result.error))
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result.error})
                    # Count unknown-tool calls too, so a model that hallucinates the
                    # same missing tool repeatedly is caught by the loop guard.
                    tool_seq.append(_sig(tc.name, tc.args))
                    err_seq.append(result.error)
                    continue
                context_factory = runtime_tool_context_factory
                if callable(runtime_tool_context_factory):
                    def _request_context_factory(req: ToolRequest) -> ToolContext:
                        return runtime_tool_context_factory(req, session=session, turn=turn, runtime=self)
                    context_factory = _request_context_factory
                else:
                    def _request_context_factory(req: ToolRequest) -> ToolContext:
                        return ToolContext(
                            name=req.handler.name,
                            args=dict(req.args or {}),
                            capabilities=req.handler.required_capabilities(),
                            workspace_root=session.cwd,
                            permission_mode=self.config.approval_profile,
                            turn_id=turn.id,
                            tool_call_id=req.call_id,
                            tool_registry=self.tool_registry,
                            activate_skills=self._make_activate_skills(session),
                        )
                    context_factory = _request_context_factory
                outcome = self.invoke_tool(
                    session,
                    tc.name,
                    dict(tc.args),
                    turn_id=turn.id,
                    call_id=tc.id,
                    turn=turn,
                    context_factory=context_factory,
                    record_request_event=False,
                )
                result = turn.tool_results[-1]
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result.result or result.error, ensure_ascii=False, default=str)})

                tool_seq.append(_sig(tc.name, tc.args))
                err_seq.append(str(result.error) if result.error else "")

            # No-progress = the SAME call (or the SAME error) repeating CONSECUTIVELY.
            # Interleaving distinct/idempotent calls resets the run, so legitimate
            # progress is never killed.
            # Consecutive identical tool calls, OR consecutive identical errors (a
            # success "" in err_seq breaks the error run).
            err_run = _trailing_run(err_seq) if (err_seq and err_seq[-1]) else 0
            stuck = max(_trailing_run(tool_seq), err_run)
            if stuck >= 6:
                self._record_event(
                    session,
                    AgentEvent.make(
                        AgentEventType.ERROR,
                        session_id=session.session_id,
                        turn_id=turn.id,
                        message=f"no-progress loop detected (same tool call/error repeated {stuck}x in a row); stopping turn",
                        round=round_index,
                    ),
                )
                break
            if stuck >= 3 and not nudged:
                # Nudge as a separate system message. Appending to the last
                # tool-result message would corrupt its JSON content; a trailing
                # system message preserves tool-call adjacency and is accepted by
                # OpenAI-compatible providers.
                nudged = True
                messages.append({
                    "role": "system",
                    "content": (
                        "You have repeated the same action with no progress. Stop repeating it: change your "
                        "approach (e.g. if a path was rejected, use one INSIDE the workspace root), or, if you "
                        "cannot proceed, write a final answer explaining what is blocking you."
                    ),
                })

        turn.completed_at = time.time()
        # Completed iff the model converged to a final answer and the provider did
        # not fail. Recovered tool errors do not fail the turn (they remain visible
        # as tool_call_completed events for audit / evolution feedback).
        turn_ok = converged and not provider_error
        turn.status = RuntimeStatus.COMPLETED.value if turn_ok else RuntimeStatus.FAILED.value
        session.state.active_turn_id = None
        session.state.status = RuntimeStatus.COMPLETED if turn.status == RuntimeStatus.COMPLETED.value else RuntimeStatus.FAILED
        self._record_event(session, AgentEvent.make(AgentEventType.USER_TURN_COMPLETED, session_id=session.session_id, turn_id=turn.id, status=turn.status))
        self.create_checkpoint(session, "completion" if turn.status == RuntimeStatus.COMPLETED.value else "error", summary=turn.status)
        self.save_session(session)
        return session

    def to_trajectory(self, session: AgentSession) -> list[RuntimeTrajectoryRecord]:
        records: list[RuntimeTrajectoryRecord] = []
        for seq, event in enumerate(session.events):
            records.append(RuntimeTrajectoryRecord(session_id=session.session_id, event=AgentEvent.from_dict(event), seq=seq))
        return records

    def _record_event(self, session: AgentSession, event: AgentEvent) -> None:
        payload = event.to_dict()
        payload["schema_version"] = session.schema_version
        session.events.append(payload)
        session.touch()
        self.bus.publish_nowait(event)

    def _checkpoint_summary(self, session: AgentSession, reason: str) -> str:
        if session.state.goal and session.state.goal.objective:
            return f"{reason}: {session.state.goal.objective[:160]}"
        if session.state.plan and session.state.plan.objective:
            return f"{reason}: {session.state.plan.objective[:160]}"
        return reason


def _title_from_text(text: str) -> str:
    compact = " ".join(str(text or "").split())
    return compact[:80] or "Interactive session"


def _tool_may_change_files(handler: Any, name: str) -> bool:
    if str(name or "") in {"file_write", "file_edit", "apply_patch"}:
        return True
    try:
        metadata = handler.spec().metadata()
    except Exception:
        metadata = {}
    action_classes = {str(item) for item in metadata.get("action_classes") or []}
    safety_class = str(metadata.get("safety_class") or "")
    workspace_scope = str(metadata.get("workspace_scope") or "")
    if workspace_scope not in {"workspace", "reports"}:
        return False
    return bool({"write", "self_modification"} & action_classes) or safety_class in {"write", "self_modification"}


__all__ = [
    "AgentRuntime",
    "AgentSession",
    "FakeProvider",
    "ProviderRouter",
    "SessionStore",
]
