"""Runtime harness for interactive CLI trajectories.

Harness tasks are versioned, synthetic, and invariant-driven. They are meant to
exercise the same ``InteractiveShell``/``AgentRuntime`` path used by humans,
while still letting tests provide fake providers and fake tools.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from biobank_agent.core.events import AgentEventType
from biobank_agent.core.tools.protocol import Capability, ToolContext, ToolSpec

from .audit import audit_session, redact_secrets
from .engine import FakeProvider, ProviderResponse
from .replay import replay_trajectory
from .types import ToolCall


HARNESS_SCHEMA_VERSION = 1


@dataclass
class HarnessStep:
    """One line fed into the interactive shell."""

    kind: str
    text: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarnessStep":
        return cls(kind=str(data.get("kind") or "message"), text=str(data.get("text") or ""))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HarnessExpectations:
    """Invariant checks for a harness task."""

    event_types: list[str] = field(default_factory=list)
    forbidden_event_types: list[str] = field(default_factory=list)
    min_turns: int = 0
    min_events: int = 0
    tool_calls: list[str] = field(default_factory=list)
    approvals: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    verification_statuses: list[str] = field(default_factory=list)
    repair_statuses: list[str] = field(default_factory=list)
    action_graph_node_types: list[str] = field(default_factory=list)
    checkpoint_reasons: list[str] = field(default_factory=list)
    provider_roles: list[str] = field(default_factory=list)
    file_change_tools: list[str] = field(default_factory=list)
    require_trajectory: bool = True
    require_session_saved: bool = True
    require_replay_ok: bool = True
    require_no_errors: bool = True
    forbidden_shortcuts: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarnessExpectations":
        raw = dict(data or {})
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in raw.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HarnessTask:
    """Versioned task spec for runtime harness execution."""

    task_id: str
    user_inputs: list[str] = field(default_factory=list)
    slash_commands: list[str] = field(default_factory=list)
    steps: list[HarnessStep] = field(default_factory=list)
    fake_provider_script: list[dict[str, Any]] = field(default_factory=list)
    fake_tool_responses: dict[str, dict[str, Any]] = field(default_factory=dict)
    expected: HarnessExpectations = field(default_factory=HarnessExpectations)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = HARNESS_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarnessTask":
        raw = dict(data or {})
        expected = HarnessExpectations.from_dict(dict(raw.get("expected") or raw.get("expectations") or {}))
        steps = [HarnessStep.from_dict(dict(item)) for item in raw.get("steps") or []]
        return cls(
            task_id=str(raw.get("task_id") or raw.get("id") or ""),
            user_inputs=[str(item) for item in raw.get("user_inputs") or []],
            slash_commands=[str(item) for item in raw.get("slash_commands") or []],
            steps=steps,
            fake_provider_script=[dict(item) for item in raw.get("fake_provider_script") or []],
            fake_tool_responses={str(k): dict(v) for k, v in (raw.get("fake_tool_responses") or {}).items()},
            expected=expected,
            metadata=dict(raw.get("metadata") or {}),
            schema_version=int(raw.get("schema_version") or HARNESS_SCHEMA_VERSION),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "HarnessTask":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expected"] = self.expected.to_dict()
        payload["steps"] = [step.to_dict() for step in self.steps]
        return payload

    def execution_lines(self) -> list[str]:
        if self.steps:
            return [step.text for step in self.steps if step.text.strip()]
        return [line for line in [*self.slash_commands, *self.user_inputs] if str(line).strip()]


@dataclass
class HarnessRunReport:
    """Serializable report from a harness run."""

    task_id: str
    status: str
    session_id: str
    trajectory_path: str
    session_path: str
    audit: dict[str, Any] = field(default_factory=dict)
    replay: dict[str, Any] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    schema_version: int = HARNESS_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return redact_secrets(asdict(self))


class StaticHarnessTool:
    """Deterministic fake tool registered only by harness tasks."""

    def __init__(
        self,
        name: str,
        result: dict[str, Any],
        *,
        capabilities: Iterable[str] | None = None,
        mutating: bool = False,
    ) -> None:
        self._name = name
        self._result = dict(result)
        self._caps = frozenset(_capability_from_string(item) for item in (capabilities or ["read_data"]))
        self._mutating = bool(mutating)

    @property
    def name(self) -> str:
        return self._name

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self._name, description="synthetic harness tool", parameters={})

    def required_capabilities(self) -> frozenset[Capability]:
        return self._caps

    @property
    def is_mutating(self) -> bool:
        return self._mutating

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        return dict(self._result)


class RuntimeHarnessRunner:
    """Run a harness task through an initialized interactive shell."""

    def __init__(self, shell: Any) -> None:
        self.shell = shell

    def run(self, task: HarnessTask, *, output_dir: str | Path | None = None) -> HarnessRunReport:
        if not task.task_id:
            raise ValueError("harness task_id is required")
        runtime = self.shell._require_runtime()
        session = self.shell._require_session()

        self._install_fake_provider(task)
        self._install_fake_tools(task)
        for line in task.execution_lines():
            self.shell.handle_line(line)
            session = self.shell._require_session()

        runtime.save_session(session)
        audit = audit_session(runtime, session)
        trajectory_path = runtime.session_store.rollout_file(session.session_id)
        replay = replay_trajectory(trajectory_path)
        violations = self._check_expectations(task, session, audit.to_dict(), replay.to_dict())
        status = "passed" if not violations else "failed"
        report = HarnessRunReport(
            task_id=task.task_id,
            status=status,
            session_id=session.session_id,
            trajectory_path=str(trajectory_path),
            session_path=str(runtime.session_store.session_file(session.session_id)),
            audit=audit.to_dict(),
            replay=replay.to_dict(),
            violations=violations,
        )
        if output_dir is not None:
            report.artifacts.update(write_harness_report(report, output_dir))
        return report

    def _install_fake_provider(self, task: HarnessTask) -> None:
        if not task.fake_provider_script:
            return
        runtime = self.shell._require_runtime()
        responses = []
        for raw in task.fake_provider_script:
            tool_calls = [
                ToolCall(id=str(item.get("id") or ""), name=str(item.get("name") or ""), args=dict(item.get("args") or {}))
                for item in raw.get("tool_calls") or []
            ]
            responses.append(
                ProviderResponse(
                    text=str(raw.get("text") or ""),
                    tool_calls=tool_calls,
                    deltas=[str(item) for item in raw.get("deltas") or []],
                    usage=dict(raw.get("usage") or {}),
                    provider=str(raw.get("provider") or "harness"),
                    model=str(raw.get("model") or runtime.config.primary_model),
                    finish_reason=str(raw.get("finish_reason") or "stop"),
                )
            )
        model = responses[0].model or runtime.config.primary_model
        fake = FakeProvider(scripted_responses=responses, model=model)
        for configured_model in {
            runtime.config.primary_model,
            runtime.config.planner_model,
            runtime.config.critic_model,
            runtime.config.summarizer_model,
            runtime.config.safety_reviewer_model,
            model,
        }:
            runtime.provider_router.providers[configured_model] = fake

    def _install_fake_tools(self, task: HarnessTask) -> None:
        if not task.fake_tool_responses:
            return
        runtime = self.shell._require_runtime()
        for name, spec in task.fake_tool_responses.items():
            result = dict(spec.get("result") or spec)
            caps = spec.get("capabilities") or ["read_data"]
            mutating = bool(spec.get("mutating", False))
            runtime.tool_registry.register(StaticHarnessTool(name, result, capabilities=caps, mutating=mutating))

    def _check_expectations(
        self,
        task: HarnessTask,
        session: Any,
        audit: dict[str, Any],
        replay: dict[str, Any],
    ) -> list[str]:
        expected = task.expected
        violations: list[str] = []
        event_types = [str(event.get("type") or "") for event in session.events]
        event_type_set = set(event_types)
        tool_names = {
            str(item.get("tool") or "")
            for item in audit.get("tool_calls") or []
            if item.get("tool")
        }
        approval_tools = {str(item.get("tool_name") or "") for item in audit.get("approvals") or []}
        approval_caps = {str(item.get("capability") or "") for item in audit.get("approvals") or []}
        graph_types = set((audit.get("action_graph") or {}).get("node_types") or {})
        checkpoint_reasons = {str(item.get("reason") or "") for item in audit.get("checkpoints") or []}
        provider_roles = {str(item.get("role") or "") for item in (audit.get("provider_roles") or {}).get("model_requests") or []}

        if len(session.turns) < expected.min_turns:
            violations.append(f"expected at least {expected.min_turns} turn(s), found {len(session.turns)}")
        if len(session.events) < expected.min_events:
            violations.append(f"expected at least {expected.min_events} event(s), found {len(session.events)}")
        for event_type in expected.event_types:
            if event_type not in event_type_set:
                violations.append(f"missing event type: {event_type}")
        for event_type in expected.forbidden_event_types:
            if event_type in event_type_set:
                violations.append(f"forbidden event type observed: {event_type}")
        for tool in expected.tool_calls:
            if tool not in tool_names:
                violations.append(f"missing tool call: {tool}")
        for tool in expected.approvals:
            if tool not in approval_tools:
                violations.append(f"missing approval for tool: {tool}")
        for cap in expected.capabilities:
            if cap not in approval_caps:
                violations.append(f"missing approval capability: {cap}")
        for node_type in expected.action_graph_node_types:
            if node_type not in graph_types:
                violations.append(f"missing action graph node type: {node_type}")
        if expected.file_change_tools:
            changed_tools = {
                str(item.get("tool") or "")
                for item in audit.get("tool_calls") or []
                if str(item.get("tool") or "") in {"file_write", "file_edit", "apply_patch"}
            }
            for tool in expected.file_change_tools:
                if tool not in changed_tools:
                    violations.append(f"missing file-change tool evidence: {tool}")
        for reason in expected.checkpoint_reasons:
            if reason not in checkpoint_reasons:
                violations.append(f"missing checkpoint reason: {reason}")
        for role in expected.provider_roles:
            if role not in provider_roles:
                violations.append(f"missing provider role route: {role}")
        if expected.require_trajectory and not Path(audit.get("task_summary", {}).get("trajectory_path", "")).exists():
            violations.append("trajectory file missing")
        if expected.require_session_saved and not Path(audit.get("task_summary", {}).get("session_file", "")).exists():
            violations.append("session file missing")
        if expected.require_replay_ok and replay.get("status") != "ok":
            violations.append(f"replay status is {replay.get('status')}: {replay.get('violations')}")
        if expected.require_no_errors and AgentEventType.ERROR.value in event_type_set:
            violations.append("runtime error event observed")
        for shortcut in expected.forbidden_shortcuts:
            if _production_contains(shortcut):
                violations.append(f"forbidden shortcut appears in production code: {shortcut}")
        return violations


def write_harness_report(report: HarnessRunReport, output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{_safe_filename(report.task_id)}.harness.json"
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"json": str(path)}


def _capability_from_string(value: str) -> Capability:
    try:
        return Capability(str(value))
    except Exception:
        return Capability.READ_DATA


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "harness")).strip("_") or "harness"


def _production_contains(token: str) -> bool:
    text = str(token or "")
    if not text:
        return False
    root = Path("biobank_agent")
    if not root.exists():
        return False
    for path in root.rglob("*.py"):
        try:
            if text in path.read_text(encoding="utf-8"):
                return True
        except OSError:
            continue
    return False


__all__ = [
    "HARNESS_SCHEMA_VERSION",
    "HarnessStep",
    "HarnessExpectations",
    "HarnessTask",
    "HarnessRunReport",
    "RuntimeHarnessRunner",
    "StaticHarnessTool",
    "write_harness_report",
]
