"""Concurrent worker and review-state panel for the Textual TUI."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from biobank_agent.core.events import AgentEvent, AgentEventType

try:  # pragma: no cover - optional UI dependency
    from textual.widgets import Static
except Exception:  # pragma: no cover
    Static = object  # type: ignore[assignment]


@dataclass
class WorkerState:
    """UI-facing state for one active worker, external planner, or reviewer."""

    worker_id: str
    label: str
    status: str = "pending"
    phase: str = ""
    actor: str = ""
    message: str = ""
    active_tool: str = ""
    report_dir: str = ""
    transcript: str = ""
    started_at: float | None = None
    updated_at: float = field(default_factory=time.time)
    done_tools: int = 0
    failed_tools: int = 0
    choices: list[dict[str, Any]] = field(default_factory=list)


class WorkerReviewPanel(Static):  # type: ignore[misc]
    """Browse concurrent workers and plan-review actions without log scraping."""

    DEFAULT_CSS = "WorkerReviewPanel { border: solid $accent; height: 1fr; }"

    def __init__(self, *args, max_workers: int = 10, **kwargs) -> None:
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            super().__init__()
        self.max_workers = max_workers
        self.worker_states: dict[str, WorkerState] = {}
        self.worker_order: list[str] = []
        self.selected_index = 0
        self.shortcut_log: list[str] = []

    def on_mount(self) -> None:
        self._safe_update(self.render_text())

    def apply_event(self, event: AgentEvent) -> None:
        if event.type == AgentEventType.TURN_STARTED:
            worker = self._ensure_worker("main", "main", event.ts)
            worker.status = "running"
            worker.phase = "Turn"
            worker.message = "started"
            worker.started_at = event.ts
            worker.updated_at = event.ts
        elif event.type == AgentEventType.TURN_FINISHED:
            worker = self._ensure_worker("main", "main", event.ts)
            worker.status = "complete"
            worker.phase = "Turn"
            worker.message = "finished"
            worker.updated_at = event.ts
        elif event.type == AgentEventType.PLAN_PHASE:
            self._apply_plan_phase(event)
        elif event.type in {
            AgentEventType.TOOL_STARTED,
            AgentEventType.TOOL_PROGRESS,
            AgentEventType.TOOL_RESULT,
            AgentEventType.TOOL_ERROR,
        }:
            self._apply_tool_event(event)
        elif event.type == AgentEventType.TOOL_CONFIRMATION_REQ:
            worker = self._ensure_worker("main", "main", event.ts)
            worker.status = "awaiting approval"
            worker.message = str(event.payload.get("reason") or event.payload.get("skill") or "tool confirmation")
            worker.updated_at = event.ts
        elif event.type == AgentEventType.TOOL_CONFIRMATION_DONE:
            worker = self._ensure_worker("main", "main", event.ts)
            worker.status = "approved" if event.payload.get("confirmed") else "rejected"
            worker.message = str(event.payload.get("skill") or "tool confirmation")
            worker.updated_at = event.ts
        else:
            return
        self._trim()
        self._safe_update(self.render_text())

    def record_shortcut(self, command: str, status: str = "sent") -> None:
        """Record a keyboard-triggered command for operator feedback."""
        self.shortcut_log.append(f"{command}: {status}")
        self.shortcut_log = self.shortcut_log[-10:]
        self._safe_update(self.render_text())

    def select_next(self) -> None:
        if self.worker_order:
            self.selected_index = (self.selected_index + 1) % len(self.worker_order)
        self._safe_update(self.render_text())

    def select_previous(self) -> None:
        if self.worker_order:
            self.selected_index = (self.selected_index - 1) % len(self.worker_order)
        self._safe_update(self.render_text())

    def selected_worker(self) -> WorkerState | None:
        if not self.worker_order:
            return None
        self.selected_index = max(0, min(self.selected_index, len(self.worker_order) - 1))
        return self.worker_states.get(self.worker_order[self.selected_index])

    def render_text(self) -> str:
        rows = ["Worker/review queue"]
        if not self.worker_order:
            rows.append("- no active workers yet")
            if self.shortcut_log:
                rows.extend(["", "Shortcut log"])
                rows.extend(f"- {entry}" for entry in self.shortcut_log[-10:])
            rows.extend(["", "Shortcuts: F5 approve, F6 resume, F7 repair A, F8 review hooks, F9 abort, Ctrl+J/K select"])
            return "\n".join(rows)

        for idx, worker_id in enumerate(self.worker_order[: self.max_workers]):
            worker = self.worker_states[worker_id]
            marker = ">" if idx == self.selected_index else " "
            elapsed = self._elapsed(worker)
            label = worker.label or worker.worker_id
            active = f" tool={worker.active_tool}" if worker.active_tool else ""
            rows.append(
                f"{marker} {label}: {worker.status} {elapsed}s | {worker.phase or '-'} | "
                f"{worker.message[:90]}{active}"
            )

        selected = self.selected_worker()
        if selected is not None:
            rows.extend(["", "Selected"])
            rows.append(f"- id: {selected.worker_id}")
            rows.append(f"- actor: {selected.actor or selected.label}")
            if selected.report_dir:
                rows.append(f"- report: {selected.report_dir}")
            if selected.transcript:
                rows.append(f"- transcript: {selected.transcript}")
            if selected.choices:
                rows.append("- choices:")
                for choice in selected.choices[:5]:
                    key = str(choice.get("key") or "?")
                    title = str(choice.get("title") or "")
                    rows.append(f"  {key}: {title}")
        if self.shortcut_log:
            rows.extend(["", "Shortcut log"])
            rows.extend(f"- {entry}" for entry in self.shortcut_log[-10:])
        rows.extend(["", "Shortcuts: F5 approve, F6 resume, F7 repair A, F8 review hooks, F9 abort, Ctrl+J/K select"])
        return "\n".join(rows)

    def _apply_plan_phase(self, event: AgentEvent) -> None:
        payload = event.payload
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        phase = str(payload.get("phase") or "")
        actor = str(payload.get("actor") or "biobank")
        worker_id = self._worker_id(payload, metadata, phase, actor)
        label = str(metadata.get("case") or metadata.get("label") or actor or worker_id)
        worker = self._ensure_worker(worker_id, label, event.ts)
        worker.status = str(payload.get("status") or "running")
        worker.phase = phase
        worker.actor = actor
        worker.message = str(payload.get("message") or "")
        worker.updated_at = event.ts
        worker.report_dir = str(metadata.get("report_dir") or worker.report_dir)
        worker.transcript = str(metadata.get("transcript") or worker.transcript)
        choices = metadata.get("choices")
        if isinstance(choices, list):
            worker.choices = [dict(item) for item in choices if isinstance(item, dict)]

    def _apply_tool_event(self, event: AgentEvent) -> None:
        metadata = event.payload.get("metadata") if isinstance(event.payload.get("metadata"), dict) else {}
        worker_id = str(metadata.get("worker_id") or metadata.get("worker") or "main")
        label = str(metadata.get("case") or metadata.get("label") or worker_id)
        skill = str(event.payload.get("skill") or event.payload.get("actor") or "tool")
        worker = self._ensure_worker(worker_id, label, event.ts)
        worker.phase = "Execution"
        worker.actor = skill
        worker.updated_at = event.ts
        if worker.started_at is None:
            worker.started_at = event.ts
        if event.type == AgentEventType.TOOL_STARTED:
            worker.status = "running"
            worker.active_tool = skill
            worker.message = "started"
        elif event.type == AgentEventType.TOOL_PROGRESS:
            worker.status = "running"
            worker.active_tool = skill
            worker.message = str(event.payload.get("message") or event.payload.get("state") or "running")
        elif event.type == AgentEventType.TOOL_RESULT:
            worker.status = "running"
            worker.done_tools += 1
            worker.active_tool = ""
            worker.message = f"{skill} done"
        elif event.type == AgentEventType.TOOL_ERROR:
            worker.status = "failed"
            worker.failed_tools += 1
            worker.active_tool = ""
            worker.message = str(event.payload.get("error") or f"{skill} failed")

    def _ensure_worker(self, worker_id: str, label: str, ts: float) -> WorkerState:
        worker_id = str(worker_id or "main")
        if worker_id not in self.worker_states:
            self.worker_states[worker_id] = WorkerState(worker_id=worker_id, label=str(label or worker_id), started_at=ts, updated_at=ts)
            self.worker_order.append(worker_id)
        worker = self.worker_states[worker_id]
        if label and worker.label == worker.worker_id:
            worker.label = str(label)
        return worker

    @staticmethod
    def _worker_id(payload: dict[str, Any], metadata: dict[str, Any], phase: str, actor: str) -> str:
        explicit = metadata.get("worker_id") or metadata.get("worker") or payload.get("worker_id")
        if explicit:
            return str(explicit)
        if phase.lower() in {"external council", "review hooks"} and actor:
            return str(actor)
        if actor and actor.lower() not in {"biobank", "agent"} and phase.lower() in {"planning", "merge", "external council"}:
            return str(actor)
        return "main"

    def _trim(self) -> None:
        if len(self.worker_order) <= self.max_workers:
            return
        keep = set(self.worker_order[-self.max_workers:])
        self.worker_order = [worker_id for worker_id in self.worker_order if worker_id in keep]
        self.worker_states = {worker_id: worker for worker_id, worker in self.worker_states.items() if worker_id in keep}
        self.selected_index = min(self.selected_index, max(0, len(self.worker_order) - 1))

    @staticmethod
    def _elapsed(worker: WorkerState) -> int:
        if worker.started_at is None:
            return 0
        return max(0, int((worker.updated_at or time.time()) - worker.started_at))

    def _safe_update(self, value: str) -> None:
        update = getattr(self, "update", None)
        if callable(update):
            update(value)


__all__ = ["WorkerReviewPanel", "WorkerState"]
