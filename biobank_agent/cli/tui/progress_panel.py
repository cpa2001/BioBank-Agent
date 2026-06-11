"""Plan progress panel for the Textual TUI."""

from __future__ import annotations

import time

from biobank_agent.core.events import AgentEvent, AgentEventType

try:  # pragma: no cover - optional UI dependency
    from textual.widgets import Static
except Exception:  # pragma: no cover
    Static = object  # type: ignore[assignment]


class ProgressPanel(Static):  # type: ignore[misc]
    DEFAULT_CSS = "ProgressPanel { border: solid $accent; height: 1fr; }"

    def __init__(self, *args, max_lines: int = 12, **kwargs) -> None:
        try:
            super().__init__(*args, **kwargs)
        except TypeError:  # object fallback when Textual is unavailable
            super().__init__()
        self.max_lines = max_lines
        self.lines: list[str] = []
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.run_status = "idle"
        self.current_activity = "No active plan."
        self.phase_state: dict[str, dict[str, str | float]] = {}
        self.active_tools: dict[str, float] = {}
        self.tool_done = 0
        self.tool_failed = 0

    def on_mount(self) -> None:
        self._safe_update(self.render_text())

    def apply_event(self, event: AgentEvent) -> None:
        """Update progress state from an AgentEvent."""
        line = ""
        if event.type == AgentEventType.PLAN_PHASE:
            p = event.payload
            phase = str(p.get("phase", "Plan"))
            actor = str(p.get("actor", "biobank"))
            status = str(p.get("status", "running"))
            message = str(p.get("message", ""))
            line = f"{phase} | {actor} | {status} | {message}".strip()
            self._mark_started(event)
            self.run_status = self._status_from_event(status)
            self.current_activity = f"{phase}: {message or status}"
            self.phase_state[phase] = {
                "actor": actor,
                "status": status,
                "message": message,
                "ts": event.ts,
            }
        elif event.type == AgentEventType.TURN_STARTED:
            self.started_at = event.ts
            self.finished_at = None
            self.run_status = "running"
            self.current_activity = "Turn started"
            self.phase_state.clear()
            self.active_tools.clear()
            self.tool_done = 0
            self.tool_failed = 0
            line = "Turn | biobank | running | started"
        elif event.type == AgentEventType.TURN_FINISHED:
            self.finished_at = event.ts
            self.run_status = "complete"
            self.current_activity = "Turn finished"
            line = "Turn | biobank | success | finished"
        elif event.type == AgentEventType.CANCELLED:
            self.finished_at = event.ts
            self.run_status = "cancelled"
            self.current_activity = "Run cancelled"
            line = "Runtime | biobank | cancelled | stopped"
        elif event.type == AgentEventType.ERROR:
            self.finished_at = event.ts
            self.run_status = "failed"
            message = str(event.payload.get("message", "error"))
            self.current_activity = message
            line = f"Runtime | biobank | failed | {message}"
        elif event.type == AgentEventType.TOOL_STARTED:
            skill = self._skill(event)
            self._mark_started(event)
            self.run_status = "running"
            self.active_tools[skill] = event.ts
            self.current_activity = f"Running tool: {skill}"
            line = f"Execution | {skill} | running | started"
        elif event.type == AgentEventType.TOOL_PROGRESS:
            skill = self._skill(event)
            message = str(event.payload.get("message") or event.payload.get("state") or "running")
            self._mark_started(event)
            self.current_activity = f"{skill}: {message}"
            line = f"Execution | {skill} | running | {message}"
        elif event.type == AgentEventType.TOOL_RESULT:
            skill = self._skill(event)
            self.active_tools.pop(skill, None)
            self.tool_done += 1
            self.current_activity = f"Tool completed: {skill}"
            line = f"Execution | {skill} | success | done"
        elif event.type == AgentEventType.TOOL_ERROR:
            skill = self._skill(event)
            self.active_tools.pop(skill, None)
            self.tool_failed += 1
            self.run_status = "failed"
            error = str(event.payload.get("error", "tool failed"))
            self.current_activity = f"{skill} failed"
            line = f"Execution | {skill} | failed | {error}"
        elif event.type == AgentEventType.ORCHESTRATION_STRATEGY:
            strategy = str(event.payload.get("strategy") or event.payload.get("mode") or "orchestration")
            self._mark_started(event)
            self.current_activity = f"Orchestration: {strategy}"
            line = f"Orchestration | biobank | running | {strategy}"
        elif event.type == AgentEventType.SCHEMA_VIOLATION:
            reason = str(event.payload.get("reason", "schema violation"))
            self.run_status = "failed"
            self.current_activity = reason
            line = f"Validation | study_spec | failed | {reason}"
        if not line:
            return
        self.lines.append(line)
        self.lines = self.lines[-self.max_lines:]
        self._safe_update(self.render_text())

    def render_text(self) -> str:
        elapsed = self._elapsed_seconds()
        active = ", ".join(sorted(self.active_tools)) if self.active_tools else "none"
        header = [
            "Run dashboard",
            (
                f"Status: {self.run_status} | elapsed: {elapsed}s | "
                f"tools: {self.tool_done} done, {self.tool_failed} failed, "
                f"{len(self.active_tools)} active"
            ),
            f"Current: {self.current_activity}",
            f"Active tools: {active}",
        ]
        phases = self._render_phases()
        recent = self.lines[-self.max_lines:]
        if not phases and not recent:
            return "\n".join(header + ["", "Recent", "- No active plan."])
        out = header
        if phases:
            out.extend(["", "Phases", *phases])
        if recent:
            out.extend(["", "Recent", *recent])
        return "\n".join(out)

    def _mark_started(self, event: AgentEvent) -> None:
        if self.started_at is None:
            self.started_at = event.ts
        self.finished_at = None

    @staticmethod
    def _status_from_event(status: str) -> str:
        normalized = str(status or "").strip().lower()
        if normalized in {"success", "done", "complete", "completed"}:
            return "running"
        if normalized in {"failed", "error"}:
            return "failed"
        if normalized in {"paused", "blocked"}:
            return "paused"
        return "running"

    @staticmethod
    def _skill(event: AgentEvent) -> str:
        return str(event.payload.get("skill") or event.payload.get("actor") or event.payload.get("phase") or "tool")

    def _elapsed_seconds(self) -> int:
        if self.started_at is None:
            return 0
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0, int(end - self.started_at))

    def _render_phases(self) -> list[str]:
        rows: list[str] = []
        for phase, state in self.phase_state.items():
            actor = str(state.get("actor", "biobank"))
            status = str(state.get("status", "running"))
            message = str(state.get("message", ""))
            rows.append(f"- {phase}: {status} ({actor}) {message}".rstrip())
        return rows[-6:]

    def _safe_update(self, value: str) -> None:
        update = getattr(self, "update", None)
        if callable(update):
            update(value)


__all__ = ["ProgressPanel"]
