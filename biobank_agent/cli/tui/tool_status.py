"""Active tool status panel for the Textual TUI."""

from __future__ import annotations

from biobank_agent.core.events import AgentEvent, AgentEventType

try:  # pragma: no cover - optional UI dependency
    from textual.widgets import Static
except Exception:  # pragma: no cover
    Static = object  # type: ignore[assignment]


class ToolStatusPanel(Static):  # type: ignore[misc]
    DEFAULT_CSS = "ToolStatusPanel { border: solid $accent; height: 1fr; }"

    def __init__(self, *args, max_recent: int = 10, **kwargs) -> None:
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            super().__init__()
        self.max_recent = max_recent
        self.current: dict[str, str] = {}
        self.recent: list[str] = []

    def on_mount(self) -> None:
        self._safe_update(self.render_text())

    def apply_event(self, event: AgentEvent) -> None:
        """Update active-tool status from an AgentEvent."""
        if event.type not in {
            AgentEventType.TOOL_STARTED,
            AgentEventType.TOOL_PROGRESS,
            AgentEventType.TOOL_RESULT,
            AgentEventType.TOOL_ERROR,
            AgentEventType.TOOL_CONFIRMATION_REQ,
            AgentEventType.TOOL_CONFIRMATION_DONE,
            AgentEventType.SCHEMA_VIOLATION,
        }:
            return
        skill = str(event.payload.get("skill", event.payload.get("phase", "tool")))
        if event.type == AgentEventType.TOOL_STARTED:
            detail = "started"
            self.current[skill] = detail
        elif event.type == AgentEventType.TOOL_PROGRESS:
            detail = str(event.payload.get("message") or event.payload.get("state") or "running")
            self.current[skill] = detail
        elif event.type == AgentEventType.TOOL_CONFIRMATION_REQ:
            detail = "awaiting approval"
            self.current[skill] = detail
        elif event.type == AgentEventType.TOOL_CONFIRMATION_DONE:
            detail = "approval granted" if event.payload.get("confirmed") else "approval denied"
            self.current[skill] = detail
        elif event.type == AgentEventType.TOOL_RESULT:
            detail = "done"
            self.current.pop(skill, None)
        elif event.type == AgentEventType.SCHEMA_VIOLATION:
            detail = "schema violation: " + str(event.payload.get("reason", "invalid args"))
            self.current.pop(skill, None)
        else:
            detail = "failed: " + str(event.payload.get("error", "tool failed"))
            self.current.pop(skill, None)
        self.recent.append(f"{skill}: {detail}")
        self.recent = self.recent[-self.max_recent:]
        self._safe_update(self.render_text())

    def render_text(self) -> str:
        active = [f"- {skill}: {detail}" for skill, detail in sorted(self.current.items())]
        recent = [f"- {line}" for line in self.recent[-self.max_recent:]]
        parts = ["Current tool"]
        parts.extend(active or ["Idle."])
        if recent:
            parts.extend(["", "Recent", *recent])
        return "\n".join(parts)

    def _safe_update(self, value: str) -> None:
        update = getattr(self, "update", None)
        if callable(update):
            update(value)


__all__ = ["ToolStatusPanel"]
