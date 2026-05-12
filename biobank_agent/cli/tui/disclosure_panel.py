"""Disclosure and governance panel for the Textual TUI."""

from __future__ import annotations

from biobank_agent.core.events import AgentEvent, AgentEventType

try:  # pragma: no cover - optional UI dependency
    from textual.widgets import Static
except Exception:  # pragma: no cover
    Static = object  # type: ignore[assignment]


class DisclosurePanel(Static):  # type: ignore[misc]
    DEFAULT_CSS = "DisclosurePanel { border: solid $accent; height: 1fr; }"

    def __init__(self, *args, max_lines: int = 10, **kwargs) -> None:
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            super().__init__()
        self.max_lines = max_lines
        self.lines: list[str] = ["Internal de-identified mode."]

    def on_mount(self) -> None:
        self._safe_update(self.render_text())

    def apply_event(self, event: AgentEvent) -> None:
        """Update disclosure/governance state from an AgentEvent."""
        line = ""
        if event.type == AgentEventType.DISCLOSURE_LAYER:
            status = event.payload.get("status") or event.payload.get("layer") or "disclosure"
            detail = event.payload.get("message") or event.payload.get("summary") or event.payload
            line = f"{status}: {detail}"
        elif event.type == AgentEventType.REPRODUCIBILITY_CHECKPOINT:
            line = "checkpoint: " + str(event.payload.get("checkpoint_id") or event.payload.get("path") or "recorded")
        elif event.type == AgentEventType.EVIDENCE_LINKED:
            line = "evidence: " + str(event.payload.get("claim_id") or event.payload.get("summary") or "linked")
        elif event.type == AgentEventType.CONTEXT_WINDOW_WARNING:
            line = "context warning: " + str(event.payload.get("message") or event.payload)
        elif event.type == AgentEventType.SCHEMA_VIOLATION:
            line = "schema gate: " + str(event.payload.get("reason", "invalid plan/tool args"))
        if not line:
            return
        self.lines.append(line)
        self.lines = self.lines[-self.max_lines:]
        self._safe_update(self.render_text())

    def render_text(self) -> str:
        return "Disclosure\n" + "\n".join(f"- {line}" for line in self.lines)

    def _safe_update(self, value: str) -> None:
        update = getattr(self, "update", None)
        if callable(update):
            update(value)


__all__ = ["DisclosurePanel"]
