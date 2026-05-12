"""Rich renderer for ``AgentEvent`` streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from biobank_agent.core.events import AgentEvent, AgentEventType


@dataclass
class StreamingRenderer:
    """Small stateful renderer for CLI streaming events.

    This is intentionally independent from the legacy REPL so the same renderer
    can be reused by future command modules or SDK examples.
    """

    console: Console = field(default_factory=Console)
    max_recent: int = 8
    recent: list[tuple[str, str, str]] = field(default_factory=list)
    text: str = ""

    def handle(self, event: AgentEvent) -> None:
        """Render one event."""
        if event.type == AgentEventType.MESSAGE_DELTA:
            delta = str(event.payload.get("delta", ""))
            self.text += delta
            self.console.print(delta, end="")
            return
        if event.type == AgentEventType.MESSAGE_COMPLETE:
            final = str(event.payload.get("text", ""))
            if final and not self.text:
                self.console.print(final)
            self.text = final or self.text
            return
        if event.type in {
            AgentEventType.TOOL_STARTED,
            AgentEventType.TOOL_PROGRESS,
            AgentEventType.TOOL_RESULT,
            AgentEventType.TOOL_ERROR,
            AgentEventType.SCHEMA_VIOLATION,
            AgentEventType.REPRODUCIBILITY_CHECKPOINT,
            AgentEventType.EVIDENCE_LINKED,
        }:
            self._record(event)
            self.console.print(self.snapshot())

    def _record(self, event: AgentEvent) -> None:
        skill = str(event.payload.get("skill", event.payload.get("phase", "event")))
        if event.type == AgentEventType.TOOL_ERROR:
            status = "failed"
            detail = str(event.payload.get("error", "tool failed"))
        elif event.type == AgentEventType.TOOL_RESULT:
            status = "done"
            detail = _compact(event.payload.get("summary", {}))
        elif event.type == AgentEventType.TOOL_PROGRESS:
            status = "running"
            detail = str(event.payload.get("state", "executing"))
        else:
            status = event.type.value
            detail = _compact(event.payload)
        self.recent.append((skill, status, detail))
        self.recent = self.recent[-self.max_recent:]

    def snapshot(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_column(ratio=3)
        for skill, status, detail in self.recent:
            table.add_row(skill, status, detail)
        return Panel(table, title="Agent Stream")


def _compact(value: Any, limit: int = 160) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


__all__ = ["StreamingRenderer"]
