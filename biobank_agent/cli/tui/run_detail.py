"""Run-detail panel for plan repair choices and output artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from biobank_agent.core.events import AgentEvent, AgentEventType

try:  # pragma: no cover - optional UI dependency
    from textual.widgets import Static
except Exception:  # pragma: no cover
    Static = object  # type: ignore[assignment]


class RunDetailPanel(Static):  # type: ignore[misc]
    """Show the parts of a long run that need operator attention."""

    DEFAULT_CSS = "RunDetailPanel { border: solid $accent; height: 1fr; }"

    def __init__(self, *args, max_items: int = 8, **kwargs) -> None:
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            super().__init__()
        self.max_items = max_items
        self.pause_reason = ""
        self.repair_choices: list[dict[str, Any]] = []
        self.repair_log: list[str] = []
        self.external_status: dict[str, str] = {}
        self.report_dir = ""
        self.report_artifacts: list[str] = []
        self.review_hooks: list[str] = []

    def on_mount(self) -> None:
        self._safe_update(self.render_text())

    def apply_event(self, event: AgentEvent) -> None:
        if event.type == AgentEventType.PLAN_PHASE:
            self._apply_plan_phase(event)
        elif event.type == AgentEventType.ORCHESTRATION_STRATEGY:
            strategy = str(event.payload.get("strategy") or event.payload.get("mode") or "active")
            self.external_status["orchestration"] = strategy
        elif event.type == AgentEventType.TOOL_CONFIRMATION_REQ:
            skill = str(event.payload.get("skill") or "tool")
            self.repair_log.append(f"{skill}: awaiting approval")
        elif event.type == AgentEventType.TOOL_CONFIRMATION_DONE:
            skill = str(event.payload.get("skill") or "tool")
            status = "approved" if event.payload.get("confirmed") else "rejected"
            self.repair_log.append(f"{skill}: approval {status}")
        else:
            return
        self.repair_log = self.repair_log[-self.max_items:]
        self._safe_update(self.render_text())

    def _apply_plan_phase(self, event: AgentEvent) -> None:
        phase = str(event.payload.get("phase") or "")
        actor = str(event.payload.get("actor") or "biobank")
        status = str(event.payload.get("status") or "")
        message = str(event.payload.get("message") or "")
        metadata = event.payload.get("metadata") if isinstance(event.payload.get("metadata"), dict) else {}

        if phase.lower() == "external council" or actor.lower() in {"codex", "claude", "claude_code", "claude-code"}:
            self.external_status[actor] = f"{status}: {message}".strip(": ")
        if phase.lower() in {"repair", "validation"}:
            if metadata.get("pause_reason"):
                self.pause_reason = str(metadata.get("pause_reason"))
            elif status.lower() in {"failed", "paused", "warning"} and message:
                self.pause_reason = message
            if message:
                self.repair_log.append(f"{actor}: {status} - {message}")
        if metadata.get("choices"):
            choices = metadata.get("choices")
            if isinstance(choices, list):
                self.repair_choices = [dict(item) for item in choices if isinstance(item, dict)]
        if metadata.get("report_dir"):
            self.report_dir = str(metadata.get("report_dir"))
        if metadata.get("report_artifacts"):
            artifacts = metadata.get("report_artifacts")
            if isinstance(artifacts, list):
                self.report_artifacts = [str(item) for item in artifacts]
        if phase.lower() == "review hooks":
            self.review_hooks.append(f"{actor}: {status} - {message}")

    def render_text(self) -> str:
        parts = ["Run details"]
        parts.extend(self._render_external())
        parts.extend(self._render_pause())
        parts.extend(self._render_choices())
        parts.extend(self._render_report_artifacts())
        parts.extend(self._render_review_hooks())
        if len(parts) == 1:
            parts.append("No run details yet.")
        return "\n".join(parts)

    def _render_external(self) -> list[str]:
        if not self.external_status:
            return []
        rows = ["", "External/subagents"]
        rows.extend(f"- {name}: {status}" for name, status in sorted(self.external_status.items()))
        return rows[-(self.max_items + 2):]

    def _render_pause(self) -> list[str]:
        if not self.pause_reason and not self.repair_log:
            return []
        rows = ["", "Repair state"]
        if self.pause_reason:
            rows.append(f"- Pause: {self.pause_reason}")
        rows.extend(f"- {line}" for line in self.repair_log[-self.max_items:])
        return rows

    def _render_choices(self) -> list[str]:
        if not self.repair_choices:
            return []
        rows = ["", "Choices"]
        for item in self.repair_choices[: self.max_items]:
            key = str(item.get("key") or "?")
            title = str(item.get("title") or "")
            detail = str(item.get("detail") or "")
            rows.append(f"- {key}: {title}")
            if detail:
                rows.append(f"  {detail}")
        return rows

    def _render_report_artifacts(self) -> list[str]:
        if not self.report_dir and not self.report_artifacts:
            return []
        rows = ["", "Report artifacts"]
        if self.report_dir:
            rows.append(f"- dir: {self.report_dir}")
        for artifact in self.report_artifacts[: self.max_items]:
            rows.append(f"- {Path(artifact).name}")
        if len(self.report_artifacts) > self.max_items:
            rows.append(f"- ... {len(self.report_artifacts) - self.max_items} more")
        return rows

    def _render_review_hooks(self) -> list[str]:
        if not self.review_hooks:
            return []
        rows = ["", "Review hooks"]
        rows.extend(f"- {line}" for line in self.review_hooks[-self.max_items:])
        return rows

    def _safe_update(self, value: str) -> None:
        update = getattr(self, "update", None)
        if callable(update):
            update(value)


__all__ = ["RunDetailPanel"]
