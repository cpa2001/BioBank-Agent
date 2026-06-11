"""Textual main screen for Biobank Agent."""

from __future__ import annotations

import asyncio
import inspect
from io import StringIO
from typing import Any, Callable

from biobank_agent.core.events import AgentEvent

try:  # pragma: no cover - exercised only when textual is installed
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Footer, Header, Input, RichLog
except Exception:  # pragma: no cover
    App = object  # type: ignore[assignment]
    ComposeResult = object  # type: ignore[assignment]
    Horizontal = Vertical = Header = Footer = Input = RichLog = None  # type: ignore[assignment]


CommandHandler = Callable[[str], Any]


class _StreamingLogFile:
    """File-like bridge that streams Rich console writes into the TUI log."""

    def __init__(self, app: "BiobankTuiApp") -> None:
        self.app = app
        self.capture = StringIO()
        self._pending = ""

    def write(self, text: str) -> int:
        value = str(text or "")
        self.capture.write(value)
        if not value:
            return 0
        self._pending += value.replace("\r", "\n")
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            clean = line.strip()
            if clean:
                self.app._submit_log_from_worker(clean)
        return len(value)

    def flush(self) -> None:
        clean = self._pending.strip()
        if clean:
            self.app._submit_log_from_worker(clean)
        self._pending = ""

    def isatty(self) -> bool:
        return False

    def getvalue(self) -> str:
        self.flush()
        return self.capture.getvalue()


class BiobankTuiApp(App):  # type: ignore[misc]
    """Compact multi-panel TUI shell.

    The first production milestone is visibility: plan progress, active tool,
    disclosure status, and logs are separated so long analyses do not look
    frozen.
    """

    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #left, #right { width: 1fr; }
    #log { height: 12; border: solid $accent; }
    #command { dock: bottom; }
    RunDetailPanel { min-height: 10; }
    WorkerReviewPanel { min-height: 10; }
    """

    BINDINGS = [
        ("f5", "approve_plan", "Approve plan"),
        ("f6", "resume_plan", "Resume plan"),
        ("f7", "repair_a", "Repair A"),
        ("f8", "repair_b", "Review hooks"),
        ("f9", "repair_n", "Abort"),
        ("ctrl+j", "next_worker", "Next worker"),
        ("ctrl+k", "previous_worker", "Previous worker"),
    ]

    def __init__(
        self,
        *,
        settings: Any | None = None,
        command_handler: CommandHandler | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.command_handler = command_handler
        self.agent = None
        self.planner = None
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.log_lines: list[str] = []
        self.log_widget = None
        self.input_widget = None
        from .disclosure_panel import DisclosurePanel
        from .progress_panel import ProgressPanel
        from .run_detail import RunDetailPanel
        from .tool_status import ToolStatusPanel
        from .worker_review import WorkerReviewPanel

        self.progress_panel = ProgressPanel()
        self.tool_status_panel = ToolStatusPanel()
        self.run_detail_panel = RunDetailPanel()
        self.disclosure_panel = DisclosurePanel()
        self.worker_review_panel = WorkerReviewPanel()

    def compose(self) -> "ComposeResult":
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield self.progress_panel
                yield self.tool_status_panel
            with Vertical(id="right"):
                yield self.disclosure_panel
                yield self.worker_review_panel
                yield self.run_detail_panel
        if RichLog is not None:
            self.log_widget = RichLog(id="log", wrap=True, highlight=False)
            yield self.log_widget
        if Input is not None:
            self.input_widget = Input(
                placeholder="Ask a question or type a slash command, e.g. /status or /plan ...",
                id="command",
            )
            yield self.input_widget
        yield Footer()

    def on_mount(self) -> None:  # pragma: no cover - requires Textual runtime
        """Focus the command input so the TUI is usable immediately."""
        input_widget = self.input_widget
        if input_widget is not None:
            focus = getattr(input_widget, "focus", None)
            if callable(focus):
                focus()

    def handle_agent_event(self, event: AgentEvent) -> None:
        """Route one AgentEvent to all mounted TUI panels."""
        for panel in (
            self.progress_panel,
            self.tool_status_panel,
            self.disclosure_panel,
            self.worker_review_panel,
            self.run_detail_panel,
        ):
            apply_event = getattr(panel, "apply_event", None)
            if callable(apply_event):
                apply_event(event)

    def write_log(self, text: str) -> None:
        """Append text to the TUI log and retain it for tests/fallbacks."""
        clean = str(text or "").rstrip()
        if not clean:
            return
        self.log_lines.append(clean)
        self.log_lines = self.log_lines[-200:]
        widget = self.log_widget
        write = getattr(widget, "write", None)
        if callable(write):
            write(clean)

    def _submit_log_from_worker(self, text: str) -> None:
        """Thread-safe-ish log entrypoint for legacy commands run in workers."""
        call_from_thread = getattr(self, "call_from_thread", None)
        if callable(call_from_thread):
            try:
                call_from_thread(self.write_log, text)
                return
            except Exception:
                pass
        self.write_log(text)

    def _submit_agent_event_from_worker(self, event: AgentEvent) -> None:
        """Forward legacy plan events into TUI panels from worker threads."""
        call_from_thread = getattr(self, "call_from_thread", None)
        if callable(call_from_thread):
            try:
                call_from_thread(self.handle_agent_event, event)
                return
            except Exception:
                pass
        self.handle_agent_event(event)

    async def handle_text_async(self, text: str) -> str:
        """Handle one user-entered line from the Textual input widget.

        Tests can call this directly without Textual installed. In production,
        ``on_input_submitted`` schedules it as a worker.
        """
        text = str(text or "").strip()
        if not text:
            return ""
        self.write_log(f"biobank > {text}")
        if text.lower() in {"quit", "exit", "q"}:
            exit_fn = getattr(self, "exit", None)
            if callable(exit_fn):
                exit_fn()
            return "Goodbye."

        handler = self.command_handler or self._default_command_handler
        result = handler(text)
        if inspect.isawaitable(result):
            result = await result
        output = "" if result is None else str(result)
        if output:
            self.write_log(output)
        return output

    async def _run_shortcut_command(self, command: str) -> str:
        """Run a plan-review shortcut command and surface the action in TUI state."""
        self.worker_review_panel.record_shortcut(command, "sent")
        try:
            output = await self.handle_text_async(command)
        except Exception as exc:
            self.worker_review_panel.record_shortcut(command, f"failed: {exc}")
            raise
        self.worker_review_panel.record_shortcut(command, "done")
        return output

    async def action_approve_plan(self) -> None:
        await self._run_shortcut_command("/plan-approve")

    async def action_resume_plan(self) -> None:
        await self._run_shortcut_command("/plan-resume")

    async def action_repair_a(self) -> None:
        await self._run_shortcut_command("/plan-option A")

    async def action_repair_b(self) -> None:
        await self._run_shortcut_command("/plan-option B")

    async def action_repair_n(self) -> None:
        await self._run_shortcut_command("/plan-option N")

    def action_next_worker(self) -> None:
        self.worker_review_panel.select_next()

    def action_previous_worker(self) -> None:
        self.worker_review_panel.select_previous()

    def on_input_submitted(self, event: Any) -> None:  # pragma: no cover - requires Textual
        value = str(getattr(event, "value", "") or "")
        input_widget = getattr(event, "input", None) or self.input_widget
        if input_widget is not None and hasattr(input_widget, "value"):
            input_widget.value = ""
        worker = getattr(self, "run_worker", None)
        if callable(worker):
            worker(self.handle_text_async(value), exclusive=False)
            return
        asyncio.create_task(self.handle_text_async(value))

    async def _default_command_handler(self, text: str) -> str:
        await asyncio.to_thread(self._ensure_runtime)
        if text.startswith("/"):
            return await asyncio.to_thread(self._run_registry_command_capture, text, True)
        return await self._run_agent_stream(text)

    def _ensure_runtime(self) -> None:
        if self.agent is not None and self.planner is not None:
            return
        from biobank_agent.agent import Agent
        from biobank_agent.config import get_settings
        from biobank_agent.planner import PlanMode

        settings = self.settings or get_settings()
        settings.ensure_dirs()
        # Textual has no blocking stdin prompt. Use deterministic defaults and
        # keep optional review hooks explicit via slash commands.
        if hasattr(settings, "plan_clarification_enabled"):
            settings.plan_clarification_enabled = False
        if hasattr(settings, "plan_review_hook_mode"):
            settings.plan_review_hook_mode = "never"
        self.settings = settings
        self.write_log("Loading Biobank Agent runtime...")
        self.agent = Agent(settings)
        skill_names = [s["name"] for s in self.agent.registry.list_skills()]
        self.planner = PlanMode(
            plans_dir=settings.plans_dir,
            llm=self.agent.llm,
            available_skills=skill_names,
            tool_schemas=self.agent.registry.tool_schemas(),
        )
        self.write_log(f"Ready. {len(skill_names)} skills loaded.")

    def _run_registry_command_capture(self, text: str, stream_to_log: bool = False) -> str:
        """Run one slash command through the shared v3 command registry.

        The registry handlers still call host actions implemented by the Rich
        CLI module, but the TUI no longer depends on the monolithic
        ``_handle_command`` dispatcher. This keeps plugin commands and built-in
        commands on the same path.
        """
        from rich.console import Console

        import biobank_agent.cli_legacy as legacy_cli

        old_console = legacy_cli.console
        file_obj: StringIO | _StreamingLogFile
        file_obj = _StreamingLogFile(self) if stream_to_log else StringIO()
        legacy_cli.console = Console(file=file_obj, force_terminal=False, width=120)
        previous_callback = None
        previous_confirm_fn = None
        custom_data = getattr(getattr(self.agent, "state", None), "custom_data", None)
        if isinstance(custom_data, dict):
            from .confirm_modal import auto_merger_confirm_fn

            previous_callback = custom_data.get("tui_event_callback")
            previous_confirm_fn = custom_data.get("tui_confirm_fn")
            custom_data["tui_event_callback"] = self._submit_agent_event_from_worker
            custom_data["tui_confirm_fn"] = auto_merger_confirm_fn(self)
        try:
            legacy_cli._dispatch_registered_command_line(text, self.agent, self.planner, self.token_usage)
        finally:
            if isinstance(custom_data, dict):
                if previous_callback is None:
                    custom_data.pop("tui_event_callback", None)
                else:
                    custom_data["tui_event_callback"] = previous_callback
                if previous_confirm_fn is None:
                    custom_data.pop("tui_confirm_fn", None)
                else:
                    custom_data["tui_confirm_fn"] = previous_confirm_fn
            flush = getattr(file_obj, "flush", None)
            if callable(flush):
                flush()
            legacy_cli.console = old_console
        if stream_to_log:
            return ""
        return file_obj.getvalue().strip()

    def _run_legacy_command_capture(self, text: str, stream_to_log: bool = False) -> str:
        """Compatibility wrapper for older tests/extensions."""
        return self._run_registry_command_capture(text, stream_to_log=stream_to_log)

    async def _run_agent_stream(self, text: str) -> str:
        from biobank_agent.core.events import AgentEventType
        from biobank_agent.core.runtime import AsyncAgent

        safe, reason = AsyncAgent.is_safe_for_async(self.agent)
        if not safe:
            self.write_log(f"Streaming disabled for this turn: {reason}")
            return await asyncio.to_thread(self.agent.run, text)

        runtime = AsyncAgent(self.agent)
        final_text = ""
        deltas: list[str] = []
        async for event in runtime.stream_events(text):
            self.handle_agent_event(event)
            if event.type == AgentEventType.MESSAGE_DELTA:
                chunk = str(event.payload.get("text", "") or "")
                if chunk:
                    deltas.append(chunk)
            elif event.type == AgentEventType.MESSAGE_COMPLETE:
                final_text = str(event.payload.get("text", "") or final_text)
            elif event.type == AgentEventType.TURN_FINISHED:
                final_text = str(event.payload.get("final_text", "") or final_text)
        return final_text or "".join(deltas)


__all__ = ["BiobankTuiApp"]
