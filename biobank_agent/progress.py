"""Rich-based progress display for plan mode execution.

Provides live-updating panels that show plan review, execution progress,
and final reports. Uses Rich Live for real-time terminal updates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Event, RLock, Thread
from typing import TYPE_CHECKING, Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

if TYPE_CHECKING:
    from .planner import LongHorizonPlan, PlanStep


@dataclass
class StepResult:
    """Result of executing a single plan step."""

    step_id: str
    step_description: str
    skill: str = ""
    success: bool = True
    result: dict = field(default_factory=dict)
    error: str = ""
    duration_s: float = 0.0


@dataclass
class PlanRunEvent:
    """User-visible event emitted while a plan is being designed or executed."""

    phase: str
    actor: str = "biobank"
    status: str = "running"
    message: str = ""
    elapsed_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChoiceOption:
    """A selectable recovery/review option shown in CLI panels."""

    key: str
    title: str
    detail: str


STATUS_ICONS = {
    "pending": "⬚",
    "running": "⟳",
    "done": "✓",
    "failed": "✗",
    "skipped": "⊘",
}

STATUS_STYLES = {
    "pending": "dim",
    "running": "bold yellow",
    "done": "green",
    "failed": "bold red",
    "skipped": "dim strikethrough",
}


DASHBOARD_STATUS_ICONS = {
    "pending": "⬚",
    "running": "⟳",
    "success": "✓",
    "done": "✓",
    "failed": "✗",
    "warning": "!",
    "skipped": "⊘",
}

DASHBOARD_STATUS_STYLES = {
    "pending": "dim",
    "running": "bold yellow",
    "success": "green",
    "done": "green",
    "failed": "bold red",
    "warning": "yellow",
    "skipped": "dim",
}


class PlanRunDashboard:
    """Live dashboard for the higher-level plan lifecycle.

    This sits above per-step execution progress and shows what the agent is
    doing during planning, external-agent consultation, validation, repair, and
    review-hook phases.
    """

    PHASES = [
        "Preflight",
        "Clarification",
        "Research setup",
        "Planning",
        "External council",
        "Merge",
        "Validation",
        "Review",
        "Execution",
        "Repair",
        "Report",
        "Review hooks",
    ]

    def __init__(self, console: Console | None = None, title: str = "Plan Run") -> None:
        self.console = console or Console()
        self.title = title
        self.events: list[PlanRunEvent] = []
        self._phase_status: dict[str, PlanRunEvent] = {}
        self._live: Live | None = None
        self._start_time: float = 0.0
        self._refresh_per_second = 10.0
        self._refresh_stop = Event()
        self._refresh_thread: Thread | None = None
        self._refresh_lock = RLock()
        self._last_render_second = -1

    def start(self, refresh_per_second: float = 10.0) -> None:
        """Start live rendering."""
        if self._live:
            return
        self._refresh_per_second = max(float(refresh_per_second or 10.0), 1.0)
        if not self._start_time:
            self._start_time = time.time()
        self._live = Live(
            self._build_panel(),
            console=self.console,
            refresh_per_second=self._refresh_per_second,
            transient=False,
        )
        self._live.start()
        self._start_background_refresh()

    def stop(self) -> None:
        """Stop live rendering."""
        self._stop_background_refresh()
        if self._live:
            self._live.stop()
            self._live = None

    def _start_background_refresh(self) -> None:
        """Rebuild the renderable on a timer so elapsed time advances without new events."""
        self._refresh_stop.clear()

        def _loop() -> None:
            interval = 1.0 / max(self._refresh_per_second, 1.0)
            while not self._refresh_stop.wait(interval):
                self.refresh()

        self._refresh_thread = Thread(target=_loop, name="biobank-plan-dashboard-refresh", daemon=True)
        self._refresh_thread.start()

    def _stop_background_refresh(self) -> None:
        self._refresh_stop.set()
        thread = self._refresh_thread
        if thread and thread.is_alive():
            thread.join(timeout=0.5)
        self._refresh_thread = None

    def refresh(self) -> None:
        """Force a live redraw if rendering is active."""
        self._refresh(force=False)

    def _refresh(self, *, force: bool) -> None:
        """Redraw at most once per displayed second unless an event forces it."""
        live = self._live
        if not live:
            return
        current_second = int(time.time() - self._start_time) if self._start_time else 0
        if not force and current_second == self._last_render_second:
            return
        with self._refresh_lock:
            self._last_render_second = current_second
            live.update(self._build_panel())

    def record(
        self,
        phase: str,
        actor: str = "biobank",
        status: str = "running",
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PlanRunEvent:
        """Record and render a phase-level event."""
        if not self._start_time:
            self._start_time = time.time()
        event = PlanRunEvent(
            phase=phase,
            actor=actor,
            status=status,
            message=message,
            elapsed_s=time.time() - self._start_time,
            metadata=metadata or {},
        )
        self.events.append(event)
        self._phase_status[phase] = event
        self._refresh(force=True)
        return event

    def _build_panel(self) -> Panel:
        phase_table = Table(show_header=False, box=None, padding=(0, 1))
        phase_table.add_column("Phase", ratio=1)
        phase_table.add_column("Status", ratio=3)

        known_phases = list(self.PHASES)
        for phase in self._phase_status:
            if phase not in known_phases:
                known_phases.append(phase)

        for phase in known_phases:
            event = self._phase_status.get(phase)
            if event:
                icon = DASHBOARD_STATUS_ICONS.get(event.status, "•")
                style = DASHBOARD_STATUS_STYLES.get(event.status, "")
                actor = f"[dim]{event.actor}[/dim] " if event.actor else ""
                message = event.message or event.status
                status_text = (
                    f"[{style}]{icon}[/{style}] {actor}{message}"
                    if style
                    else f"{icon} {actor}{message}"
                )
            else:
                status_text = "[dim]⬚ pending[/dim]"
            phase_table.add_row(f"[bold]{phase}[/bold]", status_text)

        recent_lines = []
        for event in self.events[-5:]:
            icon = DASHBOARD_STATUS_ICONS.get(event.status, "•")
            recent_lines.append(
                f"[dim]{int(event.elapsed_s):5d}s[/dim] {icon} [bold]{event.phase}[/bold] "
                f"[cyan]{event.actor}[/cyan]: {event.message}"
            )
        recent = Text.from_markup("\n".join(recent_lines) if recent_lines else "[dim]waiting...[/dim]")

        elapsed = time.time() - self._start_time if self._start_time else 0
        return Panel(
            Group(phase_table, Text(""), recent),
            title=f"[bold blue]{self.title}[/bold blue] [dim]{int(elapsed)}s[/dim]",
            border_style="blue",
            padding=(1, 2),
        )


def default_failure_choices() -> list[ChoiceOption]:
    """Default recovery choices for a blocked or failed plan step."""
    return [
        ChoiceOption(
            "A",
            "Auto-repair or re-plan from here",
            "Agent diagnoses the failed step, edits/creates needed skills when useful, then retries.",
        ),
        ChoiceOption(
            "B",
            "Ask Codex/Claude/Gemini for repair advice",
            "Runs external review hooks against the current stalled trajectory before changing the plan.",
        ),
        ChoiceOption(
            "C",
            "Resume after an explicit fix",
            "Use this only after you have typed repair instructions or approved a revised plan.",
        ),
        ChoiceOption("N", "Abort", "Stop plan execution and keep artifacts for inspection."),
    ]


def _format_choices(choices: list[ChoiceOption | dict[str, Any]]) -> str:
    lines = []
    for raw_choice in choices:
        if isinstance(raw_choice, ChoiceOption):
            key, title, detail = raw_choice.key, raw_choice.title, raw_choice.detail
        else:
            key = str(raw_choice.get("key", "?"))
            title = str(raw_choice.get("title", "Option"))
            detail = str(raw_choice.get("detail", ""))
        lines.append(f"  [bold cyan]{key}[/bold cyan]: {title}")
        if detail:
            lines.append(f"     [dim]{detail}[/dim]")
    lines.append("  [bold cyan]Other[/bold cyan]: Type your own repair instructions")
    return "\n".join(lines)


class PlanProgressDisplay:
    """Live-updating Rich display for plan execution.

    Handles three display modes:
    1. Plan review: tree-formatted plan for user approval
    2. Execution progress: live-updating step tracker
    3. Final report: summary table of results
    """

    def __init__(self, plan: LongHorizonPlan, console: Console | None = None) -> None:
        self.plan = plan
        self.console = console or Console()
        self._live: Live | None = None
        self._step_status: dict[str, str] = {}
        self._current_detail: str = ""
        self._recent_details: list[str] = []
        self._start_time: float = 0.0
        self._refresh_per_second = 10.0
        self._refresh_stop = Event()
        self._refresh_thread: Thread | None = None
        self._refresh_lock = RLock()
        self._last_render_second = -1

    # ─── Plan Review Display ─────────────────────────────────────

    def show_plan_for_review(self, revision: int = 1) -> None:
        """Display the plan in a clear tree format for user approval."""
        tree = Tree("[bold]Execution Plan[/bold]")
        for i, step in enumerate(self.plan.steps, 1):
            deps = f" [dim](after: {', '.join(step.depends_on)})[/dim]" if step.depends_on else ""
            # Show status icon for completed steps (during re-plan)
            icon = STATUS_ICONS.get(step.status, "⬚")
            style = STATUS_STYLES.get(step.status, "")
            label = f"[{style}]{icon} {step.description}{deps}[/{style}]" if style else f"{icon} {step.description}{deps}"
            id_tag = f" [dim]({step.id})[/dim]" if step.id else ""
            node = tree.add(f"[dim]{i}.[/dim]{id_tag} {label}")
            node.add(f"[dim]Skill:[/dim] [cyan]{step.skill}[/cyan]")
            if step.args:
                args_str = ", ".join(f"{k}={v!r}" for k, v in step.args.items())
                if len(args_str) > 80:
                    args_str = args_str[:77] + "..."
                node.add(f"[dim]Args:[/dim] {args_str}")
            if step.error:
                node.add(f"[red]Error:[/red] {step.error}")

        subtitle_parts = [
            "[green]/plan-approve[/green] execute",
            "type feedback to refine",
            "[red]/plan-exit[/red] cancel",
        ]
        subtitle = " │ ".join(subtitle_parts)

        version_tag = f" (v{revision})" if revision > 1 else ""
        panel = Panel(
            tree,
            title=f"[bold blue]Plan Review{version_tag}[/bold blue]",
            subtitle=f"[dim]{subtitle}[/dim]",
            border_style="blue",
            padding=(1, 2),
        )
        self.console.print(panel)

    # ─── Execution Progress Display ──────────────────────────────

    def start_execution_display(self, refresh_per_second: float = 10.0) -> None:
        """Start the live-updating execution panel."""
        self._start_time = time.time()
        self._refresh_per_second = max(float(refresh_per_second or 10.0), 1.0)
        self._live = Live(
            self._build_execution_panel(),
            console=self.console,
            refresh_per_second=self._refresh_per_second,
            transient=False,
        )
        self._live.start()
        self._start_background_refresh()

    def _start_background_refresh(self) -> None:
        """Keep elapsed time and current activity visible during long blocking skills."""
        self._refresh_stop.clear()

        def _loop() -> None:
            interval = 1.0 / max(self._refresh_per_second, 1.0)
            while not self._refresh_stop.wait(interval):
                self.refresh()

        self._refresh_thread = Thread(target=_loop, name="biobank-plan-execution-refresh", daemon=True)
        self._refresh_thread.start()

    def _stop_background_refresh(self) -> None:
        self._refresh_stop.set()
        thread = self._refresh_thread
        if thread and thread.is_alive():
            thread.join(timeout=0.5)
        self._refresh_thread = None

    def refresh(self) -> None:
        """Force a live redraw if rendering is active."""
        self._refresh(force=False)

    def _refresh(self, *, force: bool) -> None:
        """Redraw at most once per displayed second unless progress changes."""
        live = self._live
        if not live:
            return
        current_second = int(time.time() - self._start_time) if self._start_time else 0
        if not force and current_second == self._last_render_second:
            return
        with self._refresh_lock:
            self._last_render_second = current_second
            live.update(self._build_execution_panel())

    def update_step(self, step_id: str, status: str, detail: str = "") -> None:
        """Update a step's display status.

        Args:
            step_id: The step ID to update
            status: One of 'running', 'done', 'failed', 'skipped', 'pending'
            detail: Optional detail string for current activity
        """
        self._step_status[step_id] = status
        if detail:
            self._current_detail = detail
        self._refresh(force=True)

    def update_activity(self, detail: str) -> None:
        """Update the current intra-skill activity line."""
        detail = str(detail or "").strip()
        if not detail:
            return
        self._current_detail = detail
        self._recent_details.append(detail)
        self._recent_details = self._recent_details[-5:]
        self._refresh(force=True)

    def stop_display(self) -> None:
        """Stop the live display."""
        self._stop_background_refresh()
        if self._live:
            self._live.stop()
            self._live = None

    def _build_execution_panel(self) -> Panel:
        """Build the Rich renderable showing execution state."""
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("#", width=3, justify="right")
        table.add_column("", width=2)  # status icon
        table.add_column("Step", ratio=3)
        table.add_column("Skill", ratio=1, style="cyan")

        total = len(self.plan.steps)
        completed = sum(1 for s in self._step_status.values() if s in ("done", "skipped"))

        for i, step in enumerate(self.plan.steps, 1):
            status = self._step_status.get(step.id, "pending")
            icon = STATUS_ICONS[status]
            style = STATUS_STYLES[status]
            table.add_row(
                str(i),
                f"[{style}]{icon}[/{style}]",
                f"[{style}]{step.description}[/{style}]",
                step.skill,
            )

        # Progress bar (text-based)
        bar_width = 30
        filled = int(bar_width * completed / total) if total > 0 else 0
        bar = f"[green]{'█' * filled}[/green][dim]{'░' * (bar_width - filled)}[/dim]"

        elapsed = time.time() - self._start_time if self._start_time else 0
        progress_line = f"\n{bar} {completed}/{total} steps ({int(elapsed)}s elapsed)"
        if self._current_detail:
            progress_line += f"\n[dim italic]↳ {self._current_detail}[/dim italic]"
        if self._recent_details:
            recent = "\n".join(f"[dim]  - {item}[/dim]" for item in self._recent_details[-4:])
            progress_line += f"\n[dim]Recent activity:[/dim]\n{recent}"

        return Panel(
            Group(table, Text.from_markup(progress_line)),
            title=f"[bold green]Executing Plan[/bold green] [{completed}/{total}]",
            border_style="green",
            padding=(1, 2),
        )

    # ─── Failure Display ─────────────────────────────────────────

    def show_step_failure(
        self,
        step_description: str,
        error: str,
        choices: list[ChoiceOption | dict[str, Any]] | None = None,
    ) -> None:
        """Show a failure panel with options for the user."""
        choice_text = _format_choices(choices or default_failure_choices())
        self.console.print(
            Panel(
                f"[red bold]Step failed:[/red bold] {step_description}\n\n"
                f"[red]{error}[/red]\n\n"
                "[dim]Choose an option or type repair instructions:[/dim]\n"
                f"{choice_text}\n\n"
                "[dim]Commands:[/dim] [green]/plan-option A[/green], "
                "[green]/plan-option B[/green], [green]/plan-resume[/green] after repair, "
                "[yellow]/plan-exit[/yellow]",
                title="[bold red]Execution Issue[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
        )

    # ─── Pause Display ───────────────────────────────────────────

    def show_paused(self, reason: str = "") -> None:
        """Show paused state with resume options."""
        reason_text = f"\n[dim]Reason: {reason}[/dim]" if reason else ""
        completed = sum(1 for s in self._step_status.values() if s in ("done", "skipped"))
        total = len(self.plan.steps)

        self.console.print(
            Panel(
                f"[yellow bold]⏸ Plan execution paused[/yellow bold]{reason_text}\n\n"
                f"Progress: {completed}/{total} steps completed\n\n"
                "[dim]Options:[/dim]\n"
                "  • Type feedback to repair or modify the remaining plan\n"
                "  • [green]/plan-option A[/green] to ask the agent to auto-repair/re-plan\n"
                "  • [green]/plan-option B[/green] to ask Codex/Claude/Gemini for repair advice\n"
                "  • [green]/plan-resume[/green] to continue after the repair is in place\n"
                "  • [green]/plan-skip <step_id>[/green] to skip an optional/diagnostic step only\n"
                "  • [green]/plan-approve[/green] to restart with modified plan\n"
                "  • [yellow]/plan-exit[/yellow] to abort",
                title="[bold yellow]Paused[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    # ─── Final Report ────────────────────────────────────────────

    def show_report(self, results: list[StepResult]) -> None:
        """Show execution summary after completion."""
        table = Table(
            title="[bold]Execution Report[/bold]",
            show_header=True,
            header_style="bold",
            padding=(0, 1),
        )
        table.add_column("#", width=3, justify="right")
        table.add_column("Step", ratio=3)
        table.add_column("Result", width=10, justify="center")
        table.add_column("Time", width=8, justify="right")

        for i, r in enumerate(results, 1):
            if r.success:
                result_text = "[green]✓ Done[/green]"
            else:
                result_text = "[red]✗ Failed[/red]"
            table.add_row(
                str(i),
                r.step_description,
                result_text,
                f"{r.duration_s:.1f}s",
            )

        total_time = sum(r.duration_s for r in results)
        succeeded = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)

        summary_parts = [f"[bold]{succeeded}[/bold] succeeded"]
        if failed:
            summary_parts.append(f"[red]{failed}[/red] failed")
        summary_parts.append(f"in [bold]{total_time:.1f}s[/bold] total")

        self.console.print(table)
        self.console.print(f"\n  {' │ '.join(summary_parts)}")
        self.console.print()

    def show_review_hook_prompt(self, report_dir: str) -> None:
        """Show the optional post-run external review hook choices."""
        self.console.print(
            Panel(
                f"[bold]Reports written to:[/bold] {report_dir}\n\n"
                "[dim]Optional external review hooks:[/dim]\n"
                "  [bold cyan]A[/bold cyan]: Full review council (Codex + Claude + Gemini)\n"
                "  [bold cyan]B[/bold cyan]: Codex only\n"
                "  [bold cyan]C[/bold cyan]: Claude only\n"
                "  [bold cyan]D[/bold cyan]: Gemini only\n"
                "  [bold cyan]N[/bold cyan]: Skip review hooks",
                title="[bold blue]Review Hooks[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
        )
