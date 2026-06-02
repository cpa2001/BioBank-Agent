"""Rich-based progress display for plan mode execution.

Provides live-updating panels that show plan review, execution progress,
and final reports. Uses Rich Live for real-time terminal updates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Event, RLock, Thread
from typing import TYPE_CHECKING, Any

from rich.cells import cell_len
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape as _rich_escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

if TYPE_CHECKING:
    from .planner import LongHorizonPlan, PlanStep

logger = logging.getLogger(__name__)


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
    "error": "✗",
    "warning": "!",
    "skipped": "⊘",
    "cancelled": "⊘",
}

DASHBOARD_STATUS_STYLES = {
    "pending": "dim",
    "running": "bold yellow",
    "success": "green",
    "done": "green",
    "failed": "bold red",
    "error": "bold red",
    "warning": "yellow",
    "skipped": "dim",
    "cancelled": "dim",
}


_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Map a provider model id (e.g. "moonshotai/kimi-k2.6") to a short, friendly
# name shown live in the dashboard ("kimi"). Falls back to the trailing path
# segment so unknown models still render something sensible.
_MODEL_SHORT_KEYS = ("deepseek", "kimi", "moonshot", "glm", "qwen", "gpt", "claude", "gemini", "llama", "mistral")


def _short_model(model: str) -> str:
    raw = (model or "").strip()
    if not raw:
        return "model"
    low = raw.lower()
    for key in _MODEL_SHORT_KEYS:
        if key in low:
            return "kimi" if key == "moonshot" else key
    return raw.split("/")[-1] or "model"


def _elapsed_str(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _styled(text: Any, style: str) -> str:
    """Wrap ``text`` in a Rich style tag — but ONLY when ``style`` is non-empty.

    A status that is absent from ``DASHBOARD_STATUS_STYLES`` used to default to
    ``""``, so ``f"[{style}]{icon}[/{style}]"`` rendered ``[]<icon>[/]`` and the
    unmatched ``[/]`` raised ``MarkupError("closing tag '[/]' has nothing to
    close")`` — the live ``/plan`` crash. When the style is empty we instead
    escape the text and emit no tag, so no dynamic style can ever inject an
    unbalanced close tag. ``text`` is assumed already markup-safe (a fixed icon)
    or is escaped here; callers pass dynamic prose through ``_rich_escape``/
    ``_sanitize`` separately."""
    return f"[{style}]{text}[/{style}]" if style else _rich_escape(str(text))


# Activity verb shown per stage so an active row reads "kimi · drafting · 12s".
_STAGE_ACTIVITY = {
    "Clarification": "clarifying",
    "Research setup": "scoping",
    "Planning": "drafting",
    "External council": "critiquing",
    "Debate": "debating",
    "Merge": "merging",
    "Validation": "validating",
    "Review": "reviewing",
}


class PlanRunDashboard:
    """Live dashboard for the higher-level plan lifecycle.

    This sits above per-step execution progress and shows what the agent is
    doing during planning, external-agent consultation, validation, repair, and
    review-hook phases. When stages fan out to parallel model calls it also
    renders one live row per model (name · activity · persona · ticking timer)
    plus a scrolling history of completed calls with their durations.
    """

    PHASES = [
        "Preflight",
        "Clarification",
        "Research setup",
        "Planning",
        "External council",
        "Debate",
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
        # Per-subagent live state. ``_active`` maps a subagent label (e.g.
        # "candidate-2") to its live row; ``_history`` keeps completed rows.
        self._active: dict[str, dict[str, Any]] = {}
        self._history: list[dict[str, Any]] = []
        self._live: Live | None = None
        self._start_time: float = 0.0
        self._refresh_per_second = 10.0
        self._refresh_stop = Event()
        self._refresh_thread: Thread | None = None
        self._refresh_lock = RLock()
        self._last_render_second = -1
        self._last_render_ts = 0.0

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
            # transient=True so the live panel is erased on stop instead of being
            # committed to scrollback. This prevents the dashboard from stacking
            # above the final plan render (and above any later live region).
            transient=True,
        )
        self._live.start()
        self._start_background_refresh()

    def stop(self) -> None:
        """Stop live rendering."""
        self._stop_background_refresh()
        if self._live:
            self._live.stop()
            self._live = None
        # Clear live rows under the lock so any orphaned worker still streaming
        # after teardown (its HTTP read is not interruptible) finds no active row
        # in note_partial and is dropped — atomic w.r.t. the delta path.
        with self._refresh_lock:
            self._active.clear()

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

    # Min interval between *background* (timer/spinner) redraws. ~8 Hz keeps the
    # spinner (which steps at 8 fps) and the live timers smooth without rebuilding
    # the whole panel 10x/s. Milestone redraws (force=True) bypass this floor.
    _MIN_REDRAW_INTERVAL = 0.12

    def _refresh(self, *, force: bool) -> None:
        """Redraw the live panel. The background thread paces calls at
        ``refresh_per_second``; this floors background rebuilds at ~8 Hz to avoid
        CPU waste/flicker. ``record`` forces an immediate redraw on milestones.
        Both go through the lock the mutators also hold."""
        live = self._live
        if not live:
            return
        with self._refresh_lock:
            now = time.time()
            if not force and (now - self._last_render_ts) < self._MIN_REDRAW_INTERVAL:
                return
            self._last_render_ts = now
            # Defense in depth: a single malformed frame (e.g. a Rich MarkupError
            # from some future dynamic string) must NEVER kill the background
            # refresh thread or abort a milestone record(). Degrade to a skipped
            # frame and keep going — the next event/tick repaints.
            try:
                live.update(self._build_panel())
            except Exception:  # pragma: no cover - render must be crash-proof
                logger.debug("dashboard frame render failed; skipping frame", exc_info=True)

    _TERMINAL_STATUSES = {"success", "error", "done", "failed", "warning", "skipped", "cancelled"}

    def _update_subagent(self, phase: str, status: str, now: float, metadata: dict[str, Any]) -> None:
        """Track per-model live rows from events carrying a ``subagent`` label.
        A running event opens/updates an active row (timestamped); a terminal
        event moves it to history with its final duration. Called under the
        refresh lock by ``record`` so the refresh thread never reads a row that
        is being mutated."""
        label = metadata.get("subagent")
        if not label:
            return
        if status in self._TERMINAL_STATUSES:
            row = self._active.pop(label, None)
            start = row.get("start_ts", now) if row else now
            elapsed = metadata.get("elapsed_s")
            if elapsed is None:
                elapsed = now - start
            self._history.append({
                "label": label,
                "stage": phase,
                "status": status,
                "model": metadata.get("model") or (row or {}).get("model", ""),
                "persona": metadata.get("persona") or (row or {}).get("persona"),
                "elapsed_s": float(elapsed),
            })
            if len(self._history) > 200:  # bound memory on long runs
                self._history = self._history[-200:]
        else:
            row = self._active.get(label) or {"start_ts": now, "partial": ""}
            row["stage"] = phase
            if metadata.get("model"):
                row["model"] = metadata["model"]
            if metadata.get("role"):
                row["role"] = metadata["role"]
            if metadata.get("persona"):
                row["persona"] = metadata["persona"]
            if metadata.get("activity"):
                row["activity"] = metadata["activity"]
            self._active[label] = row

    def record(
        self,
        phase: str,
        actor: str = "biobank",
        status: str = "running",
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PlanRunEvent:
        """Record and render a phase-level event."""
        # Mutate shared state under the same lock the background refresh thread
        # uses to read it in _build_panel, so concurrent record()/refresh() can't
        # iterate _phase_status / _active while they are being modified.
        md = metadata or {}
        with self._refresh_lock:
            if not self._start_time:
                self._start_time = time.time()
            now = time.time()
            event = PlanRunEvent(
                phase=phase,
                actor=actor,
                status=status,
                message=message,
                elapsed_s=now - self._start_time,
                metadata=md,
            )
            self.events.append(event)
            self._phase_status[phase] = event
            self._update_subagent(phase, status, now, md)
        self._refresh(force=True)
        return event

    def note_partial(self, label: str, text: str, *, tail_cells: int = 160) -> None:
        """Append streamed text to an active model row's partial buffer (Phase C).

        Updates state ONLY (no forced redraw) so token-rate streaming cannot
        saturate rendering; the background refresh thread picks it up. Drops the
        delta if the row already finished. Text is sanitised before display in
        ``_build_panel``."""
        # This is the AUTHORITATIVE, atomic guard against late/orphaned deltas:
        # the check-and-append happens under the same lock that ``record`` uses to
        # pop a finished row to history (and that ``stop`` uses to clear all rows),
        # so a delta that slipped past the council's (lock-free, best-effort)
        # cancel checks still cannot land on a completed row or after the dashboard
        # has stopped — closing the TOCTOU window at the consumer.
        with self._refresh_lock:
            row = self._active.get(label)
            if row is None:  # completed / timed-out / dashboard stopped -> drop
                return
            buf = (row.get("partial", "") + (text or ""))
            # keep only the visible tail (by display cells)
            while buf and cell_len(buf) > tail_cells:
                buf = buf[1:]
            row["partial"] = buf

    @staticmethod
    def _sanitize(text: str, limit: int = 200) -> str:
        """Strip newlines/CR/control chars and escape markup for safe single-line
        display of model-derived text."""
        cleaned = "".join(ch for ch in (text or "") if ch == " " or (ch.isprintable() and ch not in "\r\n"))
        cleaned = cleaned.strip()
        if cell_len(cleaned) > limit:
            while cell_len(cleaned) > limit and cleaned:
                cleaned = cleaned[1:]
            cleaned = "…" + cleaned
        return _rich_escape(cleaned)

    def _active_renderable(self, max_rows: int = 8) -> Table | None:
        if not self._active:
            return None
        now = time.time()
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(width=2)              # spinner
        table.add_column(min_width=8)          # model
        table.add_column(width=14, no_wrap=True, overflow="ellipsis")  # activity · persona
        table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")   # streamed partial
        table.add_column(justify="right", width=7)  # timer
        frame = _SPINNER_FRAMES[int(now * 8) % len(_SPINNER_FRAMES)]
        # Stable order: by start time. Cap rows so a wide fan-out cannot overflow.
        rows = sorted(self._active.items(), key=lambda kv: kv[1].get("start_ts", now))[: max(1, max_rows)]
        for _label, row in rows:
            model = _rich_escape(_short_model(row.get("model", "")))
            # An explicit per-job activity verb (e.g. "revising R2") takes
            # precedence over the stage default; it is model/round-derived, so
            # escape it. Falls back to the stage verb ("drafting"/"debating").
            activity = _rich_escape(str(row.get("activity") or _STAGE_ACTIVITY.get(row.get("stage", ""), "working")))
            persona = row.get("persona")
            label = f"[cyan]{activity}[/cyan]"
            if persona:
                label += f" [dim]· {_rich_escape(str(persona))}[/dim]"
            partial = row.get("partial", "")
            stream = f"[white]{self._sanitize(partial, 100)}[/white]" if partial else ""
            timer = _elapsed_str(now - row.get("start_ts", now))
            table.add_row(f"[green]{frame}[/green]", f"[bold]{model}[/bold]", label, stream, f"[dim]{timer}[/dim]")
        return table

    def _phases_renderable(self) -> Table:
        # Compact multi-column grid of the whole pipeline so each phase name
        # stays intact on one cell (no mid-word wrap) while keeping the panel
        # short — 12 phases across 3 columns is 4 rows.
        grid = Table.grid(padding=(0, 3))
        cols = 3
        for _ in range(cols):
            grid.add_column()
        cells: list[str] = []
        for phase in self.PHASES:
            event = self._phase_status.get(phase)
            status = event.status if event else "pending"
            icon = DASHBOARD_STATUS_ICONS.get(status, "⬚") if event else "⬚"
            style = DASHBOARD_STATUS_STYLES.get(status, "dim")
            cells.append(f"{_styled(icon, style)} {_rich_escape(str(phase))}")
        for i in range(0, len(cells), cols):
            chunk = cells[i:i + cols] + [""] * (cols - len(cells[i:i + cols]))
            grid.add_row(*[Text.from_markup(c) for c in chunk])
        return grid

    def _recent_renderable(self, limit: int = 5) -> Text:
        # Last few milestone events with their messages. Skip per-subagent
        # "dispatch" chatter (running rows live in the Active models table);
        # keep phase milestones and subagent completions.
        rows = [
            e for e in self.events
            if not (e.metadata.get("subagent") and e.status not in self._TERMINAL_STATUSES)
        ][-limit:]
        if not rows:
            return Text.from_markup("[dim]starting…[/dim]")
        lines: list[str] = []
        for e in rows:
            icon = DASHBOARD_STATUS_ICONS.get(e.status, "•")
            style = DASHBOARD_STATUS_STYLES.get(e.status, "dim")
            msg = self._sanitize(e.message, 80)
            lines.append(
                f"[dim]{_elapsed_str(e.elapsed_s):>6}[/dim] {_styled(icon, style)} "
                f"[bold]{_rich_escape(str(e.phase))}[/bold] [dim]{msg}[/dim]"
            )
        return Text.from_markup("\n".join(lines))

    def _history_renderable(self, limit: int = 6) -> Text | None:
        if not self._history:
            return None
        lines: list[str] = []
        for row in self._history[-limit:]:
            icon = DASHBOARD_STATUS_ICONS.get(row.get("status", ""), "•")
            style = DASHBOARD_STATUS_STYLES.get(row.get("status", ""), "dim")
            model = _rich_escape(_short_model(row.get("model", "")))
            persona = row.get("persona")
            tag = f" [dim]· {_rich_escape(str(persona))}[/dim]" if persona else ""
            dur = _elapsed_str(row.get("elapsed_s", 0))
            lines.append(
                f"[dim]{dur:>6}[/dim] {_styled(icon, style)} [bold]{model}[/bold] "
                f"[dim]{_rich_escape(str(row.get('label', '')))}[/dim]{tag}"
            )
        return Text.from_markup("\n".join(lines))

    def _build_panel(self) -> Panel:
        # Budget rows against the terminal height so a full fan-out never makes
        # the live panel taller than the viewport (which destabilises Rich's
        # in-place redraw). Reserve ~12 rows for borders/padding/phase-grid/
        # section headers, then split the rest between active + recent.
        height = getattr(self.console.size, "height", 24) or 24
        budget = max(4, height - 12)
        max_active = max(2, min(8, (budget + 1) // 2))
        max_recent = max(2, min(5, budget - max_active))

        sections: list[Any] = []
        active = self._active_renderable(max_active)
        if active is not None:
            sections.append(Text.from_markup("[bold]Active models[/bold]"))
            sections.append(active)
            sections.append(Text(""))
        sections.append(self._phases_renderable())
        sections.append(Text(""))
        sections.append(Text.from_markup("[bold]Recent[/bold]"))
        sections.append(self._recent_renderable(max_recent))

        elapsed = time.time() - self._start_time if self._start_time else 0
        spin = _SPINNER_FRAMES[int(time.time() * 8) % len(_SPINNER_FRAMES)]
        return Panel(
            Group(*sections),
            title=f"[green]{spin}[/green] [bold blue]{self.title}[/bold blue] [dim]{_elapsed_str(elapsed)}[/dim]",
            border_style="blue",
            padding=(1, 2),
        )

    def render_activity_summary(self) -> Panel | None:
        """A static panel of completed model calls, printed to scrollback after
        the (transient) live dashboard stops so the per-model history the user
        asked for is preserved."""
        with self._refresh_lock:
            history = self._history_renderable(limit=40)
            if history is None:
                return None
            return Panel(history, title="[bold]Council activity[/bold]", border_style="dim", padding=(0, 2))


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
        # key/title/detail may be plan-derived (e.g. a step id in the title), so
        # escape them — an unbalanced "[/]" must not raise MarkupError here.
        lines.append(f"  [bold cyan]{_rich_escape(str(key))}[/bold cyan]: {_rich_escape(str(title))}")
        if detail:
            lines.append(f"     [dim]{_rich_escape(str(detail))}[/dim]")
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
            # Plan fields are LLM-derived; escape every interpolated value so a
            # stray "[" / "[/]" can never raise MarkupError mid-render.
            deps = f" [dim](after: {_rich_escape(', '.join(step.depends_on))})[/dim]" if step.depends_on else ""
            # Show status icon for completed steps (during re-plan)
            icon = STATUS_ICONS.get(step.status, "⬚")
            style = STATUS_STYLES.get(step.status, "")
            desc = _rich_escape(str(step.description))
            label = f"[{style}]{icon} {desc}{deps}[/{style}]" if style else f"{icon} {desc}{deps}"
            id_tag = f" [dim]({_rich_escape(str(step.id))})[/dim]" if step.id else ""
            node = tree.add(f"[dim]{i}.[/dim]{id_tag} {label}")
            node.add(f"[dim]Skill:[/dim] [cyan]{_rich_escape(str(step.skill))}[/cyan]")
            if step.args:
                args_str = ", ".join(f"{k}={v!r}" for k, v in step.args.items())
                if len(args_str) > 80:
                    args_str = args_str[:77] + "..."
                node.add(f"[dim]Args:[/dim] {_rich_escape(args_str)}")
            if step.error:
                node.add(f"[red]Error:[/red] {_rich_escape(str(step.error))}")

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
            icon = STATUS_ICONS.get(status, "⬚")
            style = STATUS_STYLES.get(status, "dim")
            table.add_row(
                str(i),
                _styled(icon, style),
                _styled(_rich_escape(str(step.description)), style),
                _rich_escape(str(step.skill)),
            )

        # Progress bar (text-based)
        bar_width = 30
        filled = int(bar_width * completed / total) if total > 0 else 0
        bar = f"[green]{'█' * filled}[/green][dim]{'░' * (bar_width - filled)}[/dim]"

        elapsed = time.time() - self._start_time if self._start_time else 0
        progress_line = f"\n{bar} {completed}/{total} steps ({int(elapsed)}s elapsed)"
        if self._current_detail:
            progress_line += f"\n[dim italic]↳ {_rich_escape(str(self._current_detail))}[/dim italic]"
        if self._recent_details:
            recent = "\n".join(f"[dim]  - {_rich_escape(str(item))}[/dim]" for item in self._recent_details[-4:])
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
                f"[red bold]Step failed:[/red bold] {_rich_escape(str(step_description))}\n\n"
                f"[red]{_rich_escape(str(error))}[/red]\n\n"
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
        reason_text = f"\n[dim]Reason: {_rich_escape(str(reason))}[/dim]" if reason else ""
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
