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


def _short_tokens(n: int) -> str:
    """Compact completion-token count for a live row header ('1.2k tok')."""
    n = max(0, int(n))
    if n >= 1000:
        return f"{n / 1000:.1f}k tok"
    return f"{n} tok"


def _styled(text: Any, style: str) -> str:
    """Wrap ``text`` in a Rich style tag — but only when ``style`` is non-empty.

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
            # from some future dynamic string) must never kill the background
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
            if metadata.get("tool_uses") is not None:
                row["tool_uses"] = metadata["tool_uses"]
            if metadata.get("tokens") is not None:
                row["tokens"] = metadata["tokens"]
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
        # This is the authoritative, atomic guard against late/orphaned deltas:
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
        table.add_column(min_width=10)         # model
        table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")  # what it is doing now · persona
        table.add_column(justify="right", width=7)  # live timer
        frame = _SPINNER_FRAMES[int(now * 8) % len(_SPINNER_FRAMES)]
        # Stable order: by start time. Cap rows so a wide fan-out cannot overflow.
        rows = sorted(self._active.items(), key=lambda kv: kv[1].get("start_ts", now))[: max(1, max_rows)]
        for _label, row in rows:
            model = _rich_escape(_short_model(row.get("model", "")))
            # A clear "what it is doing now" verb (e.g. "revising R2") instead of raw model tokens —
            # an explicit per-job activity takes precedence over the stage default ("drafting"/"debating").
            activity = _rich_escape(str(row.get("activity") or _STAGE_ACTIVITY.get(row.get("stage", ""), "working")))
            persona = row.get("persona")
            label = f"[cyan]{activity}[/cyan]"
            if persona:
                label += f" [dim]· {_rich_escape(str(persona))}[/dim]"
            timer = _elapsed_str(now - row.get("start_ts", now))
            table.add_row(f"[green]{frame}[/green]", f"[bold]{model}[/bold]", label, f"[dim]{timer}[/dim]")
        return table

    def _active_header_markup(self, row: dict[str, Any], now: float, *, frame: str) -> str:
        """One-line header for an active subagent: spinner · model · what-it-is-doing-now · metrics · timer."""
        model = _rich_escape(_short_model(row.get("model", "")))
        activity = _rich_escape(str(row.get("activity") or _STAGE_ACTIVITY.get(row.get("stage", ""), "working")))
        parts = [f"[green]{frame}[/green]", f"[bold]{model}[/bold]", f"[cyan]{activity}[/cyan]"]
        persona = row.get("persona")
        if persona:
            parts.append(f"[dim]· {_rich_escape(str(persona))}[/dim]")
        tools = row.get("tool_uses")
        if tools:
            try:
                n = int(tools)
                parts.append(f"[dim]· {n} tool{'s' if n != 1 else ''}[/dim]")
            except (TypeError, ValueError):
                pass
        tokens = row.get("tokens")
        if tokens:
            try:
                parts.append(f"[dim]· {_short_tokens(int(tokens))}[/dim]")
            except (TypeError, ValueError):
                pass
        parts.append(f"[dim]{_elapsed_str(now - row.get('start_ts', now))}[/dim]")
        return " ".join(parts)

    def _panel_text_width(self, *, reserve: int = 8) -> int:
        """Usable inner width for transcript/tree text (panel padding/borders/tree-guides reserved)."""
        width = getattr(getattr(self.console, "size", None), "width", 100) or 100
        return max(20, int(width) - max(0, reserve))

    def _transcript_body(self, partial: str, *, max_lines: int = 3, width: int = 100) -> Text:
        """The live streamed tail as up to ``max_lines`` sanitised, cell-bounded lines — so a long or
        wrapping stream can never destabilise the panel height (the wrap/refresh problem)."""
        raw = (partial or "").strip()
        if not raw:
            return Text.from_markup("[dim]…[/dim]")
        lines = [ln for ln in raw.splitlines() if ln.strip()] or [raw]
        rendered = [self._sanitize(ln, width) for ln in lines[-max_lines:]]
        return Text.from_markup("\n".join(f"[dim]{ln}[/dim]" for ln in rendered) or "[dim]…[/dim]")

    def _transcript_renderable(self, max_lines: int = 3) -> Group:
        """Single active subagent → a flowing Codex-style transcript: a header line plus the live output
        tail (what the model or its tool is producing right now). ``max_lines`` is capped by the caller's
        height budget so the transcript never overflows a short viewport."""
        now = time.time()
        frame = _SPINNER_FRAMES[int(now * 8) % len(_SPINNER_FRAMES)]
        _label, row = next(iter(self._active.items()))
        header = Text.from_markup(self._active_header_markup(row, now, frame=frame))
        body = self._transcript_body(row.get("partial", ""), max_lines=max(1, max_lines),
                                     width=self._panel_text_width(reserve=8))
        return Group(header, body)

    def _active_tree_renderable(self, max_rows: int = 8) -> Tree:
        """2+ active subagents → a Claude-Code-style task tree: one node per subagent (spinner · model ·
        activity · metrics · timer) plus a dim child showing its live output tail."""
        now = time.time()
        frame = _SPINNER_FRAMES[int(now * 8) % len(_SPINNER_FRAMES)]
        width = self._panel_text_width(reserve=12)
        tree = Tree("[bold]Active[/bold]", guide_style="dim")
        active = sorted(self._active.items(), key=lambda kv: kv[1].get("start_ts", now))
        n = len(active)
        budget = max(2, int(max_rows))
        # Height-safe budgeting: the Tree root costs 1 line; a node costs 1 (header) + 1 if it shows a
        # live tail. Show tails ONLY when every node's header+tail fits the budget (a small fan-out on a
        # tall enough panel); otherwise render nodes header-only so a fan-out shows every agent it can
        # without overflowing the viewport (the wrap/refresh problem).
        show_tails = n <= 6 and (1 + n * 2) <= budget
        shown = active if show_tails else active[: max(1, budget - 1)]
        for _label, row in shown:
            node = tree.add(Text.from_markup(self._active_header_markup(row, now, frame=frame)))
            partial = (row.get("partial") or "").strip()
            if show_tails and partial:
                last = partial.splitlines()[-1] if partial.splitlines() else partial
                node.add(Text.from_markup(f"[dim]{self._sanitize(last, width)}[/dim]"))
        return tree

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
        # Auto-switch: a single active subagent renders as a Codex-style streaming transcript (header +
        # live output tail); a parallel fan-out renders as a Claude-Code-style task tree (one node each,
        # each with its own live tail). Both surface the model/tool output the user wants to watch.
        active_count = len(self._active)
        if active_count == 1:
            sections.append(Text.from_markup("[bold]Now[/bold]"))
            # Cap the transcript body to the active-row budget (header takes one line) so it never
            # overflows a short viewport.
            sections.append(self._transcript_renderable(max_lines=max(1, min(3, max_active - 1))))
            sections.append(Text(""))
        elif active_count >= 2:
            sections.append(self._active_tree_renderable(max_active))
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
        """Show execution summary after completion.

        DEPRECATED (legacy engine): the live v3 CLI renders execution via
        ``InteractiveShell._finalize_execution`` + ``cli/render.render_result_payload``
        (which surface each step's real output). This method is reached only by the
        legacy PlanMode path; fix live output rendering in the v3 engine.
        """
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
