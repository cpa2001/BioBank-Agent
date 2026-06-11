"""Rich-based CLI for Biobank Agent.

Entry point: `biobank` command (configured in pyproject.toml).
Subcommands: `biobank rebuild-parquet` for batch parquet rebuild.
Slash commands: /skills, /status, /history, /plan, /compact, /clear, /cost,
                /model, /export, /help, /figures, /cohorts, /models,
                /record, /pipelines, /errors, /memory, /models-available,
                /routing-status, /evidence
"""

from __future__ import annotations

import json
import logging
import re
import select
import shlex
import subprocess
import sys
import time
import asyncio
import termios
import tty
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

try:  # pragma: no cover - optional runtime dependency
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import FormattedText, HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style as PTStyle
    _PROMPT_TOOLKIT_AVAILABLE = True
except Exception:  # pragma: no cover - import fallback
    _PROMPT_TOOLKIT_AVAILABLE = False

    class PromptSession:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs

        def prompt(self, *args, **kwargs):
            raise RuntimeError("prompt_toolkit is not available")

    class AutoSuggestFromHistory:  # type: ignore[override]
        pass

    class Completer:  # type: ignore[override]
        pass

    class Completion:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FormattedText(list):  # type: ignore[override]
        pass

    def HTML(text):  # type: ignore[override]
        return text

    class FileHistory:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs

    class PTStyle:  # type: ignore[override]
        @staticmethod
        def from_dict(data):
            return data
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import get_settings
from .plan_state import DEFAULT_CHECKPOINT_FILE, PlanCheckpoint
from .plan_executor import PlanExecutor
from .planner import LongHorizonPlanner, PlanMode, PlanState

run_interactive_shell = None
Agent = None

console = Console()
_PENDING_STDIN_LINES: list[str] = []

DEFAULT_EVAL_REVIEWER = "codex-gpt-5.5-xhigh"


def _plan_checkpoint_path(settings) -> Path:
    """Return the plan checkpoint path for the current configured workspace."""
    return Path(settings.plans_dir) / DEFAULT_CHECKPOINT_FILE


def _sync_plan_execution_log(agent, planner: PlanMode) -> None:
    """Expose plan execution results to report generation as sanitized rows."""
    rows = []
    for item in getattr(planner, "execution_log", []) or []:
        rows.append({
            "step": item.step_description,
            "tool": item.skill,
            "status": "success" if item.success else "failed",
            "duration_s": round(float(item.duration_s or 0.0), 3),
            "note": "" if item.success else item.error,
        })
    try:
        agent.state.execution_log = rows
        agent.state.custom_data["execution_log"] = rows
    except Exception:
        pass


def _emit_tui_plan_event(
    agent,
    phase: str,
    actor: str,
    status: str,
    message: str,
    metadata: dict | None = None,
) -> None:
    """Forward legacy plan/review-hook progress to Textual panels when present."""
    try:
        callback = getattr(agent.state, "custom_data", {}).get("tui_event_callback")
    except Exception:
        callback = None
    if not callable(callback):
        return
    try:
        from .core.events import AgentEvent, AgentEventType

        callback(
            AgentEvent.make(
                AgentEventType.PLAN_PHASE,
                phase=phase,
                actor=actor,
                status=status,
                message=message,
                metadata=metadata or {},
            )
        )
    except Exception:
        logger.debug("Failed to forward TUI plan event", exc_info=True)


def _stream_agent_response(agent, query: str) -> tuple[str, bool]:
    """Run one normal agent turn through AsyncAgent and render events live.

    Returns ``(final_text, streamed)``. ``streamed=False`` means the caller
    should fall back to the legacy synchronous rendering path.
    """
    if not hasattr(agent.settings, "async_runtime_enabled"):
        return "", False
    if not getattr(agent.settings, "async_runtime_enabled", True):
        return "", False

    try:
        from .core.events import AgentEventType
        from .core.runtime import AsyncAgent
    except Exception:
        return "", False

    safe, reason = AsyncAgent.is_safe_for_async(agent)
    if not safe:
        console.print(f"[dim]Streaming disabled for this turn: {reason}[/dim]")
        return "", False

    async def _run() -> tuple[str, bool]:
        runtime = AsyncAgent(agent)
        final_text = ""
        saw_text_delta = False
        saw_any_event = False
        tool_line_open = False

        async for event in runtime.stream_events(query):
            saw_any_event = True
            if event.type == AgentEventType.MESSAGE_DELTA:
                text = str(event.payload.get("text", "") or "")
                if not text:
                    continue
                saw_text_delta = True
                tool_line_open = False
                console.print(Text(text), end="")
            elif event.type == AgentEventType.TOOL_STARTED:
                if saw_text_delta and not tool_line_open:
                    console.print()
                skill = event.payload.get("skill", "tool")
                args = event.payload.get("args") or {}
                arg_keys = ", ".join(sorted(map(str, args.keys())))
                suffix = f"({arg_keys})" if arg_keys else ""
                console.print(f"[dim]→ {skill}{suffix}[/dim]")
                tool_line_open = True
            elif event.type == AgentEventType.TOOL_PROGRESS:
                message = event.payload.get("message") or event.payload.get("state") or "running"
                skill = event.payload.get("skill", "tool")
                console.print(f"[dim]  {skill}: {message}[/dim]")
            elif event.type == AgentEventType.TOOL_ERROR:
                skill = event.payload.get("skill", "tool")
                error = event.payload.get("error", "tool failed")
                console.print(f"[red]  {skill} failed: {error}[/red]")
            elif event.type == AgentEventType.TOOL_RESULT:
                summary = event.payload.get("summary") or {}
                if summary:
                    skill = event.payload.get("skill", "tool")
                    console.print(f"[dim]  {skill} result: {str(summary)[:240]}[/dim]")
            elif event.type == AgentEventType.MESSAGE_COMPLETE:
                final_text = str(event.payload.get("text", "") or final_text)
            elif event.type == AgentEventType.TURN_FINISHED:
                final_text = str(event.payload.get("final_text", "") or final_text)

        if saw_text_delta:
            console.print()
        elif final_text:
            console.print()
            console.print(Markdown(final_text))
            console.print()
        return final_text, saw_any_event

    try:
        return asyncio.run(_run())
    except RuntimeError:
        return "", False


@dataclass(frozen=True)
class EvalArgs:
    """Parsed arguments for the lightweight eval subcommand parser."""
    suite: str = "research_eval_v1"
    mode: str = "baseline"
    policy: str = "all"
    enforce_gate: bool = False
    ab_compare: bool = False
    baseline_report: str = ""
    review_loop: bool = False
    primary_reviewer: str = DEFAULT_EVAL_REVIEWER
    include_claude: bool = False
    review_timeout_s: int = 600
    evolution_history: str = ""
    fail_on_evolution_patterns: bool = False
    run_dir: str = ""
    hpp_ckb_rap_readiness: str = ""
    mcp_compat_evidence: str = ""
    remote_ci_evidence: str = ""
    high_pr_evidence: str = ""
    external_evidence_dir: str = ""
    collect_external: bool = False
    collect: str = "all"
    banks: str = "ukb,hpp,ckb,ukb_rap"
    icd10_code: str = "E11"
    probe_fields: str = "hba1c,bmi,glucose"
    mcp_config: str = ""
    mcp_call_args: str = ""
    mcp_min_servers: int = 2
    workflow: str = "biobank-scheduled-eval.yml"
    pr_url: str = ""
    branch: str = ""

@dataclass(frozen=True)
class CommandHint:
    """Command metadata for help rendering and slash completion."""
    command: str
    usage: str
    description: str

    @property
    def insert_text(self) -> str:
        """Text inserted by autocomplete."""
        return f"{self.command} " if self.usage != self.command else self.command

HELP_SECTIONS = [
    ("Session", [
        ("/help", "Show this help message"),
        ("/status", "Session state, platform info, memory summary"),
        ("/routing-status", "Show latest orchestration trace, claims, and safety status"),
        ("/cost", "Show token usage and estimated cost"),
        ("/compact", "Compress conversation history (keep last 10 turns)"),
        ("/clear", "Reset session state (cohorts, models, figures)"),
        ("/export [format]", "Export session as JSON or Markdown"),
    ]),
    ("Analysis", [
        ("/skills", "List all available analysis tools"),
        ("/history", "Show analysis history"),
        ("/figures", "List all generated figures"),
        ("/cohorts", "List active cohorts with summary stats"),
        ("/models", "List trained models with AUC"),
        ("/evidence <claim_id>", "Show evidence chain for a claim from Action Graph"),
        ("/replicate <paper_path_or_text>", "Draft a review-only paper replication StudySpec and plan"),
    ]),
    ("Planning", [
        ("/plan <task>", "Design and execute a structured plan"),
        ("/plan-approve", "Approve plan and begin execution"),
        ("/plan-edit <feedback>", "Refine plan with natural language"),
        ("/plan-title <title>", "Rename the active plan/report title"),
        ("/plan-rename <title>", "Alias for /plan-title"),
        ("/plan-pause", "Pause plan execution"),
        ("/plan-resume", "Resume paused plan execution after repair"),
        ("/plan-option <A|B|C|N>", "Choose a suggested repair/review option after a block"),
        ("/plan-skip <step_id>", "Explicitly skip an optional/diagnostic step"),
        ("/plan-exit", "Exit plan mode"),
        ("/plans", "List all saved plans (diagnostic)"),
    ]),
    ("MCP", [
        ("/mcp-list", "List configured MCP servers and loaded MCP tools"),
        ("/mcp-start", "Start configured STDIO MCP servers and register remote tools"),
        ("/mcp-health [--repair]", "Probe MCP servers and optionally reconnect unhealthy ones"),
        ("/mcp-call <tool> [json|key=value...]", "Call a loaded MCP tool directly for audit/debugging"),
        ("/mcp-stop", "Stop active MCP clients and unregister remote tools"),
    ]),
    ("Pipelines & Memory", [
        ("/record <name>", "Save current session as a replayable pipeline"),
        ("/pipelines", "List saved pipelines"),
        ("/memory", "Show long-term memory summary"),
        ("/errors", "Show error catalog from long-term memory"),
        ("/replay <hash|prefix|prov_id> [--strict|--dry-run]", "Replay checkpointed skill (full SHA-256, 6+ char prefix, or 8-char Provenance id)"),
        ("/evolve [--dry-run|--pause|--resume|--write-history|--scheduled-run]", "Inspect repeated tool failures and proposed safe improvements"),
    ]),
    ("Configuration", [
        ("/model <name>", "Switch LLM model at runtime"),
        ("/models-pool", "Show available models in the pool"),
        ("/models-available", "Fetch model list from current relay endpoint"),
        ("/strategy <mode>", "Set routing: auto, single, debate, ensemble"),
        ("/debate <query>", "Force multi-model debate for a query"),
    ]),
    ("General", [
        ("quit / exit / q", "Exit the agent"),
    ]),
]


def _command_hints() -> list[CommandHint]:
    """Flatten HELP_SECTIONS into command hints."""
    hints: list[CommandHint] = []
    for _, rows in HELP_SECTIONS:
        for usage, description in rows:
            command = usage.split()[0]
            if command.startswith("/"):
                hints.append(CommandHint(command=command, usage=usage, description=description))
    return hints


class SlashCommandCompleter(Completer):
    """Autocomplete slash commands such as /help, /status, /plan."""

    def __init__(self, hints: list[CommandHint]) -> None:
        self.hints = hints

    def get_completions(self, document, complete_event):  # noqa: D401
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return

        # Only autocomplete the command token.
        if " " in text:
            if text.endswith(" "):
                return
            fragment = text.split(None, 1)[0]
        else:
            fragment = text

        for hint in self.hints:
            if hint.command.startswith(fragment):
                yield Completion(
                    hint.insert_text,
                    start_position=-len(fragment),
                    display=hint.usage,
                    display_meta=hint.description,
                )


def _build_prompt_session(settings) -> PromptSession:
    """Create an interactive prompt session with slash autocomplete."""
    if not _PROMPT_TOOLKIT_AVAILABLE:
        raise RuntimeError("prompt_toolkit is not available")
    settings.memory_dir.mkdir(parents=True, exist_ok=True)
    history_path = settings.memory_dir / "cli_history.txt"
    hints = _command_hints()
    completer = SlashCommandCompleter(hints)
    style = PTStyle.from_dict(
        {
            "prompt.main": "bold #56d4dd",
            "prompt.plan": "bold #f59e0b",
            "prompt.sep": "#768194",
            "bottom-toolbar": "fg:#9aa4b2 bg:#1f2937",
            "completion-menu.completion": "fg:#cbd5e1 bg:#0f172a",
            "completion-menu.completion.current": "fg:#ffffff bg:#334155",
            "completion-menu.meta.completion": "fg:#94a3b8 bg:#0f172a",
            "completion-menu.meta.completion.current": "fg:#e2e8f0 bg:#334155",
        }
    )

    return PromptSession(
        history=FileHistory(str(history_path)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=completer,
        complete_while_typing=True,
        complete_in_thread=True,
        reserve_space_for_menu=8,
        style=style,
        bottom_toolbar=lambda: HTML(
            "<b>Tab</b> autocomplete  •  <b>↑/↓</b> history  •  <b>/help</b> command palette"
        ),
    )


def _prompt_message(is_plan_mode: bool, plan_state: str = "") -> FormattedText:
    """Prompt prefix with mode-aware visual style."""
    if is_plan_mode:
        state_tag = f":{plan_state.lower()}" if plan_state else ""
        return FormattedText(
            [
                ("class:prompt.plan", f"plan{state_tag}"),
                ("class:prompt.sep", " › "),
            ]
        )
    return FormattedText(
        [
            ("class:prompt.main", "biobank"),
            ("class:prompt.sep", " › "),
        ]
    )


def _read_query(prompt_session: PromptSession | None, planner: PlanMode) -> str:
    """Read user input with prompt-toolkit on TTY, fallback otherwise."""
    if _PENDING_STDIN_LINES:
        return _PENDING_STDIN_LINES.pop(0)
    if prompt_session is None:
        prompt_str = "[bold magenta]plan>[/] " if planner.is_active else "[bold cyan]biobank>[/] "
        return console.input(prompt_str)
    plan_status = getattr(planner, "status", "") if planner.is_active else ""
    return prompt_session.prompt(_prompt_message(planner.is_active, plan_status))


def _read_multiline_plan_goal() -> str:
    """Read a natural-language task after a bare /plan command."""
    console.print(Panel(
        "\n".join([
            "[bold]Describe the task to plan.[/bold]",
            "",
            "Type a short natural-language request. Finish with an empty line.",
            "Example: 分析这批白癜风WGS数据，比较青少年白癜风和白癜风组",
        ]),
        title="New Plan",
        border_style="cyan",
    ))
    lines: list[str] = []
    while True:
        try:
            raw = console.input("[bold cyan]task[/] > " if not lines else "[dim]...[/] ")
        except EOFError:
            break
        if not raw.strip():
            break
        lines.append(raw.rstrip())
        if len(lines) >= 200:
            break
    return "\n".join(lines).strip()


def _looks_like_plan_paste_start(query: str) -> bool:
    """Heuristic for a first line that is probably the header of a pasted plan."""
    stripped = (query or "").strip()
    if not stripped.lower().startswith("/plan "):
        return False
    arg = stripped.split(None, 1)[1].strip()
    if "\n" in arg:
        return False
    lower = arg.lower()
    return (
        arg.endswith(":")
        or len(arg.split()) <= 8
        and any(token in lower for token in ("task", "workflow", "paper", "论文", "复现", "研究"))
    )


def _collect_pasted_command_lines(query: str) -> str:
    """Merge immediately pasted continuation lines into a slash command.

    A common human pattern is pasting:

        /plan Title:
        paragraph one
        paragraph two

    Line-oriented REPLs otherwise execute the first line immediately and treat
    the rest as unrelated input. This function only drains lines already waiting
    on stdin, so normal typed commands are not delayed beyond a short grace.
    """
    if not query or not query.lstrip().startswith("/"):
        return query
    if "\n" in query:
        return query

    # Only use a longer grace period for /plan-like headers. Other commands get
    # a tiny drain window for bracketed paste/newline bursts without changing
    # ordinary interactive behavior.
    initial_wait = 0.35 if _looks_like_plan_paste_start(query) else 0.03
    continuation: list[str] = []
    try:
        stream = sys.stdin
        stream.fileno()
        wait = initial_wait
        while True:
            ready, _, _ = select.select([stream], [], [], wait)
            if not ready:
                break
            line = stream.readline()
            if line == "":
                break
            stripped = line.strip()
            if stripped.startswith("/") or stripped.lower() in {"quit", "exit", "q"}:
                _PENDING_STDIN_LINES.append(line.rstrip("\n"))
                break
            continuation.append(line.rstrip("\n"))
            wait = 0.02
            if len(continuation) >= 200:
                break
    except Exception:
        return query

    if not continuation:
        return query
    if query.lower().startswith("/plan "):
        console.print(f"[dim]Collected {len(continuation)} pasted continuation line(s) for /plan.[/dim]")
    return query.rstrip("\n") + "\n" + "\n".join(continuation)


def _extract_pdf_path_from_text(text: str) -> str | None:
    """Extract a PDF path from pasted natural-language text."""
    match = re.search(r"((?:/|~/|\./|\../)?[^\s'\"<>]+\.pdf)", text or "", flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).rstrip(".,;:)")


def _absorb_pasted_plan_continuation(planner: PlanMode, text: str) -> bool:
    """Absorb non-command lines that belong to a just-pasted /plan goal.

    Prompt-toolkit can accept the first line of a pasted block as a complete
    command and return later lines on subsequent prompts. In plan review, those
    later lines should extend the original goal, not trigger LLM refinement.
    """
    continuation = (text or "").strip()
    if not continuation or continuation.startswith("/"):
        return False
    if planner.state != PlanState.REVIEW or not planner.plan:
        return False
    if not getattr(planner, "_awaiting_pasted_plan_continuation", False):
        return False

    pdf_path = _extract_pdf_path_from_text(continuation)
    # Once the REPL has identified a pasted /plan header, keep absorbing
    # following non-command lines until the next slash command. Some useful
    # continuations are short ("E11", "use tabular biomarkers") and should
    # still affect the final re-planned goal.

    if continuation not in planner.goal:
        planner.goal = f"{planner.goal.rstrip()}\n{continuation}".strip()
        planner.plan.goal = planner.goal

    if pdf_path:
        for step in planner.plan.steps:
            if step.skill != "read_paper":
                continue
            args = dict(step.args or {})
            current = str(args.get("paper_path_or_doi", ""))
            if not current.lower().endswith(".pdf") or current == planner.goal.splitlines()[0].strip():
                args["paper_path_or_doi"] = pdf_path
                step.args = args
            break

    setattr(planner, "_needs_replan_from_pasted_goal", True)
    planner.revision += 1
    planner.validate_current_plan()
    try:
        planner._save_plan_file()
    except Exception:
        logger.debug("Failed to save plan after pasted continuation", exc_info=True)
    return True


def _refresh_plan_from_pasted_goal(planner: PlanMode) -> str | None:
    """Rebuild a just-created plan using the full pasted goal before approval."""
    if not getattr(planner, "_needs_replan_from_pasted_goal", False):
        return None
    if planner.state != PlanState.REVIEW or not planner.plan:
        return None

    try:
        try:
            new_plan = planner.planner.decompose(
                goal=planner.goal,
                available_skills=planner.available_skills,
                tool_schemas=planner.tool_schemas,
                spec=getattr(planner, "current_study_spec", None),
            )
        except TypeError:
            new_plan = planner.planner.decompose(
                goal=planner.goal,
                available_skills=planner.available_skills,
                tool_schemas=planner.tool_schemas,
            )
        planner.plan = new_plan
        if hasattr(planner, "_attach_study_spec_metadata"):
            planner._attach_study_spec_metadata()
        planner.revision += 1
        planner.validate_current_plan()
        planner._save_plan_file()
        setattr(planner, "_needs_replan_from_pasted_goal", False)
        setattr(planner, "_awaiting_pasted_plan_continuation", False)
        return f"Plan updated from pasted multi-line goal (v{planner.revision})."
    except Exception as exc:
        logger.warning("Failed to refresh plan from pasted goal: %s", exc)
        return "Could not rebuild plan from pasted goal; keeping the current reviewed plan."


def _read_single_key() -> str:
    """Read one raw key sequence from stdin for lightweight TTY selectors."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = sys.stdin.read(1)
        if first == "\x1b":
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if ready:
                second = sys.stdin.read(1)
                if second == "[":
                    ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                    third = sys.stdin.read(1) if ready else ""
                    return first + second + third
                return first + second
        return first
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _keyboard_select_option(question: dict, options: list[dict]) -> dict | None:
    """Interactive up/down/space/enter selector for single-choice clarifications."""
    result = _keyboard_select(question, options, multi=False)
    if isinstance(result, dict):
        return result
    return None


def _keyboard_select(question: dict, options: list[dict], *, multi: bool = False) -> dict | list[dict] | None:
    """Interactive up/down/space/enter selector for plan clarification choices."""
    if not _stdin_is_interactive():
        return None
    try:
        selected = 0
        checked: set[int] = {idx for idx, opt in enumerate(options) if opt.get("default_checked")}
        if not checked:
            checked = {0}

        def render() -> None:
            mode = "multi-select" if multi else "single-select"
            rows = [
                f"[bold]{question.get('question', 'Clarification needed')}[/bold]",
                "",
            ]
            for idx, opt in enumerate(options):
                cursor = ">" if idx == selected else " "
                marker = "●" if idx in checked else "○"
                rows.append(
                    f"{cursor} {marker} {opt.get('label', f'Option {idx + 1}')} - "
                    f"{opt.get('description', '')}"
                )
            rows.extend([
                "",
                f"[dim]{mode}: Use ↑/↓ to move, Space to select, Enter to confirm. "
                "Press 1-9 for direct choice.[/dim]",
            ])
            console.print(Panel(
                "\n".join(rows),
                title=f"Clarification: {question.get('header', question.get('id', 'Plan'))}",
                border_style="yellow",
            ))

        while True:
            render()
            key = _read_single_key()
            if key in ("\r", "\n"):
                selected_options = [options[idx] for idx in sorted(checked) if 0 <= idx < len(options)]
                if multi:
                    return selected_options or [options[0]]
                return selected_options[0] if selected_options else options[0]
            if key == " ":
                if multi:
                    if selected in checked and len(checked) > 1:
                        checked.remove(selected)
                    else:
                        checked.add(selected)
                else:
                    checked = {selected}
                continue
            if key in ("\x1b[A", "k"):
                selected = (selected - 1) % len(options)
                continue
            if key in ("\x1b[B", "j"):
                selected = (selected + 1) % len(options)
                continue
            if key and key.isdigit():
                index = int(key) - 1
                if 0 <= index < len(options):
                    if multi:
                        return [options[index]]
                    return options[index]
            if key in ("\x03", "\x04"):
                raise KeyboardInterrupt
    except (OSError, termios.error):
        return None


def _collect_plan_clarifications(planner: PlanMode, goal: str) -> tuple[str, list[dict]]:
    """Ask only critical plan-shaping questions before decomposition."""
    questions = planner.identify_clarifications(goal)
    if not questions:
        return goal, []

    answers: list[dict] = []
    for question in questions:
        options = list(question.get("options", []) or [])
        if not options:
            continue
        selected = options[0]
        is_multi = bool(question.get("multi"))
        selected_multi: list[dict] | None = None
        if _stdin_is_interactive():
            if is_multi:
                keyboard_selected = _keyboard_select(question, options, multi=True)
                if keyboard_selected:
                    selected_multi = [item for item in keyboard_selected if isinstance(item, dict)]
                    selected = keyboard_selected[0]
            else:
                keyboard_selected = _keyboard_select_option(question, options)
                if keyboard_selected is not None:
                    selected = keyboard_selected
            if not is_multi and keyboard_selected is None:
                console.print()
                console.print(Panel(
                    "\n".join(
                        [
                            f"[bold]{question.get('question', 'Clarification needed')}[/bold]",
                            "",
                            *[
                                f"[cyan]{idx + 1}[/cyan]. {opt.get('label', f'Option {idx + 1}')} - {opt.get('description', '')}"
                                for idx, opt in enumerate(options)
                            ],
                            "",
                            "[dim]Press Enter to accept option 1, or type a number/custom answer.[/dim]",
                        ]
                    ),
                    title=f"Clarification: {question.get('header', question.get('id', 'Plan'))}",
                    border_style="yellow",
                ))
                raw = console.input("[bold yellow]Choice[/] > ").strip()
                if raw:
                    try:
                        index = int(raw) - 1
                        if 0 <= index < len(options):
                            selected = options[index]
                        else:
                            selected = {
                                "label": "Custom",
                                "value": raw,
                                "description": "User supplied custom clarification.",
                            }
                    except ValueError:
                        selected = {
                            "label": "Custom",
                            "value": raw,
                            "description": "User supplied custom clarification.",
                        }
        else:
            console.print()
            console.print(Panel(
                "\n".join(
                    [
                        f"[bold]{question.get('question', 'Clarification needed')}[/bold]",
                        "",
                        *[
                            f"[cyan]{idx + 1}[/cyan]. {opt.get('label', f'Option {idx + 1}')} - {opt.get('description', '')}"
                            for idx, opt in enumerate(options)
                        ],
                    ]
                ),
                title=f"Clarification: {question.get('header', question.get('id', 'Plan'))}",
                border_style="yellow",
            ))
            console.print(f"[dim]Non-interactive mode: using default clarification {selected.get('label')}.[/dim]")

        if is_multi and selected_multi:
            answer_value = "\n".join(
                str(item.get("value") or item.get("description") or item.get("label") or "")
                for item in selected_multi
            )
            answer_label = ", ".join(str(item.get("label") or "") for item in selected_multi if item.get("label"))
        else:
            answer_value = selected.get("value") or selected.get("description") or selected.get("label") or ""
            answer_label = str(selected.get("label", ""))
        answers.append({
            "id": str(question.get("id", "")),
            "question": str(question.get("question", "")),
            "label": answer_label,
            "answer": str(answer_value),
            "source": str(question.get("source") or "planner"),
        })

    if not answers:
        return goal, []

    clarification_text = "\n".join(f"- {item['id']}: {item['answer']}" for item in answers)
    is_wgs_vitiligo = bool(planner.planner._is_wgs_vitiligo_goal(goal))
    is_juvenile_mechanism = bool(planner.planner._is_juvenile_hair_mechanism_goal(goal))
    if is_juvenile_mechanism and not is_wgs_vitiligo:
        clarification_text = (
            f"{clarification_text}\n"
            "- juvenile_hair_mechanism_policy: Treat this as a Juvenile hair-whitening multi-omics "
            "mechanism analysis. Infer the WGS candidate variant, TF binding, scATAC accessibility, "
            "scRNA expression, Stereo spatial, Action Graph, workflow-gap and report trajectory from "
            "registered skills rather than requiring the user to enumerate each analysis step."
        )
    elif is_wgs_vitiligo:
        clarification_text = (
            f"{clarification_text}\n"
            "- wgs_agent_policy: Treat this as a VirtualCell vitiligo WGS analysis. "
            "Infer the executable workflow from available WGS skills and data inventory rather than requiring "
            "the user to enumerate QC, annotation, population genetics, association, burden, enrichment, "
            "model diagnostics, literature, Action Graph, reproducibility, or report steps."
        )
    elif planner.planner._is_virtualcell_multimodal_goal(goal):
        clarification_text = (
            f"{clarification_text}\n"
            "- virtualcell_agent_policy: Treat this as a VirtualCell/BWhair multimodal analysis. "
            "Infer WGS, Stereo-seq, scRNA-seq, scATAC-seq, donor linkage, backed h5ad inspection, "
            "literature context, Action Graph, reproducibility and report steps from the registered skills "
            "rather than requiring the user to enumerate them."
        )
    clarified_goal = f"{goal.rstrip()}\n\nClarifications:\n{clarification_text}".strip()
    return clarified_goal, answers


def _is_broad_metabolic_showcase_goal(goal: str) -> bool:
    """Recognize the grand-challenge style broad research question."""
    lower = str(goal or "").lower()
    data_context = any(token in lower for token in ("ukb", "biobank", "biobank data", "cohort data"))
    return (
        data_context
        and any(token in lower for token in ("metabolic health", "cardiometabolic", "cardio-metabolic"))
        and any(token in lower for token in ("type 2 diabetes", "t2d", "t2dm", "diabetes risk"))
        and any(token in lower for token in ("biomarker", "biomarkers", "actionable"))
        and any(token in lower for token in ("trajectory", "trajectories", "progression", "risk prediction", "prediction"))
    )


def _collect_research_setup(agent: Agent, goal: str) -> tuple[str, list[dict], dict[str, Any]]:
    """Collect benchmark-grade research setup without forcing a long user prompt."""
    if not _is_broad_metabolic_showcase_goal(goal):
        return goal, [], {}

    trajectory_policy = "fallback"
    model_policy = "auto"
    dataset_scope = "ukb"

    def _read_setup_choice() -> str:
        return console.input("[bold cyan]Choice[/] > ").strip()

    def _parse_numbers(raw: str) -> set[str]:
        return {part.strip() for part in re.split(r"[, ]+", raw) if part.strip()}

    def render_dataset_panel() -> None:
        console.print()
        console.print(Panel(
            "\n".join([
                "[bold]Which data source should be active for this run?[/bold]",
                "",
                "1. [x] UKB only — execute against local UKB data",
                "2. [ ] HPP — future port, not used in this benchmark",
                "3. [ ] CKB — future port, not used in this benchmark",
                "4. [ ] UKB-RAP — future port, not used in this benchmark",
                "",
                "[dim]Press Enter for option 1. Future ports are recorded as scope notes, not active execution targets.[/dim]",
            ]),
            title="Research Setup 1/2: Data Scope",
            border_style="cyan",
        ))

    def render_policy_panel() -> None:
        console.print()
        console.print(Panel(
            "\n".join([
                "[bold]Which analysis strategy should Biobank Agent use?[/bold]",
                "",
                "1. [x] Feasible trajectory branch + auto model selection",
                "2. [ ] Trajectory-only feasibility report",
                "3. [ ] Interpretable model baseline first",
                "",
                "[dim]Press Enter for option 1. Choose 2 only when tabular fallback should be avoided.[/dim]",
            ]),
            title="Research Setup 2/2: Analysis Strategy",
            border_style="cyan",
        ))

    if _stdin_is_interactive():
        render_dataset_panel()
        dataset_choice = _read_setup_choice()
        if dataset_choice and "1" not in _parse_numbers(dataset_choice):
            console.print("[yellow]Only UKB is execution-ready. HPP, CKB and UKB-RAP will remain future-port scope notes.[/yellow]")

        render_policy_panel()
        policy_choice = _parse_numbers(_read_setup_choice())
        if "2" in policy_choice:
            trajectory_policy = "trajectory_only"
        if "3" in policy_choice:
            model_policy = "interpretable"
    else:
        console.print("[dim]Non-interactive mode: accepting default Research Setup for broad UKB showcase.[/dim]")

    trajectory_text = (
        "Attempt trajectory feasibility first, then fall back to governed tabular prediction if tokens are sparse."
        if trajectory_policy == "fallback" else
        "Run a trajectory-only feasibility workflow and avoid tabular fallback modelling unless trajectory data are sufficient."
    )
    model_text = (
        "Use train_model model_type=auto and report candidate models, metrics, and rationale."
        if model_policy == "auto" else
        "Prefer an interpretable model baseline unless performance is clearly inadequate."
    )
    answers = [
        {
            "id": "research_setup_dataset",
            "question": "Which data sources are active for this benchmark?",
            "label": "UKB only",
            "answer": "Use UKB as the active execution dataset. Treat HPP, CKB and UKB-RAP as future ports only.",
        },
        {
            "id": "research_setup_trajectory_policy",
            "question": "How should incomplete trajectory support be handled?",
            "label": "Feasible fallback" if trajectory_policy == "fallback" else "Trajectory only",
            "answer": trajectory_text,
        },
        {
            "id": "research_setup_model_policy",
            "question": "How should model choice be handled?",
            "label": "Auto select" if model_policy == "auto" else "Interpretable baseline",
            "answer": model_text,
        },
    ]
    setup_text = "\n".join(f"- {item['id']}: {item['answer']}" for item in answers)
    autonomous_policy = (
        "Autonomous research setup:\n"
        f"{setup_text}\n"
        "- agent_role: Act as an autonomous biobank research scientist. Inspect curated project documentation, "
        "available skills, and the current UKB data inventory before final conclusions.\n"
        "- expected_workflow: Let the planner derive the strongest feasible UKB-only workflow from the broad question; "
        "include field discovery, related work, cohort design, trajectory feasibility, model diagnostics, guardrails, "
        "and dual reports when supported by available tools."
    )
    clarified_goal = f"{goal.rstrip()}\n\n{autonomous_policy}".strip()
    return clarified_goal, answers, {
        "dataset_scope": dataset_scope,
        "trajectory_policy": trajectory_policy,
        "model_policy": model_policy,
    }


def _extract_external_planner_questions(records: list[dict], max_questions: int = 3) -> list[dict]:
    """Extract explicit blocking user questions from external planner output."""
    questions: list[dict] = []
    seen: set[tuple[str, str]] = set()
    heading_re = re.compile(r"^\s*(?:#+\s*)?blocking\s+questions?\s*:?\s*(.*)$", re.IGNORECASE)
    item_re = re.compile(r"^\s*(?:[-*]\s*|\d+[\.)]\s*)(.+)$")
    next_section_re = re.compile(r"^\s*(?:#+\s*)?[A-Z][A-Za-z0-9 /_-]{2,}:\s*$")

    for record in records or []:
        if not isinstance(record, dict) or str(record.get("status", "")).lower() != "success":
            continue
        agent_name = str(record.get("agent") or "external")
        text = str(record.get("stdout") or record.get("summary") or "")
        in_section = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            heading = heading_re.match(line)
            if heading:
                tail = heading.group(1).strip()
                in_section = True
                if tail and tail.lower().strip(".") not in {"none", "no", "n/a", "not needed"}:
                    candidate = item_re.sub(r"\1", tail).strip()
                    if "?" in candidate:
                        key = (agent_name, candidate.lower())
                        if key not in seen:
                            questions.append({"agent": agent_name, "question": candidate})
                            seen.add(key)
                continue
            if not in_section:
                continue
            if next_section_re.match(line) and "?" not in line:
                break
            candidate_match = item_re.match(line)
            candidate = (candidate_match.group(1) if candidate_match else line).strip()
            normalized = candidate.lower().strip(".")
            if normalized in {"none", "no", "n/a", "not needed", "no blocking questions"}:
                break
            if "?" not in candidate:
                continue
            key = (agent_name, candidate.lower())
            if key in seen:
                continue
            questions.append({"agent": agent_name, "question": candidate})
            seen.add(key)
            if len(questions) >= max_questions:
                return questions
    return questions[:max_questions]


def _auto_answer_external_planner_question(question: str, goal: str) -> str:
    """Answer external-planner questions only when the current goal is explicit."""
    q = str(question or "").lower()
    g = str(goal or "").lower()
    ukb_only = "ukb-only" in g or "ukb only" in g or "this benchmark is ukb-only" in g
    if ukb_only and any(term in q for term in ("hpp", "ckb", "rap", "bank", "biobank")):
        return "This run is UKB-only. Treat HPP, CKB and RAP as future ports, not active execution targets."
    if ("raw" in q or "csv" in q or "material" in q) and ("raw csv" in g or "full ukb" in g):
        return (
            "Yes. Inspect the full UKB raw CSV inventory and materialize needed UKB fields when "
            "the current parquet subset is incomplete."
        )
    if "trajectory" in q and ("feasible fallback" in g or "fallback" in g):
        return (
            "Use the feasible-fallback trajectory policy: attempt tokenization, avoid unsupported "
            "temporal claims, and fall back to valid tabular/discrimination analysis when needed."
        )
    return ""


def _collect_external_planner_clarifications(questions: list[dict], goal: str = "") -> tuple[str, list[dict]]:
    """Ask the user for unresolved questions raised by external planners."""
    if not questions:
        return "", []

    answers: list[dict] = []
    for idx, item in enumerate(questions, start=1):
        agent_name = str(item.get("agent") or "external")
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        auto_answer = _auto_answer_external_planner_question(question, goal)
        if auto_answer:
            console.print(f"[dim]Auto-answering {agent_name} planner question from the current task context.[/dim]")
            answers.append({
                "id": f"external_planner_{idx}",
                "agent": agent_name,
                "question": question,
                "answer": auto_answer,
                "label": "Auto from task context",
            })
            continue
        default_answer = (
            "Use Biobank Agent's safest conservative assumption, proceed only with "
            "schema-valid UKB-only analysis, and document the uncertainty explicitly."
        )
        console.print()
        console.print(Panel(
            "\n".join([
                f"[bold]{question}[/bold]",
                "",
                "[dim]Type an answer to pass back into the planning council.[/dim]",
                "[dim]Press Enter to let Biobank Agent use the conservative default.[/dim]",
            ]),
            title=f"External Planner Question: {agent_name}",
            border_style="yellow",
        ))
        raw = ""
        if _stdin_is_interactive():
            raw = console.input("[bold yellow]Answer[/] > ").strip()
        else:
            console.print("[dim]Non-interactive mode: using conservative default answer.[/dim]")
        answer = raw or default_answer
        answers.append({
            "id": f"external_planner_{idx}",
            "agent": agent_name,
            "question": question,
            "answer": answer,
            "label": "User answer" if raw else "Conservative default",
        })

    if not answers:
        return "", []
    text = "\n".join(
        f"- {item['agent']} asked: {item['question']} Answer: {item['answer']}"
        for item in answers
    )
    return text, answers


def _render_startup_dashboard(
    settings,
    n_skills: int,
    n_subjects: int | None,
    n_fields: int,
    model_pool: list[str] | None = None,
    available_model_count: int = 0,
) -> None:
    """Render a modern startup dashboard."""
    model_pool = model_pool or [settings.llm_model]
    model_pool_preview = ", ".join(model_pool[:3])
    if len(model_pool) > 3:
        model_pool_preview += f" (+{len(model_pool) - 3})"

    left = Table.grid(padding=(0, 1))
    left.add_row(f"[bold #56d4dd]BioBank Agent[/] [bold #94a3b8]{__version__}[/]")
    left.add_row("[#cbd5e1]Autonomous Scientific Discovery[/]")
    left.add_row(f"[#7dd3fc]Data source[/]: [bold]{settings.biobank_name}[/]")
    left.add_row("")
    left.add_row(f"[#7dd3fc]Model[/]: [#e2e8f0]{settings.llm_model}[/]")
    left.add_row(f"[#7dd3fc]Routing[/]: [#e2e8f0]{'auto multi-agent' if settings.multi_model_enabled else 'single model'}[/]")
    left.add_row(f"[#7dd3fc]Active pool[/]: [#e2e8f0]{model_pool_preview}[/]")
    left.add_row(f"[#7dd3fc]Data[/]: [#a78bfa]{settings.data_dir}[/]")

    right = Table.grid(padding=(0, 1))
    right.add_row("[bold #fb7185]Quick Start[/]")
    right.add_row("[#94a3b8]/help[/] browse all commands")
    right.add_row("[#94a3b8]type [bold]/[/] + [bold]Tab[/] for autocomplete")
    right.add_row("[#94a3b8]/status[/] session state")
    right.add_row("[#94a3b8]/skills[/] analysis tools")
    right.add_row("[#94a3b8]/mcp-list[/] MCP server inventory")
    right.add_row("[#94a3b8]/models-available[/] relay model catalog")
    if available_model_count > 0:
        right.add_row(f"[#94a3b8]relay models discovered:[/] [bold]{available_model_count}[/]")

    summary = Table.grid(expand=True)
    summary.add_column(justify="left")
    subject_text = f"{n_subjects:,}" if n_subjects is not None else "N/A"
    summary.add_row(
        f"[bold #34d399]●[/] skills [bold]{n_skills}[/]   "
        f"[bold #38bdf8]●[/] subjects [bold]{subject_text}[/]   "
        f"[bold #a78bfa]●[/] fields [bold]{n_fields:,}[/]   "
        f"[bold #f59e0b]●[/] models [bold]{len(model_pool)}[/]"
    )

    console.print(
        Columns(
            [
                Panel(
                    left,
                    title="[bold #56d4dd]Workspace[/]",
                    border_style="#334155",
                    box=box.ROUNDED,
                    padding=(1, 2),
                ),
                Panel(
                    right,
                    title="[bold #fb7185]Hints[/]",
                    border_style="#334155",
                    box=box.ROUNDED,
                    padding=(1, 2),
                ),
            ],
            equal=True,
            expand=True,
        )
    )
    console.print(
        Panel(
            summary,
            border_style="#1f2937",
            box=box.SQUARE,
            padding=(0, 1),
        )
    )
    console.print()


def _quick_external_agent_status() -> dict[str, dict]:
    """No external CLI agents are wired into the runtime; always empty."""
    return {}


def rebuild_parquet_cmd() -> None:
    """Subcommand: rebuild parquet from category CSVs."""
    from .data.parquet_builder import batch_rebuild

    settings = get_settings()
    console.print("[bold cyan]Parquet Rebuild Tool[/]")
    console.print(f"[dim]Raw CSV dir: {settings.raw_csv_dir}[/]")
    console.print(f"[dim]Output dir:  {settings.category_parquet_dir}[/]")
    console.print(f"[dim]Existing:    {settings.biomarker_parquet}[/]")
    console.print()

    # Parse category filter from args
    categories = None
    for arg in sys.argv[2:]:
        if arg.startswith("--categories="):
            categories = arg.split("=", 1)[1].split(",")

    def progress_callback(cat: str, msg: str) -> None:
        console.print(f"  [cyan]{msg}[/]")

    console.print("[bold green]Starting batch rebuild...[/]")
    with console.status("[bold green]Building parquet..."):
        result = batch_rebuild(
            raw_csv_dir=settings.raw_csv_dir,
            output_dir=settings.category_parquet_dir,
            existing_parquet_dir=settings.biomarker_parquet,
            categories=categories,
            callback=progress_callback,
        )

    console.print()
    console.print(Panel(
        f"[green]New fields:[/] {result['total_new_fields']}\n"
        f"[green]New columns:[/] {result['total_new_columns']}\n"
        f"[green]Existing fields:[/] {result['total_existing_fields']}\n"
        f"[green]Output:[/] {result['output_dir']}",
        title="Rebuild Complete",
    ))

    for cat, info in result["categories"].items():
        if info.get("skipped"):
            console.print(f"  [dim]{cat}: skipped ({info['reason']})[/]")
        else:
            console.print(
                f"  [green]{cat}:[/] {info['n_fields']} fields, "
                f"{info['n_cols']} cols, {info.get('elapsed_s', '?')}s"
            )


def build_ukb_full_parquet_cmd() -> None:
    """Subcommand: build a partitioned full UKB feature-store from raw CSVs."""
    from .data.parquet_builder import build_full_ukb_feature_store

    settings = get_settings()
    raw_dir = settings.raw_dir
    output_dir = settings.full_ukb_feature_store
    sources = None
    field_ids = None
    chunk_cols = 500
    sample_size = None
    dry_run = False
    resume = True

    args = iter(sys.argv[2:])
    for arg in args:
        if arg == "--dry-run":
            dry_run = True
        elif arg == "--no-resume":
            resume = False
        elif arg.startswith("--raw-dir="):
            raw_dir = Path(arg.split("=", 1)[1]).expanduser()
        elif arg == "--raw-dir":
            raw_dir = Path(next(args)).expanduser()
        elif arg.startswith("--out-dir="):
            output_dir = Path(arg.split("=", 1)[1]).expanduser()
        elif arg == "--out-dir":
            output_dir = Path(next(args)).expanduser()
        elif arg.startswith("--sources="):
            sources = [x.strip() for x in arg.split("=", 1)[1].split(",") if x.strip()]
        elif arg == "--sources":
            sources = [x.strip() for x in next(args).split(",") if x.strip()]
        elif arg.startswith("--field-ids="):
            field_ids = [x.strip() for x in arg.split("=", 1)[1].split(",") if x.strip()]
        elif arg == "--field-ids":
            field_ids = [x.strip() for x in next(args).split(",") if x.strip()]
        elif arg.startswith("--chunk-cols="):
            chunk_cols = int(arg.split("=", 1)[1])
        elif arg == "--chunk-cols":
            chunk_cols = int(next(args))
        elif arg.startswith("--sample-size="):
            sample_size = int(arg.split("=", 1)[1])
        elif arg == "--sample-size":
            sample_size = int(next(args))

    console.print("[bold cyan]Full UKB Feature-Store Builder[/]")
    console.print(f"[dim]Raw dir:    {raw_dir}[/]")
    console.print(f"[dim]Output dir: {output_dir}[/]")
    console.print(f"[dim]Fields:     {','.join(field_ids) if field_ids else 'ALL'}[/]")
    console.print(f"[dim]Sources:    {','.join(sources) if sources else 'ALL'}[/]")
    console.print(f"[dim]Dry run:    {dry_run}[/]")
    console.print()

    def progress_callback(source: str, msg: str) -> None:
        console.print(f"  [cyan]{source}[/] {msg}")

    result = build_full_ukb_feature_store(
        raw_dir=raw_dir,
        output_dir=output_dir,
        sources=sources,
        field_ids=field_ids,
        chunk_cols=chunk_cols,
        sample_size=sample_size,
        dry_run=dry_run,
        resume=resume,
        callback=progress_callback,
    )

    source_rows = result.get("sources", [])
    if isinstance(source_rows, dict):
        source_rows = list(source_rows.values())
    ready = sum(1 for row in source_rows if row.get("status") in {"READY", "SKIPPED_EXISTS"})
    chunks = result.get("n_chunks") or sum(
        len(row.get("chunks", [])) if "chunks" in row else int(row.get("n_chunks", 0) or 0)
        for row in source_rows
    )
    console.print()
    console.print(Panel(
        f"[green]Status:[/] {result.get('status')}\n"
        f"[green]Ready sources:[/] {ready}/{len(source_rows)}\n"
        f"[green]Selected feature columns:[/] {result.get('n_selected_feature_columns', '?')}\n"
        f"[green]Chunks:[/] {chunks}\n"
        f"[green]Manifest:[/] {result.get('manifest_path') or result.get('manifest') or output_dir / 'manifest.json'}",
        title="Full UKB Feature Store",
    ))


def _parse_eval_args(args: list[str]) -> EvalArgs:
    """Parse eval args in both --flag=value and --flag value forms."""
    suite = "research_eval_v1"
    mode = "baseline"
    policy = "all"
    enforce_gate = False
    ab_compare = False
    baseline_report = ""
    review_loop = False
    primary_reviewer = DEFAULT_EVAL_REVIEWER
    include_claude = False
    review_timeout_s = 600
    evolution_history = ""
    fail_on_evolution_patterns = False
    run_dir = ""
    hpp_ckb_rap_readiness = ""
    mcp_compat_evidence = ""
    remote_ci_evidence = ""
    high_pr_evidence = ""
    external_evidence_dir = ""
    collect_external = False
    collect = "all"
    banks = "ukb,hpp,ckb,ukb_rap"
    icd10_code = "E11"
    probe_fields = "hba1c,bmi,glucose"
    mcp_config = ""
    mcp_call_args = ""
    mcp_min_servers = 2
    workflow = "biobank-scheduled-eval.yml"
    pr_url = ""
    branch = ""

    i = 0
    while i < len(args):
        arg = args[i]
        next_value = args[i + 1].strip() if i + 1 < len(args) else ""
        if arg.startswith("--suite="):
            suite = arg.split("=", 1)[1].strip().lower()
        elif arg == "--suite" and next_value:
            suite = next_value.lower()
            i += 1
        elif arg.startswith("--mode="):
            mode = arg.split("=", 1)[1].strip().lower()
        elif arg == "--mode" and next_value:
            mode = next_value.lower()
            i += 1
        elif arg.startswith("--policy="):
            policy = arg.split("=", 1)[1].strip().lower()
        elif arg == "--policy" and next_value:
            policy = next_value.lower()
            i += 1
        elif arg.startswith("--baseline-report="):
            baseline_report = arg.split("=", 1)[1].strip()
        elif arg == "--baseline-report" and next_value:
            baseline_report = next_value
            i += 1
        elif arg.startswith("--evolution-history="):
            evolution_history = arg.split("=", 1)[1].strip()
        elif arg == "--evolution-history" and next_value:
            evolution_history = next_value
            i += 1
        elif arg.startswith("--run-dir="):
            run_dir = arg.split("=", 1)[1].strip()
        elif arg == "--run-dir" and next_value:
            run_dir = next_value
            i += 1
        elif arg.startswith("--hpp-ckb-rap-readiness="):
            hpp_ckb_rap_readiness = arg.split("=", 1)[1].strip()
        elif arg == "--hpp-ckb-rap-readiness" and next_value:
            hpp_ckb_rap_readiness = next_value
            i += 1
        elif arg.startswith("--mcp-compat-evidence="):
            mcp_compat_evidence = arg.split("=", 1)[1].strip()
        elif arg == "--mcp-compat-evidence" and next_value:
            mcp_compat_evidence = next_value
            i += 1
        elif arg.startswith("--remote-ci-evidence="):
            remote_ci_evidence = arg.split("=", 1)[1].strip()
        elif arg == "--remote-ci-evidence" and next_value:
            remote_ci_evidence = next_value
            i += 1
        elif arg.startswith("--high-pr-evidence="):
            high_pr_evidence = arg.split("=", 1)[1].strip()
        elif arg == "--high-pr-evidence" and next_value:
            high_pr_evidence = next_value
            i += 1
        elif arg.startswith("--external-evidence-dir="):
            external_evidence_dir = arg.split("=", 1)[1].strip()
        elif arg == "--external-evidence-dir" and next_value:
            external_evidence_dir = next_value
            i += 1
        elif arg == "--collect-external":
            collect_external = True
        elif arg.startswith("--collect="):
            collect = arg.split("=", 1)[1].strip().lower()
        elif arg == "--collect" and next_value:
            collect = next_value.lower()
            i += 1
        elif arg.startswith("--banks="):
            banks = arg.split("=", 1)[1].strip()
        elif arg == "--banks" and next_value:
            banks = next_value
            i += 1
        elif arg.startswith("--icd10-code="):
            icd10_code = arg.split("=", 1)[1].strip()
        elif arg == "--icd10-code" and next_value:
            icd10_code = next_value
            i += 1
        elif arg.startswith("--probe-fields="):
            probe_fields = arg.split("=", 1)[1].strip()
        elif arg == "--probe-fields" and next_value:
            probe_fields = next_value
            i += 1
        elif arg.startswith("--mcp-config="):
            mcp_config = arg.split("=", 1)[1].strip()
        elif arg == "--mcp-config" and next_value:
            mcp_config = next_value
            i += 1
        elif arg.startswith("--mcp-call-args="):
            mcp_call_args = arg.split("=", 1)[1].strip()
        elif arg == "--mcp-call-args" and next_value:
            mcp_call_args = next_value
            i += 1
        elif arg.startswith("--mcp-min-servers="):
            try:
                mcp_min_servers = max(1, int(arg.split("=", 1)[1].strip()))
            except ValueError:
                mcp_min_servers = 2
        elif arg == "--mcp-min-servers" and next_value:
            try:
                mcp_min_servers = max(1, int(next_value))
            except ValueError:
                mcp_min_servers = 2
            i += 1
        elif arg.startswith("--workflow="):
            workflow = arg.split("=", 1)[1].strip()
        elif arg == "--workflow" and next_value:
            workflow = next_value
            i += 1
        elif arg.startswith("--pr-url="):
            pr_url = arg.split("=", 1)[1].strip()
        elif arg == "--pr-url" and next_value:
            pr_url = next_value
            i += 1
        elif arg.startswith("--branch="):
            branch = arg.split("=", 1)[1].strip()
        elif arg == "--branch" and next_value:
            branch = next_value
            i += 1
        elif arg.startswith("--reviewer="):
            primary_reviewer = arg.split("=", 1)[1].strip() or DEFAULT_EVAL_REVIEWER
        elif arg == "--reviewer" and next_value:
            primary_reviewer = next_value
            i += 1
        elif arg.startswith("--review-timeout="):
            try:
                review_timeout_s = max(1, int(arg.split("=", 1)[1].strip()))
            except ValueError:
                review_timeout_s = 600
        elif arg == "--review-timeout" and next_value:
            try:
                review_timeout_s = max(1, int(next_value))
            except ValueError:
                review_timeout_s = 600
            i += 1
        elif arg == "--review-loop":
            review_loop = True
        elif arg == "--include-claude":
            include_claude = True
        elif arg == "--ab":
            ab_compare = True
        elif arg == "--enforce-gate":
            enforce_gate = True
        elif arg == "--fail-on-evolution-patterns":
            fail_on_evolution_patterns = True
        i += 1

    return EvalArgs(
        suite=suite,
        mode=mode,
        policy=policy,
        enforce_gate=enforce_gate,
        ab_compare=ab_compare,
        baseline_report=baseline_report,
        review_loop=review_loop,
        primary_reviewer=primary_reviewer,
        include_claude=include_claude,
        review_timeout_s=review_timeout_s,
        evolution_history=evolution_history,
        fail_on_evolution_patterns=fail_on_evolution_patterns,
        run_dir=run_dir,
        hpp_ckb_rap_readiness=hpp_ckb_rap_readiness,
        mcp_compat_evidence=mcp_compat_evidence,
        remote_ci_evidence=remote_ci_evidence,
        high_pr_evidence=high_pr_evidence,
        external_evidence_dir=external_evidence_dir,
        collect_external=collect_external,
        collect=collect,
        banks=banks,
        icd10_code=icd10_code,
        probe_fields=probe_fields,
        mcp_config=mcp_config,
        mcp_call_args=mcp_call_args,
        mcp_min_servers=mcp_min_servers,
        workflow=workflow,
        pr_url=pr_url,
        branch=branch,
    )


def eval_cmd() -> None:
    """Subcommand: run benchmark suites with reliability observability metrics.

    Usage:
      biobank eval --suite report_20_case --mode baseline|mas_v2
      biobank eval --suite report_20_case --review-loop [--include-claude]
      biobank eval --suite live_ukb_report_20 --review-loop
      biobank eval --suite behavioral --policy all|always|usually
      biobank eval --suite scheduled --policy all [--evolution-history failures.jsonl]
      biobank eval --suite live_artifacts --run-dir reports/biobank_live_tests/.../RUN
      biobank eval --suite external_evidence --collect bank-readiness|mcp|remote-ci|high-pr|all
      biobank eval --suite v3_completion --run-dir reports/biobank_live_tests/.../RUN
    """
    from .eval.benchmarks import (
        AgentReportWorkflowBenchmark,
        BiomedQABenchmark,
        LiveUKBReport20Benchmark,
        Report20CaseBenchmark,
        ReportQualityBenchmark,
        ResearchEvalV1,
        SkillCallBenchmark,
        SkillSchemaBenchmark,
    )
    from .eval.harness import EvalHarness

    eval_args = _parse_eval_args(sys.argv[2:])
    suite = eval_args.suite
    mode = eval_args.mode
    enforce_gate = eval_args.enforce_gate
    ab_compare = eval_args.ab_compare
    baseline_report = eval_args.baseline_report

    settings = get_settings()
    settings.ensure_dirs()
    if mode == "baseline":
        settings.multi_model_enabled = False
    elif mode == "mas_v2":
        settings.multi_model_enabled = True
    else:
        console.print(f"[red]Unknown mode: {mode}. Use baseline or mas_v2.[/]")
        raise SystemExit(2)

    if suite == "behavioral":
        from .eval.behavioral import run_behavioral_eval

        if eval_args.policy not in {"always", "usually", "all"}:
            console.print("[red]Unknown behavioral policy: "
                          f"{eval_args.policy}. Use always, usually, or all.[/]")
            raise SystemExit(2)
        policies = ("always", "usually") if eval_args.policy == "all" else (eval_args.policy,)
        result = run_behavioral_eval(
            output_dir=settings.reports_dir / "eval" / "behavioral",
            policies=policies,
        )
        console.print(Panel(
            f"Status: {result.status}\n"
            f"Cases: {result.n_passed}/{result.n_total}\n"
            f"Pass rate: {result.pass_rate:.2%}\n"
            f"Latest: {result.artifacts.get('latest_json', '')}",
            title="Behavioral Evaluation",
        ))
        for artifact_name, path in result.artifacts.items():
            console.print(f"[green]{artifact_name}:[/] {path}")
        if enforce_gate and result.status != "PASS":
            raise SystemExit(3)
        return

    if suite == "scheduled":
        from .eval.scheduled import run_scheduled_quality_gates

        if eval_args.policy not in {"always", "usually", "all"}:
            console.print("[red]Unknown scheduled behavioral policy: "
                          f"{eval_args.policy}. Use always, usually, or all.[/]")
            raise SystemExit(2)
        policies = ("always", "usually") if eval_args.policy == "all" else (eval_args.policy,)
        result = run_scheduled_quality_gates(
            output_dir=settings.reports_dir / "eval" / "scheduled",
            behavioral_output_dir=settings.reports_dir / "eval" / "behavioral",
            evolution_output_dir=settings.reports_dir / "eval" / "evolution",
            behavioral_policies=policies,
            evolution_history=eval_args.evolution_history or None,
            fail_on_evolution_patterns=eval_args.fail_on_evolution_patterns,
        )
        console.print(Panel(
            f"Status: {result.status}\n"
            f"Behavioral: {result.behavioral_status}\n"
            f"Evolution: {result.evolution_status}\n"
            f"Latest: {result.artifacts.get('latest_json', '')}",
            title="Scheduled Quality Gates",
        ))
        for artifact_name, path in result.artifacts.items():
            console.print(f"[green]{artifact_name}:[/] {path}")
        if enforce_gate and result.status not in {"PASS", "NEEDS_REVIEW"}:
            raise SystemExit(3)
        if enforce_gate and result.status == "NEEDS_REVIEW" and eval_args.fail_on_evolution_patterns:
            raise SystemExit(3)
        return

    if suite == "live_artifacts":
        from .eval.live_artifact_audit import run_live_artifact_audit, write_audit_artifacts

        if not eval_args.run_dir:
            console.print("[red]Missing --run-dir for live_artifacts suite.[/]")
            raise SystemExit(2)
        result = write_audit_artifacts(run_live_artifact_audit(eval_args.run_dir))
        console.print(Panel(
            f"Status: {result.status}\n"
            f"Failures: {result.n_failed}\n"
            f"Warnings: {result.n_warning}\n"
            f"Markdown: {result.artifacts.get('markdown', '')}",
            title="Strict Live Artifact Audit",
        ))
        for artifact_name, path in result.artifacts.items():
            console.print(f"[green]{artifact_name}:[/] {path}")
        if enforce_gate and result.status != "PASS":
            raise SystemExit(3)
        return

    if suite == "v3_completion":
        from .eval.v3_completion import collect_external_evidence_for_completion, run_v3_completion_audit

        external_evidence_dir = eval_args.external_evidence_dir or str(settings.reports_dir / "eval" / "external_evidence")
        if eval_args.collect_external:
            collected = collect_external_evidence_for_completion(
                output_dir=external_evidence_dir,
                banks=eval_args.banks,
                icd10_code=eval_args.icd10_code,
                probe_fields=eval_args.probe_fields,
                mcp_config=eval_args.mcp_config or None,
                mcp_call_args=eval_args.mcp_call_args or None,
                mcp_min_servers=eval_args.mcp_min_servers,
                workflow=eval_args.workflow,
                repo_root=Path.cwd(),
                pr_url=eval_args.pr_url,
                branch=eval_args.branch,
            )
            for artifact_name, path in collected.items():
                console.print(f"[green]collected {artifact_name}:[/] {path}")

        result = run_v3_completion_audit(
            repo_root=Path.cwd(),
            live_run_dir=eval_args.run_dir or None,
            hpp_ckb_rap_readiness=eval_args.hpp_ckb_rap_readiness or None,
            mcp_compat_evidence=eval_args.mcp_compat_evidence or None,
            remote_ci_evidence=eval_args.remote_ci_evidence or None,
            high_pr_evidence=eval_args.high_pr_evidence,
            external_evidence_dir=external_evidence_dir,
            output_dir=settings.reports_dir / "eval" / "v3_completion",
        )
        console.print(Panel(
            f"Status: {result.status}\n"
            f"Passed: {result.n_passed}\n"
            f"Blocked external: {result.n_blocked_external}\n"
            f"Failed: {result.n_failed}\n"
            f"Markdown: {result.artifacts.get('run_md', '')}",
            title="v3 Completion Audit",
        ))
        for artifact_name, path in result.artifacts.items():
            console.print(f"[green]{artifact_name}:[/] {path}")
        if enforce_gate and result.status != "COMPLETE":
            raise SystemExit(3)
        return

    if suite == "external_evidence":
        from .eval.external_evidence import (
            collect_bank_readiness_evidence,
            collect_high_pr_evidence,
            collect_mcp_compatibility_evidence,
            collect_remote_ci_evidence,
        )

        artifacts = []
        out_dir = settings.reports_dir / "eval" / "external_evidence"
        if eval_args.collect in {"bank-readiness", "all"}:
            artifacts.append(collect_bank_readiness_evidence(
                output_dir=out_dir,
                banks=eval_args.banks,
                icd10_code=eval_args.icd10_code,
                probe_fields=eval_args.probe_fields,
            ))
        if eval_args.collect in {"mcp", "all"}:
            artifacts.append(collect_mcp_compatibility_evidence(
                output_dir=out_dir,
                config_path=eval_args.mcp_config or None,
                call_args_path=eval_args.mcp_call_args or None,
                min_servers=eval_args.mcp_min_servers,
            ))
        if eval_args.collect in {"remote-ci", "all"}:
            artifacts.append(collect_remote_ci_evidence(
                output_dir=out_dir,
                workflow=eval_args.workflow,
                repo_root=Path.cwd(),
            ))
        if eval_args.collect in {"high-pr", "all"}:
            artifacts.append(collect_high_pr_evidence(
                output_dir=out_dir,
                pr_url=eval_args.pr_url,
                branch=eval_args.branch,
                repo_root=Path.cwd(),
            ))
        if not artifacts:
            console.print("[red]Nothing collected. Use --collect mcp|remote-ci|high-pr|all.[/]")
            raise SystemExit(2)
        summary = "\n".join(f"{item.kind}: {item.status}\n  {item.path}" for item in artifacts)
        console.print(Panel(summary, title="External Evidence"))
        if enforce_gate and not all(item.status in {"PASS", "SUCCESS", "PR_OPENED"} for item in artifacts):
            raise SystemExit(3)
        return

    benchmarks = {
        "agent_report_workflow": AgentReportWorkflowBenchmark,
        "live_ukb_report_20": LiveUKBReport20Benchmark,
        "research_eval_v1": ResearchEvalV1,
        "report_20_case": Report20CaseBenchmark,
        "report_quality": ReportQualityBenchmark,
        "skill_schemas": SkillSchemaBenchmark,
        "biomedical_qa": BiomedQABenchmark,
        "skill_calls": SkillCallBenchmark,
    }
    if suite not in benchmarks:
        console.print(f"[red]Unknown suite: {suite}.[/]")
        console.print(f"[dim]Available: {', '.join(sorted([*benchmarks, 'behavioral', 'external_evidence', 'live_artifacts', 'scheduled', 'v3_completion']))}[/]")
        raise SystemExit(2)

    benchmark = benchmarks[suite]()
    harness = EvalHarness()
    baseline_obs = {}

    # Optional A/B baseline run or load from report.
    if baseline_report:
        p = Path(baseline_report)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                baseline_obs = data.get("observability", {}) or {}
            except Exception as e:
                console.print(f"[yellow]Failed to parse baseline report ({p}): {e}[/]")
    elif ab_compare and mode == "mas_v2":
        baseline_settings = get_settings()
        baseline_settings.ensure_dirs()
        baseline_settings.multi_model_enabled = False
        with console.status("[bold green]Running baseline for A/B..."):
            baseline_agent = Agent(baseline_settings)
            baseline_result = harness.run(
                benchmark=benchmark,
                agent=baseline_agent,
                mode="baseline",
                enforce_gate=False,
            )
            baseline_obs = baseline_result.observability

    with console.status("[bold green]Loading agent for evaluation..."):
        agent = Agent(settings)
    model_pool_ids = [spec.model_id for spec in getattr(agent.orchestrator, "model_pool", [])]
    available_models = list(getattr(agent, "available_models", []) or [])
    missing_pool_models = [
        {
            "model": model_id,
            "reason": "not_returned_by_relay_models_endpoint",
        }
        for model_id in model_pool_ids
        if available_models and model_id not in set(available_models)
    ]
    if not available_models:
        missing_pool_models = [
            {
                "model": model_id,
                "reason": "model_list_unavailable_or_not_configured",
            }
            for model_id in model_pool_ids
        ]

    with console.status(f"[bold green]Running {suite} ({mode}) ..."):
        result = harness.run(
            benchmark=benchmark,
            agent=agent,
            mode=mode,
            enforce_gate=enforce_gate,
            baseline_observability=baseline_obs,
            review_loop=eval_args.review_loop,
            primary_reviewer=eval_args.primary_reviewer,
            include_claude=eval_args.include_claude,
            review_timeout_s=eval_args.review_timeout_s,
        )

    console.print(Panel(result.summary(), title="Evaluation Summary"))
    obs = result.observability
    table = Table(title="Reliability Dashboard")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("success_rate", f"{obs.get('success_rate', 0.0):.2%}")
    table.add_row("evidence_coverage", f"{obs.get('evidence_coverage', 0.0):.2%}")
    table.add_row("stat_guardrail_violation", f"{obs.get('stat_guardrail_violation', 0.0):.2%}")
    table.add_row("wrong_consensus_rate", f"{obs.get('wrong_consensus_rate', 0.0):.2%}")
    table.add_row("complex_task_success", f"{obs.get('complex_task_success', 0.0):.2%}")
    table.add_row("token_cost(tokens)", f"{obs.get('token_cost', 0.0):,.0f}")
    table.add_row("p95_latency_s", f"{obs.get('p95_latency_s', 0.0):.2f}")
    console.print(table)
    model_table = Table(title="Model Diagnostics")
    model_table.add_column("Field", style="cyan")
    model_table.add_column("Value", style="green")
    model_table.add_row("default_model", agent.settings.llm_model)
    model_table.add_row("model_pool", ", ".join(model_pool_ids))
    model_table.add_row("relay_model_count", str(len(available_models)))
    model_table.add_row(
        "pool_model_failures",
        json.dumps(missing_pool_models, ensure_ascii=False) if missing_pool_models else "[]",
    )
    console.print(model_table)
    if result.comparative:
        cmp = result.comparative
        cmp_table = Table(title="A/B Delta vs Baseline")
        cmp_table.add_column("Metric", style="cyan")
        cmp_table.add_column("Value", style="green", justify="right")
        cmp_table.add_row(
            "complex_task_success_uplift",
            f"{cmp.get('complex_task_success_uplift', 0.0):+.2%}",
        )
        cmp_table.add_row(
            "wrong_consensus_reduction_ratio",
            f"{cmp.get('wrong_consensus_reduction_ratio', 0.0):+.2%}",
        )
        console.print(cmp_table)

    out_dir = settings.reports_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{suite}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "suite": suite,
        "mode": mode,
        "summary": result.summary(),
        "results": [
            {
                "case_id": getattr(case_result, "case_id", ""),
                "passed": bool(getattr(case_result, "passed", False)),
                "score": float(getattr(case_result, "score", 0.0)),
                "actual_skills": list(getattr(case_result, "actual_skills", []) or []),
                "actual_text": getattr(case_result, "actual_text", ""),
                "errors": list(getattr(case_result, "errors", []) or []),
                "elapsed_s": float(getattr(case_result, "elapsed_s", 0.0)),
                "metadata": getattr(case_result, "metadata", {}) or {},
            }
            for case_result in getattr(result, "results", []) or []
        ],
        "observability": obs,
        "baseline_observability": result.baseline_observability,
        "comparative": result.comparative,
        "review_loop": result.review_loop,
        "gate_passed": result.gate_passed,
        "gate_failures": result.gate_failures,
        "n_total": result.n_total,
        "n_passed": result.n_passed,
        "timestamp": result.timestamp,
        "model_diagnostics": {
            "base_url": settings.llm_base_url,
            "default_model": agent.settings.llm_model,
            "model_pool": model_pool_ids,
            "available_model_count": len(available_models),
            "pool_model_failures": missing_pool_models,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    console.print(f"[green]Saved report:[/] {out_path}")

    if enforce_gate and not result.gate_passed:
        raise SystemExit(3)


def main() -> None:
    """CLI entry point."""
    global Agent
    if Agent is None:
        from .agent import Agent as _Agent

        Agent = _Agent
    # Check for subcommands
    if len(sys.argv) > 1 and sys.argv[1] == "rebuild-parquet":
        rebuild_parquet_cmd()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "build-ukb-full-parquet":
        build_ukb_full_parquet_cmd()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        eval_cmd()
        return

    # Parse optional args. The single interactive `biobank` REPL is the only
    # interaction mode: the former --tui (Textual) and --legacy-repl modes were
    # removed during consolidation; any leftover dashes are ignored.
    model = None
    workspace = ""
    positional_task: list[str] = []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:], 1):
        if skip_next:
            skip_next = False  # this token was consumed as a flag value
            continue
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]
        elif arg == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
            skip_next = True
        elif arg.startswith("--workspace=") or arg.startswith("--cwd="):
            workspace = arg.split("=", 1)[1]
        elif arg in ("--workspace", "--cwd") and i + 1 < len(sys.argv):
            workspace = sys.argv[i + 1]
            skip_next = True
        elif arg.startswith("-"):
            if positional_task and positional_task[0].startswith("/"):
                positional_task.append(arg)
            continue
        else:
            positional_task.append(arg)

    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s: %(message)s",
    )

    # Load settings
    settings = get_settings()
    settings.ensure_dirs()
    if model:
        settings.llm_model = model

    global run_interactive_shell
    if run_interactive_shell is None:
        from .cli.interactive import run_interactive_shell as _run_interactive_shell
        run_interactive_shell = _run_interactive_shell
    if run_interactive_shell is None:
        raise RuntimeError("Interactive shell is unavailable")
    # Final safety net: the shell loop already contains command-level interrupts,
    # but guarantee that no interrupt at any layer ever surfaces as a raw traceback.
    try:
        run_interactive_shell(settings, initial_task=" ".join(positional_task).strip(), console=console,
                              workspace=workspace)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Goodbye.[/]")
    return
    # main() returns above; the live entrypoint is the v3 InteractiveShell
    # (run_interactive_shell).




def _handle_command(
    query: str,
    agent: Agent,
    planner: PlanMode,
    token_usage: dict,
) -> None:
    """Dispatch slash commands."""
    _dispatch_registered_command_line(query, agent, planner, token_usage)


# ── Command implementations ─────────────────────────────────


def _dispatch_registered_command_line(
    query: str,
    agent: Agent,
    planner: PlanMode,
    token_usage: dict,
) -> bool:
    """Dispatch one slash-command line through the importable registry."""
    parts = str(query or "").split(None, 1)
    if not parts:
        return False
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if _dispatch_registered_command(cmd, arg, agent, planner, token_usage):
        return True

    console.print(f"[yellow]Unknown command: {cmd}. Type /help for available commands.[/]")
    return False


def _build_registered_command_actions(
    agent: Agent,
    planner: PlanMode,
    token_usage: dict,
) -> dict[str, Callable[..., Any]]:
    """Build host actions shared by the Rich CLI and Textual registry dispatch."""
    return {
        "help": _show_help,
        "status": lambda: _show_status(agent, planner, token_usage),
        "routing_status": lambda: _show_routing_status(agent),
        "cost": lambda: _show_cost(token_usage, agent),
        "compact": lambda: _compact(agent),
        "clear": lambda: _clear(agent),
        "export": lambda fmt: _export(agent, fmt),
        "skills": lambda: _show_skills(agent),
        "history": lambda: _show_history(agent),
        "figures": lambda: _show_figures(agent),
        "cohorts": lambda: _show_cohorts(agent),
        "models": lambda: _show_models(agent),
        "show_evidence": lambda claim_id: _show_evidence(agent, claim_id),
        "replicate": lambda value: _replicate_paper(agent, value, planner=planner),
        "plan": lambda value: _cmd_plan(agent, planner, value),
        "plan_approve": lambda: _cmd_plan_approve(agent, planner, token_usage),
        "plan_edit": lambda value: _cmd_plan_edit(agent, planner, value),
        "plan_title": lambda value: _cmd_plan_title(agent, planner, value),
        "plan_pause": lambda: _cmd_plan_pause(agent, planner),
        "plan_resume": lambda: _cmd_plan_resume(agent, planner, token_usage),
        "plan_option": lambda value: _cmd_plan_option(agent, planner, value, token_usage),
        "plan_skip": lambda value: _cmd_plan_skip(agent, planner, value, token_usage),
        "plan_exit": lambda: _cmd_plan_exit(agent, planner),
        "plans": lambda: _cmd_list_plans(planner),
        "model": lambda value: _switch_model(agent, value),
        "models_pool": lambda: _show_model_pool(agent),
        "models_available": lambda: _show_available_llm_models(agent),
        "strategy": lambda value: _set_strategy(agent, value),
        "mcp_list": lambda: _show_mcp_status(agent),
        "mcp_start": lambda: _start_mcp(agent),
        "mcp_health": lambda value="": _mcp_health(agent, str(value or "")),
        "mcp_call": lambda value: _call_mcp_tool(agent, value),
        "mcp_stop": lambda: _stop_mcp(agent),
        "replay": lambda value: _replay_provenance(agent, value),
        "evolve": lambda value: _show_evolution_status(agent, value),
        "record": lambda value: _record_pipeline(agent, value),
        "pipelines": lambda: _show_pipelines(agent),
        "errors": lambda: _show_errors(agent),
        "memory": lambda: _show_memory(agent),
        "debate": lambda value: _force_debate(agent, value, token_usage),
    }


def _dispatch_registered_command(
    cmd: str,
    arg: str,
    agent: Agent,
    planner: PlanMode,
    token_usage: dict,
) -> bool:
    """Dispatch commands through the importable v3 command registry."""
    from .cli.commands import CommandContext, build_core_registry

    command = build_core_registry().get(cmd)
    if command is None:
        return False

    ctx = CommandContext(
        agent=agent,
        planner=planner,
        token_usage=token_usage,
        console=console,
        actions=_build_registered_command_actions(agent, planner, token_usage),
    )
    command.handle(ctx, arg)
    return True


def _cmd_plan(agent: Agent, planner: PlanMode, arg: str) -> None:
    """Registry-backed implementation of /plan."""
    if not arg:
        if planner.state == PlanState.INACTIVE:
            if not _stdin_is_interactive():
                console.print("[dim]No active plan. Use /plan <goal> to start.[/]")
                return
            arg = _read_multiline_plan_goal()
            if not arg:
                console.print("[dim]No task entered.[/]")
                return
        else:
            console.print(f"[bold]Plan:[/bold] {planner.plan.title if planner.plan and planner.plan.title else planner.goal}")
            console.print(f"[bold]Original goal:[/bold] {planner.goal}")
            console.print(f"[bold]State:[/bold] {planner.state.value} (v{planner.revision})")
            if planner.plan:
                console.print(f"[bold]Progress:[/bold] {planner.plan.done_steps}/{planner.plan.total_steps} steps done")
            return

    if planner.is_active:
        console.print("[yellow]A plan is already active. Use /plan-exit first.[/]")
        return
    try:
        agent.state.current_report_dir = None
    except Exception:
        pass

    from .progress import PlanProgressDisplay, PlanRunDashboard

    dashboard = PlanRunDashboard(console, title="Plan Design")

    def event_sink(phase: str, actor: str, status: str, message: str, metadata: dict | None = None) -> None:
        _emit_tui_plan_event(agent, phase, actor, status, message, metadata)
        dashboard.record(phase, actor, status, message, metadata)

    dashboard.start()
    result = ""
    clarified_goal = arg
    clarification_answers: list[dict] = []
    research_setup_meta: dict[str, Any] = {}
    try:
        event_sink("Preflight", "biobank", "success", "loaded live tool schemas")
        if _settings_bool(agent.settings, "plan_clarification_enabled", True):
            event_sink("Clarification", "biobank", "running", "checking for critical ambiguities")
            questions = planner.identify_clarifications(arg)
            if questions:
                event_sink("Clarification", "biobank", "warning", f"{len(questions)} critical question(s)")
                dashboard.stop()
                clarified_goal, clarification_answers = _collect_plan_clarifications(planner, arg)
                dashboard.start()
                event_sink("Clarification", "user", "success", f"{len(clarification_answers)} answer(s) recorded")
            else:
                event_sink("Clarification", "biobank", "success", "no blocking ambiguity")
        else:
            event_sink("Clarification", "biobank", "skipped", "clarification gate disabled")

        if _settings_bool(agent.settings, "plan_research_setup_enabled", True) and _is_broad_metabolic_showcase_goal(clarified_goal):
            event_sink("Research setup", "biobank", "running", "confirming autonomous research defaults")
            dashboard.stop()
            clarified_goal, setup_answers, research_setup_meta = _collect_research_setup(agent, clarified_goal)
            dashboard.start()
            if setup_answers:
                clarification_answers.extend(setup_answers)
                event_sink(
                    "Research setup",
                    "user",
                    "success",
                    f"{len(setup_answers)} setting(s) recorded",
                    research_setup_meta,
                )
            else:
                event_sink("Research setup", "biobank", "skipped", "not needed for this task")
        else:
            event_sink("Research setup", "biobank", "skipped", "not a broad autonomous research setup task")

        event_sink("Planning", "biobank", "running", "decomposing goal into executable steps")
        result = planner.start(clarified_goal)
        if clarification_answers:
            planner.record_clarification_answers(clarification_answers)
        event_sink("Planning", "biobank", "success", result)

        if planner.plan:
            planner.capture_base_plan_snapshot()
            if _should_run_external_planning_council(agent.settings, clarified_goal):
                event_sink("External council", "biobank", "running", "consulting optional Codex/Claude/Gemini planners")
                council_records = _collect_external_planning_council(agent, clarified_goal, event_sink=event_sink)
            else:
                council_records = []
                event_sink("External council", "biobank", "skipped", "not requested for this plan")
            external_questions = _extract_external_planner_questions(council_records)
            if external_questions:
                event_sink(
                    "Clarification",
                    "external council",
                    "warning",
                    f"{len(external_questions)} external planner question(s)",
                )
                dashboard.stop()
                external_text, external_answers = _collect_external_planner_clarifications(
                    external_questions,
                    clarified_goal,
                )
                dashboard.start()
                if external_answers:
                    clarified_goal = (
                        f"{clarified_goal.rstrip()}\n\n"
                        f"External planner clarifications:\n{external_text}"
                    ).strip()
                    all_answers = [*clarification_answers, *external_answers]
                    event_sink(
                        "Clarification",
                        "user",
                        "success",
                        f"{len(external_answers)} external planner answer(s) recorded",
                    )
                    event_sink("Planning", "biobank", "running", "replanning with external planner clarifications")
                    result = planner.start(clarified_goal)
                    planner.record_clarification_answers(all_answers)
                    planner.capture_base_plan_snapshot()
                    event_sink("Planning", "biobank", "success", result)
                    if _should_run_external_planning_council(agent.settings, clarified_goal):
                        event_sink(
                            "External council",
                            "biobank",
                            "running",
                            "rerunning external planners with clarification answers",
                        )
                        council_records = _collect_external_planning_council(agent, clarified_goal, event_sink=event_sink)
            if council_records:
                planner.attach_planning_council(council_records)
                event_sink("Merge", "biobank", "running", "merging schema-safe external advice")
                merge_result = planner.merge_external_plans()
                event_sink("Merge", "biobank", "success", merge_result)
                PlanCheckpoint.from_plan_mode(planner).save(_plan_checkpoint_path(agent.settings))
            else:
                event_sink("External council", "biobank", "skipped", "no external advice attached")

            planner.validate_current_plan()
            if planner.validation_issues:
                event_sink("Validation", "biobank", "failed", f"{len(planner.validation_issues)} schema issue(s)")
            else:
                event_sink("Validation", "biobank", "success", "plan is schema-valid")
            event_sink("Review", "biobank", "success", "awaiting /plan-approve or feedback")
    finally:
        dashboard.stop()

    console.print(f"[green]{result}[/green]\n")
    if planner.plan and planner.validation_issues:
        console.print(f"[yellow]{planner.validation_summary()}[/yellow]\n")
    if planner.plan:
        PlanCheckpoint.from_plan_mode(planner).save(_plan_checkpoint_path(agent.settings))
        display = PlanProgressDisplay(planner.plan, console)
        display.show_plan_for_review(revision=planner.revision)


def _cmd_plan_approve(agent: Agent, planner: PlanMode, token_usage: dict) -> None:
    """Registry-backed implementation of /plan-approve."""
    if planner.state == PlanState.REVIEW:
        refresh_msg = _refresh_plan_from_pasted_goal(planner)
        if refresh_msg:
            console.print(f"[dim]{refresh_msg}[/dim]")
            PlanCheckpoint.from_plan_mode(planner).save(_plan_checkpoint_path(agent.settings))
        result = planner.approve()
        if "Cannot" in result:
            console.print(f"[yellow]{result}[/]")
        else:
            console.print(f"[green]{result}[/green]\n")
            _execute_plan(agent, planner, token_usage)
    elif planner.state == PlanState.PAUSED:
        planner.state = PlanState.REVIEW
        result = planner.approve()
        if "Cannot" in result:
            console.print(f"[yellow]{result}[/]")
        else:
            console.print("[green]Plan re-approved. Resuming execution...[/green]\n")
            _execute_plan(agent, planner, token_usage)
    else:
        console.print(f"[yellow]Cannot approve: state is {planner.state.value} (need REVIEW or PAUSED).[/]")


def _cmd_plan_edit(agent: Agent, planner: PlanMode, arg: str) -> None:
    """Registry-backed implementation of /plan-edit."""
    if not arg:
        console.print("[yellow]Usage: /plan-edit <modification instructions>[/]")
        return
    if planner.state not in (PlanState.REVIEW, PlanState.PAUSED):
        console.print(f"[yellow]Cannot edit: state is {planner.state.value} (need REVIEW or PAUSED).[/]")
        return

    refresh_msg = _refresh_plan_from_pasted_goal(planner)
    if refresh_msg:
        console.print(f"[dim]{refresh_msg}[/dim]")
        PlanCheckpoint.from_plan_mode(planner).save(_plan_checkpoint_path(agent.settings))
    with console.status("[bold blue]Refining plan...[/bold blue]", spinner="dots"):
        result = planner.refine(arg)
    console.print(f"[green]{result}[/green]\n")
    if planner.plan:
        PlanCheckpoint.from_plan_mode(planner).save(_plan_checkpoint_path(agent.settings))
        from .progress import PlanProgressDisplay
        display = PlanProgressDisplay(planner.plan, console)
        display.show_plan_for_review(revision=planner.revision)


def _cmd_plan_title(agent: Agent, planner: PlanMode, arg: str) -> None:
    """Rename the active plan/report title."""
    result = planner.rename(arg)
    if result.startswith("Plan title updated"):
        PlanCheckpoint.from_plan_mode(planner).save(_plan_checkpoint_path(agent.settings))
        console.print(f"[green]{result}[/green]\n")
        if planner.plan:
            from .progress import PlanProgressDisplay
            PlanProgressDisplay(planner.plan, console).show_plan_for_review(revision=planner.revision)
    else:
        console.print(f"[yellow]{result}[/yellow]")


def _cmd_plan_pause(agent: Agent, planner: PlanMode) -> None:
    if planner.state == PlanState.EXECUTING:
        result = planner.pause("User requested")
        console.print(f"[yellow]{result}[/yellow]")
        PlanCheckpoint.from_plan_mode(planner).save(_plan_checkpoint_path(agent.settings))
    else:
        console.print(f"[yellow]Cannot pause: state is {planner.state.value} (need EXECUTING).[/]")


def _cmd_plan_resume(agent: Agent, planner: PlanMode, token_usage: dict) -> None:
    if planner.state == PlanState.PAUSED:
        console.print("[green]Resuming plan execution after repair checks...[/green]\n")
        _execute_plan(agent, planner, token_usage)
    else:
        console.print(f"[yellow]Cannot resume: state is {planner.state.value} (need PAUSED).[/]")


def _cmd_plan_option(agent: Agent, planner: PlanMode, arg: str, token_usage: dict) -> None:
    if not arg:
        console.print("[yellow]Usage: /plan-option <A|B|C|N>[/]")
        return
    _handle_plan_option(agent, planner, arg.strip(), token_usage)


def _cmd_plan_skip(agent: Agent, planner: PlanMode, arg: str, token_usage: dict) -> None:
    if not arg:
        console.print("[yellow]Usage: /plan-skip <step_id>[/]")
        return
    _handle_plan_skip(agent, planner, arg.strip(), token_usage)


def _cmd_plan_exit(agent: Agent, planner: PlanMode) -> None:
    if planner.is_active or planner.state == PlanState.DONE:
        result = planner.exit()
        PlanCheckpoint.clear(_plan_checkpoint_path(agent.settings))
        console.print(f"[dim]{result}[/dim]")
    else:
        console.print("[dim]Not in plan mode.[/]")


def _cmd_list_plans(planner: PlanMode) -> None:
    plans = planner.list_plans()
    if not plans:
        console.print("[dim]No saved plans.[/]")
        return
    for p in plans:
        status_color = {"DONE": "green", "EXECUTING": "yellow", "PAUSED": "red"}.get(p["status"], "dim")
        console.print(f"  [{status_color}]{p['status']}[/] {p['file']}")


def _handle_plan_option(agent: Agent, planner: PlanMode, choice: str, token_usage: dict) -> None:
    """Handle the explicit A/B/C/N choices shown after plan stalls/failures."""
    normalized = (choice or "").strip().upper()[:1]
    if planner.state != PlanState.PAUSED:
        console.print(f"[yellow]Plan option is only available while PAUSED (current: {planner.state.value}).[/]")
        return

    if normalized == "A":
        repaired = _auto_repair_plan_graph(planner)
        if repaired and planner.plan and not planner.validation_issues:
            from .progress import PlanProgressDisplay
            console.print("[green]Plan dependency graph repaired deterministically.[/green]\n")
            display = PlanProgressDisplay(planner.plan, console)
            display.show_plan_for_review(revision=planner.revision)
            PlanCheckpoint.from_plan_mode(planner).save(_plan_checkpoint_path(agent.settings))
            console.print("[dim]Use /plan-approve to execute the repaired plan.[/dim]\n")
            return

        repair_instruction = (
            "Auto-repair from the paused state. Diagnose the failed or blocked step, "
            "use existing skills when possible, create a custom skill only if the missing "
            "capability is bounded and safe, then preserve governed report dependencies."
        )
        with console.status("[bold blue]Re-planning repair path...[/bold blue]", spinner="dots"):
            result = planner.refine(repair_instruction)
        console.print(f"[green]{result}[/green]\n")
        if planner.plan:
            from .progress import PlanProgressDisplay
            display = PlanProgressDisplay(planner.plan, console)
            display.show_plan_for_review(revision=planner.revision)
            PlanCheckpoint.from_plan_mode(planner).save(_plan_checkpoint_path(agent.settings))
        console.print("[dim]Use /plan-approve to execute the repaired plan.[/dim]\n")
        return

    if normalized == "B":
        console.print("[dim]External review hooks have been removed. Type repair instructions directly.[/dim]\n")
        return

    if normalized == "C":
        console.print(
            "[yellow]Type the concrete repair instruction now, or use /plan-edit <instruction>. "
            "Then run /plan-approve or /plan-resume after the plan is valid.[/yellow]"
        )
        return

    if normalized == "N":
        result = planner.exit()
        PlanCheckpoint.clear(_plan_checkpoint_path(agent.settings))
        console.print(f"[dim]{result}[/dim]")
        return

    console.print("[yellow]Unknown option. Use A, B, C, or N.[/]")


def _handle_plan_skip(agent: Agent, planner: PlanMode, step_id: str, token_usage: dict) -> None:
    """Explicitly skip an optional/diagnostic step and resume dependency checks."""
    if planner.state not in (PlanState.PAUSED, PlanState.REVIEW):
        console.print(f"[yellow]Can only skip while REVIEW or PAUSED (current: {planner.state.value}).[/]")
        return
    if not planner.plan:
        console.print("[yellow]No active plan.[/]")
        return
    planner.validate_current_plan()
    step = next((s for s in planner.plan.steps if s.id == step_id), None)
    if step is None:
        console.print(f"[yellow]No step with id {step_id!r}.[/]")
        return
    if getattr(step, "criticality", "required") == "required":
        console.print(
            f"[red]Refusing to skip required step {step.id}.[/] "
            "Use /plan-option A or /plan-edit to repair/re-plan instead."
        )
        return
    planner.plan.mark_skipped(step.id, "Explicit user skip of optional/diagnostic step")
    planner.validate_current_plan()
    PlanCheckpoint.from_plan_mode(planner).save(_plan_checkpoint_path(agent.settings))
    console.print(f"[green]Skipped {step.id} ({step.description}).[/green]")
    if planner.state == PlanState.PAUSED:
        console.print("[dim]Use /plan-resume to continue execution after the explicit skip.[/dim]")


def _auto_repair_plan_graph(planner: PlanMode) -> bool:
    """Deterministically fix missing deps/self deps/cycles before LLM repair."""
    if not planner.plan:
        return False
    changed = False
    ids = {s.id for s in planner.plan.steps}
    for step in planner.plan.steps:
        cleaned = []
        for dep in step.depends_on:
            if dep in ids and dep != step.id and dep not in cleaned:
                cleaned.append(dep)
        if cleaned != step.depends_on:
            step.depends_on = cleaned
            changed = True

    # Re-apply the report contract after removing impossible edges.
    try:
        planner.plan = planner.planner._enforce_report_contract(planner.plan, planner.available_skills)
    except Exception:
        pass

    for _ in range(20):
        issues = planner.validate_current_plan()
        cycle_issue = next((issue for issue in issues if "Dependency cycle detected:" in issue.message), None)
        if not cycle_issue:
            break
        chain_text = cycle_issue.message.split("Dependency cycle detected:", 1)[1].strip().rstrip(".")
        chain = [part.strip() for part in chain_text.split("->") if part.strip()]
        if len(chain) < 2:
            break
        source, dep = chain[0], chain[1]
        step = next((s for s in planner.plan.steps if s.id == source), None)
        if not step or dep not in step.depends_on:
            break
        step.depends_on = [item for item in step.depends_on if item != dep]
        changed = True

    if changed:
        planner.revision += 1
        planner.state = PlanState.REVIEW
        planner.validate_current_plan()
        try:
            planner._save_plan_file()
        except Exception:
            pass
    return changed


def _execute_plan(agent: Agent, planner: PlanMode, token_usage: dict) -> None:
    """Execute the approved plan with progress tracking.

    Creates a PlanExecutor and runs the plan step-by-step.
    Handles the full lifecycle including pause on failure.
    """
    if not planner.plan:
        console.print("[red]No plan to execute.[/red]")
        return

    from .plan_executor import PlanExecutor
    from datetime import datetime

    active_report_dir = getattr(planner, "report_dir", "") or getattr(agent.state, "current_report_dir", None)
    report_dir = Path(active_report_dir) if active_report_dir else agent.settings.reports_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        agent.state.current_report_dir = report_dir
        planner.report_dir = str(report_dir)
    except Exception:
        pass

    def ctx_builder():
        """Build a context for direct registry fallback paths."""
        return agent._build_ctx(report_dir)

    def skill_executor(skill_name: str, args: dict) -> dict:
        """Run plan steps through the agent recorder so reports see them."""
        _sync_plan_execution_log(agent, planner)
        progress_callback = getattr(planner, "_active_step_progress", None)
        if callable(progress_callback):
            agent.state.custom_data["plan_progress_callback"] = progress_callback
        try:
            return agent._execute_skill_and_record(
                skill_name,
                args,
                report_dir,
                allow_retry=True,
            )
        finally:
            try:
                if agent.state.custom_data.get("plan_progress_callback") is progress_callback:
                    agent.state.custom_data.pop("plan_progress_callback", None)
            except Exception:
                pass

    repair_strategy = _build_plan_repair_strategy(agent, planner)
    plan_events: list[dict] = []

    def event_sink(phase: str, actor: str, status: str, message: str, metadata: dict | None = None) -> None:
        event = {
            "phase": phase,
            "actor": actor,
            "status": status,
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        plan_events.append(event)
        try:
            agent.state.custom_data["plan_events"] = list(plan_events)
        except Exception:
            pass
        _emit_tui_plan_event(agent, phase, actor, status, message, metadata)

    executor = PlanExecutor(
        plan_mode=planner,
        registry=agent.registry,
        ctx_builder=ctx_builder,
        skill_executor=skill_executor,
        console=console,
        checkpoint_dir=planner.plans_dir,
        on_log_update=lambda _log: _sync_plan_execution_log(agent, planner),
        repair_strategy=repair_strategy,
        refresh_tools=lambda: planner.refresh_tools(agent.registry),
        repair_budget_per_step=getattr(agent.settings, "plan_repair_budget_per_step", 3),
        repair_budget_total=getattr(agent.settings, "plan_repair_budget_total", 8),
        goal_acceptance_enabled=getattr(agent.settings, "plan_goal_acceptance_enabled", True),
        event_sink=event_sink,
    )

    try:
        results = executor.execute()
        _sync_plan_execution_log(agent, planner)

        # Update token usage
        tu = agent.state.token_usage
        token_usage["prompt_tokens"] = tu.prompt_tokens
        token_usage["completion_tokens"] = tu.completion_tokens

        if planner.state == PlanState.DONE:
            completion_warnings = list(getattr(planner, "completion_warnings", []) or [])
            if completion_warnings:
                console.print("\n[bold green]✓ Plan execution complete.[/bold green]")
                console.print("[bold yellow]Critical findings require attention:[/bold yellow]")
                for warning in completion_warnings[:5]:
                    console.print(f"[yellow]- {warning}[/yellow]")
            else:
                console.print("\n[bold green]✓ Plan execution complete.[/bold green]")
            selected_reviewers = _review_hook_selection(agent, report_dir)
            if selected_reviewers:
                review_records = _run_external_review_hooks(
                    agent,
                    selected_reviewers,
                    focus=planner.goal or "post-run plan execution review",
                    report_dir=report_dir,
                )
                _run_review_repair_loop(
                    agent,
                    review_records,
                    focus=planner.goal or "post-run plan execution review",
                    report_dir=report_dir,
                )
            console.print("[dim]Ask follow-up questions or start a new /plan.[/dim]\n")

    except KeyboardInterrupt:
        planner.pause("Interrupted by user")
        PlanCheckpoint.from_plan_mode(planner).save(_plan_checkpoint_path(agent.settings))
        _sync_plan_execution_log(agent, planner)
        console.print("\n[yellow]⏸ Plan execution interrupted. Use /plan-resume to continue.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Plan execution error: {e}[/red]")
        planner.pause(f"Error: {e}")
        PlanCheckpoint.from_plan_mode(planner).save(_plan_checkpoint_path(agent.settings))
        _sync_plan_execution_log(agent, planner)


def _build_plan_repair_strategy(agent: Agent, planner: PlanMode):
    """Build a bounded repair callback for PlanExecutor."""

    def repair_strategy(step, result, plan, context):
        # First use learned successful parameters when there is enough history.
        try:
            suggestions = agent.tool_learner.suggest_params(
                step.skill,
                {"error": result.error, "goal": planner.goal},
            )
            if suggestions:
                repaired_args = dict(step.args or {})
                changed = False
                for suggestion in suggestions:
                    if suggestion.param in repaired_args and repaired_args[suggestion.param] != suggestion.suggested_value:
                        repaired_args[suggestion.param] = suggestion.suggested_value
                        changed = True
                if changed:
                    return {
                        "action": "retry_args",
                        "reason": "Retrying with parameter values learned from prior successful runs.",
                        "args": repaired_args,
                    }
        except Exception:
            pass

        # Then ask the current model for one constrained repair action.
        try:
            prompt = _plan_repair_prompt(agent, planner, step, result, plan, context)
            response = agent.llm.chat(
                messages=[
                    {"role": "system", "content": (
                        "You repair failed biobank-agent plan steps. Return one JSON object only. "
                        "Prefer retry_args or insert_prerequisite_steps. Use create_custom_skill only "
                        "for low-risk missing analysis capability and write only custom skill code."
                    )},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1800,
            )
            action = _extract_json_object(getattr(response, "text", ""))
            return _normalize_cli_repair_action(action, agent)
        except Exception as e:
            return {
                "action": "pause_with_trace",
                "reason": f"Could not synthesize a repair action: {e}",
            }

    return repair_strategy


def _plan_repair_prompt(agent: Agent, planner: PlanMode, step, result, plan, context: dict) -> str:
    schema_summary = _compact_tool_schema_summary(agent)
    plan_steps = [
        {
            "id": s.id,
            "skill": s.skill,
            "args": s.args,
            "description": s.description,
            "depends_on": s.depends_on,
            "status": s.status,
        }
        for s in plan.steps
    ]
    payload = {
        "goal": planner.goal,
        "failed_step": {
            "id": step.id,
            "skill": step.skill,
            "args": step.args,
            "description": step.description,
            "depends_on": step.depends_on,
        },
        "error": result.error,
        "result": result.result,
        "repair_context": context,
        "mutation_mode": getattr(agent.settings, "plan_code_mutation_mode", "review_only"),
        "plan_steps": plan_steps,
        "available_tool_schemas": schema_summary,
    }
    return (
        "Choose exactly one repair action for the failed step.\n"
        "Allowed JSON shapes:\n"
        '{"action":"retry_args","reason":"...","args":{...}}\n'
        '{"action":"insert_prerequisite_steps","reason":"...","steps":[{"id":"r1","skill":"...","args":{},"description":"...","depends_on":[]}]}\n'
        '{"action":"create_custom_skill","reason":"...","skill":{"name":"...","description":"...","parameters":"{\\"x\\":{\\"type\\":\\"string\\",\\"description\\":\\"...\\"}}","code_body":"return {\\"status\\": \\"success\\"}"},"replacement_step":{"skill":"...","args":{}}}\n'
        '{"action":"replan_remaining","reason":"...","steps":[...]}\n'
        '{"action":"pause_with_trace","reason":"..."}\n\n'
        "Constraints: all steps must use exact registered schemas. create_custom_skill may only generate a review artifact "
        "unless mutation_mode explicitly allows activation with user approval. Generated code must be safe, deterministic, "
        "and use ctx/state/data APIs only; do not modify repository source.\n\n"
        f"Context JSON:\n{json.dumps(payload, ensure_ascii=False, default=str)[:12000]}"
    )


def _compact_tool_schema_summary(agent: Agent, limit: int = 60) -> list[dict]:
    schemas = []
    for schema in agent.registry.tool_schemas()[:limit]:
        func = schema.get("function", {})
        params = func.get("parameters", {})
        props = params.get("properties", {}) or {}
        schemas.append({
            "name": func.get("name"),
            "required": params.get("required", []) or [],
            "args": sorted(props.keys()),
        })
    return schemas


def _extract_json_object(text: str) -> dict:
    clean = (text or "").strip()
    if "```" in clean:
        for part in clean.split("```"):
            candidate = part.strip().removeprefix("json").strip()
            if candidate.startswith("{"):
                clean = candidate
                break
    if not clean.startswith("{"):
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            clean = clean[start:end + 1]
    data = json.loads(clean)
    return data if isinstance(data, dict) else {}


def _normalize_cli_repair_action(action: dict, agent: Agent) -> dict | None:
    if not isinstance(action, dict):
        return None
    allowed = {"retry_args", "insert_prerequisite_steps", "create_custom_skill", "replan_remaining", "pause_with_trace"}
    if action.get("action") not in allowed:
        return None
    if action.get("action") == "create_custom_skill":
        if getattr(agent.settings, "plan_code_mutation_mode", "review_only") != "custom_skills":
            return {
                "action": "pause_with_trace",
                "reason": "Code mutation mode does not allow automatic custom skill creation.",
            }
        skill = dict(action.get("skill") or {})
        if isinstance(skill.get("parameters"), dict):
            skill["parameters"] = json.dumps(skill["parameters"], ensure_ascii=False)
        action["skill"] = skill
    return action


def _show_help() -> None:
    """Render slash command help using Rich tables."""
    console.print(
        Panel(
            "[bold #56d4dd]Command Palette[/]\n"
            "[dim]Tip: type [bold]/[/] then press [bold]Tab[/] for autocomplete.[/]",
            border_style="#334155",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    for section, rows in HELP_SECTIONS:
        console.print(f"[bold #7dd3fc]{section}[/]")
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold #94a3b8",
            pad_edge=True,
            show_edge=True,
            show_lines=False,
            padding=(0, 1),
        )
        table.add_column("Command", style="#67e8f9", no_wrap=True)
        table.add_column("Description", style="#e2e8f0")

        for command, description in rows:
            table.add_row(Text(command), description)

        console.print(table)
        console.print()


def _show_skills(agent: Agent) -> None:
    table = Table(title="Available Skills", show_lines=False, padding=(0, 1))
    table.add_column("Skill", style="cyan", no_wrap=True)
    table.add_column("Description", style="dim")
    for s in agent.registry.list_skills():
        table.add_row(s["name"], s["description"][:80])
    console.print(table)


def _show_status(agent: Agent, planner: PlanMode, token_usage: dict) -> None:
    from .utils.platform import platform_summary
    lines = [agent.state.context_summary()]
    pool = [spec.model_id for spec in getattr(agent.orchestrator, "model_pool", [])]
    routing_mode = "auto multi-agent" if agent.settings.multi_model_enabled else "single model"

    if planner.is_active:
        lines.append(f"\nPlan Mode: [magenta]{planner.status}[/]")
        if planner.goal:
            lines.append(f"Plan Goal: {planner.goal[:60]}")
        if planner.plan:
            lines.append(f"Plan Progress: {planner.plan.done_steps}/{planner.plan.total_steps} steps (v{planner.revision})")
        if planner.current_plan_file:
            lines.append(f"Plan File: {planner.current_plan_file.name}")

    lines.extend([
        "",
        f"Platform: {platform_summary()}",
        f"Model: {agent.settings.llm_model}",
        f"Routing: {routing_mode}",
        f"Model Pool: {', '.join(pool) if pool else agent.settings.llm_model}",
        f"Data: {agent.settings.data_dir}",
        f"Skills: {len(agent.registry)}",
        f"Tokens: {token_usage['prompt_tokens']:,} prompt + {token_usage['completion_tokens']:,} completion",
    ])

    mem_summary = agent.memory.summary()
    if mem_summary:
        lines.append(mem_summary)
    console.print(Panel("\n".join(lines), title="Session Status"))


def _show_routing_status(agent: Agent) -> None:
    """Display the latest orchestration trace and safety summary."""
    trace = getattr(agent.state, "last_orchestration", {}) or {}
    if not trace:
        console.print("[dim]No routing trace yet. Run one query first.[/]")
        return
    claims = trace.get("claims", []) if isinstance(trace, dict) else []
    evidence_links = trace.get("evidence_links", []) if isinstance(trace, dict) else []
    debate_trace = trace.get("debate_trace", {}) if isinstance(trace, dict) else {}
    safety_status = trace.get("safety_status", "PASS") if isinstance(trace, dict) else "PASS"
    strategy = debate_trace.get("strategy", "single") if isinstance(debate_trace, dict) else "single"
    disagreement = bool(debate_trace.get("disagreement", False)) if isinstance(debate_trace, dict) else False
    initial_strategy = trace.get("initial_strategy") if isinstance(trace, dict) else None
    final_strategy = trace.get("final_strategy") if isinstance(trace, dict) else None
    turn_orchestrations = trace.get("turn_orchestrations", []) if isinstance(trace, dict) else []

    table = Table(title="Routing Status")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("strategy", str(strategy))
    if initial_strategy or final_strategy:
        table.add_row("initial_strategy", str(initial_strategy or strategy))
        table.add_row("final_strategy", str(final_strategy or strategy))
    if turn_orchestrations:
        table.add_row("turn_rounds", str(len(turn_orchestrations)))
    table.add_row("safety_status", str(safety_status))
    table.add_row("claims", str(len(claims)))
    table.add_row("evidence_links", str(len(evidence_links)))
    table.add_row("disagreement", "yes" if disagreement else "no")
    if claims:
        claim_ids = ", ".join(str(c.get("claim_id", "")) for c in claims[:4] if isinstance(c, dict))
        if claim_ids:
            table.add_row("claim_ids", claim_ids)
    if isinstance(debate_trace, dict) and debate_trace.get("participants"):
        table.add_row("participants", ", ".join(str(x) for x in debate_trace.get("participants", [])[:6]))
    console.print(table)


def _show_evidence(agent: Agent, claim_id: str) -> None:
    """Display claim evidence chain from Action Graph."""
    claim_id = claim_id.strip()
    text = agent.memory.explain_claim(claim_id, limit=10)
    if "no linked evidence" in text.lower():
        console.print(f"[yellow]{text}[/]")
        return
    console.print(Panel(text, title=f"Evidence: {claim_id}"))


def _replicate_paper(agent: Agent, arg: str, *, planner: PlanMode | None = None) -> None:
    """Run the review-only paper replication planning skill."""
    if not arg:
        console.print("[yellow]Usage: /replicate <paper_path_or_text>[/]")
        return
    report_dir = agent.settings.reports_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        agent.state.current_report_dir = report_dir
    except Exception:
        pass
    execution = agent._execute_skill_and_record(
        "replicate_paper",
        {"source": arg, "source_type": "auto"},
        report_dir,
        allow_retry=False,
    )
    result = execution.get("result", {}) if isinstance(execution, dict) else {}
    if execution.get("is_error"):
        console.print(f"[red]Replication planning failed: {result.get('error', 'unknown error')}[/]")
        return
    spec = result.get("proposed_spec", {}) if isinstance(result, dict) else {}
    artifact = result.get("artifact", "")
    console.print(Panel(
        f"Status: {result.get('status', 'AWAITING_USER_APPROVAL')}\n"
        f"Design: {spec.get('design') or 'unknown'}\n"
        f"N: {spec.get('n') or 'unknown'}\n"
        f"Outcomes: {', '.join(spec.get('outcomes') or []) or 'not extracted'}\n"
        f"Artifact: {artifact or report_dir}",
        title="Paper Replication Draft",
    ))
    if planner is not None:
        _load_replication_plan_for_review(agent, planner, arg, result)


def _load_replication_plan_for_review(
    agent: Agent,
    planner: PlanMode,
    source: str,
    result: dict,
) -> None:
    """Load a replicate_paper review plan into PlanMode for approval."""
    plan_rows = result.get("plan") if isinstance(result, dict) else None
    if not isinstance(plan_rows, list) or not plan_rows:
        return
    if planner.is_active:
        console.print("[yellow]Replication plan artifacts were written, but an active plan already exists. Use /plan-exit before loading it for execution.[/]")
        return

    from .planner import LongHorizonPlan, PlanStep
    from .progress import PlanProgressDisplay

    steps: list[PlanStep] = []
    for idx, row in enumerate(plan_rows, start=1):
        if not isinstance(row, dict):
            continue
        steps.append(PlanStep(
            id=str(row.get("id") or f"s{idx}"),
            skill=str(row.get("skill") or "think"),
            args=dict(row.get("args") or {}),
            description=str(row.get("description") or row.get("skill") or f"Replication step {idx}"),
            depends_on=[str(dep) for dep in (row.get("depends_on") or [])],
            can_parallelize=bool(row.get("can_parallelize", False)),
            criticality=str(row.get("criticality") or "required"),
        ))
    if not steps:
        return

    planner.goal = f"Execute reviewed UKB paper replication for {source[:180]}"
    planner.plan = LongHorizonPlan(goal=planner.goal, steps=steps)
    planner.state = PlanState.REVIEW
    planner.revision += 1
    planner.report_dir = str(getattr(agent.state, "current_report_dir", "") or "")
    planner.validation_issues = planner.validate_current_plan()
    try:
        planner._save_plan_file()
    except Exception:
        pass
    PlanCheckpoint.from_plan_mode(planner).save(_plan_checkpoint_path(agent.settings))
    if planner.validation_issues:
        console.print(f"[yellow]{planner.validation_summary()}[/yellow]\n")
    console.print("[green]Replication plan loaded for review. Use /plan-approve to execute.[/green]\n")
    PlanProgressDisplay(planner.plan, console).show_plan_for_review(revision=planner.revision)


def _tool_ctx(agent: Agent, report_dir: Path | None = None):
    """Build a skill context for CLI-triggered tool calls (MCP and others)."""
    if report_dir is None:
        report_dir = getattr(agent.settings, "reports_dir", Path("./reports")) / "tools"
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    build_ctx = getattr(agent, "_build_ctx", None)
    if callable(build_ctx):
        try:
            return build_ctx(report_dir)
        except Exception:
            pass
    return SimpleNamespace(
        settings=agent.settings,
        state=agent.state,
        memory=getattr(agent, "memory", None),
        report_dir=report_dir,
    )


def _settings_bool(settings: Any, name: str, default: bool = False) -> bool:
    value = getattr(settings, name, None)
    return value if isinstance(value, bool) else default


def _settings_int(settings: Any, name: str, default: int) -> int:
    value = getattr(settings, name, None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _settings_str(settings: Any, name: str, default: str) -> str:
    value = getattr(settings, name, None)
    return value if isinstance(value, str) else default


def _settings_float(settings: Any, name: str, default: float) -> float:
    value = getattr(settings, name, None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _configured_external_agents(settings: Any, name: str, default: str = "") -> list[str]:
    """External CLI agents are no longer wired into the planner."""
    return []


def _should_run_external_planning_council(settings: Any, goal: str) -> bool:
    """External planning councils have been removed; always returns False."""
    return False


def _stdin_is_interactive() -> bool:
    isatty = getattr(sys.stdin, "isatty", None)
    try:
        return bool(isatty and isatty())
    except Exception:
        return False


def _collect_external_planning_council(
    agent: Agent,
    task: str,
    event_sink: Callable[[str, str, str, str, dict | None], None] | None = None,
) -> list[dict]:
    """External planning councils have been removed; always returns no records."""
    if event_sink:
        event_sink("External council", "biobank", "skipped", "external planning council removed", {})
    return []


def _run_external_agent_skill(agent: Agent, skill_name: str, args: dict) -> None:
    """External agent skills have been removed; this is a no-op."""
    return None


def _review_hook_selection(agent: Agent, report_dir: Path) -> list[str]:
    """External review hooks have been removed; always returns no agents."""
    return []


def _run_external_review_hooks(
    agent: Agent,
    agents: list[str],
    *,
    focus: str = "",
    report_dir: Path | None = None,
) -> list[dict]:
    """External review hooks have been removed; always returns no records."""
    return []


def _review_output_text(record: dict) -> str:
    return str(record.get("stdout") or record.get("stderr") or record.get("error") or "")


def _summarize_external_reviews(records: list[dict]) -> dict:
    """Extract a small actionable summary from external review hook text."""
    combined = "\n\n".join(_review_output_text(record) for record in records)
    lower = combined.lower()
    severity_counts = {
        "critical": len(re.findall(r"\bcritical\b|\bblocker\b|\bblock\b", lower)),
        "high": len(re.findall(r"\bhigh severity\b|\bhigh:\b|\*\*high", lower)),
        "medium": len(re.findall(r"\bmedium severity\b|\bmedium:\b|\*\*medium", lower)),
        "warning": len(re.findall(r"\bwarning\b|\bwarn\b", lower)),
    }
    risk_keywords = [
        "leakage",
        "incident",
        "prevalent",
        "in-sample",
        "holdout",
        "5-fold",
        "cross-validation",
        "overstate",
        "unsupported",
        "shap",
        "beeswarm",
        "forecasting not achieved",
        "not a valid",
        "not support",
    ]
    keyword_hits = sorted({keyword for keyword in risk_keywords if keyword in lower})
    findings: list[str] = []
    for line in combined.splitlines():
        clean = line.strip()
        if not clean:
            continue
        marker = clean.lower()
        if (
            "severity" in marker
            or marker.startswith(("1.", "2.", "3.", "4.", "- ", "* "))
            or any(keyword in marker for keyword in keyword_hits)
        ):
            findings.append(clean[:500])
        if len(findings) >= 16:
            break
    failed_agents = [
        str(record.get("agent", "external"))
        for record in records
        if str(record.get("status", "")).lower() not in {"success", "ok", "pass"}
    ]
    actionable = bool(
        failed_agents
        or severity_counts["critical"]
        or severity_counts["high"]
        or severity_counts["medium"]
        or keyword_hits
    )
    return {
        "status": "needs_repair" if actionable else "clean",
        "agents": [str(record.get("agent", "external")) for record in records],
        "failed_agents": failed_agents,
        "severity_counts": severity_counts,
        "keyword_hits": keyword_hits,
        "actionable_findings": findings,
        "n_records": len(records),
    }


def _write_review_repair_artifacts(report_dir: Path, summary: dict) -> None:
    """Persist structured review findings and the safe repair plan."""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "external_review_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    actions = [
        "Re-run statistical_review over the full session so new scientific-design checks are captured.",
        "Re-run safety_check before publication artifacts are regenerated.",
        "Re-run world_model_audit with association-conditioned framing when trajectory claims are present.",
        "Regenerate the paired technical and Nature-style reports in the same report directory.",
        "Carry unresolved external-review findings into the regenerated report as explicit limitations.",
    ]
    finding_lines = "\n".join(f"- {item}" for item in summary.get("actionable_findings", [])[:12]) or "- None"
    action_lines = "\n".join(f"- {item}" for item in actions)
    (report_dir / "review_repair_plan.md").write_text(
        "# External Review Repair Plan\n\n"
        f"Status: {summary.get('status')}\n\n"
        "## Findings\n\n"
        f"{finding_lines}\n\n"
        "## Safe Repair Actions\n\n"
        f"{action_lines}\n",
        encoding="utf-8",
    )


def _run_review_repair_loop(
    agent: Agent,
    review_records: list[dict],
    *,
    focus: str,
    report_dir: Path,
) -> None:
    """Turn external review hooks into a bounded safe repair/report loop."""
    if not review_records:
        return
    mode = _settings_str(agent.settings, "plan_review_repair_mode", "auto_safe").strip().lower()
    if mode in {"never", "off", "disabled", "skip"}:
        return

    summary = _summarize_external_reviews(review_records)
    try:
        agent.state.custom_data["external_review_records"] = review_records
        agent.state.custom_data["external_review_summary"] = summary
    except Exception:
        pass
    _write_review_repair_artifacts(report_dir, summary)

    if summary.get("status") != "needs_repair":
        console.print("[green]External review found no actionable repair items.[/green]")
        return

    max_loops = max(_settings_int(agent.settings, "plan_review_repair_max_loops", 2), 1)
    loops = int(getattr(agent.state, "custom_data", {}).get("review_repair_loops", 0) or 0)
    if loops >= max_loops:
        console.print(
            "[yellow]External review still has actionable findings, but the repair loop budget is exhausted. "
            "Unresolved items were saved into the report directory.[/yellow]"
        )
        return
    try:
        agent.state.custom_data["review_repair_loops"] = loops + 1
    except Exception:
        pass

    console.print(Panel(
        "External review found actionable issues. Applying safe repair loop:\n"
        "  1. statistical_review(scope='session')\n"
        "  2. safety_check(scope='session')\n"
        "  3. world_model_audit(task=..., association_conditioned_forecast)\n"
        "  4. generate_report(format='dual') in the same report directory\n\n"
        f"Repair plan: {report_dir / 'review_repair_plan.md'}",
        title="[bold yellow]Review Repair Loop[/bold yellow]",
        border_style="yellow",
    ))

    ctx_report_dir = report_dir
    repair_steps = [
        ("statistical_review", {"scope": "session"}),
        ("safety_check", {"scope": "session"}),
        (
            "world_model_audit",
            {
                "task": focus or "post-run Biobank Agent trajectory and model claim audit",
                "simulation_type": "association_conditioned_forecast",
                "calibration_status": "unknown",
                "external_validation_status": "not_validated",
            },
        ),
        (
            "generate_report",
            {
                "title": "UKB longitudinal trajectory feasibility report",
                "format": "dual",
                "output_dir": str(report_dir),
            },
        ),
    ]
    for skill_name, args in repair_steps:
        console.print(f"[dim]Review repair: running {skill_name}...[/dim]")
        try:
            execution = agent._execute_skill_and_record(
                skill_name,
                args,
                ctx_report_dir,
                allow_retry=True,
            )
            if isinstance(execution, dict) and execution.get("is_error"):
                result = execution.get("result", {})
                console.print(f"[yellow]Review repair step {skill_name} returned an error: {result}[/yellow]")
        except Exception as exc:
            console.print(f"[yellow]Review repair step {skill_name} failed: {exc}[/yellow]")
    console.print(f"[green]Review repair artifacts updated in:[/] {report_dir}")


def _get_mcp_manager(agent: Agent):
    """Return a cached MCP manager for the current session."""
    custom_data = getattr(agent.state, "custom_data", None)
    if custom_data is None:
        agent.state.custom_data = {}
        custom_data = agent.state.custom_data
    manager = custom_data.get("mcp_manager")
    registry = custom_data.get("mcp_tool_registry")
    if manager is not None and registry is not None:
        return manager, registry

    from .core.tools.registry import ToolRegistry
    from .extensions.mcp_manager import McpManager

    registry = ToolRegistry(legacy=agent.registry)
    registry.hydrate_from_legacy()
    configured_path = str(getattr(agent.settings, "mcp_config_path", "") or "").strip()
    manager = McpManager(registry, config_path=Path(configured_path) if configured_path else None)
    custom_data["mcp_manager"] = manager
    custom_data["mcp_tool_registry"] = registry
    return manager, registry


def _show_mcp_status(agent: Agent) -> None:
    """Render configured MCP servers and currently loaded remote tools."""
    manager, registry = _get_mcp_manager(agent)
    table = Table(title=f"MCP Servers ({manager.config_path})")
    table.add_column("Name", style="cyan")
    table.add_column("Transport", style="green")
    table.add_column("Enabled", justify="center")
    table.add_column("Running", justify="center")
    table.add_column("Tools", justify="right")
    table.add_column("Restarts", justify="right")
    table.add_column("Command / URL", overflow="fold")
    table.add_column("Last error", overflow="fold")
    configs = manager.load_config()
    status_by_name = {row["name"]: row for row in manager.status()}
    if configs:
        for cfg in configs:
            transport = str(cfg.transport).strip().lower()
            target = cfg.url if transport in {"http", "http-sse", "sse", "streamable-http"} else " ".join([cfg.command, *cfg.args]).strip()
            status = status_by_name.get(cfg.name, {})
            table.add_row(
                cfg.name,
                cfg.transport,
                "yes" if cfg.enabled else "no",
                "yes" if status.get("running") else "no",
                str(status.get("loaded_tools", 0)),
                str(status.get("restarts", 0)),
                target,
                str(status.get("error", "")),
            )
    else:
        table.add_row("(none)", "", "", "", "", "", "Create ~/.biobank_agent/mcp_servers.json", "")
    console.print(table)

    remote_handlers = [
        h for h in registry.list_handlers()
        if h.name.startswith("mcp_")
    ]
    tools = Table(title=f"Loaded MCP Tools ({len(remote_handlers)})")
    tools.add_column("Tool", style="cyan")
    tools.add_column("Capabilities", style="green")
    for handler in remote_handlers:
        caps = ", ".join(c.value for c in handler.required_capabilities())
        tools.add_row(handler.name, caps)
    if not remote_handlers:
        tools.add_row("(none)", "Run /mcp-start after configuring an MCP server")
    console.print(tools)


def _unregister_mcp_legacy_tools(agent: Agent) -> None:
    """Remove previously bridged MCP tools from the legacy skill registry."""
    custom_data = getattr(agent.state, "custom_data", {}) or {}
    names = list(custom_data.get("mcp_legacy_tool_names") or [])
    for name in names:
        try:
            agent.registry.unregister(name)
        except Exception:
            logger.debug("Failed to unregister MCP legacy tool %s", name)
    custom_data["mcp_legacy_tool_names"] = []


def _register_mcp_handlers_in_legacy(agent: Agent, handlers: list) -> int:
    """Expose loaded MCP handlers to the legacy LLM tool schema path.

    The v3 ``ToolRegistry`` is the native home for MCP tools, but the
    production ReAct loop still builds OpenAI schemas from
    ``agent.registry``. Without this bridge ``/mcp-start`` looks
    successful while the model cannot actually call the remote tools.
    """
    import asyncio

    from .core.tools.protocol import ToolContext

    _unregister_mcp_legacy_tools(agent)
    custom_data = getattr(agent.state, "custom_data", None)
    if custom_data is None:
        agent.state.custom_data = {}
        custom_data = agent.state.custom_data

    registered: list[str] = []

    def _make_callable(handler):
        def _call(ctx=None, **kwargs):
            report_dir = getattr(ctx, "report_dir", None)
            tool_ctx = ToolContext(
                name=handler.name,
                args=dict(kwargs or {}),
                capabilities=handler.required_capabilities(),
                settings=agent.settings,
                duckdb_conn=getattr(agent.dm, "conn", None),
                catalog=getattr(agent, "catalog", None),
                state=getattr(agent, "state", None),
                memory=getattr(agent, "memory", None),
                report_dir=report_dir,
                emit_progress=getattr(ctx, "emit_progress", None),
                turn_id=getattr(ctx, "turn_id", None),
                tool_call_id=getattr(ctx, "tool_call_id", None),
            )
            return asyncio.run(handler.handle(tool_ctx))

        _call.__name__ = handler.name
        return _call

    for handler in handlers:
        if not handler.name.startswith("mcp_"):
            continue
        try:
            agent.registry.register(
                handler.name,
                _make_callable(handler),
                handler.spec().to_openai_schema(),
            )
            registered.append(handler.name)
        except Exception as exc:
            logger.warning("Failed to bridge MCP tool %s into legacy registry: %s", handler.name, exc)

    custom_data["mcp_legacy_tool_names"] = registered
    return len(registered)


def _start_mcp(agent: Agent) -> None:
    """Start MCP servers from config and register remote tools."""
    import asyncio

    manager, _registry = _get_mcp_manager(agent)
    if manager.clients:
        console.print("[yellow]MCP is already running. Use /mcp-stop before restarting.[/yellow]")
        _show_mcp_status(agent)
        return
    try:
        loaded = asyncio.run(manager.start())
    except RuntimeError:
        console.print("[red]Cannot start MCP from inside a running event loop.[/]")
        return
    bridged = _register_mcp_handlers_in_legacy(agent, manager.handlers)
    console.print(
        f"[green]MCP start complete: {loaded} remote tool(s) registered, "
        f"{bridged} visible to the agent.[/green]"
    )
    _show_mcp_status(agent)


def _mcp_health(agent: Agent, arg: str = "") -> None:
    """Probe MCP servers and optionally reconnect unhealthy ones."""
    import asyncio

    manager, _registry = _get_mcp_manager(agent)
    repair = "--repair" in str(arg or "").split()
    try:
        rows = asyncio.run(manager.health_check(repair=repair))
    except RuntimeError:
        console.print("[red]Cannot health-check MCP from inside a running event loop.[/]")
        return
    table = Table(title=f"MCP Health ({'repair' if repair else 'probe'})")
    table.add_column("Name", style="cyan")
    table.add_column("OK", justify="center")
    table.add_column("Running", justify="center")
    table.add_column("Tools", justify="right")
    table.add_column("Error", overflow="fold")
    if rows:
        for row in rows:
            table.add_row(
                str(row.get("name", "")),
                "yes" if row.get("ok") else "no",
                "yes" if row.get("running") else "no",
                str(row.get("tools", "")),
                str(row.get("error", "")),
            )
    else:
        table.add_row("(none)", "", "", "", "Create ~/.biobank_agent/mcp_servers.json")
    console.print(table)
    _register_mcp_handlers_in_legacy(agent, manager.handlers)


def _parse_mcp_call_args(raw: str) -> tuple[str, dict[str, Any]]:
    """Parse ``/mcp-call`` arguments without requiring shell quoting."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Usage: /mcp-call <tool> [json|key=value...]")
    parts = shlex.split(text)
    if not parts:
        raise ValueError("Usage: /mcp-call <tool> [json|key=value...]")
    tool_name = parts[0]
    rest = text[len(tool_name):].strip()
    if not rest:
        return tool_name, {}

    json_candidates = [rest]
    if len(parts) == 2:
        json_candidates.append(parts[1])
    for candidate in json_candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        if (candidate.startswith("'") and candidate.endswith("'")) or (
            candidate.startswith('"') and candidate.endswith('"')
        ):
            candidate = candidate[1:-1]
        if candidate.startswith("{"):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                raise ValueError("MCP tool arguments must be a JSON object")
            return tool_name, parsed

    parsed_args: dict[str, Any] = {}
    for token in parts[1:]:
        if "=" not in token:
            raise ValueError("Arguments must be JSON or key=value pairs")
        key, value = token.split("=", 1)
        parsed_args[key] = value
    return tool_name, parsed_args


def _call_mcp_tool(agent: Agent, arg: str) -> None:
    """Call a loaded MCP tool from the CLI and record an auditable row."""
    try:
        tool_name, tool_args = _parse_mcp_call_args(arg)
    except ValueError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return

    if not tool_name.startswith("mcp_"):
        tool_name = f"mcp_{tool_name}"
    if tool_name not in getattr(agent.registry, "_schemas", {}):
        console.print(
            f"[yellow]MCP tool {tool_name!r} is not loaded. Run /mcp-start and /mcp-list first.[/yellow]"
        )
        return

    ctx = _tool_ctx(agent, getattr(agent.settings, "reports_dir", Path("./reports")) / "mcp_calls")
    try:
        result = agent.registry.execute(tool_name, tool_args, ctx=ctx)
    except Exception as exc:
        console.print(Panel(str(exc), title=f"MCP call failed: {tool_name}", border_style="red"))
        try:
            agent.state.records.append({
                "skill": tool_name,
                "args": dict(tool_args),
                "result": {"error": str(exc)},
                "is_error": True,
                "source": "mcp_call",
            })
        except Exception:
            pass
        return

    try:
        agent.state.records.append({
            "skill": tool_name,
            "args": dict(tool_args),
            "result": result,
            "is_error": False,
            "source": "mcp_call",
        })
    except Exception:
        pass
    console.print(Panel(json.dumps(result, indent=2, ensure_ascii=False), title=f"MCP result: {tool_name}", border_style="green"))


def _stop_mcp(agent: Agent) -> None:
    """Stop active MCP clients."""
    import asyncio

    manager, _registry = _get_mcp_manager(agent)
    _unregister_mcp_legacy_tools(agent)
    try:
        asyncio.run(manager.stop())
    except RuntimeError:
        console.print("[red]Cannot stop MCP from inside a running event loop.[/]")
        return
    console.print("[green]MCP clients stopped.[/green]")
    _show_mcp_status(agent)


def _show_cost(token_usage: dict, agent: Agent) -> None:
    """Show token usage and estimated cost."""
    prompt = token_usage["prompt_tokens"]
    completion = token_usage["completion_tokens"]
    total = prompt + completion

    # Rough cost estimates (per 1M tokens)
    model = agent.settings.llm_model.lower()
    if "claude" in model and "sonnet" in model:
        cost_per_m_in, cost_per_m_out = 3.0, 15.0
    elif "gpt-4" in model:
        cost_per_m_in, cost_per_m_out = 2.5, 10.0
    else:
        cost_per_m_in, cost_per_m_out = 1.0, 3.0

    est_cost = (prompt * cost_per_m_in + completion * cost_per_m_out) / 1_000_000

    console.print(Panel(
        f"Prompt tokens:     {prompt:>10,}\n"
        f"Completion tokens: {completion:>10,}\n"
        f"Total tokens:      {total:>10,}\n"
        f"Estimated cost:    ${est_cost:>9.4f}\n"
        f"Model:             {agent.settings.llm_model}",
        title="Token Usage",
    ))


def _compact(agent: Agent) -> None:
    """Compress conversation history, keeping system + last 10 turns."""
    before = len(agent.messages)
    if before <= 20:
        console.print(f"[dim]History already compact ({before} messages).[/]")
        return

    # Keep the last 20 messages (roughly 10 turns)
    agent.messages = agent.messages[-20:]
    after = len(agent.messages)
    console.print(f"[green]Compacted:[/] {before} → {after} messages ({before - after} removed)")


def _clear(agent: Agent) -> None:
    """Reset session state."""
    agent.messages.clear()
    agent.state.cohorts.clear()
    agent.state.models.clear()
    agent.state.model_metadata.clear()
    agent.state.figures.clear()
    agent.state.records.clear()
    agent.state.feature_matrix = None
    agent.state.labels = None
    agent.state.current_report_dir = None
    agent.state.embeddings.clear()
    agent.state.custom_data.clear()
    console.print("[green]Session cleared.[/] All cohorts, models, figures, and history reset.")


def _export(agent: Agent, fmt: str) -> None:
    """Export session as JSON or Markdown."""
    export_dir = agent.settings.reports_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if fmt == "json":
        data = {
            "timestamp": timestamp,
            "model": agent.settings.llm_model,
            "records": [
                {
                    "timestamp": r.timestamp,
                    "skill": r.skill,
                    "args": r.args,
                    "key_results": {k: str(v) for k, v in r.key_results.items()},
                    "figures": r.figure_paths,
                }
                for r in agent.state.records
            ],
            "cohorts": {name: len(df) for name, df in agent.state.cohorts.items()},
            "models": list(agent.state.model_metadata.keys()),
            "n_figures": len(agent.state.figures),
        }
        path = export_dir / f"session_{timestamp}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
    elif fmt in ("md", "markdown"):
        lines = [f"# Biobank Agent Session Export\n", f"**Date:** {timestamp}\n"]
        for r in agent.state.records:
            lines.append(f"## {r.skill}\n")
            lines.append(f"**Args:** {r.args}\n")
            for k, v in r.key_results.items():
                lines.append(f"- {k}: {v}\n")
            lines.append("")
        path = export_dir / f"session_{timestamp}.md"
        path.write_text("\n".join(lines))
    else:
        console.print(f"[yellow]Unknown format: {fmt}. Use 'json' or 'md'.[/]")
        return

    console.print(f"[green]Session exported to:[/] {path}")


def _show_history(agent: Agent) -> None:
    if not agent.state.records:
        console.print("[dim]No analyses performed yet.[/]")
        return
    table = Table(title="Analysis History", show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Time", style="dim", width=16)
    table.add_column("Skill", style="cyan")
    table.add_column("Key Results", style="white", max_width=60)
    for i, r in enumerate(agent.state.records, 1):
        results_str = ", ".join(f"{k}={v}" for k, v in list(r.key_results.items())[:3])
        table.add_row(str(i), r.timestamp[:16], r.skill, results_str[:60])
    console.print(table)


def _show_figures(agent: Agent) -> None:
    """List all generated figures."""
    if not agent.state.figures:
        console.print("[dim]No figures generated yet.[/]")
        return
    table = Table(title="Generated Figures")
    table.add_column("#", style="dim", width=3)
    table.add_column("Path", style="cyan")
    table.add_column("Format", style="green", width=5)
    for i, fig_path in enumerate(agent.state.figures, 1):
        p = Path(fig_path)
        table.add_row(str(i), str(p.name), p.suffix.lstrip(".").upper())
    console.print(table)
    console.print(f"[dim]Total: {len(agent.state.figures)} figures[/]")


def _show_cohorts(agent: Agent) -> None:
    """List active cohorts with summary stats."""
    if not agent.state.cohorts:
        console.print("[dim]No active cohorts.[/]")
        return
    table = Table(title="Active Cohorts")
    table.add_column("Name", style="cyan")
    table.add_column("Subjects", style="green", justify="right")
    table.add_column("Cases", style="yellow", justify="right")
    table.add_column("Controls", style="dim", justify="right")
    for name, df in agent.state.cohorts.items():
        n_total = len(df)
        if "label" in df.columns:
            n_cases = int(df["label"].sum())
            n_controls = n_total - n_cases
        else:
            n_cases, n_controls = "?", "?"
        table.add_row(name, f"{n_total:,}", str(n_cases), str(n_controls))
    console.print(table)


def _show_models(agent: Agent) -> None:
    """List trained models with AUC."""
    if not agent.state.model_metadata:
        console.print("[dim]No trained models.[/]")
        return
    table = Table(title="Trained Models")
    table.add_column("Key", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("AUC", style="yellow", justify="right")
    table.add_column("Cases", justify="right")
    table.add_column("Features", justify="right")
    for key, meta in agent.state.model_metadata.items():
        auc = meta.get("auc")
        auc_str = f"{float(auc):.4f}" if auc is not None else "N/A"
        table.add_row(
            key,
            meta.get("model_type", "?"),
            auc_str,
            str(meta.get("n_cases", "?")),
            str(meta.get("n_features", "?")),
        )
    console.print(table)


def _record_pipeline(agent: Agent, name: str) -> None:
    """Save current session's skill calls as a named pipeline."""
    if not agent.state.records:
        console.print("[yellow]No skill calls to record.[/]")
        return
    steps = []
    skip = {"think", "record_macro", "replay_pipeline", "list_pipelines"}
    for r in agent.state.records:
        if r.skill not in skip:
            steps.append({"skill": r.skill, "args": r.args})
    if not steps:
        console.print("[yellow]No recordable skill calls found.[/]")
        return
    agent.memory.save_pipeline(name, steps)
    console.print(f"[green]Saved pipeline '{name}' with {len(steps)} steps.[/]")


def _show_pipelines(agent: Agent) -> None:
    """List saved pipelines from long-term memory."""
    pipelines = agent.memory.list_pipelines()
    if not pipelines:
        console.print("[dim]No saved pipelines.[/]")
        return
    for p in pipelines:
        steps = agent.memory.get_pipeline(p)
        n = len(steps) if steps else 0
        console.print(f"  [cyan]{p}[/]: {n} steps")


def _show_errors(agent: Agent) -> None:
    """Show most common errors from long-term memory."""
    errors = agent.memory.most_common_errors(10)
    if not errors:
        console.print("[dim]No errors recorded.[/]")
        return
    table = Table(title="Error Catalog")
    table.add_column("Error", style="red")
    table.add_column("Skill", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Last Seen", style="dim")
    for e in errors:
        table.add_row(
            e["error_type"],
            e["skill"],
            str(e["count"]),
            e["last_seen"][:10] if e["last_seen"] else "?",
        )
    console.print(table)


def _show_memory(agent: Agent) -> None:
    """Show long-term memory summary."""
    summary = agent.memory.summary()
    if not summary:
        console.print("[dim]Long-term memory is empty.[/]")
        return
    console.print(Panel(summary, title="Long-term Memory"))


def _show_evolution_status(agent: Agent, raw_arg: str = "") -> None:
    """Dry-run evolution dashboard for repeated tool failures."""
    flags = set((raw_arg or "").split())
    custom_data = getattr(agent.state, "custom_data", None)
    if custom_data is None:
        agent.state.custom_data = {}
        custom_data = agent.state.custom_data

    if "--pause" in flags:
        custom_data["evolve_paused"] = True
        console.print("[yellow]Evolution loop paused. No auto-improvement proposals will be applied.[/]")
        return
    if "--resume" in flags:
        custom_data["evolve_paused"] = False
        console.print("[green]Evolution loop resumed in review-only dry-run mode.[/green]")

    paused = bool(custom_data.get("evolve_paused", False))
    learner = getattr(agent, "tool_learner", None)
    if learner is None:
        console.print("[yellow]Tool learner unavailable.[/]")
        return

    patterns = learner.mine_failure_patterns(min_count=3)
    proposals = learner.auto_propose_skill_improvement(min_count=3)

    status = "paused" if paused else "review-only"
    console.print(Panel(
        f"Mode: {status}\n"
        "This command is review-only by default. It surfaces repeated failure patterns "
        "and creates review-only candidate patch plans when requested with "
        "`--write-proposals`. Use `--apply-low` to route only LOW-risk generated "
        "proposal artifacts through the auto-merger allow-list. Use "
        "`--apply-medium` only from the Textual TUI or another host that "
        "provides an explicit confirmation callback. Use `--scheduled-run` "
        "or `--write-history` from cron/CI to persist pass-rate style pattern "
        "history under `reports/eval/evolution` without modifying code.",
        title="Evolution",
    ))

    table = Table(title=f"Repeated Failure Patterns ({len(patterns)})")
    table.add_column("Skill", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Signature", overflow="fold")
    table.add_column("Suggested action", overflow="fold")
    for pattern in patterns:
        table.add_row(
            pattern.skill_name,
            str(pattern.count),
            pattern.error_signature,
            pattern.suggested_action,
        )
    if not patterns:
        table.add_row("(none)", "0", "Need at least 3 repeated failures", "")
    console.print(table)

    if proposals:
        console.print("[dim]Review-only proposals:[/dim]")
        for p in proposals:
            console.print(
                f"  - {p['skill']}: {p['suggested_action']} "
                f"(failures={p['failure_count']}, risk={p['risk']}, target={p.get('target_path', '')})"
            )
        if "--write-proposals" in flags:
            written = _write_evolution_proposals(agent, proposals)
            console.print(f"[green]Evolution proposals written: {written}[/green]")
        if "--apply-low" in flags and not paused:
            outcomes = _apply_low_evolution_proposals(agent, proposals)
            for outcome in outcomes:
                color = "green" if outcome.get("status") == "merged" else "yellow"
                console.print(
                    f"[{color}]LOW proposal {outcome.get('skill')}: "
                    f"{outcome.get('status')} {outcome.get('target_path')} "
                    f"{outcome.get('error', '')}[/]"
                )
        elif "--apply-low" in flags and paused:
            console.print("[yellow]Evolution is paused; LOW proposals were not applied.[/]")
        if "--apply-medium" in flags and not paused:
            confirm_fn = custom_data.get("tui_confirm_fn") or custom_data.get("evolve_confirm_fn")
            if confirm_fn is None:
                console.print(
                    "[yellow]MEDIUM proposals require an explicit confirmation callback. "
                    "Provide `state.custom_data['evolve_confirm_fn']` to approve them.[/]"
                )
            outcomes = _apply_evolution_proposals(
                agent,
                proposals,
                risk_levels={"medium"},
                confirm_fn=confirm_fn,
                label="MEDIUM",
            )
            for outcome in outcomes:
                color = "green" if outcome.get("status") == "merged" else "yellow"
                console.print(
                    f"[{color}]MEDIUM proposal {outcome.get('skill')}: "
                    f"{outcome.get('status')} {outcome.get('target_path')} "
                    f"{outcome.get('error', '')}[/]"
                )
        elif "--apply-medium" in flags and paused:
            console.print("[yellow]Evolution is paused; MEDIUM proposals were not applied.[/]")
    if "--scheduled-run" in flags or "--write-history" in flags:
        run = _write_evolution_pattern_history(agent, min_count=3)
        console.print(
            "[green]Evolution pattern history written:[/] "
            f"{run.artifacts.get('run_json', '')} "
            f"({run.status}, patterns={run.n_patterns}, proposals={run.n_proposals})"
        )


def _write_evolution_pattern_history(agent: Agent, *, min_count: int = 3):
    """Write scheduled self-evolution pattern mining artifacts."""
    from .core.evolution.pattern_mining import run_pattern_mining

    reports_dir = Path(getattr(agent.settings, "reports_dir", Path("./reports")))
    output_dir = reports_dir / "eval" / "evolution"
    return run_pattern_mining(
        getattr(agent, "tool_learner", None),
        output_dir=output_dir,
        min_count=min_count,
        write_artifacts=True,
    )


def _write_evolution_proposals(agent: Agent, proposals: list[dict]) -> Path:
    """Write review-only evolution proposals under reports/generated_skills."""
    root = Path(getattr(agent.settings, "reports_dir", Path("./reports"))) / "generated_skills"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"evolve_proposals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    lines = [
        "# Biobank Agent Evolution Proposals",
        "",
        "These are review-only patch plans. No code was modified.",
        "",
    ]
    for idx, proposal in enumerate(proposals, start=1):
        lines.extend([
            f"## {idx}. {proposal.get('skill', 'unknown')}",
            "",
            f"- Failures: {proposal.get('failure_count', 0)}",
            f"- Risk: {proposal.get('risk', 'unknown')}",
            f"- Risk reason: {proposal.get('risk_reason', '')}",
            f"- Target path: `{proposal.get('target_path', '')}`",
            f"- Error signature: `{proposal.get('error_signature', '')}`",
            f"- Suggested action: {proposal.get('suggested_action', '')}",
            "",
            "```diff",
            str(proposal.get("candidate_patch", "")).rstrip(),
            "```",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _apply_low_evolution_proposals(agent: Agent, proposals: list[dict]) -> list[dict]:
    """Apply only LOW-risk generated proposal artifacts through AutoMerger.

    This intentionally writes review artifacts under ``reports/generated_skills``
    and does not activate custom skills. MEDIUM/HIGH proposals remain manual.
    """
    return _apply_evolution_proposals(
        agent,
        proposals,
        risk_levels={"low"},
        confirm_fn=None,
        label="LOW",
    )


def _apply_evolution_proposals(
    agent: Agent,
    proposals: list[dict],
    *,
    risk_levels: set[str],
    confirm_fn=None,
    label: str = "proposal",
) -> list[dict]:
    """Apply generated proposal artifacts after eval and risk-specific gates."""
    try:
        from .core.evolution.auto_merger import AutoMerger, Patch
    except Exception as e:
        return [{"skill": "(system)", "status": "error", "target_path": "", "error": str(e)}]

    normalized_risks = {str(item).lower() for item in risk_levels}
    repo_root = Path.cwd()
    merger = AutoMerger(
        repo_root=repo_root,
        confirm_fn=confirm_fn,
        write_allow_paths=("reports/generated_skills/",),
        audit_log_path=repo_root / "reports" / "generated_skills" / "evolution_decisions.jsonl",
    )
    selected = [p for p in proposals if str(p.get("risk", "")).lower() in normalized_risks]
    eval_result = _run_evolution_apply_gate(repo_root, selected, risk_levels=normalized_risks)
    if not getattr(eval_result, "all_passed", False):
        return [{
            "skill": "(eval-gate)",
            "target_path": "",
            "status": "skipped",
            "risk": "",
            "error": getattr(eval_result, "summary", "evolution eval gate failed"),
        }]
    outcomes: list[dict] = []
    for proposal in selected:
        proposal_risk = str(proposal.get("risk", "")).lower()
        candidate_patch = str(proposal.get("candidate_patch", ""))
        patch_text = (
            _candidate_patch_file_content(candidate_patch)
            if proposal_risk == "low"
            else candidate_patch
        )
        patch = Patch(
            target_path=str(proposal.get("target_path", "")),
            unified_diff=patch_text,
            target_skill=str(proposal.get("skill", "")),
            summary=str(proposal.get("suggested_action", "")),
            metadata={"source": "evolve", "proposal_risk": proposal_risk},
        )
        try:
            outcome = asyncio.run(merger.review_and_merge(patch, eval_result=eval_result))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                outcome = loop.run_until_complete(merger.review_and_merge(patch, eval_result=eval_result))
            finally:
                loop.close()
        outcomes.append({
            "skill": proposal.get("skill", ""),
            "target_path": proposal.get("target_path", ""),
            "status": outcome.status,
            "risk": getattr(outcome.risk, "value", str(outcome.risk)),
            "error": outcome.error,
        })
    if not outcomes:
        outcomes.append({
            "skill": "(none)",
            "target_path": "",
            "status": "skipped",
            "risk": "",
            "error": f"no {label}-risk proposals available",
        })
    return outcomes


def _run_evolution_apply_gate(
    repo_root: Path,
    proposals: list[dict],
    *,
    risk_levels: set[str] | None = None,
) -> SimpleNamespace:
    """Run the required lightweight gate before applying LOW proposals."""
    allowed = {str(item).lower() for item in (risk_levels or {"low"})}
    selected = [p for p in proposals if str(p.get("risk", "")).lower() in allowed]
    for proposal in selected:
        target = str(proposal.get("target_path", ""))
        patch = str(proposal.get("candidate_patch", ""))
        if not target.startswith("reports/generated_skills/"):
            return SimpleNamespace(all_passed=False, summary=f"target outside generated-skills allow-list: {target}")
        if not patch.strip():
            return SimpleNamespace(all_passed=False, summary=f"empty candidate patch for {proposal.get('skill', '')}")

    behavioral = repo_root / "tests" / "eval" / "behavioral" / "test_behavioral_always.py"
    if behavioral.exists():
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    str(behavioral),
                    "--capture=no",
                    "-p",
                    "no:cacheprovider",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except Exception as exc:
            return SimpleNamespace(all_passed=False, summary=f"ALWAYS_PASSES eval gate could not run: {exc}")
        if proc.returncode != 0:
            output = (proc.stdout + "\n" + proc.stderr).strip()
            return SimpleNamespace(
                all_passed=False,
                summary=f"ALWAYS_PASSES eval gate failed: {output[:500]}",
            )
        return SimpleNamespace(all_passed=True, summary="ALWAYS_PASSES eval gate passed")

    return SimpleNamespace(
        all_passed=True,
        summary="static generated-skill proposal gate passed; behavioral eval file not present in this repo root",
    )


def _candidate_patch_file_content(candidate_patch: str) -> str:
    """Convert ToolLearner's review diff into the file body to store."""
    lines = []
    in_hunk = False
    for line in str(candidate_patch or "").splitlines():
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:])
    body = "\n".join(lines).rstrip()
    return (body + "\n") if body else str(candidate_patch or "")


def _replay_provenance(agent: Agent, raw_arg: str) -> None:
    """``/replay <provenance_id> [--strict] [--dry-run]`` slash command.

    Loads a checkpointed skill execution and re-runs it after verifying
    the data fingerprint hasn't drifted. By default soft drift produces
    a warning and proceeds; ``--strict`` refuses on any drift; severe
    drift always refuses.
    """
    parts = (raw_arg or "").split()
    if not parts:
        console.print(
            "[yellow]Usage: /replay <provenance_id> [--strict] [--dry-run][/]"
        )
        return
    pid = parts[0]
    flags = {p.strip() for p in parts[1:]}
    allow_drift = "--strict" not in flags
    dry_run = "--dry-run" in flags

    try:
        from .core.safety.replay import replay as _replay_impl
    except Exception as e:
        console.print(f"[red]Replay subsystem unavailable: {e}[/]")
        return

    try:
        outcome = _replay_impl(
            provenance_id=pid,
            legacy_agent=agent,
            allow_data_drift=allow_drift,
            dry_run=dry_run,
        )
    except Exception as e:
        console.print(f"[red]Replay failed: {e}[/]")
        return

    status = outcome.get("status", "unknown")
    drift = outcome.get("drift", "?")
    title = f"Replay {pid[:12]} — {status} (drift={drift})"
    body_lines = [f"skill: [cyan]{outcome.get('skill', '?')}[/]"]
    args = outcome.get("args") or {}
    if args:
        body_lines.append(f"args: {args}")
    diff = outcome.get("diff") or {}
    if diff:
        body_lines.append(f"fingerprint diff: {diff}")
    if "result" in outcome:
        body_lines.append(f"result: {str(outcome['result'])[:400]}")
    if outcome.get("error"):
        body_lines.append(f"[red]error: {outcome['error']}[/]")
    if outcome.get("report_dir"):
        body_lines.append(f"report_dir: {outcome['report_dir']}")
    console.print(Panel("\n".join(body_lines), title=title))


def _show_model_pool(agent: Agent) -> None:
    """Show available models in the orchestrator pool."""
    pool = agent.orchestrator.model_pool
    enabled = agent.settings.multi_model_enabled
    table = Table(title=f"Model Pool ({'enabled' if enabled else 'disabled'})")
    table.add_column("Model", style="cyan")
    table.add_column("Role", style="green")
    table.add_column("Priority", justify="right")
    table.add_column("Default", style="yellow")
    for spec in pool:
        is_default = "yes" if spec.model_id == agent.settings.llm_model else ""
        table.add_row(spec.model_id, spec.role, str(spec.priority), is_default)
    console.print(table)
    if not enabled:
        console.print("[dim]Enable with MULTI_MODEL_ENABLED=true in .env[/]")
        console.print("[dim]Add models with MODEL_POOL=model1,model2,model3[/]")
    console.print("[dim]Run /models-available to fetch relay-supported models.[/]")


def _show_available_llm_models(agent: Agent) -> None:
    """Fetch and display model IDs available from relay /v1/models."""
    models: list[str] = []
    fetch_fn = getattr(agent, "refresh_available_models", None)
    health = "unknown"
    try:
        if callable(fetch_fn):
            models = fetch_fn()
            health = "ok"
        else:
            llm = getattr(agent, "llm", None)
            if llm and hasattr(llm, "list_models"):
                models = llm.list_models(refresh=True)
                health = "ok"
            else:
                models = list(getattr(agent, "available_models", []))
    except Exception as e:
        console.print(f"[red]Failed to fetch model list: {e}[/]")
        health = "degraded"
        return

    if not models:
        console.print("[yellow]No models returned by relay endpoint.[/]")
        console.print("[dim]Check LLM_BASE_URL / API key, or try again later.[/]")
        return

    pool_ids = {spec.model_id for spec in getattr(agent.orchestrator, "model_pool", [])}
    llm = getattr(agent, "llm", None)
    cached_count = len(getattr(llm, "_model_cache", []) or []) if llm else 0
    cache_ts = float(getattr(llm, "_model_cache_ts", 0.0) or 0.0) if llm else 0.0
    cache_age_s = 0.0
    if cache_ts > 0:
        import time as _time
        cache_age_s = max(0.0, _time.time() - cache_ts)

    console.print(
        Panel(
            f"Endpoint: {getattr(agent.settings, 'llm_base_url', '(unknown)')}\n"
            f"Health: {health}\n"
            f"Returned models: {len(models)}\n"
            f"Cached models: {cached_count}\n"
            f"Cache age: {cache_age_s:.1f}s",
            title="Relay Health",
        )
    )

    table = Table(title=f"Relay Models ({len(models)})")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Model ID", style="cyan")
    table.add_column("In Pool", style="green")
    table.add_column("Default", style="yellow")

    for i, model_id in enumerate(models, 1):
        table.add_row(
            str(i),
            model_id,
            "yes" if model_id in pool_ids else "",
            "yes" if model_id == agent.settings.llm_model else "",
        )
    console.print(table)


def _switch_model(agent: Agent, arg: str) -> None:
    """Switch the active model, preserving legacy mock compatibility."""
    if not arg:
        console.print(f"Current model: [cyan]{agent.settings.llm_model}[/]")
        return
    old = agent.settings.llm_model
    switch_fn = getattr(agent, "switch_model", None)
    if callable(switch_fn):
        switch_fn(arg)
    # Keep direct assignments for compatibility with mocks/tests.
    agent.settings.llm_model = arg
    agent.llm.model = arg
    console.print(f"Model switched: {old} → [cyan]{arg}[/]")


def _set_strategy(agent: Agent, arg: str) -> None:
    """Set the multi-model routing strategy."""
    from .complexity import Strategy
    if not arg:
        current = "auto" if agent.settings.multi_model_enabled else "single"
        console.print(f"Current strategy: [cyan]{current}[/]")
        console.print("[dim]Options: auto, single, debate, ensemble[/]")
        return
    arg = arg.lower()
    if arg == "single":
        agent.settings.multi_model_enabled = False
        console.print("[green]Strategy set to single-model[/]")
    elif arg in ("auto", "debate", "ensemble"):
        agent.settings.multi_model_enabled = True
        console.print(f"[green]Strategy set to {arg}[/]")
    else:
        console.print(f"[yellow]Unknown strategy: {arg}. Use: auto, single, debate, ensemble[/]")


def _force_debate(agent: Agent, query: str, token_usage: dict) -> None:
    """Force a multi-model debate for a query."""
    from .complexity import Strategy
    from rich.markdown import Markdown

    if len(agent.orchestrator.model_pool) < 2:
        console.print("[yellow]Debate requires 2+ models. Add MODEL_POOL=model1,model2 to .env[/]")
        return

    console.print(f"[bold cyan]Debate:[/] {query}")
    console.print(f"[dim]Models: {', '.join(m.model_id for m in agent.orchestrator.model_pool[:3])}[/]")

    try:
        messages = [agent._system_message()] + agent.messages + [
            {"role": "user", "content": query}
        ]
        with console.status("[bold green]Models debating..."):
            response = agent.orchestrator.debate(
                query=query,
                messages=messages,
                tools=agent.registry.tool_schemas() or None,
            )
        console.print()
        console.print(Markdown(response.text))
        if getattr(response, "safety_status", "PASS") != "PASS":
            console.print(f"[yellow]safety_status={response.safety_status}[/]")
    except Exception as e:
        console.print(f"[red]Debate failed: {e}[/]")


if __name__ == "__main__":
    main()
