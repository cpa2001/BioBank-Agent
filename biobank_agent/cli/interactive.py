"""Default interactive CLI shell for BioBank-Agent."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import select
import shlex
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from rich import box
from rich.cells import cell_len
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape as _rich_escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from biobank_agent import __version__
from biobank_agent.cli.commands import CommandContext
from biobank_agent.cli.commands.registry import SlashCommandRegistry
from biobank_agent.cli.render import render_result_payload, render_run_tree
from biobank_agent.core.events import AgentEvent, AgentEventType
from biobank_agent.core.memory.action_graph import ActionGraph
from biobank_agent.core.tools.registry import ToolRegistry
from biobank_agent.extensions.mcp_manager import McpManager
from biobank_agent.runtime import (
    ActionGraphNode,
    AgentRuntime,
    AgentSession,
    HarnessTask,
    RuntimeHarnessRunner,
    ProviderRequest,
    ProviderResponse,
    ProviderRole,
    ProviderRouter,
    PlanStatus,
    RuntimeConfig,
    RuntimeStatus,
    SessionStore,
    ToolCall,
)
from biobank_agent.runtime.audit import RuntimeAuditReport, audit_session, write_audit_report
from biobank_agent.runtime.run_eval import default_evaluators, evaluate_run_tree
from biobank_agent.runtime.run_tree import build_run_tree, summarize_run_tree
from biobank_agent.runtime.evolution import LearningReport, learn_from_session, reject_persistent_apply, write_learning_report
from biobank_agent.runtime.completion import CompletionGate
from biobank_agent.runtime.council import CouncilError
from biobank_agent.runtime.harness import write_harness_report
from biobank_agent.runtime.jobs import JobManager
from biobank_agent.runtime.planner import RuntimePlanner
from biobank_agent.runtime.replay import replay_trajectory
from biobank_agent.runtime.researcher import RuntimeResearcher
from biobank_agent.progress import PlanRunDashboard


class PlanStepTimeoutError(BaseException):
    """Raised when one autonomous plan step exceeds the configured wall-clock cap.

    Inherits ``BaseException`` (not ``Exception``), like ``KeyboardInterrupt``: the SIGALRM handler
    raises this from deep inside ``runtime.run_turn``, whose broad ``except Exception`` tool/agent-loop
    guards would otherwise swallow the deadline before the step's ``except`` handler can run."""


class PlanBuildTimeoutError(BaseException):
    """Raised when plan drafting exceeds the configured wall-clock cap. ``BaseException`` for the same
    reason — the planner/council's broad ``except Exception`` guards must not swallow the deadline."""


# Active inactivity watchdogs. ``mark_activity()`` refreshes them on every sign of progress (an LLM
# token, a step, a phase change), so a model that is actively streaming is never judged as timed out —
# the deadline measures SILENCE, not total wall-clock.
_ACTIVE_TIMEOUTS: list[dict] = []


def mark_activity() -> None:
    """Signal forward progress so an INACTIVITY watchdog re-arms instead of firing. No-op when idle.

    Only refreshes activity-based timeouts (plan drafting). Hard per-step deadlines are left untouched
    so a real provider stall during a step still fires regardless of unrelated activity elsewhere."""
    if not _ACTIVE_TIMEOUTS:
        return
    now = time.monotonic()
    for state in _ACTIVE_TIMEOUTS:
        if state.get("activity_based"):
            state["last_activity"] = now


@contextmanager
def _wall_clock_timeout(timeout_s: float, exc_type: type[BaseException], label: str,
                        *, activity_based: bool = False):
    if timeout_s <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    state = {"last_activity": time.monotonic(), "activity_based": activity_based}

    def _handler(_signum, _frame):
        if activity_based:
            idle = time.monotonic() - state["last_activity"]
            if idle < timeout_s:
                # Progress since the last tick — re-arm for the remaining idle window instead of firing.
                signal.setitimer(signal.ITIMER_REAL, max(0.05, timeout_s - idle))
                return
            raise exc_type(f"{label} stalled for {timeout_s:.0f}s with no model activity")
        # Hard deadline: a real provider stall must fire regardless of activity elsewhere.
        raise exc_type(f"{label} exceeded {timeout_s:.0f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_s)
    _ACTIVE_TIMEOUTS.append(state)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        try:
            _ACTIVE_TIMEOUTS.remove(state)
        except ValueError:
            pass
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _plan_step_wall_clock_timeout(timeout_s: float):
    # Hard per-step deadline: a stalled provider during a step must fire regardless of other activity.
    return _wall_clock_timeout(timeout_s, PlanStepTimeoutError, "plan step")


def _plan_build_wall_clock_timeout(timeout_s: float):
    # Inactivity-based: while the planner's models are actively streaming, drafting is not "stalled".
    return _wall_clock_timeout(timeout_s, PlanBuildTimeoutError, "plan drafting", activity_based=True)


class LLMProvider:
    """Thin runtime provider adapter for the existing OpenAI-compatible LLM client."""

    def __init__(self, llm: "LLMClient", *, provider_name: str = "openai-compatible") -> None:
        self.llm = llm
        self.provider_name = provider_name

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        stream_cb = getattr(request, "stream_cb", None)
        # Stream only for plain-text calls (no tools). Tool-call turns keep the
        # non-streaming path unchanged. The generator yields text deltas and
        # returns the final LLMResponse (text/tool_calls/usage) on StopIteration.
        if stream_cb is not None and not request.tools:
            gen = self.llm.stream(messages=request.messages, tools=None)
            deltas: list[str] = []
            final = None
            while True:
                try:
                    chunk = next(gen)
                except StopIteration as stop:
                    final = stop.value
                    break
                deltas.append(chunk)
                mark_activity()  # a streaming token is progress — refresh the inactivity watchdog
                try:
                    stream_cb(chunk)
                except Exception:
                    pass  # a progress hook must never break generation
            return ProviderResponse(
                text=(final.text if final else "".join(deltas)),
                tool_calls=[ToolCall(id=tc.id, name=tc.name, args=dict(tc.args or {})) for tc in (final.tool_calls if final else [])],
                deltas=deltas,
                usage=dict((final.usage if final else {}) or {}),
                provider=self.provider_name,
                model=request.model or self.llm.model,
            )
        response = self.llm.chat(messages=request.messages, tools=request.tools or None)
        return ProviderResponse(
            text=response.text,
            tool_calls=[ToolCall(id=tc.id, name=tc.name, args=dict(tc.args or {})) for tc in response.tool_calls],
            usage=dict(response.usage or {}),
            provider=self.provider_name,
            model=request.model or self.llm.model,
        )


class LazyLLMProvider:
    """Deferred provider wrapper that only imports the LLM client on demand."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        provider_name: str,
        tool_call_content_mode: str = "null",
        request_timeout_s: float = 60.0,
        max_retries: int = 3,
        retry_base_delay_s: float = 2.0,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.provider_name = provider_name
        self.tool_call_content_mode = tool_call_content_mode
        self.request_timeout_s = float(request_timeout_s or 60.0)
        self.max_retries = max(0, int(max_retries))
        self.retry_base_delay_s = max(0.0, float(retry_base_delay_s))
        self._provider: LLMProvider | None = None

    def set_request_timeout(self, request_timeout_s: float) -> float:
        previous = self.request_timeout_s
        next_timeout = float(request_timeout_s or previous or 60.0)
        if abs(next_timeout - previous) > 1e-9:
            self.request_timeout_s = next_timeout
            self._provider = None
        return previous

    def set_llm_policy(
        self,
        *,
        request_timeout_s: float | None = None,
        max_retries: int | None = None,
    ) -> tuple[float, int]:
        previous = (self.request_timeout_s, self.max_retries)
        changed = False
        if request_timeout_s is not None:
            next_timeout = float(request_timeout_s or self.request_timeout_s or 60.0)
            if abs(next_timeout - self.request_timeout_s) > 1e-9:
                self.request_timeout_s = next_timeout
                changed = True
        if max_retries is not None:
            next_retries = max(0, int(max_retries))
            if next_retries != self.max_retries:
                self.max_retries = next_retries
                changed = True
        if changed:
            self._provider = None
        return previous

    def _ensure(self) -> LLMProvider:
        if self._provider is None:
            from biobank_agent.llm import LLMClient

            llm = LLMClient(
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                request_timeout_s=self.request_timeout_s,
                max_retries=self.max_retries,
                retry_base_delay_s=self.retry_base_delay_s,
            )
            if hasattr(llm, "tool_call_content_mode"):
                llm.tool_call_content_mode = self.tool_call_content_mode
            self._provider = LLMProvider(llm, provider_name=self.provider_name)
        return self._provider

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        return self._ensure().complete(request)


class _CommandToolTurn:
    def __init__(self, turn_id: str) -> None:
        self.id = turn_id


class _LineExecutionView:
    """No-TTY fallback for the live execution view: exposes the same
    ``start()/record()/stop()`` API as :class:`PlanRunDashboard` but logs one
    line per event (no ``Live``, no daemon refresh thread) so autonomous
    execution still runs + reports progress when stdout is not a terminal."""

    _ICONS = {"success": "✓", "done": "✓", "failed": "✗", "error": "✗", "cancelled": "⊘", "warning": "!"}

    def __init__(self, console: Console) -> None:
        self.console = console

    def start(self, *_args: Any, **_kw: Any) -> None:  # no-op
        pass

    def record(self, phase: str, actor: str = "biobank", status: str = "running",
               message: str = "", metadata: dict[str, Any] | None = None) -> None:
        mark_activity()  # recorded progress refreshes the inactivity watchdog
        icon = self._ICONS.get(status, "•")
        self.console.print(f"[dim]{_rich_escape(str(phase))}[/] {icon} {_rich_escape(str(message))}")

    def stop(self, *_args: Any, **_kw: Any) -> None:  # no-op
        pass


# Synthetic final row appended to every clarification menu. Selecting it (or
# pressing Tab on any option) opens a free-text prompt so the user can type a
# custom answer / correct an option (inline "type something" UX).
_CUSTOM_ANSWER_LABEL = "Type a custom answer / 以上都不是"


@dataclass(frozen=True)
class PlanReviewChoice:
    key: str
    label: str
    description: str
    action: str


@dataclass
class InteractiveShell:
    """Runtime-backed prompt/composer loop used by the ``biobank`` command."""

    settings: Any
    console: Console = field(default_factory=Console)
    # Initial workspace (cwd) for the session; defaults to the launch directory.
    # Set via the `--workspace`/`--cwd` flag (issue #4).
    workspace: str = ""
    runtime: AgentRuntime | None = None
    session: AgentSession | None = None
    legacy_agent: Agent | None = None
    slash_registry: SlashCommandRegistry = field(default_factory=SlashCommandRegistry)
    prompt_session: PromptSession | None = None
    token_usage: dict[str, Any] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    exit_requested: bool = False
    legacy_agent_factory: Any | None = None
    mcp_manager: McpManager | None = None
    tool_registry: ToolRegistry | None = None
    plan_review_choice_provider: Any | None = None
    _researcher: Any = None
    _completion_gate: Any = None
    _activity: str = "idle"
    _activity_started_at: float = field(default_factory=time.time)
    # The live execution view during _execute_plan_autonomously, so streamed
    # subprocess lines (ctx.emit_line) can surface on it in real time (issue #2).
    _active_exec_view: Any = None
    # Background job manager for run_job / job_* tools and /jobs (issue #1).
    job_manager: Any = None

    def initialize(self, *, initial_title: str = "Interactive session") -> AgentSession:
        if self.legacy_agent_factory is None:
            from biobank_agent.agent import Agent
        else:
            Agent = self.legacy_agent_factory
        self.settings.ensure_dirs()
        runtime_config = RuntimeConfig.from_settings(self.settings)
        permission_mode = getattr(self.settings, "approval_profile", "") or runtime_config.approval_profile
        runtime_config.approval_profile = str(permission_mode or "yolo")
        factory = self.legacy_agent_factory or Agent
        self.legacy_agent = factory(self.settings)
        tool_registry = ToolRegistry()
        tool_registry.hydrate_from_legacy()
        self.tool_registry = tool_registry
        providers = {}
        for model_name in {
            runtime_config.primary_model,
            runtime_config.planner_model,
            runtime_config.critic_model,
            runtime_config.summarizer_model,
            runtime_config.safety_reviewer_model,
        }:
            providers[model_name] = LazyLLMProvider(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                model=model_name,
                provider_name=model_name,
                tool_call_content_mode=getattr(self.settings, "tool_call_content_mode", "null"),
                request_timeout_s=float(getattr(self.settings, "llm_request_timeout_s", 60.0) or 60.0),
                max_retries=int(getattr(self.settings, "llm_max_retries", 3) or 0),
                retry_base_delay_s=float(getattr(self.settings, "llm_retry_base_delay_s", 2.0) or 0.0),
            )
        session_root = Path(self.settings.memory_dir) / "runtime_sessions"
        provider_router = ProviderRouter(providers, runtime_config)
        self.runtime = AgentRuntime(
            provider_router=provider_router,
            tool_registry=tool_registry,
            session_store=SessionStore(session_root),
            action_graph=ActionGraph(Path(self.settings.memory_dir) / "action_graph.db"),
            config=runtime_config,
            tool_context_factory=self._build_tool_context,
            planner=RuntimePlanner(
                provider_router,
                runtime_config,
                num_candidates=int(getattr(self.settings, "council_plan_candidates", 3)),
                timeout_s=float(getattr(self.settings, "council_timeout_s", 360.0)),
                clock=time.time,
            ),
        )
        initial_cwd = str(Path.cwd())
        if str(self.workspace or "").strip():
            candidate = Path(str(self.workspace)).expanduser()
            if candidate.is_dir():
                initial_cwd = str(candidate.resolve())
            else:
                self.console.print(
                    f"[yellow]--workspace {self.workspace!r} is not a directory; "
                    f"using the launch directory instead.[/]"
                )
        self.session = self.runtime.create_session(title=initial_title, cwd=initial_cwd)
        # Background job manager rooted under the active workspace (issue #1),
        # with completion notifications (durable JSONL + optional command) so an
        # unattended long run pings the operator when it finishes (issue #3).
        jobs_dir = str(getattr(self.settings, "jobs_dir_name", ".biobank_jobs") or ".biobank_jobs")
        from biobank_agent.runtime.notify import notify as _notify

        def _job_notifier(event: str, *, title: str = "", body: str = "") -> None:
            _notify(event, title=title, body=body, settings=self.settings)

        self.job_manager = JobManager(Path(self.session.cwd) / jobs_dir, notifier=_job_notifier)
        try:
            self.job_manager.reattach()
        except Exception:
            pass
        self.mcp_manager = McpManager(
            tool_registry,
            config_path=Path(str(getattr(self.settings, "mcp_config_path", "")).strip()).expanduser()
            if str(getattr(self.settings, "mcp_config_path", "") or "").strip()
            else None,
        )
        self.runtime.config.active_role = runtime_config.active_role
        if self.legacy_agent is not None:
            try:
                self.legacy_agent.state.custom_data.setdefault("mcp_manager", self.mcp_manager)
                self.legacy_agent.state.custom_data.setdefault("mcp_tool_registry", tool_registry)
            except Exception:
                pass
        self.runtime.save_session(self.session)
        if sys.stdin.isatty():
            self.prompt_session = self._build_prompt_session()
        return self.session

    def startup_banner(self) -> Panel:
        session = self._require_session()
        runtime = self._require_runtime()
        roles = [
            ("primary", runtime.config.primary_model),
            ("planner", runtime.config.planner_model),
            ("critic", runtime.config.critic_model),
            ("summarizer", runtime.config.summarizer_model),
            ("safety", runtime.config.safety_reviewer_model),
        ]
        mcp_loaded = len(self.mcp_manager.handlers) if self.mcp_manager is not None else 0
        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("version", __version__)
        table.add_row("workspace", session.cwd)
        table.add_row("session", session.session_id)
        table.add_row("active role", runtime.config.active_role)
        table.add_row("permissions", runtime.config.approval_profile)
        table.add_row("provider roles", ", ".join(f"{name}={model}" for name, model in roles))
        tool_summary = self._tool_summary_snapshot()
        table.add_row("tools", tool_summary["summary"])
        table.add_row("wgs", tool_summary["wgs_summary"])
        table.add_row("mcp", f"{mcp_loaded} loaded tool(s)")
        table.add_row(
            "commands",
            "/help, /status, /graph, /plan, /plan-diagnose, /plan-retry, /plan-use, /goal, /resume, /cd, /jobs, /job-tail, /artifacts, /compact, /diff, /permissions, /doctor, /tools, /skills, /agent, /subagents, /review, /audit, /harness, /learn, /replay, /evolve, /quit",
        )
        return Panel(table, title="BioBank Agent", border_style="cyan", box=box.ROUNDED)

    def run(self, *, initial_task: str = "") -> None:
        self.initialize(initial_title=_title_from_task(initial_task) if initial_task else "Interactive session")
        self.console.print(self.startup_banner())
        if initial_task:
            self.console.print("[dim]Compatibility shortcut: running positional task as the first interactive turn.[/dim]")
            self.handle_line(initial_task)
            if not sys.stdin.isatty():
                return
        while not self.exit_requested:
            try:
                line = self._read_line()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]Goodbye.[/]")
                break
            # A KeyboardInterrupt raised *inside* command handling (e.g. a genuine
            # Ctrl-C during a /plan clarification) must cancel the command and drop
            # back to the prompt — never crash the REPL with a raw traceback. The
            # terminal is already clean by here: _read_single_key restores termios
            # in its finally, and _draft_plan_with_live_progress stops the dashboard
            # (restoring sys.stdout) in its finally before re-raising.
            try:
                self.handle_line(line)
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Cancelled.[/] (command interrupted)")
                continue

    def handle_line(self, line: str) -> list[AgentEvent]:
        text = str(line or "").strip()
        if not text:
            return []
        if text.lower() in {"quit", "exit", "q"}:
            text = "/quit"
        if text.startswith("/"):
            return self.handle_slash(text)
        return self.handle_message(text)

    def handle_message(self, text: str) -> list[AgentEvent]:
        if self._plan_message_should_intercept(text):
            return self._handle_plan_natural_language(text)
        runtime = self._require_runtime()
        session = self._require_session()
        before = len(session.events)
        before_graph_refs = len(session.action_graph_refs)
        try:
            runtime.run_turn(session, text)
        except Exception as exc:
            self._record_event(
                AgentEvent.make(
                    AgentEventType.ERROR,
                    session_id=session.session_id,
                    message=str(exc),
                )
            )
            runtime.save_session(session)
            self.console.print(f"[red]Error:[/] {_rich_escape(str(exc))}")
            return [AgentEvent.from_dict(e) for e in session.events[before:]]
        self.session = session
        new_events = [AgentEvent.from_dict(e) for e in session.events[before:]]
        plan_changed = self._sync_plan_execution_from_events(new_events)
        compacted = runtime.maybe_auto_compact(
            session,
            event_threshold=int(getattr(self.settings, "auto_compact_event_threshold", 120)),
            turn_threshold=int(getattr(self.settings, "auto_compact_turn_threshold", 20)),
        )
        if compacted:
            runtime.save_session(session)
        self._render_event_progress(new_events, title="Execution Progress")
        if plan_changed and session.state.plan is not None:
            self._render_plan_progress(session.state.plan, title="Plan Execution Progress")
            runtime.save_session(session)
        if len(session.action_graph_refs) > before_graph_refs:
            self._render_action_graph(session=session, limit=8)
        self._render_latest_assistant()
        self._update_token_usage()
        self.console.print(
            f"[dim]session: {session.session_id[:12]} events: {len(session.events)}[/]"
        )
        return new_events

    def handle_slash(self, text: str) -> list[AgentEvent]:
        runtime = self._require_runtime()
        session = self._require_session()
        parsed = self.slash_registry.parse(text)
        if parsed is None:
            return []
        before = len(session.events)
        self._record_event(
            AgentEvent.make(
                AgentEventType.COMMAND_STARTED,
                session_id=session.session_id,
                command=parsed.name,
                arg=parsed.arg,
            )
        )
        try:
            ctx = CommandContext(
                agent=None,
                planner=None,
                token_usage=self.token_usage,
                console=self.console,
                actions=self._command_actions(),
            )
            result = self.slash_registry.dispatch(text, ctx)
            self._record_event(
                AgentEvent.make(
                    AgentEventType.COMMAND_FINISHED,
                    session_id=session.session_id,
                    command=parsed.name,
                    arg=parsed.arg,
                    result=self._normalize_command_result(result),
                )
            )
        except KeyError:
            self.console.print(f"[yellow]Unknown command: {_rich_escape(str(parsed.name))}. Type /help.[/]")
            self._record_event(
                AgentEvent.make(
                    AgentEventType.COMMAND_FAILED,
                    session_id=session.session_id,
                    command=parsed.name,
                    arg=parsed.arg,
                    error="unknown command",
                )
            )
        except Exception as exc:
            self.console.print(f"[red]Command failed:[/] {_rich_escape(str(exc))}")
            self._record_event(
                AgentEvent.make(
                    AgentEventType.COMMAND_FAILED,
                    session_id=session.session_id,
                    command=parsed.name,
                    arg=parsed.arg,
                    error=str(exc),
                )
            )
        runtime.save_session(session)
        return [AgentEvent.from_dict(e) for e in session.events[before:]]

    def _command_actions(self) -> dict[str, Any]:
        return {
            "help": self._cmd_help,
            "status": self._cmd_status,
            "graph": self._cmd_graph,
            "compact": lambda: self._cmd_compact(),
            "goal": self._cmd_goal,
            "resume": self._cmd_resume,
            "new": self._cmd_new,
            "fork": self._cmd_fork,
            "cd": self._cmd_cd,
            "jobs": self._cmd_jobs,
            "job_tail": self._cmd_job_tail,
            "diff": self._cmd_diff,
            "permissions": self._cmd_permissions,
            "doctor": self._cmd_doctor,
            "agent": self._cmd_agent,
            "subagents": self._cmd_subagents,
            "review": self._cmd_review,
            "audit": self._cmd_audit,
            "trace": self._cmd_trace,
            "harness": self._cmd_harness,
            "learn": self._cmd_learn,
            "verify": self._cmd_verify,
            "quit": self._cmd_quit,
            "mcp_list": lambda: self._cmd_mcp("list"),
            "mcp_start": lambda: self._cmd_mcp("start"),
            "mcp_stop": lambda: self._cmd_mcp("stop"),
            "mcp_health": lambda arg="": self._cmd_mcp("health", arg),
            "mcp_call": lambda arg="": self._cmd_mcp("call", arg),
            "plan": self._cmd_plan,
            "research": self._cmd_research,
            "plan_approve": self._cmd_plan_approve,
            "plan_edit": self._cmd_plan_edit,
            "plan_reject": self._cmd_plan_reject,
            "plan_title": lambda arg="": self._cmd_plan_title(arg),
            "plan_pause": self._cmd_plan_pause,
            "plan_resume": self._cmd_plan_resume,
            "plan_diagnose": self._cmd_plan_diagnose,
            "plan_retry": self._cmd_plan_retry,
            "plan_use": self._cmd_plan_use,
            "plan_option": self._cmd_plan_option,
            "plan_skip": self._cmd_plan_skip,
            "plan_exit": self._cmd_plan_exit,
            "plans": self._cmd_plans,
            "artifacts": self._cmd_artifacts,
            "routing_status": self._cmd_routing_status,
            "cost": self._cmd_cost,
            "clear": self._cmd_clear,
            "export": self._cmd_export,
            "skills": self._cmd_skills,
            "tools": self._cmd_tools,
            "history": self._cmd_history,
            "figures": self._cmd_figures,
            "cohorts": self._cmd_cohorts,
            "models": self._cmd_models,
            "show_evidence": lambda claim_id: self._cmd_audit(claim_id),
            "replicate": self._cmd_replicate,
            "model": self._cmd_model,
            "models_pool": self._cmd_models_pool,
            "models_available": self._cmd_models_available,
            "strategy": self._cmd_strategy,
            "replay": lambda arg="": self._cmd_replay(arg),
            "evolve": lambda arg="": self._cmd_evolve(arg),
            "record": self._cmd_record,
            "pipelines": self._cmd_pipelines,
            "errors": self._cmd_errors,
            "memory": self._cmd_memory,
            "debate": self._cmd_debate,
        }

    def _reports_root(self, session: AgentSession | None = None) -> Path:
        """Root for user-visible outputs for a session.

        When reports_follow_workspace is enabled, outputs follow the active
        workspace so `/cd` and `--workspace` keep scripts, logs, and reports
        together. Otherwise use the configured global reports_dir.
        """
        sess = session or self._require_session()
        if getattr(self.settings, "reports_follow_workspace", True):
            return Path(sess.cwd).expanduser() / "reports"
        return Path(self.settings.reports_dir).expanduser()

    def _session_report_dir(self, session: AgentSession | None = None) -> Path:
        sess = session or self._require_session()
        return self._reports_root(sess) / sess.session_id

    def _reroot_job_manager(self, session: AgentSession | None = None) -> None:
        """Make background-job commands follow the active/resumed workspace."""
        if self.job_manager is None:
            return
        sess = session or self._require_session()
        jobs_dir = str(getattr(self.settings, "jobs_dir_name", ".biobank_jobs") or ".biobank_jobs")
        self.job_manager.jobs_root = Path(sess.cwd).expanduser() / jobs_dir
        try:
            self.job_manager.reattach()
        except Exception:
            pass

    def _cmd_help(self) -> None:
        table = Table(title="Slash commands", box=box.SIMPLE_HEAVY)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Usage")
        table.add_column("Description")
        for name in sorted(self.slash_registry.commands):
            command = self.slash_registry.commands[name]
            table.add_row(command.name, command.usage, command.description)
        self.console.print(table)
        return {"commands": sorted(self.slash_registry.commands)}

    def _cmd_status(self, *_args: Any) -> None:
        session = self._require_session()
        runtime = self._require_runtime()
        mcp_loaded = len(self.mcp_manager.handlers) if self.mcp_manager is not None else 0
        table = Table(title="Runtime Status", box=box.SIMPLE)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value")
        table.add_row("session", session.session_id)
        table.add_row("title", session.title)
        table.add_row("workspace", session.cwd)
        table.add_row("status", session.state.status.value)
        table.add_row("active role", runtime.config.active_role)
        table.add_row("turns", str(len(session.turns)))
        table.add_row("events", str(len(session.events)))
        table.add_row("trajectory", str(runtime.session_store.rollout_file(session.session_id)))
        table.add_row("permissions", runtime.config.approval_profile)
        table.add_row("plan", session.state.plan.status.value if session.state.plan else "none")
        table.add_row("goal", session.state.goal.completion_state if session.state.goal else "none")
        table.add_row("checkpoints", str(len(session.state.checkpoints)))
        table.add_row("mcp tools", str(mcp_loaded))
        self.console.print(table)
        return {
            "session_id": session.session_id,
            "turns": len(session.turns),
            "events": len(session.events),
            "plan_status": session.state.plan.status.value if session.state.plan else "none",
            "goal_status": session.state.goal.completion_state if session.state.goal else "none",
        }

    def _apply_planning_provider_caps(self, runtime, build_timeout_s: float) -> list:
        """Temporarily cap each provider's request timeout and disable retries during planning, so a
        single slow/stalled provider can't outlast the plan-build deadline. Returns a restore list."""
        restore: list[tuple[Any, tuple[float, int] | float]] = []
        if build_timeout_s <= 0:
            return restore
        request_cap = max(0.05, build_timeout_s)
        for provider in getattr(runtime.provider_router, "providers", {}).values():
            policy_setter = getattr(provider, "set_llm_policy", None)
            if callable(policy_setter):
                try:
                    current = float(getattr(provider, "request_timeout_s", request_cap) or request_cap)
                    restore.append((provider, policy_setter(
                        request_timeout_s=request_cap if request_cap < current else current,
                        max_retries=0)))
                except Exception:
                    continue
                continue
            timeout_setter = getattr(provider, "set_request_timeout", None)
            if callable(timeout_setter):
                try:
                    current = float(getattr(provider, "request_timeout_s", request_cap) or request_cap)
                    if request_cap < current:
                        restore.append((provider, timeout_setter(request_cap)))
                except Exception:
                    continue
        return restore

    def _restore_planning_provider_caps(self, restore: list) -> None:
        """Undo the temporary planning caps applied by _apply_planning_provider_caps."""
        for provider, previous_policy in restore:
            policy_setter = getattr(provider, "set_llm_policy", None)
            if callable(policy_setter) and isinstance(previous_policy, tuple):
                try:
                    policy_setter(request_timeout_s=previous_policy[0], max_retries=previous_policy[1])
                except Exception:
                    pass
                continue
            timeout_setter = getattr(provider, "set_request_timeout", None)
            if callable(timeout_setter):
                try:
                    timeout_setter(float(previous_policy))
                except Exception:
                    pass

    def _cmd_plan(self, arg: str = "") -> dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session()
        if not arg:
            if session.state.plan:
                # _render_plan already includes a Plan Steps table; the separate
                # progress panel here was a redundant third steps view — dropped.
                self._render_plan(session.state.plan)
                self._render_plan_flow(session.state.plan)
                if session.state.plan.status == PlanStatus.DRAFT:
                    self._render_plan_review_menu_notice()
                return session.state.plan.to_dict()
            else:
                self.console.print("[dim]Use /plan <task> to draft a structured plan.[/]")
                return {"status": "none"}
        previous_mode = runtime.config.approval_profile
        # Remember the user's real profile (captured before drafting flips the
        # runtime to the read-only "plan" profile) so approval can restore it for
        # autonomous execution. session.config is runtime.config (a shared
        # object), so it gets mutated to "plan" too — it must not be used as the
        # restore source; this captured string is the safe source.
        self._pre_plan_profile = previous_mode
        build_timeout_s = max(0.0, float(getattr(self.settings, "plan_build_timeout_s", 240.0)))
        provider_timeout_restore = self._apply_planning_provider_caps(runtime, build_timeout_s)
        plan = None  # defined for the finally so an aborted draft can detect "no plan produced"
        try:
            with _plan_build_wall_clock_timeout(build_timeout_s):
                plan = self._draft_plan_with_live_progress(arg)
        except PlanBuildTimeoutError:
            self._set_activity("idle")
            timeout_label = f"{build_timeout_s:g}"
            reason = (
                f"Plan generation stalled for {timeout_label}s with no model activity "
                f"(plan_build_timeout_s={timeout_label}s) before producing a draft. "
                "The planner is likely blocked on an unreachable provider or an over-broad task; retry or narrow the request."
            )
            session.state.custom_data["plan_last_diagnosis"] = {
                "status": "failed",
                "step_id": "planning",
                "step_title": "Plan generation",
                "step_status": "timeout",
                "reason": reason,
                "objective": arg,
                "blocked": True,
                "repair_options": [
                    "Retry /plan with a narrower objective or explicit input file paths.",
                    f"Increase PLAN_BUILD_TIMEOUT_S or plan_build_timeout_s if the planner needs more than {timeout_label}s.",
                    "Use /doctor to verify provider/API connectivity before retrying.",
                ],
            }
            self._emit_plan_phase("Planning", status="error", message=reason, metadata={"objective": arg, "timeout_s": build_timeout_s})
            runtime.save_session(session)
            self.console.print(f"[red]Planning timed out:[/] {_rich_escape(reason)}")
            return {"status": "failed", "error": reason, "step_status": "timeout"}
        except CouncilError as exc:
            self._set_activity("idle")
            session.state.custom_data["plan_last_diagnosis"] = {
                "status": "failed",
                "step_id": "planning",
                "step_title": "Plan generation",
                "step_status": "failed",
                "reason": str(exc),
                "objective": arg,
                "blocked": True,
                "repair_options": [
                    "Use /plan-edit <feedback> to simplify or constrain the task.",
                    "Retry /plan after checking planner credentials with /doctor.",
                    "For tabular files, mention the file type and desired output table explicitly.",
                ],
            }
            self._emit_plan_phase("Planning", status="error", message=str(exc), metadata={"objective": arg})
            runtime.save_session(session)
            self.console.print(f"[red]Planning failed:[/] {_rich_escape(str(exc))}")
            return {"status": "failed", "error": str(exc)}
        finally:
            self._restore_planning_provider_caps(provider_timeout_restore)
            if plan is None:
                # Drafting aborted (timeout / council error) before any plan existed: drafting had
                # flipped the runtime to the read-only "plan" profile. Restore the EXACT profile the
                # user had before /plan — never widen it. (Unlike approval, nothing executes here, so a
                # user already in read-only "plan" mode must NOT be escalated to yolo by a failed draft.)
                try:
                    runtime.set_approval_profile(previous_mode)
                except Exception:
                    pass
        self.console.print(f"[dim]Plan mode is read-only. Previous permission mode was {previous_mode!r}.[/]")
        review = self._run_plan_review_loop(plan, previous_mode=previous_mode)
        final_plan = session.state.plan or plan
        result = {
            "status": final_plan.status.value,
            "plan": final_plan.to_dict(),
            "previous_permission_mode": previous_mode,
            "permission_mode": runtime.config.approval_profile,
        }
        if review:
            result["review"] = review
        return result

    def _cmd_plan_approve(self, *_args: Any) -> dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session()
        plan = runtime.approve_plan(session)
        if plan is None:
            self.console.print("[yellow]No draft plan to approve.[/]")
            return {"status": "no_plan"}
        # RESTORE the user's real permission profile (e.g. yolo) so the autonomous
        # execution runs with their permissions. Plan mode temporarily set the
        # read-only "plan" profile; the old code instead switched to
        # "ask_before_edits" (which blocked nothing — the scheduler has no
        # confirm_fn — and made the flow look like it was waiting for input).
        # We must NOT restore session.config.approval_profile: session.config is
        # the SAME object as runtime.config, so it was mutated to "plan" too.
        # Use the profile captured before plan mode (falling back to the durable
        # settings profile), and never execute under the read-only "plan" profile.
        restore_profile = (
            getattr(self, "_pre_plan_profile", "")
            or getattr(self.settings, "approval_profile", "")
            or "yolo"
        )
        if restore_profile == "plan":
            restore_profile = getattr(self.settings, "approval_profile", "") or "yolo"
        runtime.set_approval_profile(restore_profile)
        self._mark_plan_execution_ready(plan)
        self._record_plan_review_graph(plan, action="approve", choice="implement", detail="Approved; executing autonomously")
        self._emit_plan_phase(
            "Execution",
            actor="plan",
            status="running",
            message=f"executing {len(plan.steps)} step(s)",
            metadata={
                "completed_steps": sum(1 for step in plan.steps if step.status in {"done", "completed"}),
                "total_steps": len(plan.steps),
                "current_step": next((step.id for step in plan.steps if step.status not in self._TERMINAL_STEP_STATUS), ""),
            },
        )
        runtime.save_session(session)
        self._set_activity("executing plan")
        self.console.print("[green]Plan approved.[/] Executing autonomously — Ctrl-C to cancel.")
        summary = self._execute_plan_autonomously(plan)
        return {"plan_status": plan.status.value, **(summary or {})}

    # --------------------------------------------------------- plan execution
    def _execute_plan_autonomously(self, plan) -> dict[str, Any]:
        """Run every plan step autonomously via ``run_turn`` (the agent executes
        each step from its natural-language purpose + tool scope), updating a
        single live progress view. No per-step user input. Ctrl-C cancels and
        returns to the prompt; the live view is always torn down."""
        runtime = self._require_runtime()
        session = self._require_session()
        # Double-execution guard (the menu-approve branch and /plan-approve both
        # funnel here; only run the loop once).
        if plan.status in {PlanStatus.COMPLETED, PlanStatus.EXECUTING}:
            return {"status": getattr(plan.status, "value", str(plan.status))}

        total = len(plan.steps)
        ordered = self._iter_steps_in_dependency_order(plan)
        plan.status = PlanStatus.EXECUTING
        # Lazy tool exposure: pre-activate the tools the planner named so the
        # executor sees their schemas immediately — no extra skill_search round for
        # planned steps. Token savings hold (only the plan's tools are added, not all).
        planned_tools: list[str] = list(getattr(plan, "proposed_tool_scope", []) or [])
        for step in plan.steps:
            planned_tools.extend(getattr(step, "tool_scope", []) or [])
        if planned_tools:
            active = session.state.custom_data.setdefault("active_skills", [])
            for tool_name in planned_tools:
                if tool_name and tool_name not in active:
                    active.append(tool_name)
        runtime.save_session(session)

        view = self._make_execution_view()
        self._active_exec_view = view  # so streamed subprocess lines surface live (#2)
        done = failed = unverified = 0
        halted: "tuple[int, Any] | None" = None
        current = None
        # Capture each completed step's real output (tool results + final text)
        # so _finalize_execution can surface it — otherwise the data the agent
        # produced is invisible (issue #7). Keyed by step id.
        step_outputs: dict[str, dict[str, Any]] = {}
        try:
            view.start()  # inside try so finally always tears the Live down,
            #              even if Ctrl-C lands during dashboard startup
            for step in ordered:
                if step.status in self._TERMINAL_STEP_STATUS:
                    continue  # e.g. context/design pre-marked done
                current = step
                position = plan.steps.index(step) + 1
                preflight = self._preflight_plan_step(plan, step)
                if preflight.get("blocked"):
                    reason = str(preflight.get("reason") or "preflight blocked the step")
                    step.status = "pending"
                    plan.status = PlanStatus.PAUSED
                    session.state.custom_data["plan_paused"] = True
                    session.state.custom_data["plan_pause_reason"] = reason
                    self._store_plan_diagnosis(plan, step, reason=reason, preflight=preflight)
                    view.record("Preflight", actor=step.id, status="warning",
                                message=f"step {position}/{total} needs input: {reason}",
                                metadata={"subagent": step.id})
                    halted = (position, step, "paused", reason)
                    break
                step.status = "running"
                view.record("Execution", actor=step.id, status="running",
                            message=f"step {position}/{total}: {step.title}",
                            metadata={"subagent": step.id, "current_step": step.id,
                                      "total_steps": total, "completed_steps": done})
                runtime.save_session(session)

                # Stage 6: run the step with bounded autonomous recovery (retry →
                # augmented-retry). The agent may call ask_user mid-attempt; if it
                # does, we stop and hand the question to the user (Stage 5).
                status_, reason, tool_calls, question = self._run_step_with_recovery(
                    plan, step, position, total, view
                )
                if status_ == "ok":
                    captured = self._capture_step_output()
                    step_outputs[step.id] = captured
                    # Evidence Contract: a step that declared a verification but
                    # produced no observable artifact is "unverified", not "done" — so a
                    # confidently-narrated step with nothing behind it can't pass silently.
                    verified, ev_reason = self._assess_step_evidence(step, captured)
                    if verified:
                        step.status = "done"
                        done += 1
                        view.record("Execution", actor=step.id, status="success",
                                    message=f"step {position}/{total} done: {step.title} ({tool_calls} tool call(s))",
                                    metadata={"subagent": step.id, "current_step": step.id,
                                              "total_steps": total, "completed_steps": done})
                    else:
                        step.status = "unverified"
                        unverified += 1
                        self._store_plan_diagnosis(plan, step, reason=f"unverified: {ev_reason}")
                        view.record("Execution", actor=step.id, status="warning",
                                    message=f"step {position}/{total} UNVERIFIED: {ev_reason}",
                                    metadata={"subagent": step.id, "current_step": step.id,
                                              "total_steps": total, "completed_steps": done})
                elif status_ == "needs_input":
                    # Agent asked the user a question — pause and hand control back.
                    reason = reason or f"step {position}/{total} needs input"
                    step.status = "pending"
                    plan.status = PlanStatus.PAUSED
                    session.state.custom_data["plan_paused"] = True
                    session.state.custom_data["plan_pause_reason"] = reason
                    session.state.custom_data["plan_pending_question"] = question
                    self._store_plan_diagnosis(
                        plan, step, reason=reason,
                        preflight={
                            "blocked": True,
                            "reason": reason,
                            "pending_question": question,
                            "repair_options": [
                                "Type your answer directly and I'll resume the step with it.",
                                "Or use /plan-use key=value to supply a path/option, then /plan-retry.",
                                "Run /doctor for dependency and data-readiness details.",
                            ],
                        },
                    )
                    view.record("Repair", actor=step.id, status="warning",
                                message=f"step {position}/{total} asked: {question[:80]}",
                                metadata={"subagent": step.id})
                    halted = (position, step, "paused", reason)
                    break
                else:  # "failed" — autonomous recovery exhausted; pause and hand back
                    step.status = "failed"
                    failed += 1
                    reason = reason or f"step {position}/{total} did not converge"
                    self._store_plan_diagnosis(plan, step, reason=reason)
                    view.record("Repair", actor=step.id, status="failed",
                                message=f"step {position}/{total} failed after retries: {step.title}",
                                metadata={"subagent": step.id})
                    halted = (position, step, "failed", reason)
                    break  # hand control back to the user with a diagnosis + menu
                session.state.pending_work = [
                    s.to_dict() for s in plan.steps if s.status not in {"done", "completed", "skipped"}
                ]
                runtime.save_session(session)
        except KeyboardInterrupt:
            if current is not None and current.status == "running":
                current.status = "cancelled"
            plan.status = PlanStatus.PAUSED
            runtime.save_session(session)
            view.stop()
            pos = (plan.steps.index(current) + 1) if current is not None else 0
            self.console.print(f"\n[yellow]Execution cancelled at step {pos}/{total}.[/] {done} done, {failed} failed.")
            return {"status": "cancelled", "done": done, "failed": failed}
        finally:
            self._active_exec_view = None
            view.stop()  # idempotent — the Live must always be torn down

        if halted and len(halted) > 2 and halted[2] == "paused":
            plan.status = PlanStatus.PAUSED
        else:
            plan.status = PlanStatus.FAILED if halted else PlanStatus.COMPLETED
        runtime.save_session(session)
        return self._finalize_execution(plan, done, failed, halted, total, step_outputs, unverified)

    def _finalize_execution(self, plan, done: int, failed: int, halted, total: int,
                            step_outputs: dict[str, dict] | None = None,
                            unverified: int = 0) -> dict[str, Any]:
        step_outputs = step_outputs or {}
        unv = f", [yellow]{unverified} unverified[/]" if unverified else ""
        if halted:
            paused = len(halted) > 2 and halted[2] == "paused"
            style = "yellow" if paused else "red"
            reason = str(halted[3]) if len(halted) > 3 else ""
            suffix = f" Reason: {_rich_escape(reason)}" if reason else ""
            self.console.print(f"[{style}]Execution halted at step {halted[0]}/{total}.[/] {done} done, {failed} failed{unv}.{suffix}")
            self._render_plan_diagnosis(plan)
            # Notify (durable JSONL + optional command + bell) so an unattended run
            # surfaces that it is waiting for input / has failed (issues #1, #3).
            try:
                from biobank_agent.runtime.notify import notify
                notify("awaiting_user" if paused else "plan_failed",
                       title=f"plan {'paused' if paused else 'failed'} at step {halted[0]}/{total}",
                       body=reason or "", settings=self.settings)
            except Exception:
                pass
        else:
            self.console.print(f"[green]Execution complete.[/] {done} done, {failed} failed{unv}.")
            if unverified:
                self.console.print(
                    "[dim]Unverified steps ran but produced no artifact matching their declared "
                    "verification — treat their results as unconfirmed.[/]"
                )
        rows = [
            {"label": f"{i}/{total} {step.title}", "status": step.status, "detail": (step.purpose or "")[:160]}
            for i, step in enumerate(plan.steps, 1)
        ]
        self._render_progress_rows("Plan Execution Summary", rows)
        artifacts = self._collect_artifact_index(plan, step_outputs)
        self._render_artifact_index(artifacts)
        try:
            self._require_session().state.custom_data["last_artifact_index"] = artifacts
            self._require_runtime().save_session(self._require_session())
        except Exception:
            pass
        # Surface what each step produced + a final answer.
        self._render_step_outputs(plan, step_outputs)
        return {
            "status": getattr(plan.status, "value", str(plan.status)),
            "done": done,
            "failed": failed,
            "unverified": unverified,
            "artifacts": artifacts,
        }

    def _capture_step_output(self) -> dict[str, Any]:
        """Snapshot the most recent turn's tool outputs + final text for a step.

        The step ran via ``run_turn``; its real output lives in the turn's
        ``tool_results`` (each a name + result dict) and the agent's closing
        ``assistant_messages`` text. Captured immediately after the step
        completes, so ``session.turns[-1]`` is still that step's turn."""
        session = self._require_session()
        turns = getattr(session, "turns", None) or []
        if not turns:
            return {}
        turn = turns[-1]
        tool_results: list[tuple[str, dict]] = []
        for result in (getattr(turn, "tool_results", None) or []):
            payload = getattr(result, "result", None)
            if isinstance(payload, dict) and payload:
                tool_results.append((str(getattr(result, "name", "")), payload))
        messages = getattr(turn, "assistant_messages", None) or []
        text = messages[-1].text.strip() if messages and getattr(messages[-1], "text", "") else ""
        return {"tool_results": tool_results, "text": text}

    @staticmethod
    def _turn_terminal_tool_failure(session) -> str:
        turns = getattr(session, "turns", None) or []
        if not turns:
            return ""
        results = getattr(turns[-1], "tool_results", None) or []
        if not results:
            return ""
        result = results[-1]
        raw_error = str(getattr(result, "error", "") or "").strip()
        payload = getattr(result, "result", None) or {}
        if isinstance(payload, dict):
            if payload.get("awaiting_user"):
                return ""
            status = str(payload.get("status") or "").strip().lower()
            if status in {"error", "failed", "cancelled", "requires_confirmation"}:
                return str(payload.get("error") or payload.get("stderr") or raw_error or status)
            if payload.get("ok") is False:
                return str(payload.get("error") or payload.get("reason") or raw_error or "tool returned ok=false")
            if payload.get("error") and status not in {"ok", "success", "done", "completed"}:
                return str(payload.get("error"))
        return raw_error

    def _render_step_outputs(self, plan, step_outputs: dict[str, dict]) -> None:
        """Render each completed step's real output and a final-answer panel.

        Runs after the live dashboard is torn down, printing to ``self.console``
        so both the interactive console and the headless line-logger capture it.
        Output is bounded (``render_result_payload`` caps rows/cols/chars; at most
        a few tool results per step) so long runs do not flood the terminal."""
        if not step_outputs:
            return
        for index, step in enumerate(plan.steps, 1):
            captured = step_outputs.get(step.id)
            if not captured:
                continue
            body: list[Any] = []
            for name, result in (captured.get("tool_results") or [])[:6]:
                payload = render_result_payload(result)
                if payload is not None:
                    body.append(Text(f"[{name}]", style="bold dim"))
                    body.append(payload)
            if not body:
                continue
            self.console.print(Panel(
                Group(*body),
                title=f"[bold]Step {index} output[/] — {_rich_escape(str(step.title))}",
                border_style="blue",
                padding=(0, 1),
            ))
        final = self._final_answer_text(plan, step_outputs)
        if final:
            self.console.print(Panel(
                Markdown(final),
                title="[bold green]Final Answer[/]",
                border_style="green",
                padding=(1, 2),
            ))

    _ARTIFACT_PATH_KEYS = {
        "path", "output", "output_path", "out_path", "report_path", "report_dir", "figure",
        "artifact", "file", "saved_to", "log_path", "all_table", "top_table",
        "annotation_table", "effect_table", "result_dir",
    }
    _ARTIFACT_LIST_KEYS = {"figures", "artifacts", "files", "outputs", "output_paths", "report_paths", "logs"}

    @classmethod
    def _collect_artifact_paths(cls, payload: Any, *, prefix: str = "") -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                label = f"{prefix}.{key}" if prefix else str(key)
                key_l = str(key).lower()
                if key_l in cls._ARTIFACT_PATH_KEYS and isinstance(value, (str, Path)) and str(value).strip():
                    found.append((label, str(value)))
                    continue
                if key_l in cls._ARTIFACT_LIST_KEYS and isinstance(value, (list, tuple, set)):
                    for item in value:
                        if isinstance(item, (str, Path)) and str(item).strip():
                            found.append((label, str(item)))
                        else:
                            found.extend(cls._collect_artifact_paths(item, prefix=label))
                    continue
                if isinstance(value, (dict, list, tuple)):
                    found.extend(cls._collect_artifact_paths(value, prefix=label))
        elif isinstance(payload, (list, tuple)):
            for item in payload:
                found.extend(cls._collect_artifact_paths(item, prefix=prefix))
        return found

    def _collect_artifact_index(self, plan, step_outputs: dict[str, dict]) -> list[dict[str, str]]:
        index: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for step in getattr(plan, "steps", []) or []:
            captured = step_outputs.get(step.id) or {}
            for tool_name, result in captured.get("tool_results") or []:
                for label, path in self._collect_artifact_paths(result):
                    key = (str(step.id), str(tool_name), path)
                    if key in seen:
                        continue
                    seen.add(key)
                    index.append({
                        "step_id": str(step.id),
                        "step_title": str(getattr(step, "title", "")),
                        "tool": str(tool_name),
                        "kind": label,
                        "path": path,
                    })
        return index

    def _render_artifact_index(self, artifacts: list[dict[str, str]]) -> None:
        if not artifacts:
            return
        table = Table(title="Output Artifacts", box=box.SIMPLE)
        table.add_column("Step", style="cyan", no_wrap=True)
        table.add_column("Tool", style="magenta", no_wrap=True)
        table.add_column("Kind", no_wrap=True)
        table.add_column("Path", overflow="fold")
        for item in artifacts[:30]:
            table.add_row(
                str(item.get("step_id") or "-"),
                str(item.get("tool") or "-"),
                str(item.get("kind") or "-")[-40:],
                str(item.get("path") or "-"),
            )
        if len(artifacts) > 30:
            table.caption = f"{len(artifacts) - 30} additional artifact(s) omitted from display; full index is in last_artifact_index."
        self.console.print(table)
        full_paths = [
            f"{item.get('step_id') or '-'} {item.get('tool') or '-'} {item.get('kind') or '-'}: {item.get('path') or '-'}"
            for item in artifacts[:30]
        ]
        if full_paths:
            self.console.print(Panel("\n".join(_rich_escape(p) for p in full_paths), title="Full Artifact Paths", border_style="blue"))

    @staticmethod
    def _final_answer_text(plan, step_outputs: dict[str, dict]) -> str:
        """The closing text of the last executed step that produced one — the
        agent's brief 'what I did' answer that the plan summary alone never showed."""
        for step in reversed(list(plan.steps)):
            captured = step_outputs.get(step.id)
            if captured and captured.get("text"):
                return str(captured["text"])
        return ""

    _EVIDENCE_ARTIFACT_KEYS = (
        "path", "output_path", "out_path", "report_path", "report_dir", "figure", "figures",
        "artifact", "artifacts", "file", "files", "saved_to", "log_path",
        "columns", "column_names", "rows", "head", "preview", "results", "n_rows",
    )

    def _assess_step_evidence(self, step, captured: dict) -> tuple[bool, str]:
        """Evidence Contract: a finished step must have produced a real artifact
        matching its declared verification. The exemption is derived from the absence of
        a declared verification — not from a self-assigned step.kind — so 'just call it a
        design step' cannot bypass the gate. Conservative: any concrete artifact (a file
        path, structured columns/rows, or substantial console output) counts as evidence,
        so only a step that declared a deliverable yet produced nothing observable is
        flagged unverified."""
        if not getattr(self.settings, "evidence_contract_enabled", True):
            return True, "evidence contract disabled"
        declared = [str(v).strip() for v in (getattr(step, "verification", None) or []) if str(v).strip()]
        if not declared:
            return True, "no verification contract declared"
        tool_results = (captured or {}).get("tool_results") or []

        def _has_artifact(result) -> bool:
            if not isinstance(result, dict):
                return False
            for key in self._EVIDENCE_ARTIFACT_KEYS:
                val = result.get(key)
                if val not in (None, "", [], {}, 0):
                    return True
            out = str(result.get("stdout") or result.get("output") or result.get("stdout_tail") or "")
            return len(out.strip()) >= 40

        if any(_has_artifact(r) for _name, r in tool_results):
            return True, "artifact produced"
        if not tool_results:
            return False, "declared a verification but ran no tools and produced no artifact"
        return False, "declared a verification but produced no observable artifact/output"

    def _make_execution_view(self):
        """Live dashboard on a TTY; a line-logger shim otherwise (so execution
        still runs + logs without a Live/daemon thread in non-interactive runs)."""
        if self._interactive_terminal():
            return PlanRunDashboard(self.console, title="Executing Plan")
        return _LineExecutionView(self.console)

    @staticmethod
    def _iter_steps_in_dependency_order(plan) -> list:
        """Kahn topological order over ``step.dependencies`` (stable among ready
        nodes), falling back to list order for any step left by a cycle/unknown
        dep. Emits every step EXACTLY once — never waits on an unsatisfiable dep."""
        steps = list(plan.steps)
        by_id = {s.id: s for s in steps}
        deps = {s.id: [d for d in (getattr(s, "dependencies", None) or []) if d in by_id and d != s.id] for s in steps}
        indeg = {sid: len(set(dlist)) for sid, dlist in deps.items()}
        ready = [s for s in steps if indeg[s.id] == 0]
        ordered: list = []
        emitted: set = set()
        while ready:
            node = ready.pop(0)
            if node.id in emitted:
                continue
            ordered.append(node)
            emitted.add(node.id)
            for s in steps:
                if s.id not in emitted and node.id in deps[s.id]:
                    indeg[s.id] -= 1
                    if indeg[s.id] <= 0 and s not in ready:
                        ready.append(s)
        for s in steps:  # any cycle/unresolved remainder -> original order
            if s.id not in emitted:
                ordered.append(s)
        return ordered

    @staticmethod
    def _build_step_instruction(plan, step, position: int, total: int) -> str:
        tools = ", ".join(getattr(step, "tool_scope", None) or []) or "any available tool"
        files = ", ".join(getattr(step, "file_scope", None) or []) or "the workspace"
        verify = "; ".join(getattr(step, "verification", None) or [])
        lines = [
            f"Execute plan step {position}/{total} now: {step.title}",
            f"Purpose: {step.purpose}" if getattr(step, "purpose", "") else "",
            f"Allowed tools: {tools}",
            f"Files in scope: {files}",
            f"How to verify: {verify}" if verify else "",
            "Use the available tools to complete this step end to end, then give a brief "
            "final answer confirming what you did. If required inputs are missing, either "
            "call the ask_user tool with ONE concrete question, or stop with a concise "
            "diagnosis naming the exact value/path/tool needed to continue.",
        ]
        return "\n".join(line for line in lines if line)

    def _build_recovery_instruction(self, plan, step, position: int, total: int,
                                    reason: str, attempt: int, max_retries: int) -> str:
        """Augment the step instruction after a failed attempt: name the observed
        failure and tell the agent to change approach (or ask the user if truly
        blocked) rather than repeat the same failing action."""
        base = self._build_step_instruction(plan, step, position, total)
        lines = [
            base,
            f"\nThe previous attempt did not succeed (auto-retry {attempt}/{max_retries}).",
            f"Observed problem: {reason}" if reason else "",
            "Read the error carefully and CHANGE your approach — do NOT repeat the same "
            "failing action or arguments. If a path or tool was missing, try an alternative "
            "inside the workspace. If you are genuinely blocked on a fact only the user can "
            "provide (e.g. a data path or an ambiguous definition), call the ask_user tool "
            "with ONE concrete question instead of guessing or repeating. Otherwise finish "
            "the step and give a brief final answer.",
        ]
        return "\n".join(line for line in lines if line)

    @staticmethod
    def _pending_ask_user_question(session) -> str:
        """If the most recent turn ended by asking the user (``ask_user`` OR
        ``pause_and_ask`` — any tool result flagged ``awaiting_user``), return the
        enriched question; otherwise "". Detection is via the LAST tool result, so
        an agent that asked and then kept working (self-resolved) is not treated as
        blocked."""
        turns = getattr(session, "turns", None) or []
        if not turns:
            return ""
        results = getattr(turns[-1], "tool_results", None) or []
        if not results:
            return ""
        payload = getattr(results[-1], "result", None) or {}
        if not (isinstance(payload, dict) and payload.get("awaiting_user")):
            return ""
        question = str(payload.get("question") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        options = payload.get("options") or []
        if reason and reason.lower() not in question.lower():
            question = f"{question}  (reason: {reason})"
        if isinstance(options, list) and options:
            question = f"{question}  [options: {', '.join(str(o) for o in options)}]"
        return question.strip()

    @staticmethod
    def _plan_answer_suffix(session) -> str:
        """Render any user-provided answers (from ask_user round-trips) as a context
        block to fold into the step instruction so the resumed step uses them."""
        try:
            answers = session.state.custom_data.get("plan_user_answers") or []
        except Exception:
            answers = []
        lines = ["The user has answered earlier questions — use these facts:"]
        for qa in list(answers)[-3:]:
            q = str((qa or {}).get("q") or "").strip()
            a = str((qa or {}).get("a") or "").strip()
            if a:
                lines.append(f"- Q: {q}\n  A: {a}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _run_step_with_recovery(self, plan, step, position: int, total: int, view) -> tuple[str, str, int, str]:
        """Run one plan step with bounded autonomous recovery.

        Attempt 0 runs the step; on a non-convergent/errored turn we automatically
        retry up to ``plan_step_max_retries`` times (default 2) with an instruction
        that names the observed failure and tells the agent to change approach. If
        the agent calls ``ask_user`` (genuinely blocked) we stop and return
        ``needs_input`` so the loop pauses and hands the question to the user.
        Returns ``(status, reason, tool_calls, question)`` with status one of
        ``ok`` | ``needs_input`` | ``failed``."""
        runtime = self._require_runtime()
        session = self._require_session()
        try:
            max_retries = max(0, int(getattr(self.settings, "plan_step_max_retries", 2)))
        except (TypeError, ValueError):
            max_retries = 2
        try:
            step_timeout_s = max(0.0, float(getattr(self.settings, "plan_step_timeout_s", 240.0)))
        except (TypeError, ValueError):
            step_timeout_s = 240.0
        answer_suffix = self._plan_answer_suffix(session)
        reason = ""
        attempt = 0
        while True:
            if attempt == 0:
                instruction = self._build_step_instruction(plan, step, position, total)
            else:
                instruction = self._build_recovery_instruction(
                    plan, step, position, total, reason, attempt, max_retries
                )
            if answer_suffix:
                instruction = f"{instruction}\n\n{answer_suffix}"
            before = len(session.events)
            try:
                with _plan_step_wall_clock_timeout(step_timeout_s):
                    runtime.run_turn(session, instruction)  # autonomous agent; self-bounded by step_timeout_s
            except PlanStepTimeoutError:
                # If the agent PAUSED to ask the user (pause_and_ask / ask_user) before the deadline,
                # surface THAT question — a hard timeout must not swallow the agent's pause path.
                paused_question = self._pending_ask_user_question(session)
                if paused_question:
                    paused_calls = sum(1 for e in session.events[before:]
                                       if e.get("type") == AgentEventType.TOOL_CALL_COMPLETED.value)
                    return ("needs_input", f"The agent needs input: {paused_question}",
                            paused_calls, paused_question)
                timeout_label = f"{step_timeout_s:g}"
                reason = (
                    f"Step {position}/{total} exceeded the configured "
                    f"plan_step_timeout_s={timeout_label}s before producing a usable update."
                )
                if session.turns:
                    turn = session.turns[-1]
                    if getattr(turn, "status", "") == "running":
                        turn.status = "failed"
                        turn.completed_at = time.time()
                session.state.active_turn_id = None
                session.state.status = RuntimeStatus.PAUSED
                session.state.custom_data["plan_last_diagnosis"] = {
                    "status": "paused",
                    "step_id": getattr(step, "id", ""),
                    "step_title": getattr(step, "title", ""),
                    "step_status": "timeout",
                    "reason": reason,
                    "blocked": True,
                    "repair_options": [
                        "Increase PLAN_STEP_TIMEOUT_S for slow model/tool calls, then run /plan-retry.",
                        "For long bioinformatics commands, ask me to run the command as a background job and monitor it with /jobs or /job-tail.",
                        "If this was just file inspection, retry once; provider latency may have caused the stall.",
                    ],
                }
                try:
                    runtime.save_session(session)
                except Exception:
                    pass
                question = (
                    "This step timed out before producing output. Should I retry, "
                    "increase PLAN_STEP_TIMEOUT_S, or convert the slow work to a background job?"
                )
                return ("needs_input", reason, 0, question)
            new_events = session.events[before:]
            tool_calls = sum(1 for e in new_events
                             if e.get("type") == AgentEventType.TOOL_CALL_COMPLETED.value)
            question = self._pending_ask_user_question(session)
            if question:
                return ("needs_input", f"The agent needs input: {question}", tool_calls, question)
            turn_ok = bool(session.turns) and str(session.turns[-1].status) == "completed"
            if turn_ok:
                terminal_failure = self._turn_terminal_tool_failure(session)
                if not terminal_failure:
                    return ("ok", "", tool_calls, "")
                reason = terminal_failure
            event_failed = any(
                e.get("type") in {AgentEventType.ERROR.value, AgentEventType.TOOL_ERROR.value}
                or str((e.get("payload") or {}).get("state", "")).lower() in {"failed", "cancelled", "error"}
                for e in new_events
            )
            reason = reason or self._recent_plan_error_details() or f"step {position}/{total} did not converge"
            if attempt >= max_retries:
                return ("failed", reason, tool_calls, "")
            attempt += 1
            view.record("Repair", actor=step.id, status="warning",
                        message=f"step {position}/{total} auto-retry {attempt}/{max_retries}: {reason[:80]}",
                        metadata={"subagent": step.id, "current_step": step.id, "total_steps": total})
            try:
                time.sleep(min(4.0, float(2 ** (attempt - 1))))
            except Exception:
                pass

    # --------------------------------------------------------- plan repair UX
    def _plan_message_should_intercept(self, text: str) -> bool:
        session = self.session
        plan = session.state.plan if session is not None else None
        if plan is None:
            saved = dict(session.state.custom_data.get("plan_last_diagnosis") or {}) if session is not None else {}
            if saved.get("step_id") == "planning" and str(saved.get("status")) == "failed":
                return self._classify_plan_message(text) in {"diagnose", "edit", "retry"}
            return False
        if plan.status in {PlanStatus.FAILED, PlanStatus.PAUSED}:
            return True
        if bool(session.state.custom_data.get("plan_paused")):
            return True
        return any(str(step.status).lower() in {"failed", "cancelled"} for step in plan.steps)

    @staticmethod
    def _pending_reply_is_non_answer(text: str) -> bool:
        raw = str(text or "").strip()
        lower = raw.lower()
        if not raw:
            return True
        phrases = (
            "cancel", "stop", "pause", "quit", "exit", "skip",
            "not sure", "don't know", "do not know", "unknown",
            "no idea", "wait", "hold on",
        )
        chinese = ("取消", "停止", "暂停", "退出", "跳过", "不知道", "不确定", "先别", "等一下")
        return any(p in lower for p in phrases) or any(p in raw for p in chinese)

    def _handle_plan_natural_language(self, text: str) -> list[AgentEvent]:
        session = self._require_session()
        before = len(session.events)
        intent = self._classify_plan_message(text)
        plan = getattr(session.state, "plan", None)
        saved_diag = dict(session.state.custom_data.get("plan_last_diagnosis") or {})
        if plan is None and saved_diag.get("step_id") == "planning":
            objective = str(saved_diag.get("objective") or "").strip()
            if intent == "diagnose" or not objective:
                self._cmd_plan_diagnose()
            elif intent == "retry":
                self.console.print("[green]Retrying planning for the last objective.[/]")
                self._cmd_plan(objective)
            elif intent == "edit":
                self.console.print("[green]Retrying planning with your refinement.[/]")
                self._cmd_plan(f"{objective}\n\nUser refinement: {text}")
            else:
                self._cmd_plan_diagnose()
            self._require_runtime().save_session(session)
            return [AgentEvent.from_dict(e) for e in session.events[before:]]
        pending_q = str(session.state.custom_data.get("plan_pending_question") or "").strip()
        # If the agent paused via ask_user, treat this reply as the ANSWER (unless the
        # user explicitly asked to diagnose). Record it, fold any path/option into plan
        # context, and resume the step with the answer available to the agent.
        if pending_q and intent != "diagnose":
            if self._pending_reply_is_non_answer(text):
                session.state.custom_data["plan_paused"] = True
                session.state.custom_data["plan_pause_reason"] = (
                    "The plan is still waiting for a concrete answer to the pending question."
                )
                self.console.print(
                    "[yellow]Plan is still paused.[/] I kept the pending question instead of guessing. "
                    "Answer it with the path/choice to use, or run /plan-diagnose."
                )
                self._require_runtime().save_session(session)
                return [AgentEvent.from_dict(e) for e in session.events[before:]]
            answers = list(session.state.custom_data.get("plan_user_answers") or [])
            answers.append({"q": pending_q, "a": str(text or "").strip()})
            session.state.custom_data["plan_user_answers"] = answers
            session.state.custom_data.pop("plan_pending_question", None)
            session.state.custom_data.pop("plan_paused", None)
            session.state.custom_data.pop("plan_pause_reason", None)
            if intent == "use":
                self._cmd_plan_use(text)
            self.console.print("[green]Got it — resuming the step with your answer.[/]")
            self._cmd_plan_retry("")
            self._require_runtime().save_session(session)
            return [AgentEvent.from_dict(e) for e in session.events[before:]]
        if intent == "retry":
            self._cmd_plan_retry("")
        elif intent == "use":
            self._cmd_plan_use(text)
        elif intent == "edit":
            self._cmd_plan_edit(text)
        else:
            self._cmd_plan_diagnose()
            if intent == "unknown":
                self.console.print(
                    "[dim]Plan repair mode is active. Use natural language like "
                    "'continue', 'what is the problem?', 'use data/vc_wgs_vcf', "
                    "or '/plan-exit' to leave plan mode.[/dim]"
                )
        self._require_runtime().save_session(session)
        return [AgentEvent.from_dict(e) for e in session.events[before:]]

    @staticmethod
    def _classify_plan_message(text: str) -> str:
        raw = str(text or "").strip()
        lower = raw.lower()
        if any(k in lower for k in ("continue", "retry", "resume", "rerun")) or any(k in raw for k in ("继续", "重试", "恢复")):
            return "retry"
        if "vc_wgs_vcf" in lower or "vcf_wgs" in lower or "vc_wgs" in lower or "vcf_dir" in lower:
            return "use"
        if "exploratory" in lower or "degraded" in lower or any(k in raw for k in ("探索性", "降级模式")):
            return "use"
        if any(k in lower for k in ("edit", "modify", "change", "revise", "refine")) or any(k in raw for k in ("修改", "调整", "改成", "更新计划")):
            return "edit"
        if any(k in lower for k in ("problem", "why", "diagnose", "error", "failed", "issue")) or any(k in raw for k in ("问题", "原因", "为什么", "怎么回事", "诊断", "失败", "错误")):
            return "diagnose"
        if InteractiveShell._extract_existing_paths(raw):
            return "use"
        return "unknown"

    def _cmd_plan_diagnose(self, *_args: Any) -> dict[str, Any]:
        session = self._require_session()
        plan = session.state.plan
        if plan is None:
            saved = dict(session.state.custom_data.get("plan_last_diagnosis") or {})
            if saved:
                self._render_plan_diagnosis(None, diagnosis=saved)
                return saved
            self.console.print("[yellow]No active plan to diagnose.[/]")
            return {"status": "no_plan"}
        diagnosis = self._build_plan_diagnosis(plan)
        self._render_plan_diagnosis(plan, diagnosis=diagnosis)
        session.state.custom_data["plan_last_diagnosis"] = diagnosis
        self._emit_plan_phase(
            "Repair",
            actor="plan",
            status="warning" if diagnosis.get("blocked") else "success",
            message=str(diagnosis.get("reason") or "plan diagnosis"),
            metadata=diagnosis,
        )
        self._require_runtime().save_session(session)
        return diagnosis

    def _cmd_plan_retry(self, arg: str = "") -> dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session()
        plan = session.state.plan
        if plan is None:
            self.console.print("[yellow]No active plan to retry.[/]")
            return {"status": "no_plan"}
        step_id = str(arg or "").strip()
        target = None
        if step_id:
            target = next((step for step in plan.steps if step.id == step_id), None)
            if target is None:
                self.console.print(f"[yellow]Unknown plan step:[/] {_rich_escape(step_id)}")
                return {"status": "unknown_step", "step": step_id}
        if target is None:
            target = next((step for step in plan.steps if str(step.status).lower() in {"failed", "cancelled"}), None)
        if target is None:
            target = self._first_incomplete_step(plan)
        if target is None:
            self.console.print("[green]No incomplete plan step remains.[/]")
            plan.status = PlanStatus.COMPLETED
            runtime.save_session(session)
            return {"status": "completed"}

        reset = False
        for step in self._iter_steps_in_dependency_order(plan):
            if step.id == target.id:
                reset = True
            if not reset:
                continue
            if str(step.status).lower() in {"failed", "cancelled", "running", "skipped"}:
                step.status = "pending"
        session.state.custom_data.pop("plan_paused", None)
        session.state.custom_data.pop("plan_pause_reason", None)
        plan.status = PlanStatus.APPROVED
        plan.updated_at = time.time()
        runtime.save_session(session)
        self.console.print(f"[green]Retrying plan from step:[/] {_rich_escape(target.id)}")
        return self._execute_plan_autonomously(plan)

    def _cmd_plan_use(self, arg: str = "") -> dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session()
        updates: dict[str, Any] = {}
        text = str(arg or "").strip()
        context = dict(session.state.custom_data.get("plan_context") or {})

        for token in shlex.split(text):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in {"vcf_dir", "vcf_path", "vc_wgs_vcf_dir", "vc_wgs_vcf", "vc_wgs_dir", "vc_wgs"}:
                updates["VC_WGS_VCF_DIR"] = value
            elif key in {"workflow", "workflow_mode", "mode"}:
                updates["workflow_mode"] = value

        lower = text.lower()
        if "exploratory" in lower or "degraded" in lower or "探索性" in text or "降级模式" in text:
            updates["workflow_mode"] = "exploratory"
        for path in self._extract_existing_paths(text):
            if path.is_dir() and not updates.get("VC_WGS_VCF_DIR"):
                updates["VC_WGS_VCF_DIR"] = str(path)

        if "VC_WGS_VCF_DIR" in updates:
            vcf_dir = str(Path(str(updates["VC_WGS_VCF_DIR"])).expanduser())
            os.environ["VC_WGS_VCF_DIR"] = vcf_dir
            context["VC_WGS_VCF_DIR"] = vcf_dir
        if "workflow_mode" in updates:
            context["workflow_mode"] = str(updates["workflow_mode"]).strip().lower()

        if not updates:
            self.console.print("[yellow]No plan context update was detected.[/] Try: /plan-use vcf_dir=data/vc_wgs_vcf")
            return {"status": "no_update"}

        session.state.custom_data["plan_context"] = context
        session.state.custom_data.pop("plan_last_diagnosis", None)
        runtime.save_session(session)
        self.console.print("[green]Plan context updated.[/]")
        table = Table(title="Plan Context", box=box.SIMPLE)
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        for key, value in sorted(context.items()):
            table.add_row(str(key), str(value))
        self.console.print(table)
        if session.state.plan is not None:
            self._cmd_plan_diagnose()
        return {"status": "updated", "updates": updates, "context": context}

    @staticmethod
    def _extract_existing_paths(text: str) -> list[Path]:
        matches = re.findall(r"(?:~|/|\.{1,2}/|[A-Za-z0-9_.-]+/)[A-Za-z0-9_./~:=+-]+", str(text or ""))
        out: list[Path] = []
        for raw in matches:
            cleaned = raw.strip(" ,.;:，。；：'\"()[]{}<>")
            if "=" in cleaned:
                cleaned = cleaned.rsplit("=", 1)[-1]
            path = Path(cleaned).expanduser()
            if path.exists():
                out.append(path)
        return out

    def _preflight_plan_step(self, plan, step) -> dict[str, Any]:
        if not self._plan_step_mentions_wgs(plan, step):
            return {"blocked": False}
        snapshot = self._wgs_readiness_snapshot()
        if int(snapshot.get("n_samples") or 0) <= 0:
            return {
                "blocked": True,
                "reason": "No WGS VCF files were found. Set VC_WGS_VCF_DIR or use /plan-use vcf_dir=<path>.",
                "wgs": snapshot,
                "repair_options": [
                    "Use /plan-use vcf_dir=data/vc_wgs_vcf if that is the intended local VCF directory.",
                    "Use /plan-use workflow_mode=exploratory to keep built-in exploratory analysis when external tools are missing.",
                    "Run /doctor for dependency and VCF readiness details.",
                ],
            }
        return {"blocked": False, "wgs": snapshot}

    def _store_plan_diagnosis(self, plan, step, *, reason: str, preflight: dict[str, Any] | None = None) -> None:
        session = self._require_session()
        diagnosis = {
            "status": getattr(plan.status, "value", str(plan.status)),
            "step_id": getattr(step, "id", ""),
            "step_title": getattr(step, "title", ""),
            "reason": str(reason or ""),
            "blocked": bool((preflight or {}).get("blocked")),
            "preflight": dict(preflight or {}),
            "repair_options": list((preflight or {}).get("repair_options") or self._default_plan_repair_options(plan, step)),
        }
        session.state.custom_data["plan_last_diagnosis"] = diagnosis

    def _build_plan_diagnosis(self, plan) -> dict[str, Any]:
        session = self._require_session()
        saved = dict(session.state.custom_data.get("plan_last_diagnosis") or {})
        failed = next((step for step in plan.steps if str(step.status).lower() in {"failed", "cancelled"}), None)
        current = failed or self._first_incomplete_step(plan)
        reason = (
            saved.get("reason")
            or session.state.custom_data.get("plan_pause_reason")
            or self._recent_plan_error_details()
            or "The plan is waiting for a retry, a plan edit, or missing input."
        )
        diagnosis = {
            "status": getattr(plan.status, "value", str(plan.status)),
            "step_id": getattr(current, "id", "") if current is not None else "",
            "step_title": getattr(current, "title", "") if current is not None else "",
            "step_status": getattr(current, "status", "") if current is not None else "",
            "reason": str(reason),
            "blocked": plan.status == PlanStatus.PAUSED or bool(session.state.custom_data.get("plan_paused")),
            "repair_options": saved.get("repair_options") or self._default_plan_repair_options(plan, current),
        }
        if current is not None and self._plan_step_mentions_wgs(plan, current):
            diagnosis["wgs"] = self._wgs_readiness_snapshot()
        return diagnosis

    def _render_plan_diagnosis(self, plan, *, diagnosis: dict[str, Any] | None = None) -> None:
        diagnosis = diagnosis or self._build_plan_diagnosis(plan)
        table = Table(title="Plan Diagnosis", box=box.SIMPLE)
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value")
        table.add_row("plan status", str(diagnosis.get("status") or "-"))
        step = f"{diagnosis.get('step_id') or '-'}"
        if diagnosis.get("step_title"):
            step += f" - {diagnosis.get('step_title')}"
        table.add_row("current step", step)
        table.add_row("step status", str(diagnosis.get("step_status") or "-"))
        table.add_row("reason", str(diagnosis.get("reason") or "-"))
        wgs = diagnosis.get("wgs") or {}
        if wgs:
            table.add_row(
                "wgs readiness",
                f"{wgs.get('n_samples', 0)} sample(s), "
                f"{wgs.get('n_indexed', 0)}/{wgs.get('n_samples', 0)} indexed, "
                f"mode={wgs.get('workflow_mode', 'unknown')}",
            )
            missing = ", ".join(str(x) for x in (wgs.get("missing_for_standard_workflow") or []) if x)
            if missing:
                table.add_row("standard missing", missing)
        options = diagnosis.get("repair_options") or []
        if options:
            table.add_row("next actions", "\n".join(f"- {item}" for item in options))
        self.console.print(table)

    def _default_plan_repair_options(self, plan, step) -> list[str]:
        options = [
            "Type 'continue' or run /plan-retry to retry the current failed step.",
            "Use /plan-edit <feedback> to modify the plan before retrying.",
        ]
        if step is not None and self._plan_step_mentions_wgs(plan, step):
            options.insert(0, "Use /plan-use vcf_dir=<path> if the VCF directory is missing or wrong.")
            options.insert(1, "Use /plan-use workflow_mode=exploratory if standard GWAS dependencies are unavailable.")
        return options

    @staticmethod
    def _plan_step_mentions_wgs(plan, step) -> bool:
        """Whether a step genuinely needs WGS/variant-data readiness checks.

        Do not key off `gwas` in a filename alone: GWAS summary-stat CSVs are
        tabular inputs, not genotype/VCF inputs. Trigger WGS preflight only when
        the plan names variant/genotype tooling, real variant file extensions, or
        explicit WGS/VCF/genotype workflow language outside file paths.
        """
        parts = [
            getattr(step, "id", ""),
            getattr(step, "title", ""),
            getattr(step, "purpose", ""),
            " ".join(getattr(step, "tool_scope", None) or []),
            " ".join(getattr(step, "file_scope", None) or []),
        ]
        haystack = " ".join(str(part or "") for part in parts)
        lowered = haystack.lower()
        tool_names = {str(t).lower() for t in (getattr(step, "tool_scope", None) or [])}
        if any(name.startswith("vcf_") or name.startswith("wgs_") or name == "wgs_environment_check" for name in tool_names):
            return True
        if "inspection fast-path" in str(getattr(plan, "audit_summary", "")).lower():
            return False
        if "read-only preview" in lowered or "file-inspection" in lowered:
            return False

        file_text = " ".join(str(x) for x in (getattr(step, "file_scope", None) or []))
        file_refs = re.findall(
            r"((?:[~\w.\-]+)?(?:/[\w.\-]+)+|[\w.\-]+\.(?:csv|tsv|txt|parquet|xlsx?|json|vcf|gz|bgen|bed|pgen))",
            f"{haystack} {file_text}",
            flags=re.I,
        )
        variant_ext = re.compile(r"(\.vcf(\.gz)?|\.bgen|\.bed|\.pgen)$", re.I)
        tabular_ext = re.compile(r"\.(csv|tsv|txt|parquet|xlsx?|json)$", re.I)
        if any(variant_ext.search(ref) for ref in file_refs):
            return True
        non_path_text = haystack
        for ref in sorted(set(file_refs), key=len, reverse=True):
            non_path_text = non_path_text.replace(ref, " ")
        non_path = non_path_text.lower()
        if any(tabular_ext.search(ref) for ref in file_refs) and not re.search(
            r"\b(wgs|whole[- ]genome|vcf|genotyp(?:e|ing)|variant calling|raw variants?)\b|全基因组|基因型|变异文件",
            non_path,
            flags=re.I,
        ):
            return False
        return bool(re.search(
            r"\b(wgs|whole[- ]genome|vcf|genotyp(?:e|ing)|variant calling|raw variants?|run\s+gwas|gwas association)\b|全基因组|基因型|变异文件",
            non_path,
            flags=re.I,
        ))

    def _recent_plan_error_details(self) -> str:
        session = self._require_session()
        if session.turns:
            turn = session.turns[-1]
            tool_errors = [r.error for r in turn.tool_results if getattr(r, "error", "")]
            if tool_errors:
                return tool_errors[-1]
            if str(turn.status).lower() not in {"completed", "success"}:
                return f"latest model turn ended with status={turn.status}"
        for raw in reversed(session.events[-20:]):
            etype = str(raw.get("type", ""))
            if etype in {AgentEventType.ERROR.value, AgentEventType.TOOL_ERROR.value, AgentEventType.COMMAND_FAILED.value}:
                payload = raw.get("payload") or {}
                return str(payload.get("error") or raw.get("message") or payload.get("message") or etype)
        return ""

    def _tool_summary_snapshot(self) -> dict[str, str]:
        try:
            runtime = self._require_runtime()
            handlers = runtime.tool_registry.list_handlers()
        except Exception:
            handlers = []
        cap_counts: Counter[str] = Counter()
        categories: Counter[str] = Counter()
        for handler in handlers:
            name = str(getattr(handler, "name", ""))
            categories[self._tool_category(name)] += 1
            try:
                caps = handler.required_capabilities()
            except Exception:
                caps = frozenset()
            for cap in caps:
                cap_counts[getattr(cap, "value", str(cap))] += 1
        top_caps = ", ".join(f"{key}:{value}" for key, value in cap_counts.most_common(4))
        top_cats = ", ".join(f"{key}:{value}" for key, value in categories.most_common(4))
        wgs = self._wgs_readiness_snapshot()
        missing = ", ".join(str(x) for x in (wgs.get("missing_for_standard_workflow") or [])[:3] if x)
        wgs_summary = (
            f"{wgs.get('n_samples', 0)} VCF sample(s), "
            f"{wgs.get('n_indexed', 0)}/{wgs.get('n_samples', 0)} indexed, "
            f"mode={wgs.get('workflow_mode', 'unknown')}"
        )
        if missing:
            wgs_summary += f", missing: {missing}"
        return {
            "summary": f"{len(handlers)} loaded" + (f" | {top_caps}" if top_caps else "") + (f" | {top_cats}" if top_cats else ""),
            "wgs_summary": wgs_summary,
        }

    def _wgs_readiness_snapshot(self) -> dict[str, Any]:
        try:
            from biobank_agent.data.vcf_loader import discover_vcf_files
            from biobank_agent.skills.vcf_query import _get_vcf_dirs
            from biobank_agent.utils.wgs import find_executable, package_available
        except Exception as exc:
            return {"n_samples": 0, "n_indexed": 0, "workflow_mode": "unknown", "error": str(exc)}

        dirs = _get_vcf_dirs(self._require_session() if self.session is not None else None)
        files = discover_vcf_files(*dirs)
        executables = {
            "bcftools": find_executable("bcftools"),
            "tabix": find_executable("tabix"),
            "plink2": find_executable("plink2"),
            "plink": find_executable("plink"),
            "vep": find_executable("vep"),
            "snpEff": find_executable("snpEff"),
            "table_annovar.pl": find_executable("table_annovar.pl"),
            "Rscript": find_executable("Rscript"),
        }
        packages = {
            "cyvcf2": package_available("cyvcf2"),
            "pandas": package_available("pandas"),
            "scipy": package_available("scipy"),
            "gseapy": package_available("gseapy"),
        }
        has_core = bool(packages["cyvcf2"] and packages["pandas"] and executables["bcftools"] and executables["tabix"])
        has_plink = bool(executables["plink2"] or executables["plink"])
        has_annotation = bool(executables["vep"] or executables["snpEff"] or executables["table_annovar.pl"])
        has_enrichment = bool(packages["gseapy"] or executables["Rscript"])
        missing = [
            name for name, ready in {
                "bcftools": bool(executables["bcftools"]),
                "tabix": bool(executables["tabix"]),
                "plink2_or_plink": has_plink,
                "vep_or_snpeff_or_annovar": has_annotation,
                "gseapy_or_Rscript": has_enrichment,
            }.items() if not ready
        ]
        context = dict(self._require_session().state.custom_data.get("plan_context") or {}) if self.session is not None else {}
        preferred_mode = str(context.get("workflow_mode") or "").lower()
        workflow_mode = "standard" if has_core and has_plink and not preferred_mode else (preferred_mode or "exploratory")
        return {
            "n_samples": len(files),
            "n_indexed": sum(1 for item in files if item.get("has_index")),
            "total_size_gb": round(sum(float(item.get("size_mb") or 0) for item in files) / 1024, 2),
            "workflow_mode": workflow_mode,
            "missing_for_standard_workflow": missing,
            "vcf_dirs": [str(path) for path in dirs],
            "existing_vcf_dirs": [str(path) for path in dirs if path.exists()],
            "samples": files[:10],
        }

    @staticmethod
    def _tool_category(name: str) -> str:
        lowered = name.lower()
        if lowered.startswith("vcf_") or "wgs" in lowered or "gwas" in lowered:
            return "wgs"
        if "research" in lowered or "paper" in lowered or "literature" in lowered:
            return "research"
        if "report" in lowered or "plot" in lowered or "figure" in lowered:
            return "reporting"
        if "mcp" in lowered or lowered in {"shell_exec", "python_exec", "git_diff", "test_runner"}:
            return "runtime"
        if "agent" in lowered or "review" in lowered or "audit" in lowered:
            return "agentic"
        return "analysis"

    def _cmd_plan_reject(self, arg: str = "") -> None:
        runtime = self._require_runtime()
        session = self._require_session()
        plan = runtime.reject_plan(session, arg)
        if plan is None:
            self.console.print("[yellow]No draft plan to reject.[/]")
            return
        runtime.save_session(session)
        self.console.print("[yellow]Plan rejected.[/]")
        return {"status": "rejected", "reason": arg.strip()}

    def _cmd_plan_edit(self, arg: str = "") -> None:
        runtime = self._require_runtime()
        session = self._require_session()
        if session.state.plan is None:
            self.console.print("[yellow]No active plan to refine. Use /plan <task> first.[/]")
            return
        self._set_activity("refining plan")
        objective = session.state.plan.objective
        # Snapshot the plan state: drafting re-plans through the runtime and may mutate session.state
        # mid-flight (the plan AND its derived pending_work), so a failed/timed-out refinement must
        # restore both to truly leave the plan unchanged.
        original_plan = copy.deepcopy(session.state.plan)
        original_pending_work = copy.deepcopy(session.state.pending_work)
        previous_mode = runtime.config.approval_profile  # drafting flips to read-only "plan"; restore on abort
        build_timeout_s = max(0.0, float(getattr(self.settings, "plan_build_timeout_s", 240.0)))
        provider_timeout_restore = self._apply_planning_provider_caps(runtime, build_timeout_s)
        plan = None
        try:
            # Re-plan through the same council path (with live progress) so the refinement produces a
            # real, objective-specific plan — under the SAME build-timeout guard as /plan so it can't hang.
            with _plan_build_wall_clock_timeout(build_timeout_s):
                plan = self._draft_plan_with_live_progress(objective, refinement=arg)
        except PlanBuildTimeoutError:
            self._set_activity("idle")
            timeout_label = f"{build_timeout_s:g}"
            reason = (
                f"Plan refinement stalled for {timeout_label}s with no model activity "
                f"(plan_build_timeout_s={timeout_label}s); the original plan is unchanged."
            )
            self._emit_plan_phase("Plan review", status="error", message=reason, metadata={"feedback": arg.strip()})
            self.console.print(f"[red]Plan refinement timed out:[/] {_rich_escape(reason)}")
            return {"status": "failed", "error": reason, "step_status": "timeout"}
        except CouncilError as exc:
            self._set_activity("idle")
            self._emit_plan_phase("Plan review", status="error", message=str(exc), metadata={"feedback": arg.strip()})
            self.console.print(f"[red]Plan refinement failed:[/] {_rich_escape(str(exc))}")
            return {"status": "failed", "error": str(exc)}
        finally:
            self._restore_planning_provider_caps(provider_timeout_restore)
            if plan is None:
                # The draft aborted (timeout / council error): restore the pre-edit plan, its derived
                # pending_work, AND the approval profile (drafting flips to read-only "plan") so a failed
                # refinement leaves no half-mutated state behind. Restore the EXACT profile — never widen.
                try:
                    session.state.plan = original_plan
                    session.state.pending_work = original_pending_work
                    runtime.set_approval_profile(previous_mode)
                    runtime.save_session(session)
                except Exception:
                    pass
        self._record_plan_review_graph(plan, action="refine", choice="fix", detail=arg.strip())
        self._emit_plan_phase(
            "Plan review",
            actor="user",
            status="running",
            message="plan refinement requested",
            metadata={"revision": plan.revision, "feedback": arg.strip()},
        )
        runtime.save_session(session)
        self._set_activity("plan review")
        return {"status": "updated", "revision": plan.revision}

    def _cmd_plan_pause(self, *_args: Any) -> dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session()
        plan = session.state.plan
        if plan is None:
            self.console.print("[yellow]No active plan to pause.[/]")
            return {"status": "no_plan"}
        plan.status = PlanStatus.PAUSED
        plan.updated_at = time.time()
        session.state.custom_data["plan_paused"] = True
        session.state.custom_data["plan_pause_reason"] = "User requested pause"
        runtime.save_session(session)
        self.console.print("[yellow]Plan paused.[/]")
        return {"status": "paused"}

    def _cmd_plan_resume(self, *_args: Any) -> dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session()
        plan = session.state.plan
        if plan is not None and (
            plan.status in {PlanStatus.FAILED, PlanStatus.PAUSED}
            or any(str(step.status).lower() in {"failed", "cancelled"} for step in plan.steps)
        ):
            return self._cmd_plan_retry("")
        session.state.custom_data.pop("plan_paused", None)
        session.state.custom_data.pop("plan_pause_reason", None)
        runtime.save_session(session)
        self.console.print("[green]Plan resume requested.[/]")
        return {"status": "resumed"}

    def _cmd_plan_option(self, arg: str = "") -> dict[str, Any]:
        session = self._require_session()
        choice = str(arg or "").strip()
        session.state.custom_data["plan_option"] = choice
        self._require_runtime().save_session(session)
        self.console.print(f"[green]Plan option recorded:[/] {_rich_escape(str(choice or '(empty)'))}")
        return {"status": "recorded", "choice": choice}

    def _cmd_plan_skip(self, arg: str = "") -> dict[str, Any]:
        session = self._require_session()
        choice = str(arg or "").strip()
        skipped = list(session.state.custom_data.get("plan_skipped_steps") or [])
        if choice:
            skipped.append(choice)
        session.state.custom_data["plan_skipped_steps"] = skipped
        self._require_runtime().save_session(session)
        self.console.print(f"[green]Plan skip recorded:[/] {_rich_escape(str(choice or '(empty)'))}")
        return {"status": "recorded", "skipped": choice}

    def _cmd_plan_exit(self, *_args: Any) -> dict[str, Any]:
        session = self._require_session()
        session.state.custom_data["plan_mode_exited"] = True
        plan = session.state.plan
        if plan is not None and plan.status == PlanStatus.DRAFT:
            self._record_plan_review_graph(plan, action="exit", choice="exit", detail="User exited plan review")
            self._emit_plan_phase(
                "Plan review",
                actor="user",
                status="cancelled",
                message="exited plan review without approval",
                metadata={"revision": plan.revision},
            )
        self._require_runtime().save_session(session)
        self._set_activity("idle")
        self.console.print("[green]Plan mode exit recorded.[/]")
        return {"status": "exited"}

    def _cmd_plans(self, *_args: Any) -> list[dict[str, Any]]:
        runtime = self._require_runtime()
        sessions = runtime.list_sessions()
        table = Table(title="Saved Plans", box=box.SIMPLE)
        table.add_column("Session")
        table.add_column("Plan")
        table.add_column("Status")
        rows = []
        for item in sessions[:10]:
            status = item.state.plan.status.value if item.state.plan else "-"
            plan_title = item.state.plan.title if item.state.plan else item.title
            rows.append({"session_id": item.session_id, "title": plan_title, "status": status})
            table.add_row(item.session_id, plan_title, status)
        self.console.print(table)
        return rows

    def _cmd_routing_status(self, *_args: Any) -> dict[str, Any]:
        session = self._require_session()
        runtime = self._require_runtime()
        recent = session.state.last_orchestration or {}
        table = Table(title="Routing Status", box=box.SIMPLE)
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        table.add_row("active role", runtime.config.active_role)
        table.add_row("primary", runtime.config.primary_model)
        table.add_row("planner", runtime.config.planner_model)
        table.add_row("critic", runtime.config.critic_model)
        table.add_row("summarizer", runtime.config.summarizer_model)
        table.add_row("safety", runtime.config.safety_reviewer_model)
        table.add_row("strategy", str(recent.get("strategy", runtime.config.active_role)))
        table.add_row("claims", str(len(recent.get("claims") or [])))
        table.add_row("evidence", str(len(recent.get("evidence_links") or [])))
        self.console.print(table)
        return {"active_role": runtime.config.active_role, "strategy": recent.get("strategy", "")}

    def _gather_plan_context(self, plan) -> None:
        runtime = self._require_runtime()
        session = self._require_session()
        tool_names = [handler.name for handler in runtime.tool_registry.list_handlers()]
        context_items = [
            f"workspace={session.cwd}",
            f"turns={len(session.turns)}",
            f"events={len(session.events)}",
            "tools=" + ", ".join(tool_names[:20]) + (f" ... (+{len(tool_names) - 20})" if len(tool_names) > 20 else ""),
        ]
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=session.cwd,
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
            context_items.append("git_status=" + (result.stdout.strip() or "clean"))
        except Exception as exc:
            context_items.append(f"git_status_unavailable={exc}")
        plan.context_gathering.extend(context_items)
        runtime.create_checkpoint(session, "plan_context", summary="Read-only context gathered for plan.")

    def _draft_plan_with_live_progress(self, objective: str, *, refinement: str = ""):
        runtime = self._require_runtime()
        session = self._require_session()
        dashboard = PlanRunDashboard(self.console, title="Planning")
        use_live = self._interactive_terminal()
        interactive = self._interactive_terminal()
        self._set_activity("planning")
        self._emit_plan_phase(
            "Planning",
            status="running",
            message="council planning started",
            metadata={"objective": objective},
        )

        # Live-region coordinator. INVARIANT: at most one Rich Live is active at
        # a time. During the clarification stage we show a lightweight status
        # spinner (a Live) and the clarification *selector* (a Live) — never the
        # planning dashboard. The dashboard Live is started LAZILY, only when the
        # first real planning-phase event arrives (after clarification has fully
        # finished). Opening a second Live while the dashboard Live was active
        # raised rich.errors.LiveError, which _clarify swallowed -> every
        # question recorded None and the prompts blasted past without blocking.
        _PRE_PLANNING = {"Preflight", "Clarification"}
        ls: dict[str, Any] = {"status": None, "dashboard": False, "menus_shown": False}
        # Global cancel for the whole plan: set on Ctrl-C / teardown so any
        # orphaned streaming worker (HTTP read not interruptible) stops
        # forwarding deltas at the source.
        cancel_event = threading.Event()

        def _stop_status() -> None:
            st = ls["status"]
            if st is not None:
                try:
                    st.stop()
                except Exception:
                    pass
                ls["status"] = None

        def _start_dashboard() -> None:
            if not ls["dashboard"]:
                _stop_status()
                dashboard.start(refresh_per_second=10.0)
                ls["dashboard"] = True

        def emit_cb(stage: str, *, status: str = "running", message: str = "", metadata: dict[str, Any] | None = None) -> None:
            md = dict(metadata or {})
            # Streamed token deltas update the active model row only — never
            # recorded as events (token-rate record() would saturate rendering)
            # and never logged to the session.
            if "delta" in md:
                if use_live:
                    try:
                        dashboard.note_partial(str(md.get("subagent") or ""), str(md.get("delta") or ""))
                    except Exception:
                        pass
                return
            if use_live:
                # record() updates phase state even before the dashboard Live is
                # started, so once it does start it shows the full history
                # (Preflight/Clarification included).
                try:
                    dashboard.record(stage, actor="biobank", status=status, message=message, metadata=md)
                except Exception:
                    pass
                if not ls["dashboard"]:
                    if stage in _PRE_PLANNING:
                        # One transient spinner for the pre-planning phases.
                        # Suppress it once menus have been shown so the
                        # post-clarification "resolved" event cannot briefly
                        # re-open a Live next to the (now finished) prompts.
                        if ls["status"] is None and not ls["menus_shown"]:
                            try:
                                ls["status"] = self.console.status("[bold cyan]Planning[/]…", spinner="dots")
                                ls["status"].start()
                            except Exception:
                                ls["status"] = None
                        if ls["status"] is not None:
                            try:
                                ls["status"].update(f"[bold cyan]{stage}[/] {message}")
                            except Exception:
                                pass
                    else:
                        # First real planning phase: hand off spinner -> dashboard.
                        _start_dashboard()
            # Persist stage milestones (and any error) as plan-phase events; skip
            # the high-volume per-subagent dispatch chatter from the session log.
            if "subagent" not in md or status == "error":
                self._emit_plan_phase(stage, status=status, message=message, metadata=md)

        def clarifier(question: dict[str, Any]) -> "str | None":
            # The selector owns the single Live during clarification. Stop the
            # pre-planning spinner first so they never coexist; the dashboard is
            # not running yet (lazy start happens only after clarification).
            ls["menus_shown"] = True
            _stop_status()
            return self._plan_clarifier(question)

        try:
            tool_names = [handler.spec().name for handler in runtime.tool_registry.list_handlers()]
        except Exception:
            tool_names = []

        # Data-grounded planning: probe the files the objective references and
        # inject a bounded DataContext so the council plans around the actual data
        # shape (a GWAS-results CSV is a table, not genotypes — kills CSV→WGS at root).
        data_ctx = ""
        try:
            from biobank_agent.runtime.data_probe import probe_objective

            data_ctx = probe_objective(objective, cwd=session.cwd).get("text", "")
        except Exception:
            data_ctx = ""
        try:
            runtime.set_approval_profile("plan")
            plan = runtime.update_plan(
                session,
                objective,
                refinement=refinement,
                emit=emit_cb,
                clarifier=clarifier if interactive else None,
                tool_names=tool_names,
                interactive=interactive,
                context=(f"Workspace: {session.cwd}" + (f"\n\n{data_ctx}" if data_ctx else "")),
                session_id=session.session_id,
                turn_id=session.session_id,
                stream=use_live,
                cancel_event=cancel_event,
            )
            self._mark_plan_review_ready(plan)
            self._record_plan_action_graph(plan)
            self._gather_plan_context(plan)
            self._record_plan_review_graph(plan, action="opened", choice="", detail="Plan review opened")
            runtime.save_session(session)
        except KeyboardInterrupt:
            cancel_event.set()  # stop orphaned streaming workers at the source
            self._set_activity("idle")
            self._emit_plan_phase(
                "Planning",
                actor="user",
                status="cancelled",
                message="planning interrupted",
                metadata={"objective": objective},
            )
            runtime.save_session(session)
            raise
        finally:
            cancel_event.set()  # any worker still streaming past teardown is a no-op
            if use_live:
                _stop_status()
                dashboard.stop()
        self._set_activity("plan review")
        # Concise post-plan output: the live planning dashboard already showed the
        # council activity (transient), so don't reprint a static activity summary
        # or a "Plan Formulation Progress" preamble. Show the plan + the workflow
        # diagram ONCE (the diagram is no longer reprinted at approval/execution).
        self._render_plan(plan)
        self._render_plan_flow(plan)
        # Interactive terminals get the live arrow-key selector (drawn by the
        # review loop); only show the static notice when we cannot read keys.
        if not self._interactive_terminal():
            self._render_plan_review_menu_notice()
        return plan

    def _plan_clarifier(self, question: dict[str, Any]) -> "str | None":
        """Present one clarifying question via the in-place selector; return the
        chosen option label, a free-text answer the user typed, or ``None`` if
        skipped/cancelled.

        A final "Type a custom answer" row (and Tab on any option) opens a
        free-text prompt so the user can supply or correct an answer (inline
        "type something" UX). The typed text is forwarded verbatim to the
        planner (which records it as ``Q: ... A: ...``)."""
        options = [opt for opt in (question.get("options") or []) if isinstance(opt, dict) and opt.get("label")]
        if not options:
            return None
        # Append the synthetic free-text row; selecting it opens the same editor.
        display_options = options + [{"label": _CUSTOM_ANSWER_LABEL, "description": "None of these — type your own answer"}]
        custom_index = len(options)

        def render(opt: dict[str, Any], is_selected: bool, index: int) -> str:
            cursor = "[cyan]›[/cyan]" if is_selected else " "
            marker = "[green]●[/green]" if is_selected else "○"
            # Escape model-derived label/description so stray "[" / "]" can never
            # be parsed as Rich markup (raising MarkupError mid-menu).
            label = _rich_escape(str(opt.get("label", f"Option {index + 1}")))
            label = f"[bold]{label}[/bold]" if is_selected else label
            desc = _rich_escape(str(opt.get("description", "")))
            return f"{cursor} {marker} {index + 1}. {label}\n    [dim]{desc}[/dim]"

        def edit_prefill(idx: int) -> str:
            # The synthetic custom row isn't a real answer -> start blank.
            if custom_index is not None and idx == custom_index:
                return ""
            return str(display_options[idx].get("label", "")).strip()

        # KeyboardInterrupt is NOT caught here: Ctrl-C during a clarification
        # cancels the whole plan (it propagates past _clarify's `except
        # Exception`, since KeyboardInterrupt is a BaseException, up to the
        # _draft_plan_with_live_progress handler). Tab / the custom row open an
        # INLINE editor inside the same panel (no break-out prompt); Esc/empty
        # submit return to the menu without answering.
        result = self._live_single_select(
            title=f"Clarify: {_rich_escape(str(question.get('header', 'Plan')))}",
            header=_rich_escape(str(question.get("question", "Clarification needed"))),
            items=display_options,
            render_item=render,
            footer="↑/↓ or j/k move · 1-9 jump · Tab edit/correct · Enter confirm · Ctrl-C cancel",
            editable=True,
            custom_index=custom_index,
            edit_prefill=edit_prefill,
        )
        if isinstance(result, str):
            return result  # inline-typed free text (Tab edit or the custom row)
        return str(display_options[result].get("label", ""))

    # ---------------------------------------------------------------- research
    def _get_researcher(self) -> RuntimeResearcher:
        if self._researcher is None:
            runtime = self._require_runtime()
            self._researcher = RuntimeResearcher(
                runtime.provider_router,
                runtime.config,
                retriever=self._research_retriever,
                num_subqueries=int(getattr(self.settings, "research_subqueries", 3)),
                max_sources=int(getattr(self.settings, "research_max_sources", 8)),
                timeout_s=float(getattr(self.settings, "council_timeout_s", 360.0)),
                clock=time.time,
            )
        return self._researcher

    def _research_retriever(self, subquery: str, max_sources: int) -> dict[str, Any]:
        """Retrieve sources for a sub-query by calling the ``deep_research`` skill
        DIRECTLY (not via ``runtime.invoke_tool``). This is the key concurrency
        fix: the council retrieves sub-queries in parallel threads, and routing
        each through the shared ``AgentSession``/``ActionGraph`` (single SQLite
        connection, no locking) would corrupt runtime state. A direct call with a
        fresh per-call tool context touches no shared runtime state. Returns ``{}``
        on any failure so research degrades to model-only synthesis."""
        try:
            from biobank_agent.skills.deep_research import deep_research as _deep_research

            legacy = self._require_legacy_agent()
            session = self._require_session()
            report_dir = self._session_report_dir(session) / "research"
            report_dir.mkdir(parents=True, exist_ok=True)
            ctx = legacy._build_ctx(report_dir)
            try:
                ctx.workspace_root = Path(session.cwd)
            except Exception:
                pass
            return dict(_deep_research(subquery, max_sources, ctx=ctx) or {})
        except Exception:
            return {}

    def _cmd_research(self, arg: str = "") -> dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session()
        question = arg.strip()
        if not question:
            self.console.print("[dim]Use /research <question> to run a multi-agent cited research brief.[/]")
            return {"status": "none"}
        dashboard = PlanRunDashboard(self.console, title="Research")
        use_live = self._interactive_terminal()
        self._set_activity("researching")

        def emit_cb(stage: str, *, status: str = "running", message: str = "", metadata: dict[str, Any] | None = None) -> None:
            md = dict(metadata or {})
            if use_live:
                try:
                    dashboard.record(stage, actor="biobank", status=status, message=message, metadata=md)
                except Exception:
                    pass
            if "subagent" not in md or status == "error":
                self._emit_plan_phase(stage, status=status, message=message, metadata=md)

        if use_live:
            dashboard.start(refresh_per_second=4.0)
        try:
            result = self._get_researcher().research(
                question, emit=emit_cb, session_id=session.session_id, turn_id=session.session_id
            )
        except CouncilError as exc:
            self._set_activity("idle")
            self._emit_plan_phase("Research setup", status="error", message=str(exc), metadata={"question": question})
            runtime.save_session(session)
            self.console.print(f"[red]Research failed:[/] {_rich_escape(str(exc))}")
            return {"status": "failed", "error": str(exc)}
        finally:
            if use_live:
                dashboard.stop()
        self._set_activity("idle")
        self._render_research_report(result)
        runtime.save_session(session)
        return {
            "status": "ok",
            "question": question,
            "subqueries": result.get("subqueries", []),
            "sources": len(result.get("sources") or []),
            "report": result.get("report", ""),
        }

    def _render_research_report(self, result: dict[str, Any]) -> None:
        sources = result.get("sources") or []
        report = str(result.get("report") or "")
        self.console.print(
            Panel(
                Markdown(report or "*(no report generated)*"),
                title=f"Research Brief — {len(sources)} source(s)",
                border_style="cyan",
            )
        )
        verification = result.get("verification") or {}
        if verification:
            self.console.print(f"[dim]verification: {_rich_escape(str(verification))}[/]")

    def _render_plan(self, plan) -> None:
        table = Table(title=f"Plan: {plan.title}", box=box.SIMPLE)
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value")
        table.add_row("status", plan.status.value)
        table.add_row("revision", str(plan.revision))
        table.add_row("objective", plan.objective)
        table.add_row("risks", "\n".join(plan.risks) or "none")
        table.add_row("verification", "\n".join(plan.verification_plan) or "none")
        table.add_row("approvals", "\n".join(plan.required_approvals) or "none")
        table.add_row("tools", ", ".join(plan.proposed_tool_scope) or "none")
        table.add_row("files", "\n".join(plan.proposed_file_scope) or "none")
        self.console.print(table)
        steps = Table(title="Plan Steps", box=box.SIMPLE)
        steps.add_column("ID", style="cyan")
        steps.add_column("Title")
        steps.add_column("Status")
        for step in plan.steps:
            steps.add_row(step.id, step.title, self._display_plan_step_status(plan, step.status))
        self.console.print(steps)

    def _render_plan_review_menu_notice(self) -> None:
        choices = self._plan_review_choices()
        table = Table(title="Plan Review", box=box.SIMPLE_HEAVY)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Action")
        table.add_column("Meaning", overflow="fold")
        for index, choice in enumerate(choices, 1):
            table.add_row(str(index), choice.label, choice.description)
        if self._interactive_terminal():
            footer = "Use ↑/↓ to move, Space to select, Enter to confirm."
        else:
            footer = "Non-interactive mode: use /plan-approve, /plan-edit <feedback>, or /plan-exit."
        self.console.print(Panel(table, title="Awaiting Approval", subtitle=footer))

    def _render_planning_progress(self, plan) -> None:
        rows = [
            {
                "label": "Read workspace",
                "status": "success",
                "detail": self._require_session().cwd,
            },
            {
                "label": "Read tool registry",
                "status": "success",
                "detail": f"{len(self._require_runtime().tool_registry.list_handlers())} tool(s) available",
            },
            {
                "label": "Read git/data context",
                "status": "success",
                "detail": self._summarize_plan_context(plan.context_gathering),
            },
            {
                "label": "Draft plan",
                "status": "success",
                "detail": f"{len(plan.steps)} steps created",
            },
        ]
        self._render_progress_rows("Plan Formulation Progress", rows)

    def _render_plan_progress(self, plan, *, title: str = "Plan Progress") -> None:
        rows = []
        total = len(plan.steps)
        for index, step in enumerate(plan.steps, 1):
            deps = f" after {', '.join(step.dependencies)}" if getattr(step, "dependencies", None) else ""
            rows.append(
                {
                    "label": f"{index}/{total} {step.title}",
                    "status": step.status,
                    "detail": f"{step.purpose}{deps}".strip(),
                }
            )
        self._render_progress_rows(title, rows)

    def _render_event_progress(self, events: list[AgentEvent], *, title: str) -> None:
        rows = []
        for event in events:
            parsed = self._event_progress_row(event)
            if parsed is not None:
                rows.append(parsed)
        if rows:
            self._render_progress_rows(title, rows[-12:])

    def _event_progress_row(self, event: AgentEvent) -> dict[str, str] | None:
        payload = event.payload
        if event.type == AgentEventType.USER_TURN_STARTED:
            return {"label": "User turn", "status": "running", "detail": str(payload.get("text", ""))[:120]}
        if event.type == AgentEventType.MODEL_REQUEST_STARTED:
            return {
                "label": f"Model round {payload.get('round', 0)}",
                "status": "running",
                "detail": f"{payload.get('role', '')} -> {payload.get('model', '')}",
            }
        if event.type == AgentEventType.TOOL_CALL_REQUESTED:
            calls = payload.get("tool_calls") or []
            names = ", ".join(str(call.get("name", "")) for call in calls if isinstance(call, dict))
            return {"label": "Tool request", "status": "running", "detail": names or "tool call requested"}
        if event.type == AgentEventType.TOOL_CALL_STARTED:
            return {"label": "Tool started", "status": "running", "detail": str(payload.get("tool", ""))}
        if event.type == AgentEventType.TOOL_PROGRESS:
            skill = str(payload.get("skill") or payload.get("phase") or "tool")
            message = str(payload.get("message") or payload.get("state") or "running")
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            step = metadata.get("step") or ""
            total = metadata.get("total") or ""
            suffix = f" {step}/{total}" if step or total else ""
            return {"label": f"{skill}{suffix}", "status": "running", "detail": message}
        if event.type == AgentEventType.TOOL_CALL_COMPLETED:
            result = payload.get("result") or {}
            state = str(payload.get("state") or result.get("state") or "")
            status = "success" if state in {"done", "success", "completed"} else "failed"
            return {"label": "Tool completed", "status": status, "detail": f"{payload.get('tool', '')} ({state})"}
        if event.type == AgentEventType.ACTION_GRAPH_NODE_CREATED:
            return {
                "label": "Action graph",
                "status": "success",
                "detail": f"{payload.get('node_type')}:{payload.get('node_id')}",
            }
        if event.type == AgentEventType.PLAN_PHASE:
            status = str(payload.get("status", "running"))
            phase = str(payload.get("phase", "Plan"))
            message = str(payload.get("message", ""))
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            step = metadata.get("step") or metadata.get("current_step") or ""
            total = metadata.get("total") or metadata.get("total_steps") or ""
            label = f"{phase} {step}/{total}".strip(" /")
            return {"label": label, "status": status, "detail": message}
        if event.type == AgentEventType.USER_TURN_COMPLETED:
            return {"label": "User turn", "status": payload.get("status", "completed"), "detail": "turn completed"}
        if event.type == AgentEventType.COMMAND_STARTED:
            return {"label": "Command", "status": "running", "detail": str(payload.get("command", ""))}
        if event.type == AgentEventType.COMMAND_FINISHED:
            return {"label": "Command", "status": "success", "detail": str(payload.get("command", ""))}
        if event.type == AgentEventType.COMMAND_FAILED:
            return {"label": "Command", "status": "failed", "detail": str(payload.get("error", ""))}
        return None

    def _render_progress_rows(self, title: str, rows: list[dict[str, Any]]) -> None:
        total = len(rows)
        done_statuses = {"success", "done", "completed", "passed", "skipped"}
        failed_statuses = {"failed", "error", "cancelled"}
        done = sum(1 for row in rows if str(row.get("status", "")).lower() in done_statuses)
        failed = sum(1 for row in rows if str(row.get("status", "")).lower() in failed_statuses)
        current = next(
            (row for row in rows if str(row.get("status", "")).lower() not in done_statuses | failed_statuses),
            rows[-1] if rows else None,
        )
        table = Table(title=f"{title} [{done}/{total} done, {failed} failed]", box=box.SIMPLE)
        table.add_column("#", style="dim", justify="right", width=3)
        table.add_column("Status", no_wrap=True)
        table.add_column("Step")
        table.add_column("Detail", overflow="fold")
        for index, row in enumerate(rows, 1):
            status = str(row.get("status") or "pending").lower()
            table.add_row(
                str(index),
                self._format_status(status),
                str(row.get("label") or ""),
                str(row.get("detail") or ""),
            )
        bar = _text_progress_bar(done, total)
        current_text = f"Current: {current.get('label')} - {current.get('detail')}" if current else "Current: none"
        self.console.print(Panel(table, title=f"{bar} {done}/{total} steps", subtitle=current_text[:180]))

    def _render_plan_flow(self, plan) -> None:
        root = Tree(f"[bold]Plan Flowchart[/bold] {plan.title}")
        for index, step in enumerate(plan.steps, 1):
            deps = ", ".join(step.dependencies) if step.dependencies else "start"
            status = self._format_status(self._display_plan_step_status(plan, step.status))
            node = root.add(f"[cyan]{step.id}[/cyan] [{index}/{len(plan.steps)}] {step.title} - {status}")
            node.add(f"[dim]depends on:[/] {deps}")
            node.add(f"[dim]reads/tools:[/] {', '.join(step.tool_scope or []) or 'none'}")
            if step.file_scope:
                node.add(f"[dim]data/files:[/] {', '.join(step.file_scope)[:180]}")
        linear = " -> ".join(step.id for step in plan.steps) if plan.steps else "(empty)"
        self.console.print(Panel(root, title="Workflow Diagram", subtitle=f"linear view: {linear}"))

    @staticmethod
    def _display_plan_step_status(plan, status: str) -> str:
        normalized = str(status or "").lower()
        if getattr(plan, "status", None) == PlanStatus.DRAFT and normalized in {"pending", "planned", ""}:
            return "awaiting approval"
        return normalized or "pending"

    def _record_plan_action_graph(self, plan) -> None:
        session = self._require_session()
        plan_id = f"plan:{session.session_id}:{plan.revision}"
        self._record_action_graph_node(
            node_type="plan",
            node_id=plan_id,
            payload={
                "session_id": session.session_id,
                "title": plan.title,
                "objective": plan.objective,
                "revision": plan.revision,
                "steps": len(plan.steps),
            },
            score=0.8,
        )
        previous_step_id = ""
        for index, step in enumerate(plan.steps, 1):
            step_node_id = f"{plan_id}:{step.id}"
            self._record_action_graph_node(
                node_type="plan_step",
                node_id=step_node_id,
                payload={
                    "session_id": session.session_id,
                    "plan_id": plan_id,
                    "step_id": step.id,
                    "index": index,
                    "total": len(plan.steps),
                    "title": step.title,
                    "status": step.status,
                    "purpose": step.purpose,
                    "dependencies": list(step.dependencies),
                    "tool_scope": list(step.tool_scope),
                    "file_scope": list(step.file_scope),
                },
                score=0.6,
            )
            if previous_step_id:
                try:
                    self._require_runtime().action_graph.link_nodes(
                        "plan_step",
                        previous_step_id,
                        "plan_step",
                        step_node_id,
                        "next_step",
                        weight=1.0,
                        evidence={"session_id": session.session_id, "plan_revision": plan.revision},
                    )
                except Exception:
                    pass
            previous_step_id = step_node_id

    def _record_plan_review_graph(self, plan, *, action: str, choice: str = "", detail: str = "") -> str:
        session = self._require_session()
        return self._record_action_graph_node(
            node_type="plan_review",
            node_id=f"plan_review:{session.session_id}:{plan.revision}:{action}:{uuid.uuid4().hex[:8]}",
            payload={
                "session_id": session.session_id,
                "plan_revision": plan.revision,
                "plan_title": plan.title,
                "action": action,
                "choice": choice,
                "detail": detail[:500],
            },
            score=0.7,
        )

    def _mark_plan_review_ready(self, plan) -> None:
        for step in plan.steps:
            if str(step.status or "").lower() in {"pending", ""}:
                step.status = "planned"
        self._require_session().state.pending_work = []

    @staticmethod
    def _summarize_plan_context(context_items: list[str]) -> str:
        if not context_items:
            return "context recorded"
        summaries = []
        for item in context_items:
            text = str(item)
            if text.startswith("tools="):
                count_hint = ""
                if "... (+" in text:
                    count_hint = text[text.find("... (+") :].strip()
                summaries.append(f"tool registry read {count_hint}".strip())
            elif text.startswith("git_status="):
                status = text.split("=", 1)[1]
                lines = [line for line in status.splitlines() if line.strip()]
                summaries.append("git status clean" if status == "clean" else f"git status has {len(lines)} changed/untracked paths")
            else:
                summaries.append(text[:120])
        return "; ".join(summaries[-3:])

    def _render_action_graph(self, *, session: AgentSession | None = None, limit: int = 12) -> None:
        session = session or self._require_session()
        refs = session.action_graph_refs[-limit:]
        if not refs:
            self.console.print(Panel("No action graph nodes recorded yet.", title="Action Graph"))
            return
        root = Tree(f"[bold]Action Graph[/bold] session:{session.session_id[:12]}")
        flow = [f"session:{session.session_id[:8]}"]
        for index, ref in enumerate(refs, 1):
            label = f"{ref.node_type}:{ref.node_id}"
            flow.append(label)
            node = root.add(f"[cyan]{index}.[/cyan] {label} score={ref.score:.2f}")
            payload = dict(ref.payload or {})
            for key in ("name", "tool", "role", "active_role", "model", "summary"):
                if payload.get(key):
                    node.add(f"[dim]{key}:[/] {str(payload[key])[:180]}")
            if ref.node_key:
                node.add(f"[dim]node_key:[/] {ref.node_key}")
        self.console.print(Panel(root, title="Action Graph Flowchart", subtitle=" -> ".join(flow)[-220:]))

    def _cmd_graph(self, arg: str = "") -> dict[str, Any]:
        limit = 12
        raw = str(arg or "").strip()
        if raw:
            try:
                limit = max(1, int(raw.split()[0]))
            except ValueError:
                limit = 12
        session = self._require_session()
        self._render_action_graph(session=session, limit=limit)
        return {"ref_count": len(session.action_graph_refs), "shown": min(limit, len(session.action_graph_refs))}

    def _run_plan_review_loop(self, plan, *, previous_mode: str = "") -> dict[str, Any]:
        session = self._require_session()
        runtime = self._require_runtime()
        if not self._interactive_terminal() and self.plan_review_choice_provider is None:
            self._emit_plan_phase(
                "Plan review",
                status="running",
                message="awaiting explicit /plan-approve, /plan-edit, or /plan-exit",
                metadata={"revision": plan.revision, "interactive": False},
            )
            runtime.save_session(session)
            return {"mode": "non_interactive", "status": "awaiting_approval"}

        self._emit_plan_phase(
            "Plan review",
            status="running",
            message="approval menu opened",
            metadata={"revision": plan.revision, "interactive": True},
        )
        runtime.save_session(session)
        while session.state.plan and session.state.plan.status == PlanStatus.DRAFT:
            try:
                choice = self._select_plan_review_action()
            except KeyboardInterrupt:
                self._record_plan_review_graph(session.state.plan, action="interrupt", choice="exit", detail="Ctrl-C during plan review")
                self._emit_plan_phase(
                    "Plan review",
                    actor="user",
                    status="cancelled",
                    message="review interrupted; plan remains unapproved",
                    metadata={"revision": session.state.plan.revision},
                )
                session.state.custom_data["plan_review_interrupted"] = True
                runtime.save_session(session)
                self._set_activity("idle")
                self.console.print("\n[yellow]Plan review interrupted. The plan was not approved.[/]")
                return {"mode": "interactive", "status": "interrupted"}

            if choice.action == "approve":
                self._record_plan_review_graph(session.state.plan, action="selected", choice=choice.action, detail=choice.label)
                self._emit_plan_phase(
                    "Plan review",
                    actor="user",
                    status="success",
                    message="selected approve and implement",
                    metadata={"revision": session.state.plan.revision, "choice": choice.action},
                )
                self._cmd_plan_approve()
                return {"mode": "interactive", "status": "approved", "choice": choice.action}

            if choice.action == "fix":
                self._record_plan_review_graph(session.state.plan, action="selected", choice=choice.action, detail=choice.label)
                feedback = self._read_plan_refinement()
                if not feedback.strip():
                    self.console.print("[dim]No refinement entered; returning to plan review.[/]")
                    continue
                self._cmd_plan_edit(feedback)
                # The review loop redraws the live selector on the next iteration;
                # no separate static notice needed (it would stack on screen).
                continue

            self._record_plan_review_graph(session.state.plan, action="selected", choice=choice.action, detail=choice.label)
            self._cmd_plan_exit()
            return {"mode": "interactive", "status": "exited", "choice": choice.action}

        return {"mode": "interactive", "status": session.state.plan.status.value if session.state.plan else "none"}

    def _select_plan_review_action(self) -> PlanReviewChoice:
        choices = self._plan_review_choices()
        if self.plan_review_choice_provider is not None:
            raw = self.plan_review_choice_provider(choices)
            return self._coerce_plan_review_choice(raw, choices)
        return self._keyboard_select_plan_review_choice(choices)

    @staticmethod
    def _plan_review_choices() -> list[PlanReviewChoice]:
        return [
            PlanReviewChoice(
                key="1",
                label="Approve and implement plan",
                description="Approve this plan and switch to implementation-ready execution state.",
                action="approve",
            ),
            PlanReviewChoice(
                key="2",
                label="Approve and fix plan",
                description="Add feedback or missing requirements, then regenerate the plan for another review.",
                action="fix",
            ),
            PlanReviewChoice(
                key="3",
                label="Exit plan mode",
                description="Leave plan review without approving execution.",
                action="exit",
            ),
        ]

    @staticmethod
    def _coerce_plan_review_choice(raw: Any, choices: list[PlanReviewChoice]) -> PlanReviewChoice:
        if isinstance(raw, PlanReviewChoice):
            return raw
        text = str(raw or "").strip().lower()
        for choice in choices:
            if text in {choice.key.lower(), choice.action.lower(), choice.label.lower()}:
                return choice
        return choices[0]

    def _keyboard_select_plan_review_choice(self, choices: list[PlanReviewChoice]) -> PlanReviewChoice:
        def render(choice: PlanReviewChoice, is_selected: bool, index: int) -> str:
            cursor = "[cyan]›[/cyan]" if is_selected else " "
            marker = "[green]●[/green]" if is_selected else "○"
            label = f"[bold]{choice.label}[/bold]" if is_selected else choice.label
            return f"{cursor} {marker} {index + 1}. {label}\n    [dim]{choice.description}[/dim]"

        index = self._live_single_select(
            title="Plan Review",
            header="Choose how to proceed with this plan",
            items=choices,
            render_item=render,
        )
        # No edit_handler passed -> _live_single_select can only return an int.
        assert isinstance(index, int)
        return choices[index]

    def _render_edit_row(self, buffer: str, cursor: int) -> "list[Text]":
        """Render the focused row as an INLINE text field with a reverse-video
        block caret at ``cursor`` — drawn inside the same Live panel (no break-out
        prompt). Built as a styled ``Text`` via ``.append`` (NEVER ``from_markup``)
        so the user's literal '[' / ']' can never be parsed as Rich markup. A
        ``cell_len``-based horizontal window keeps the caret visible for long input
        (and accounts for wide CJK glyphs)."""
        prompt = "› "
        before = buffer[:cursor]
        at = buffer[cursor] if cursor < len(buffer) else " "
        after = buffer[cursor + 1:] if cursor < len(buffer) else ""
        avail = max(10, (self.console.size.width or 80) - 6)  # borders + padding
        lead = ""
        # Scroll the left edge so prompt + visible-before + caret cell fit.
        while before and cell_len(prompt + lead + before) + 1 > avail:
            before = before[1:]
            lead = "…"
        line = Text()
        line.append(prompt, style="cyan")
        if lead:
            line.append(lead, style="dim")
        line.append(before, style="bold")
        line.append(at, style="reverse bold")  # the visible cursor cell
        remaining = max(0, avail - cell_len(prompt + lead + before) - 1)
        shown_after = ""
        for ch in after:
            if cell_len(shown_after + ch) > remaining:
                break
            shown_after += ch
        line.append(shown_after, style="bold")
        return [line]

    def _live_single_select(
        self,
        *,
        title: str,
        header: str,
        items: list[Any],
        render_item: Any,
        footer: str = "↑/↓ or j/k move · 1-9 jump · Enter confirm · Ctrl-C cancel",
        editable: bool = False,
        custom_index: "Optional[int]" = None,
        edit_prefill: "Optional[Callable[[int], str]]" = None,
    ) -> "int | str":
        """Single-choice picker drawn in ONE in-place Rich ``Live`` panel.

        ``render_item(item, is_selected, index)`` returns the line(s) for an item.
        Returns the selected index; raises ``KeyboardInterrupt`` on Ctrl-C. Enter
        confirms the *highlighted* row. Updating a single ``Live`` surface —
        instead of ``console.print`` per keystroke — is what stops the menu from
        stacking down the screen.

        When ``editable`` is True the menu supports INLINE free-text editing inside
        the same panel: pressing **Tab** on a row (prefilled via ``edit_prefill``),
        or **Enter** on ``custom_index`` (empty), turns the focused row into an
        editable field; typing/Backspace/←/→ edit it; **Enter** submits the typed
        text (returned as a ``str``); **Esc** cancels back to the menu. No prompt
        is ever shown outside the panel. Callers that leave ``editable`` False can
        only ever get an ``int`` back, so existing int-only callers are unaffected.
        """
        if not items:
            raise ValueError("_live_single_select requires at least one item")
        # Drop any stale 1-byte pushback from a previous (possibly interrupted)
        # key read so it can't leak into this menu session.
        self._pending_byte = None
        selected = 0
        editing = False
        buffer = ""
        cursor = 0

        def build() -> Panel:
            # Build the panel body as a Group of per-line ``Text`` objects rather
            # than one ``"\n".join(markup)`` string. Rich measures each Text by
            # display *cells*, so wide CJK glyphs are accounted for correctly;
            # the joined-string path made Rich undercount lines, so its cursor-up
            # erase fell short and the stale highlight stayed on screen (the
            # "arrow keys don't move the selection" symptom).
            header_lines: list[Text] = []
            if header:
                for hline in str(header).split("\n"):
                    header_lines.append(Text.from_markup(f"[bold]{hline}[/bold]"))
                header_lines.append(Text(""))
            active_footer = "Enter submit · Esc cancel · ←/→ move · ⌫ delete" if editing else footer
            footer_lines = [Text(""), Text.from_markup(f"[dim]{active_footer}[/dim]")]
            blocks: list[list[Text]] = []
            for idx, item in enumerate(items):
                if editing and idx == selected:
                    blocks.append(self._render_edit_row(buffer, cursor))
                    continue
                block = render_item(item, idx == selected, idx)
                blocks.append([Text.from_markup(line) for line in str(block).split("\n")])

            # Bound the panel height. If the options are taller than the viewport,
            # Rich's in-place redraw cannot move the cursor above the top of the
            # screen (the menu would visibly "stack"). Show a sliding window of
            # options CENTERED on the selection so the highlighted row is always
            # visible — a plain top-slice could push the selection off-screen and
            # reproduce the original "can't see the highlight move" symptom.
            avail = max(6, (self.console.size.height or 24) - 2)
            budget = max(1, avail - len(header_lines) - len(footer_lines) - 2)  # 2 for ↑/↓ markers
            lo = hi = selected
            used = len(blocks[selected])
            while True:
                grew = False
                if hi + 1 < len(blocks) and used + len(blocks[hi + 1]) <= budget:
                    hi += 1
                    used += len(blocks[hi])
                    grew = True
                if lo - 1 >= 0 and used + len(blocks[lo - 1]) <= budget:
                    lo -= 1
                    used += len(blocks[lo])
                    grew = True
                if not grew:
                    break

            lines: list[Text] = list(header_lines)
            if lo > 0:
                lines.append(Text(f"   ↑ {lo} more", style="dim"))
            for i in range(lo, hi + 1):
                lines.extend(blocks[i])
            if hi < len(blocks) - 1:
                lines.append(Text(f"   ↓ {len(blocks) - 1 - hi} more", style="dim"))
            lines.extend(footer_lines)
            return Panel(Group(*lines), title=title, border_style="yellow")

        # transient=True erases the menu once answered so successive questions
        # never stack. auto_refresh=False means *only* our keypress-driven
        # live.update redraws — nothing repaints under the prompt.
        #
        # redirect_stdout/redirect_stderr=False is LOAD-BEARING: Rich's Live
        # defaults both to True, which swaps ``sys.stdout`` for a ``FileProxy``
        # whose ``isatty()`` returns False even on a real TTY. ``_read_single_key``
        # gates on ``_interactive_terminal()`` (``stdin.isatty() and
        # stdout.isatty()``); under the redirect that flips to False and the menu
        # raises ``KeyboardInterrupt`` on itself before reading a single key
        # (the live ``/plan`` crash). The menu only ever calls ``live.update`` and
        # ``_read_single_key`` — it never writes to ``sys.stdout`` — so there is
        # nothing for Rich to capture; leaving stdout un-swapped is both correct
        # and necessary.
        with Live(
            build(),
            console=self.console,
            auto_refresh=False,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        ) as live:
            while True:
                live.update(build(), refresh=True)
                key = self._read_single_key()
                if editing:
                    # --- inline edit mode: keystrokes edit the buffer in-panel ---
                    if key in ("\r", "\n"):
                        if buffer.strip():
                            return buffer.strip()        # typed text IS the answer
                        editing, buffer, cursor = False, "", 0  # empty -> back to menu
                    elif key == "\x1b":                  # bare Esc -> cancel edit
                        editing, buffer, cursor = False, "", 0
                    elif key in ("\x7f", "\x08"):        # Backspace / DEL
                        if cursor > 0:
                            buffer = buffer[:cursor - 1] + buffer[cursor:]
                            cursor -= 1
                    elif key == "\x1b[D":                # Left
                        cursor = max(0, cursor - 1)
                    elif key == "\x1b[C":                # Right
                        cursor = min(len(buffer), cursor + 1)
                    elif key == "\t":
                        pass                              # ignore Tab while editing
                    elif key and not key.startswith("\x1b") and key.isprintable():
                        buffer = buffer[:cursor] + key + buffer[cursor:]
                        cursor += len(key)
                    # else: up/down/other CSI/swallowed-paste -> ignored while editing
                    continue
                # --- navigation mode ---
                if key in ("\r", "\n"):
                    if editable and custom_index is not None and selected == custom_index:
                        editing, buffer, cursor = True, "", 0   # custom row -> empty edit
                        continue
                    return selected
                if key in ("\x1b[A", "k"):
                    selected = (selected - 1) % len(items)
                elif key in ("\x1b[B", "j"):
                    selected = (selected + 1) % len(items)
                elif key == "\t" and editable:
                    buffer = (edit_prefill(selected) if edit_prefill else "") or ""
                    cursor = len(buffer)
                    editing = True
                elif key and key.isdigit():
                    idx = int(key) - 1
                    if 0 <= idx < len(items):
                        return idx

    def _read_plan_refinement(self) -> str:
        self._set_activity("plan refinement")
        if self.prompt_session is not None:
            return self.prompt_session.prompt("plan feedback › ")
        return self.console.input("[bold yellow]plan feedback>[/] ")

    # Bracketed-paste markers (terminals wrap pastes in these when the mode is
    # enabled). We drain paste payloads so their bytes can never be mistaken for
    # menu keystrokes. ``_ESC_TIMEOUT`` is the per-byte wait while assembling an
    # escape sequence — generous enough for slow SSH/PTY links.
    _PASTE_START = "\x1b[200~"
    _PASTE_END = b"\x1b[201~"
    _ESC_TIMEOUT = 0.05

    def _read_single_key(self) -> str:
        """Read one key (or escape sequence) from the TTY without Python buffering.

        Reads raw bytes with ``os.read`` directly on the file descriptor instead
        of ``sys.stdin.read``. ``sys.stdin`` is a buffered ``TextIOWrapper`` that
        drains the fd into its own buffer while returning a single char, which
        defeats ``select`` on the fd — that is exactly why the old selector
        misread every arrow key (a 3-byte ``\\x1b[A`` burst arrived as three
        separate, non-matching "keys"). Reading raw bytes from the fd and polling
        the *same* fd keeps reader and poller in sync.
        """
        if not self._interactive_terminal():
            raise KeyboardInterrupt
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            # cbreak (not raw): keeps OPOST so concurrent output stays aligned and
            # keeps ISIG so Ctrl-C still raises; restore immediately with TCSANOW.
            tty.setcbreak(fd, termios.TCSANOW)
            # 1-byte pushback: a bare Esc followed by an unrelated byte (e.g. Esc
            # then Enter) must surface as TWO keys, not a merged "\x1b\r". We read
            # the trailing byte to disambiguate Esc-vs-CSI, then stash it here.
            pending = getattr(self, "_pending_byte", None)
            if pending is not None:
                self._pending_byte = None
                ch = pending
            else:
                ch = os.read(fd, 1)
            if not ch or ch == b"\x03":  # EOF, or Ctrl-C if ISIG is disabled upstream
                raise KeyboardInterrupt
            if ch == b"\x1b":
                # Parse a CSI/SS3 escape sequence precisely: consume the arrow-key
                # bytes (e.g. \x1b[A) up to the final byte but NOT any unrelated
                # byte that happens to follow (e.g. a pending Enter in \x1b[B\r).
                seq = ch
                ready, _, _ = select.select([fd], [], [], self._ESC_TIMEOUT)
                if ready:
                    nxt = os.read(fd, 1)
                    if nxt in (b"[", b"O"):  # CSI or SS3 introducer
                        seq += nxt
                        while len(seq) < 16:
                            ready, _, _ = select.select([fd], [], [], self._ESC_TIMEOUT)
                            if not ready:
                                break
                            tail = os.read(fd, 1)
                            if not tail:
                                break
                            seq += tail
                            if 0x40 <= tail[0] <= 0x7E:  # CSI/SS3 final byte
                                break
                    else:
                        # Bare Esc: ``nxt`` is a SEPARATE key, not part of the
                        # sequence. Buffer it and return Esc alone, so Esc+Enter
                        # reads as Esc THEN Enter (a merged "\x1b\r" matches no
                        # handler and would hang an inline editor waiting for a key).
                        self._pending_byte = nxt
                        return "\x1b"
                decoded = seq.decode("utf-8", errors="replace")
                if decoded == self._PASTE_START:
                    # Swallow the whole paste payload; never treat it as keystrokes.
                    self._drain_bracketed_paste(fd)
                    return ""
                return decoded
            # Complete a multi-byte UTF-8 character if continuation bytes follow.
            first = ch[0]
            extra = 3 if first >= 0xF0 else 2 if first >= 0xE0 else 1 if first >= 0xC0 else 0
            for _ in range(extra):
                ready, _, _ = select.select([fd], [], [], self._ESC_TIMEOUT)
                if not ready:
                    break
                ch += os.read(fd, 1)
            return ch.decode("utf-8", errors="replace")
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old)

    def _drain_bracketed_paste(self, fd: int) -> None:
        """Consume a bracketed-paste payload up to (and including) the ``\\x1b[201~``
        end marker, one byte at a time so we stop exactly at the marker and leave
        any genuine keystroke that follows the paste in the input stream."""
        drained = b""
        while not drained.endswith(self._PASTE_END) and len(drained) < 1_000_000:
            ready, _, _ = select.select([fd], [], [], 0.1)
            if not ready:
                break
            byte = os.read(fd, 1)
            if not byte:
                break
            drained += byte

    def _emit_plan_phase(
        self,
        phase: str,
        *,
        actor: str = "biobank",
        status: str = "running",
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AgentEvent:
        event = AgentEvent.make(
            AgentEventType.PLAN_PHASE,
            session_id=self._require_session().session_id,
            phase=phase,
            actor=actor,
            status=status,
            message=message,
            metadata=dict(metadata or {}),
        )
        self._record_event(event)
        return event

    def _mark_plan_execution_ready(self, plan) -> None:
        for step in plan.steps:
            if step.id in {"context", "design"}:
                step.status = "done"
            elif step.status not in {"done", "completed", "failed", "skipped"}:
                step.status = "pending"
        self._require_session().state.pending_work = [
            step.to_dict() for step in plan.steps if step.status not in {"done", "completed", "skipped"}
        ]

    def _sync_plan_execution_from_events(self, events: list[AgentEvent]) -> bool:
        session = self._require_session()
        plan = session.state.plan
        if plan is None:
            return False
        completed = [event for event in events if event.type == AgentEventType.TOOL_CALL_COMPLETED]
        errors = [event for event in events if event.type in {AgentEventType.TOOL_ERROR, AgentEventType.ERROR}]
        if not completed and not errors:
            return False
        failed = bool(errors) or any(
            str(event.payload.get("state", "")).lower() in {"failed", "cancelled"}
            for event in completed
        )
        # Advance the earliest not-yet-finished step. Works for council plans with
        # arbitrary step ids (not just the legacy "execute" id). Full DAG-aware
        # execution is a future enhancement; this keeps progress honest in the meantime.
        target = self._first_incomplete_step(plan)
        if target is not None:
            target.status = "failed" if failed else "done"
        if failed:
            plan.status = PlanStatus.FAILED
        elif plan.status == PlanStatus.APPROVED:
            plan.status = PlanStatus.EXECUTING
        session.state.pending_work = [
            step.to_dict() for step in plan.steps if step.status not in {"done", "completed", "skipped"}
        ]
        self._emit_plan_phase(
            "Execution",
            actor="plan",
            status="failed" if failed else "success",
            message=(f"step '{target.id}' failed" if failed else f"step '{target.id}' completed from runtime tool results")
            if target is not None
            else ("execution failed" if failed else "execution step completed"),
            metadata={
                "current_step": target.id if target is not None else "",
                "total_steps": len(plan.steps),
                "completed_steps": sum(1 for step in plan.steps if step.status in {"done", "completed"}),
            },
        )
        return True

    _TERMINAL_STEP_STATUS = {"done", "completed", "failed", "skipped", "unverified"}

    @classmethod
    def _first_incomplete_step(cls, plan):
        """First step not in a terminal state, or ``None`` if every step is
        already terminal (so callers never clobber a finished step)."""
        for step in plan.steps:
            if step.status not in cls._TERMINAL_STEP_STATUS:
                return step
        return None

    @staticmethod
    def _verification_step(plan):
        """Verification target for /verify: the LAST step whose id/title/purpose
        names verification/validation, else the final step. Narrow keywords avoid
        clobbering an early 'check data'-style step."""
        if not plan.steps:
            return None
        match = None
        for step in plan.steps:
            text = f"{step.id} {step.title} {step.purpose}".lower()
            if "verif" in text or "validat" in text:
                match = step  # keep the last verification-named step
        return match or plan.steps[-1]

    @staticmethod
    def _mark_plan_step(plan, step_id: str, status: str) -> None:
        for step in plan.steps:
            if step.id == step_id:
                step.status = status
                break

    @staticmethod
    def _format_status(status: str) -> str:
        normalized = str(status or "pending").lower()
        if normalized in {"success", "done", "completed", "passed", "approved"}:
            return "[green]done[/green]" if normalized != "approved" else "[green]approved[/green]"
        if normalized in {"running", "executing"}:
            return "[yellow]running[/yellow]"
        if normalized in {"planned", "awaiting approval", "review"}:
            return "[yellow]awaiting approval[/yellow]"
        if normalized in {"failed", "error"}:
            return "[red]failed[/red]"
        if normalized == "skipped":
            return "[dim]skipped[/dim]"
        if normalized == "rejected":
            return "[red]rejected[/red]"
        return "[dim]pending[/dim]"

    def _get_completion_gate(self) -> CompletionGate:
        if self._completion_gate is None:
            runtime = self._require_runtime()
            self._completion_gate = CompletionGate(runtime.provider_router, runtime.config, clock=time.time)
        return self._completion_gate

    def _build_completion_evidence(self, session) -> tuple[str, bool]:
        """Summarize executed tools + verifications from session events, and detect
        whether a report artifact exists."""
        tool_calls: list[str] = []
        verifications: list[str] = []
        report_present = False
        _ok_states = {"done", "success", "completed"}
        for raw in session.events:
            try:
                event = AgentEvent.from_dict(raw) if isinstance(raw, dict) else raw
            except Exception:
                continue
            if event.type == AgentEventType.TOOL_CALL_COMPLETED:
                tool = str(event.payload.get("tool", "?"))
                state = str(event.payload.get("state", "")).lower()
                tool_calls.append(f"{tool}:{state}")
                # A report only counts if the report tool succeeded.
                if "report" in tool.lower() and state in _ok_states:
                    report_present = True
            elif event.type == AgentEventType.VERIFICATION_COMPLETED:
                ver = event.payload.get("verification") or {}
                verifications.append(f"{ver.get('command', '')}={ver.get('status', '')}")
        if not report_present:
            try:
                report_dir = self._session_report_dir(session)
                if report_dir.exists():
                    scanned = 0
                    for path in report_dir.rglob("*"):  # bounded scan to cap worst-case I/O
                        scanned += 1
                        if scanned > 5000:
                            break
                        if path.is_file() and path.suffix.lower() in {".md", ".html", ".pdf", ".docx"}:
                            report_present = True
                            break
            except Exception:
                pass
        summary = "Tools run: " + (", ".join(tool_calls) or "none") + "\nVerifications: " + (", ".join(verifications) or "none")
        return summary, report_present

    def _assess_plan_completion(self) -> dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session()
        plan = session.state.plan
        if plan is None:
            self.console.print("[dim]No active plan. Use /plan <task> first, or /verify <command> to run a specific check.[/]")
            return {"status": "no_plan"}
        self._set_activity("verifying completion")
        dashboard = PlanRunDashboard(self.console, title="Completion check")
        use_live = self._interactive_terminal()

        def emit_cb(stage: str, *, status: str = "running", message: str = "", metadata: dict[str, Any] | None = None) -> None:
            md = dict(metadata or {})
            if use_live:
                try:
                    dashboard.record(stage, actor="biobank", status=status, message=message, metadata=md)
                except Exception:
                    pass
            if "subagent" not in md or status == "error":
                self._emit_plan_phase(stage, status=status, message=message, metadata=md)

        evidence, report_present = self._build_completion_evidence(session)
        gate = self._get_completion_gate()
        if use_live:
            dashboard.start(refresh_per_second=4.0)
        try:
            assessment = gate.assess(
                plan.objective, plan, evidence_summary=evidence, report_present=report_present,
                emit=emit_cb, session_id=session.session_id, turn_id=session.session_id,
            )
        finally:
            if use_live:
                dashboard.stop()
        self._render_completion_assessment(assessment)

        if not assessment.accepted:
            attempts = len(session.state.repair_attempts)
            max_attempts = int(getattr(self.settings, "max_repair_attempts", 2))
            if attempts < max_attempts:
                runtime.record_repair_attempt(
                    session,
                    reason=(assessment.reasons or "; ".join(assessment.missing) or "goal not yet accepted")[:240],
                    attempt=attempts + 1,
                    max_attempts=max_attempts,
                    status="queued" if attempts + 1 < max_attempts else "bounded_stop",
                )
                proposal = gate.propose_repair(
                    plan.objective, plan, assessment, emit=emit_cb, session_id=session.session_id, turn_id=session.session_id
                )
                if proposal:
                    self.console.print(Panel(Markdown(proposal), title="Proposed repair (bounded)", border_style="yellow"))
            else:
                self.console.print(f"[yellow]Repair budget exhausted ({attempts}/{max_attempts}); recording the limitation rather than retrying.[/]")
        self._set_activity("idle")
        runtime.save_session(session)
        return {"status": "accepted" if assessment.accepted else "incomplete", "assessment": assessment.to_dict()}

    def _render_completion_assessment(self, assessment) -> None:
        status = "[green]ACCEPTED[/green]" if assessment.accepted else "[yellow]NOT YET ACCEPTED[/yellow]"
        rows = [f"goal: {status}"]
        if assessment.reasons:
            rows.append(f"reasons: {assessment.reasons}")
        if assessment.missing:
            rows.append("missing:\n  - " + "\n  - ".join(assessment.missing))
        if assessment.warnings:
            rows.append("warnings:\n  - " + "\n  - ".join(assessment.warnings))
        rows.append(f"report required: {assessment.needs_report} · present: {assessment.report_present} · judged: {assessment.goal_checked}")
        self.console.print(Panel("\n".join(rows), title="Completion Assessment", border_style="green" if assessment.accepted else "yellow"))

    def _cmd_verify(self, arg: str = "") -> dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session()
        command = arg.strip()
        if not command:
            # No command -> assess overall plan completion (report-contract,
            # goal-acceptance, warnings, bounded LLM repair) on the run_turn model.
            return self._assess_plan_completion()
        try:
            results = runtime.run_verification_commands(
                session,
                [command],
                max_attempts=2,
                timeout_s=120,
            )
            final = results[-1] if results else None
            status = final.status if final is not None else "unknown"
            if session.state.plan is not None:
                vstep = self._verification_step(session.state.plan)
                if vstep is not None:
                    vstep.status = "done" if status == "passed" else "failed"
                runtime.save_session(session)
            self.console.print(f"Verification {status}: {command}")
            if session.state.plan is not None:
                self._render_plan_progress(session.state.plan, title="Plan Execution Progress")
            return {
                "status": status,
                "command": command,
                "results": [result.to_dict() for result in results],
            }
        except Exception as exc:
            result = runtime.record_verification(session, command=command, status="failed", summary=str(exc))
            if session.state.plan is not None:
                vstep = self._verification_step(session.state.plan)
                if vstep is not None:
                    vstep.status = "failed"
                runtime.save_session(session)
            self.console.print(f"[red]Verification failed:[/] {_rich_escape(str(exc))}")
            if session.state.plan is not None:
                self._render_plan_progress(session.state.plan, title="Plan Execution Progress")
            return {"status": "failed", "command": command, "results": [result.to_dict()], "error": str(exc)}

    def _cmd_compact(self) -> None:
        runtime = self._require_runtime()
        session = self._require_session()
        summary = session.state.summary or ""
        runtime.compact_session(session, summary)
        runtime.save_session(session)
        self.console.print(f"[green]Compacted session context.[/] {_rich_escape(str(session.state.summary))}")
        return {"summary": session.state.summary}

    def _cmd_goal(self, arg: str = "") -> None:
        runtime = self._require_runtime()
        session = self._require_session()
        if arg:
            goal = runtime.set_goal(session, arg)
            runtime.save_session(session)
            self._render_goal(goal)
        else:
            if session.state.goal:
                self._render_goal(session.state.goal)
            else:
                self.console.print("No active goal.")
        return session.state.goal.to_dict() if session.state.goal else {"status": "none"}

    def _render_goal(self, goal) -> None:
        table = Table(title="Goal", box=box.SIMPLE)
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value")
        table.add_row("objective", goal.objective)
        table.add_row("status", goal.status.value)
        table.add_row("completion", goal.completion_state)
        table.add_row("revision", str(goal.revision))
        table.add_row("risks", "\n".join(goal.risks) or "none")
        table.add_row("verification", "\n".join(goal.verification_commands) or "none")
        self.console.print(table)
        tasks = Table(title="Goal Tasks", box=box.SIMPLE)
        tasks.add_column("ID", style="cyan")
        tasks.add_column("Title")
        tasks.add_column("Status")
        for task in goal.tasks:
            tasks.add_row(str(task.get("id", "")), str(task.get("title", "")), str(task.get("status", "")))
        self.console.print(tasks)

    def _cmd_resume(self, arg: str = "") -> dict[str, Any] | list[dict[str, Any]]:
        runtime = self._require_runtime()
        sessions = runtime.list_sessions()
        current = self.session
        skip_current = (
            current is not None
            and not current.turns
            and current.state.goal is None
            and current.state.plan is None
        )
        current_id = current.session_id if current is not None and skip_current else ""
        resume_candidates = [item for item in sessions if item.session_id != current_id] or sessions
        meaningful_candidates = [
            item for item in resume_candidates
            if item.turns or item.state.goal is not None or item.state.plan is not None
        ]
        resume_candidates = meaningful_candidates or resume_candidates
        if arg.strip() and arg.strip() not in {"--last", "last"}:
            requested = arg.strip()
            self.session = runtime.load_session(arg.strip())
            runtime.config = self.session.config
            runtime.provider_router.config = self.session.config
            runtime.set_approval_profile(self.session.config.approval_profile)
            self._reroot_job_manager(self.session)
            self._warn_if_workspace_changed(self.session)
            self._record_event(
                AgentEvent.make(
                    AgentEventType.RESUME_COMPLETED,
                    session_id=self.session.session_id,
                    mode="by_id",
                )
            )
            runtime.save_session(self.session)
            self.console.print(f"[green]Resumed session:[/] {self.session.session_id}")
            return {"session_id": self.session.session_id, "requested": requested, "mode": "by_id"}
        if arg.strip() in {"--last", "last"} and resume_candidates:
            self.session = resume_candidates[0]
            runtime.config = self.session.config
            runtime.provider_router.config = self.session.config
            runtime.set_approval_profile(self.session.config.approval_profile)
            self._reroot_job_manager(self.session)
            self._warn_if_workspace_changed(self.session)
            self._record_event(
                AgentEvent.make(
                    AgentEventType.RESUME_COMPLETED,
                    session_id=self.session.session_id,
                    mode="latest",
                )
            )
            runtime.save_session(self.session)
            self.console.print(f"[green]Resumed latest session:[/] {self.session.session_id}")
            return {"session_id": self.session.session_id, "mode": "latest"}
        table = Table(title="Saved Sessions", box=box.SIMPLE)
        table.add_column("Session")
        table.add_column("Title")
        table.add_column("Turns")
        table.add_column("Goal")
        table.add_column("Plan")
        for item in sessions[:10]:
            table.add_row(
                item.session_id,
                item.title,
                str(len(item.turns)),
                item.state.goal.completion_state if item.state.goal else "-",
                item.state.plan.status.value if item.state.plan else "-",
            )
        self.console.print(table)
        return [{"session_id": item.session_id, "title": item.title} for item in sessions[:10]]

    def _warn_if_workspace_changed(self, session: AgentSession) -> None:
        runtime = self._require_runtime()
        if not session.state.checkpoints:
            return
        last = session.state.checkpoints[-1]
        current = runtime.workspace_fingerprint(session.cwd)
        if last.workspace_fingerprint and current != last.workspace_fingerprint:
            self.console.print("[yellow]Workspace has changed since the last checkpoint.[/]")

    def _cmd_new(self, arg: str = "") -> None:
        runtime = self._require_runtime()
        cwd = self._require_session().cwd if self.session is not None else str(Path.cwd())
        self.session = runtime.create_session(title=arg.strip() or "Interactive session", cwd=cwd)
        self._reroot_job_manager(self.session)
        runtime.save_session(self.session)
        self.console.print(f"[green]New session:[/] {self.session.session_id}")
        return {"session_id": self.session.session_id}

    def _cmd_fork(self, arg: str = "") -> None:
        runtime = self._require_runtime()
        source = runtime.load_session(arg.strip()) if arg.strip() else self._require_session()
        forked = runtime.create_session(title=f"{source.title} (fork)", cwd=source.cwd)
        forked.turns = list(source.turns)
        forked.state.summary = source.state.summary
        runtime.save_session(forked)
        self.session = forked
        self.console.print(f"[green]Forked session:[/] {forked.session_id}")
        return {"session_id": forked.session_id, "source": source.session_id}

    def _cmd_cd(self, arg: str = "") -> dict[str, Any]:
        """Switch the active workspace directory mid-session (issue #4).

        Subsequent tool/shell/file paths and generated outputs resolve under the
        new directory; the previous workspace is kept as an extra readable root so
        earlier files stay reachable when moving between projects."""
        runtime = self._require_runtime()
        session = self._require_session()
        target = str(arg or "").strip().strip('"').strip("'")
        if not target:
            extras = session.state.custom_data.get("workspace_extra_roots", []) or []
            self.console.print(f"[dim]workspace:[/] {session.cwd}")
            if extras:
                self.console.print(f"[dim]also readable:[/] {', '.join(extras)}")
            self.console.print("[dim]usage: /cd <path>[/]")
            return {"status": "ok", "workspace": session.cwd, "extra_roots": list(extras)}
        candidate = Path(target).expanduser()
        if not candidate.is_dir():
            self.console.print(f"[red]/cd: not a directory:[/] {target}")
            return {"status": "error", "error": "not_a_directory", "path": target}
        new_cwd = str(candidate.resolve())
        old_cwd = str(session.cwd)
        if new_cwd == old_cwd:
            self.console.print(f"[dim]Already in[/] {new_cwd}")
            return {"status": "ok", "workspace": new_cwd}
        # Keep the previous workspace readable so cross-project work isn't stranded.
        extras = list(session.state.custom_data.get("workspace_extra_roots", []) or [])
        if old_cwd not in extras:
            extras.append(old_cwd)
        session.state.custom_data["workspace_extra_roots"] = extras
        session.cwd = new_cwd
        try:
            session.workspace_fingerprint = runtime.workspace_fingerprint(new_cwd)
        except Exception:
            pass
        # Re-root background jobs under the new workspace.
        self._reroot_job_manager(session)
        runtime.save_session(session)
        try:
            self._record_event(
                AgentEvent.make(
                    AgentEventType.PLAN_PHASE,
                    session_id=session.session_id,
                    phase="Workspace",
                    actor="biobank",
                    status="success",
                    message=f"workspace -> {new_cwd}",
                    metadata={"old": old_cwd, "new": new_cwd, "extra_roots": extras},
                )
            )
        except Exception:
            pass
        self.console.print(
            f"[green]Workspace:[/] {new_cwd}\n"
            f"[dim]New files and reports go here. Previous dir kept readable: {old_cwd}[/]"
        )
        return {"status": "ok", "workspace": new_cwd, "extra_roots": extras}

    def _cmd_jobs(self, *_args: Any) -> dict[str, Any]:
        """List background jobs (run_job) with state, elapsed time and log path."""
        jm = self.job_manager
        recs = jm.list() if jm is not None else []
        if not recs:
            self.console.print(
                "[dim]No background jobs.[/]\n"
                "[dim]For long bioinformatics commands, ask the agent to use the run_job tool; "
                "then monitor with /jobs and /job-tail <job-id>.[/]"
            )
            return {"status": "ok", "jobs": []}
        table = Table(title="[bold]Background Jobs[/bold]", box=box.SIMPLE)
        table.add_column("Job"); table.add_column("State"); table.add_column("Label")
        table.add_column("Elapsed", justify="right"); table.add_column("Log", overflow="fold")
        for r in recs:
            elapsed = round((r.ended_at or time.time()) - (r.started_at or time.time()), 1)
            table.add_row(r.job_id, self._format_status(r.state), str(r.label or r.tool)[:40],
                          f"{elapsed}s", str(r.log_path))
        self.console.print(table)
        self.console.print("[dim]Tail one with /job-tail <job-id>[/]")
        return {"status": "ok", "jobs": [r.to_dict() for r in recs]}

    def _cmd_job_tail(self, arg: str = "") -> dict[str, Any]:
        """Show the tail of a background job's log."""
        jm = self.job_manager
        job_id = str(arg or "").strip()
        if jm is None or not job_id:
            self.console.print("[dim]usage: /job-tail <job-id>[/]")
            return {"status": "error", "error": "job_id is required"}
        rec = jm.status(job_id)
        if rec is None:
            self.console.print(f"[red]No such job:[/] {job_id}")
            return {"status": "error", "error": "not_found"}
        body = jm.tail(job_id, n_lines=60) or "[dim](no output yet)[/]"
        self.console.print(Panel(_rich_escape(body), title=f"job {job_id} [{rec.state}] — {rec.log_path}",
                                 border_style="blue"))
        return {"status": "ok", "job_id": job_id, "state": rec.state}

    def _cmd_diff(self, *_args: Any) -> None:
        session = self._require_session()
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=session.cwd,
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception as exc:
            self.console.print(f"[yellow]Diff unavailable:[/] {_rich_escape(str(exc))}")
            return
        body = result.stdout.strip() or "No git status changes detected."
        self.console.print(Panel(body, title="Workspace Diff"))
        return {"status": "ok", "diff": body}

    def _cmd_permissions(self, arg: str = "") -> None:
        runtime = self._require_runtime()
        if arg:
            runtime.set_approval_profile(arg.strip())
        self.console.print(f"Permission mode: [cyan]{runtime.config.approval_profile}[/]")
        return {"approval_profile": runtime.config.approval_profile}

    def _cmd_doctor(self, *_args: Any) -> None:
        runtime = self._require_runtime()
        session = self._require_session()
        tool_summary = self._tool_summary_snapshot()
        wgs = self._wgs_readiness_snapshot()
        checks = [
            ("python", sys.version.split()[0]),
            ("workspace", session.cwd),
            ("reports_dir", _path_state(self._reports_root(session))),
            ("memory_dir", _path_state(Path(self.settings.memory_dir))),
            ("sessions", _path_state(runtime.session_store.root)),
            ("llm_model", self.settings.llm_model),
            ("api_key", "configured" if bool(getattr(self.settings, "llm_api_key", "")) else "missing"),
            ("permission_mode", runtime.config.approval_profile),
            ("tools", tool_summary["summary"]),
            ("wgs_vcf_samples", str(wgs.get("n_samples", 0))),
            ("wgs_indexed", f"{wgs.get('n_indexed', 0)}/{wgs.get('n_samples', 0)}"),
            ("wgs_workflow_mode", str(wgs.get("workflow_mode", "unknown"))),
            ("wgs_standard_missing", ", ".join(str(x) for x in (wgs.get("missing_for_standard_workflow") or []) if x) or "none"),
        ]
        table = Table(title="Doctor", box=box.SIMPLE)
        table.add_column("Check", style="cyan")
        table.add_column("Result")
        for key, value in checks:
            table.add_row(key, value)
        self.console.print(table)
        return {**dict(checks), "wgs": wgs}

    def _cmd_agent(self, *_args: Any) -> None:
        session = self._require_session()
        runtime = self._require_runtime()
        arg = str(_args[0]) if _args else ""
        self._update_active_role(arg)
        table = Table(title="Active Agent", box=box.SIMPLE)
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value")
        table.add_row("session", session.session_id)
        table.add_row("workspace", session.cwd)
        table.add_row("thread", session.state.active_turn_id or "idle")
        table.add_row("active role", runtime.config.active_role)
        table.add_row("primary", runtime.config.primary_model)
        table.add_row("planner", runtime.config.planner_model)
        table.add_row("critic", runtime.config.critic_model)
        table.add_row("summarizer", runtime.config.summarizer_model)
        table.add_row("safety", runtime.config.safety_reviewer_model)
        table.add_row("permission mode", runtime.config.approval_profile)
        table.add_row("api key", "configured" if getattr(self.settings, "llm_api_key", "") else "missing")
        table.add_row("tools", str(len(runtime.tool_registry.list_handlers())))
        table.add_row("mcp tools", str(len(self.mcp_manager.handlers) if self.mcp_manager else 0))
        self.console.print(table)
        return {
            "session_id": session.session_id,
            "active_role": runtime.config.active_role,
            "primary": runtime.config.primary_model,
            "planner": runtime.config.planner_model,
            "critic": runtime.config.critic_model,
        }

    def _cmd_models_pool(self, *_args: Any) -> dict[str, str]:
        runtime = self._require_runtime()
        rows = {
            "primary": runtime.config.primary_model,
            "planner": runtime.config.planner_model,
            "critic": runtime.config.critic_model,
            "summarizer": runtime.config.summarizer_model,
            "safety": runtime.config.safety_reviewer_model,
        }
        table = Table(title="Configured Model Pool", box=box.SIMPLE)
        table.add_column("Role", style="cyan")
        table.add_column("Model")
        for role, model in rows.items():
            table.add_row(role, model)
        self.console.print(table)
        return rows

    def _cmd_models_available(self, *_args: Any) -> dict[str, Any]:
        from biobank_agent.llm import LLMClient

        runtime = self._require_runtime()
        client = LLMClient(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            model=runtime.config.primary_model,
        )
        if hasattr(client, "tool_call_content_mode"):
            client.tool_call_content_mode = getattr(self.settings, "tool_call_content_mode", "null")
        models = client.list_models(refresh=True)
        table = Table(title=f"Available Models ({len(models)})", box=box.SIMPLE)
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("Model")
        for idx, model in enumerate(models[:80], start=1):
            table.add_row(str(idx), model)
        if len(models) > 80:
            table.add_row("...", f"+{len(models) - 80} more")
        self.console.print(table)
        return {"count": len(models), "models": models}

    def _cmd_subagents(self, *_args: Any) -> None:
        runtime = self._require_runtime()
        session = self._require_session()
        arg = " ".join(str(part) for part in _args if part is not None).strip()
        if not arg or arg.lower() in {"list", "show", "--list"}:
            rows = [
                ("explorer", runtime.config.planner_model, "read-only discovery"),
                ("worker", runtime.config.primary_model, "focused execution"),
                ("verifier", runtime.config.critic_model, "adversarial review"),
            ]
            table = Table(title="Subagent Routes", box=box.SIMPLE)
            table.add_column("Role", style="cyan")
            table.add_column("Model")
            table.add_column("Purpose")
            for role, model, purpose in rows:
                table.add_row(role, model, purpose)
            self.console.print(table)
            return [{"role": role, "model": model, "purpose": purpose} for role, model, purpose in rows]

        role, task = self._parse_subagent_request(arg)
        context_pack = {
            "session_id": session.session_id,
            "workspace": session.cwd,
            "active_role": runtime.config.active_role,
            "goal": session.state.goal.objective if session.state.goal else "",
            "plan": session.state.plan.objective if session.state.plan else "",
        }
        result = self._run_subagent(role, task, context_pack=context_pack)
        summary = self._normalize_subagent_summary(result)
        node_key = self._record_action_graph_node(
            node_type="subagent",
            node_id=summary.get("run_id") or summary.get("label") or role.value,
            payload={
                "role": role.value,
                "task": task,
                "summary": summary.get("text", ""),
                "model": summary.get("model", ""),
                "session_id": session.session_id,
            },
        )
        self._record_event(
            AgentEvent.make(
                AgentEventType.PLAN_PHASE,
                session_id=session.session_id,
                phase="Subagent",
                actor=role.value,
                status="success",
                message=summary.get("text", "") or f"{role.value} completed",
                metadata={
                    "task": task,
                    "model": summary.get("model", ""),
                    "action_graph_node": node_key,
                },
            )
        )
        panel = Table(title=f"Subagent: {role.value}", box=box.SIMPLE)
        panel.add_column("Field", style="cyan", no_wrap=True)
        panel.add_column("Value")
        panel.add_row("task", task)
        panel.add_row("model", summary.get("model", ""))
        panel.add_row("summary", summary.get("text", "") or "(empty)")
        self.console.print(panel)
        return summary

    def _cmd_review(self, arg: str = "") -> None:
        self.console.print(Panel(arg or "No review focus provided.", title="Review Request"))
        return {"focus": arg.strip()}

    def _cmd_audit(self, arg: str = "") -> None:
        session = self._require_session()
        runtime = self._require_runtime()
        target = arg.strip() or session.session_id
        try:
            audited = runtime.load_session(target)
        except Exception:
            audited = session
        report = audit_session(runtime, audited)
        artifacts = write_audit_report(report, self._reports_root(audited) / "audits")
        self._record_event(
            AgentEvent.make(
                AgentEventType.PLAN_PHASE,
                session_id=audited.session_id,
                phase="Audit",
                actor="biobank",
                status="success",
                message="runtime audit generated",
                metadata={"artifacts": artifacts, "replay_status": report.replay_status},
            )
        )
        self.console.print(Panel(report.to_json(indent=2), title="Audit"))
        self._render_action_graph(session=audited, limit=12)
        return report.to_dict()

    def _cmd_trace(self, arg: str = "") -> None:
        session = self._require_session()
        runtime = self._require_runtime()
        target = arg.strip() or session.session_id
        try:
            traced = runtime.load_session(target)
        except Exception:
            traced = session
        events = [AgentEvent.from_dict(ev) for ev in traced.events if isinstance(ev, dict)]
        tree = build_run_tree(events, session_id=traced.session_id)
        summary = summarize_run_tree(tree)
        report = evaluate_run_tree(tree, default_evaluators())
        self.console.print(render_run_tree(tree))
        self.console.print(
            Panel(
                json.dumps(
                    {"summary": summary, "online_eval": report.to_dict()},
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                title="Trace",
            )
        )
        self._record_event(
            AgentEvent.make(
                AgentEventType.PLAN_PHASE,
                session_id=traced.session_id,
                phase="Trace",
                actor="biobank",
                status="success",
                message=f"run-tree trace: {summary.get('tool_calls', 0)} tool call(s)",
                metadata={"summary": summary, "online_eval": report.to_dict()},
            )
        )
        return {"summary": summary, "online_eval": report.to_dict()}

    def _cmd_replay(self, arg: str = "") -> None:
        runtime = self._require_runtime()
        session = self._require_session()
        target = arg.strip()
        trajectory = runtime.session_store.rollout_file(session.session_id)
        if target:
            candidate = Path(target).expanduser()
            if candidate.exists():
                trajectory = candidate
            else:
                maybe_session = runtime.session_store.session_dir(target) / "trajectory.jsonl"
                if maybe_session.exists():
                    trajectory = maybe_session
        report = replay_trajectory(trajectory)
        self._record_event(
            AgentEvent.make(
                AgentEventType.PLAN_PHASE,
                session_id=session.session_id,
                phase="Replay",
                actor="biobank",
                status="success" if report.status == "ok" else "failed",
                message=f"trajectory replay {report.status}",
                metadata=report.to_dict(),
            )
        )
        self.console.print(Panel(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str), title="Replay"))
        return report.to_dict()

    def _cmd_harness(self, arg: str = "") -> None:
        runtime = self._require_runtime()
        session = self._require_session()
        path = Path(arg.strip())
        if not path.exists():
            self.console.print(f"[yellow]Usage: /harness <task.json>[/]")
            return
        task = HarnessTask.from_file(path)
        runner = RuntimeHarnessRunner(self)
        report = runner.run(task, output_dir=self._reports_root(session) / "harness")
        self._record_event(
            AgentEvent.make(
                AgentEventType.PLAN_PHASE,
                session_id=session.session_id,
                phase="Harness",
                actor="biobank",
                status=report.status,
                message=f"harness {task.task_id} {report.status}",
                metadata=report.to_dict(),
            )
        )
        self.console.print(Panel(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str), title="Harness"))
        return report.to_dict()

    def _cmd_learn(self, arg: str = "") -> None:
        runtime = self._require_runtime()
        session = self._require_session()
        report = learn_from_session(runtime, session)
        artifacts = write_learning_report(report, self._reports_root(session) / "learning")
        # Curator pass — surface usage-driven skill-tier recommendations (dry-run,
        # advisory). /learn reports promotions/demotions; it never auto-rewrites the
        # manifest (tier mutation stays an explicit, reviewable action).
        curation = None
        try:
            from biobank_agent.runtime.curator import curate
            from biobank_agent.skills import manifest as _manifest

            curation = curate(
                getattr(session.state, "records", []) or [],
                lambda n: _manifest.exposure_of(n),
                _manifest._MANIFEST_PATH,
                dry_run=True,
                trust_of=_manifest.trust_of,
            )
        except Exception:
            curation = None
        self._record_event(
            AgentEvent.make(
                AgentEventType.PLAN_PHASE,
                session_id=session.session_id,
                phase="Learning",
                actor="biobank",
                status=report.status,
                message="trajectory learning completed",
                metadata={"artifacts": artifacts, "proposals": [p.to_dict() for p in report.proposals]},
            )
        )
        payload = report.to_dict()
        if curation is not None:
            payload["curation"] = {
                "recommendations": curation["recommendations"],
                "harness_tasks": curation["harness_tasks"],
            }
        self.console.print(Panel(json.dumps(payload, indent=2, ensure_ascii=False, default=str), title="Learning"))
        recs = (curation or {}).get("recommendations") or {}
        if recs.get("promote") or recs.get("demote"):
            self.console.print(
                f"[cyan]Skill curator:[/] promote={recs.get('promote')} demote={recs.get('demote')} "
                "(advisory — edit skills/manifest.json to apply)"
            )
        return payload

    def _evolution_llm(self):
        """Build an LLMClient for autonomous patch generation (planner model)."""
        from biobank_agent.llm import LLMClient

        cfg = self.runtime.config if self.runtime else None
        model = ""
        if cfg is not None:
            model = str(getattr(cfg, "planner_model", "") or getattr(cfg, "primary_model", "") or "")
        return LLMClient(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            model=model or "claude-opus-4-7",
        )

    def _cmd_evolve(self, arg: str = "") -> None:
        runtime = self._require_runtime()
        session = self._require_session()
        report = learn_from_session(runtime, session)

        from pathlib import Path as _Path

        import biobank_agent as _bb
        from biobank_agent.runtime.self_evolve import DEFAULT_APPLY_ALLOW_PATHS, apply_proposal

        repo_root = _Path(_bb.__file__).resolve().parent.parent
        # Token-exact flag detection so '--generated'/'--applyx' don't activate behavior.
        tokens = set(str(arg or "").split())
        do_generate = "--generate" in tokens
        do_apply = "--apply" in tokens or "--write" in tokens

        # Autonomous GENERATION: ask the LLM to turn each review-only proposal into
        # a concrete, applicable patch (diff + target_path + test_commands). The
        # generated patch is UNTRUSTED — it only becomes a real change by passing
        # the transactional, test-gated apply loop below.
        generation = None
        if do_generate:
            from biobank_agent.runtime.patch_generation import enrich_proposals_with_patches

            try:
                llm = self._evolution_llm()
                enriched = enrich_proposals_with_patches(
                    report.proposals, llm=llm, repo_root=repo_root,
                    allow_paths=DEFAULT_APPLY_ALLOW_PATHS,
                )
                generation = [
                    {
                        "proposal_id": getattr(p, "proposal_id", ""),
                        "category": getattr(p, "category", ""),
                        "generation": g.to_dict(),
                    }
                    for p, g in enriched
                ]
            except Exception as exc:  # never let generation crash the command
                generation = [{"status": "error", "error": str(exc)}]
            self._record_event(
                AgentEvent.make(
                    AgentEventType.PLAN_PHASE,
                    session_id=session.session_id,
                    phase="Evolution",
                    actor="biobank",
                    status="success",
                    message="autonomous patch generation completed",
                    metadata={"generation": generation},
                )
            )

        if do_apply:
            # Transactional, test-gated apply: a proposal carrying a concrete patch
            # (diff + target_path + test_commands) is applied in an isolated git
            # worktree and committed only if its tests pass; core/runtime/tests
            # edits go to a review branch, never auto-merged. Patch-less proposals
            # stay review-only.
            applicable = [
                p for p in report.proposals
                if getattr(p, "diff", "") and getattr(p, "target_path", "") and getattr(p, "test_commands", [])
            ]
            if not applicable:
                hint = (
                    " The generator did not produce an applicable patch this round."
                    if do_generate
                    else " Run /evolve --generate --apply to have the agent synthesize and verify a patch autonomously."
                )
                apply_info = {
                    "status": "no_applicable_patches",
                    "reason": (
                        "No proposal carries a concrete patch (diff + target_path + test_commands), so "
                        "evolution stays review-only. The transactional, test-gated apply path IS live for "
                        "patch-bearing proposals: applies in an isolated worktree, commits only on test pass, "
                        "and routes core/runtime/tests to a review branch." + hint
                    ),
                }
            else:
                try:
                    results = [
                        apply_proposal(p, repo_root=repo_root, allow_paths=DEFAULT_APPLY_ALLOW_PATHS).to_dict()
                        for p in applicable
                    ]
                    apply_info = {"status": "applied", "results": results}
                    self._record_event(
                        AgentEvent.make(
                            AgentEventType.PLAN_PHASE,
                            session_id=session.session_id,
                            phase="Evolution",
                            actor="biobank",
                            status="success",
                            message=f"transactional apply attempted for {len(results)} proposal(s)",
                            metadata={"apply": results},
                        )
                    )
                except Exception as exc:  # never crash the command on an apply error
                    apply_info = {"status": "error", "error": str(exc)}
            payload = {"learning": report.to_dict(), "apply": apply_info}
            if generation is not None:
                payload["generation"] = generation
            self.console.print(Panel(json.dumps(payload, indent=2, ensure_ascii=False, default=str), title="Evolve --apply"))
            return payload

        self._record_event(
            AgentEvent.make(
                AgentEventType.PLAN_PHASE,
                session_id=session.session_id,
                phase="Evolution",
                actor="biobank",
                status="success",
                message="review-only proposals generated" if not do_generate else "patches generated for review",
                metadata={"proposals": [p.to_dict() for p in report.proposals]},
            )
        )
        payload = report.to_dict() if generation is None else {"learning": report.to_dict(), "generation": generation}
        self.console.print(Panel(json.dumps(payload, indent=2, ensure_ascii=False, default=str), title="Evolve"))
        return payload

    def _cmd_mcp(self, action: str = "list", arg: str = "") -> dict[str, Any]:
        runtime = self._require_runtime()
        manager = self._require_mcp_manager()
        action = str(action or "list").strip().lower()
        arg = str(arg or "").strip()
        if action in {"", "list", "status"}:
            self._show_mcp_tables(manager, verbose="--verbose" in arg or "--schemas" in arg)
            return self._mcp_status_payload(manager, action="list")
        if action == "start":
            try:
                loaded = asyncio.run(manager.start())
            except RuntimeError:
                self.console.print("[red]Cannot start MCP from inside a running event loop.[/]")
                return {"action": "start", "status": "failed", "error": "running_event_loop"}
            self._record_event(
                AgentEvent.make(
                    AgentEventType.PLAN_PHASE,
                    session_id=self._require_session().session_id,
                    phase="MCP",
                    actor="biobank",
                    status="success",
                    message=f"started {loaded} tool(s)",
                    metadata={"action": "start", "loaded": loaded},
                )
            )
            self._record_action_graph_node(
                node_type="mcp",
                node_id=f"start:{manager.config_path}",
                payload={"action": "start", "loaded": loaded, "config_path": str(manager.config_path)},
                score=0.2,
            )
            self.console.print(f"[green]MCP started:[/] {loaded} remote tool(s) loaded")
            self._show_mcp_tables(manager, verbose="--verbose" in arg or "--schemas" in arg)
            return self._mcp_status_payload(manager, action="start", extra={"loaded": loaded})
        if action == "health":
            repair = "--repair" in arg
            try:
                rows = asyncio.run(manager.health_check(repair=repair))
            except RuntimeError:
                self.console.print("[red]Cannot health-check MCP from inside a running event loop.[/]")
                return {"action": "health", "status": "failed", "error": "running_event_loop", "repair": repair}
            self._render_mcp_health(rows, repair=repair)
            self._record_event(
                AgentEvent.make(
                    AgentEventType.PLAN_PHASE,
                    session_id=self._require_session().session_id,
                    phase="MCP",
                    actor="biobank",
                    status="success",
                    message=f"health check complete ({'repair' if repair else 'probe'})",
                    metadata={"action": "health", "repair": repair, "rows": rows},
                )
            )
            self._record_action_graph_node(
                node_type="mcp",
                node_id=f"health:{manager.config_path}",
                payload={"action": "health", "repair": repair, "rows": rows, "config_path": str(manager.config_path)},
                score=0.2,
            )
            return {"action": "health", "status": "ok", "repair": repair, "rows": rows}
        if action == "stop":
            try:
                asyncio.run(manager.stop())
            except RuntimeError:
                self.console.print("[red]Cannot stop MCP from inside a running event loop.[/]")
                return {"action": "stop", "status": "failed", "error": "running_event_loop"}
            self._record_event(
                AgentEvent.make(
                    AgentEventType.PLAN_PHASE,
                    session_id=self._require_session().session_id,
                    phase="MCP",
                    actor="biobank",
                    status="success",
                    message="stopped MCP clients",
                    metadata={"action": "stop"},
                )
            )
            self._record_action_graph_node(
                node_type="mcp",
                node_id=f"stop:{manager.config_path}",
                payload={"action": "stop", "config_path": str(manager.config_path)},
                score=0.2,
            )
            self.console.print("[green]MCP stopped.[/]")
            return self._mcp_status_payload(manager, action="stop")
        if action == "call":
            tool_name, tool_args = self._parse_mcp_call_args(arg)
            runtime_tool = self._normalize_mcp_runtime_tool_name(tool_name)
            session = self._require_session()
            outcome = runtime.invoke_tool(
                session,
                runtime_tool,
                tool_args,
                turn_id=session.state.active_turn_id,
                call_id=f"mcp:{runtime_tool}:{uuid.uuid4().hex[:8]}",
            )
            result = outcome.result if isinstance(outcome.result, dict) else {"output": outcome.result}
            result = result or {"status": "ok"}
            self.console.print(
                Panel(
                    json.dumps(result, indent=2, ensure_ascii=False, default=str),
                    title=f"MCP call: {tool_name}",
                )
            )
            self._record_event(
                AgentEvent.make(
                    AgentEventType.PLAN_PHASE,
                    session_id=session.session_id,
                    phase="MCP",
                    actor="biobank",
                    status="success",
                    message=f"called {tool_name}",
                    metadata={
                        "action": "call",
                        "tool": tool_name,
                        "runtime_tool": runtime_tool,
                        "args": tool_args,
                        "result": result,
                        "approval_profile": runtime.config.approval_profile,
                        "transport": "runtime_tool",
                    },
                )
            )
            return {
                "action": "call",
                "status": outcome.state.value,
                "tool": tool_name,
                "runtime_tool": runtime_tool,
                "result": result,
                "error": outcome.error,
                "approval_profile": runtime.config.approval_profile,
            }
        self.console.print("[yellow]Usage: /mcp [list|start|health|call|stop][/]")
        return {"action": action, "status": "unknown_action"}

    def _cmd_quit(self, *_args: Any) -> None:
        self.exit_requested = True
        self.console.print("[dim]Goodbye.[/]")
        return {"status": "quit"}

    def _build_tool_context(self, request, *, session, turn, runtime):
        legacy = self._require_legacy_agent()
        # Reports follow the active workspace by default so a --workspace/`/cd`
        # switch keeps inputs and outputs together (issue #4).
        report_dir = self._session_report_dir(session)
        report_dir.mkdir(parents=True, exist_ok=True)
        ctx = legacy._build_ctx(report_dir)
        # Critical compatibility for legacy direct skills (python_exec/shell_exec):
        # set workspace_root and project_root before the broader best-effort
        # backfill below, so an unrelated legacy-state attribute failure cannot
        # silently drop tools back to the process cwd.
        try:
            ctx.workspace_root = Path(session.cwd)
        except Exception:
            pass
        try:
            setattr(ctx.settings, "project_root", str(Path(session.cwd)))
        except Exception:
            try:
                object.__setattr__(ctx.settings, "project_root", str(Path(session.cwd)))
            except Exception:
                pass
        try:
            legacy.state.current_report_dir = report_dir
        except Exception:
            pass
        try:
            ctx.name = request.handler.name
            ctx.args = dict(request.args or {})
            ctx.capabilities = request.handler.required_capabilities()
            ctx.state = legacy.state
            ctx.state.memory = legacy.memory
            ctx.report_dir = report_dir
            ctx.workspace_root = Path(session.cwd)
            # Extra readable/writable roots the user opted into via /cd (e.g. the
            # prior project dir), so cross-project work keeps earlier files reachable.
            ctx.extra_roots = list(session.state.custom_data.get("workspace_extra_roots", []) or [])
            ctx.job_manager = self.job_manager
            # Lazy skill exposure: let skill_search rank deferred skills and
            # activate them so their schemas inject on the next round of this turn.
            ctx.tool_registry = self.tool_registry

            def _activate_skills(names: list[str], _session=session) -> None:
                active = list(_session.state.custom_data.get("active_skills", []) or [])
                for name in names:
                    if name and name not in active:
                        active.append(name)
                _session.state.custom_data["active_skills"] = active

            ctx.activate_skills = _activate_skills
            ctx.permission_mode = runtime.config.approval_profile
            ctx.turn_id = turn.id
            ctx.tool_call_id = request.call_id
            ctx.session_id = session.session_id
        except Exception:
            pass

        trajectory = getattr(legacy.state, "trajectory", None)
        if trajectory is None:
            try:
                from biobank_agent.runtime.types import TrajectoryRecorder

                trajectory = TrajectoryRecorder(session.session_id)
                legacy.state.trajectory = trajectory
            except Exception:
                trajectory = None

        def emit_progress(phase: str = "", message: str = "", metadata: dict | None = None) -> None:
            payload = {
                "skill": request.handler.name,
                "phase": str(phase or request.handler.name),
                "message": str(message or ""),
                "metadata": dict(metadata or {}),
            }
            runtime._record_event(
                session,
                AgentEvent.make(
                    AgentEventType.TOOL_PROGRESS,
                    session_id=session.session_id,
                    turn_id=turn.id,
                    tool_call_id=request.call_id,
                    **payload,
                ),
            )
            try:
                legacy.state.custom_data.setdefault("skill_progress_events", []).append(payload)
            except Exception:
                pass

        ctx.emit_progress = emit_progress

        # Live subprocess streaming (issue #2): surface stdout/stderr lines on the
        # active execution view as they arrive. Throttled (stderr always; stdout
        # sampled) so a chatty command does not saturate rendering — the FULL
        # output is always persisted to the command's log file by run_streaming.
        _line_state = {"last": 0.0, "count": 0}

        def emit_line(stream: str = "stdout", line: str = "") -> None:
            text = str(line or "").rstrip()
            if not text:
                return
            view = getattr(self, "_active_exec_view", None)
            if view is None or not hasattr(view, "record"):
                return
            _line_state["count"] += 1
            if _line_state["count"] > 2000:  # hard cap; full output is in the log
                return
            now = time.monotonic()
            if stream != "stderr" and (now - _line_state["last"]) < 0.25:
                return  # throttle stdout; always show stderr
            _line_state["last"] = now
            try:
                view.record(
                    "Output",
                    actor=str(request.call_id),
                    status="running",
                    message=f"{request.handler.name}: {text[:200]}",
                    metadata={"subagent": str(request.call_id), "stream": stream},
                )
            except Exception:
                pass

        ctx.emit_line = emit_line
        ctx.record_trajectory = lambda payload: None if trajectory is None else trajectory.add(
            AgentEvent.make(
                AgentEventType.TOOL_PROGRESS,
                session_id=session.session_id,
                turn_id=turn.id,
                tool_call_id=request.call_id,
                **{"payload": payload},
            )
        )
        ctx.record_action_graph = lambda payload: runtime.action_graph.upsert_node(
            "tool",
            request.call_id,
            payload={
                "session_id": session.session_id,
                "turn_id": turn.id,
                "payload": payload,
                "tool": request.handler.name,
            },
        )
        return ctx

    def _cmd_plan_title(self, arg: str = "") -> None:
        if not arg:
            self.console.print("[yellow]Usage: /plan-title <title>[/]")
            return
        self._require_session().title = arg.strip()
        self.console.print(f"[green]Title updated:[/] {_rich_escape(arg.strip())}")
        return {"title": arg.strip()}

    def _cmd_cost(self) -> None:
        self.console.print(json.dumps(self.token_usage, ensure_ascii=False, default=str))
        return dict(self.token_usage)

    def _cmd_clear(self) -> None:
        session = self._require_session()
        session.turns.clear()
        session.state.summary = ""
        self.console.print("[green]Session turn state cleared.[/]")
        return {"status": "cleared"}

    def _cmd_export(self, fmt: str = "json") -> None:
        session = self._require_session()
        root = self._reports_root(session) / "exports"
        root.mkdir(parents=True, exist_ok=True)
        if str(fmt or "json").lower().startswith("md"):
            path = root / f"{session.session_id}.md"
            path.write_text(f"# {session.title}\n\nSession: `{session.session_id}`\n", encoding="utf-8")
        else:
            path = root / f"{session.session_id}.json"
            path.write_text(json.dumps(session.to_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        self.console.print(f"[green]Exported:[/] {path}")
        return {"path": str(path), "format": fmt}

    def _cmd_artifacts(self, *_args: Any) -> dict[str, Any]:
        session = self._require_session()
        artifacts = list(session.state.custom_data.get("last_artifact_index") or [])
        if artifacts:
            self._render_artifact_index(artifacts)
            return {"status": "ok", "artifacts": artifacts, "count": len(artifacts)}
        report_dir = self._session_report_dir(session)
        self.console.print(
            "[dim]No captured plan artifacts yet. New plan outputs will be listed here after execution.[/]\n"
            f"[dim]Current session report directory: {report_dir}[/]"
        )
        return {"status": "empty", "artifacts": [], "report_dir": str(report_dir)}

    def _cmd_tools(self, *_args: Any) -> dict[str, Any]:
        runtime = self._require_runtime()
        from biobank_agent.core.tools.protocol import LegacySkillToolHandler
        from biobank_agent.skills import manifest as _sm

        rows: list[dict[str, Any]] = []
        category_counts: Counter[str] = Counter()
        exposure_counts: Counter[str] = Counter()
        capability_counts: Counter[str] = Counter()
        for handler in sorted(runtime.tool_registry.list_handlers(), key=lambda h: str(h.name)):
            spec = handler.spec()
            meta = spec.metadata()
            caps = sorted(getattr(cap, "value", str(cap)) for cap in handler.required_capabilities())
            for cap in caps:
                capability_counts[cap] += 1
            try:
                exposure = runtime.tool_registry._exposure(handler)
            except Exception:
                exposure = _sm.exposure_of(str(handler.name)) if isinstance(handler, LegacySkillToolHandler) else _sm.DIRECT
            category = _sm.domain_of(str(handler.name))
            if category == "other":
                category = self._tool_category(str(handler.name))
            category_counts[category] += 1
            exposure_counts[exposure] += 1
            rows.append({
                "name": str(handler.name),
                "category": category,
                "exposure": exposure,
                "capabilities": caps,
                "safety": meta.get("safety_class", ""),
                "approval": meta.get("approval_requirement", ""),
                "workspace_scope": meta.get("workspace_scope", ""),
                "mutating": bool(handler.is_mutating),
                "description": str(spec.description or ""),
            })

        summary = Table(title="Tool Registry", box=box.SIMPLE)
        summary.add_column("Metric", style="cyan", no_wrap=True)
        summary.add_column("Value")
        summary.add_row("total", str(len(rows)))
        summary.add_row("exposure", ", ".join(f"{k}:{v}" for k, v in sorted(exposure_counts.items())) or "-")
        summary.add_row("categories", ", ".join(f"{k}:{v}" for k, v in sorted(category_counts.items())) or "-")
        summary.add_row("capabilities", ", ".join(f"{k}:{v}" for k, v in capability_counts.most_common(6)) or "-")
        summary.add_row("long commands", "use run_job, then monitor with /jobs and /job-tail <job-id>")
        self.console.print(summary)

        table = Table(title="Available Tools", box=box.SIMPLE)
        table.add_column("Tool", style="cyan", no_wrap=True)
        table.add_column("Category", no_wrap=True)
        table.add_column("Exposure", no_wrap=True)
        table.add_column("Capabilities")
        table.add_column("Safety")
        table.add_column("Description")
        for row in rows[:80]:
            safety = "/".join(str(row.get(k) or "-") for k in ("safety", "approval", "workspace_scope"))
            table.add_row(
                row["name"],
                row["category"],
                row["exposure"],
                ", ".join(row["capabilities"]) or "-",
                safety,
                row["description"][:90],
            )
        if len(rows) > 80:
            table.caption = f"{len(rows) - 80} additional tool(s) omitted from display; full list returned in the command payload."
        self.console.print(table)
        return {
            "status": "ok",
            "total_tools": len(rows),
            "categories": dict(category_counts),
            "exposures": dict(exposure_counts),
            "capabilities": dict(capability_counts),
            "tools": rows,
        }

    def _cmd_skills(self) -> None:
        runtime = self._require_runtime()
        from biobank_agent.skills import manifest as _sm

        handlers = runtime.tool_registry.list_handlers()
        registry = runtime.tool_registry
        # Group by manifest domain (fallback to the legacy heuristic), tracking the
        # exposure tier so the scientist sees the full tree + what loads each turn.
        grouped: dict[str, dict[str, list[str]]] = {}
        for handler in handlers:
            name = str(handler.name)
            domain = _sm.domain_of(name)
            if domain == "other":
                domain = self._tool_category(name)
            try:
                exposure = registry._exposure(handler)
            except Exception:
                exposure = _sm.exposure_of(name)
            grouped.setdefault(domain, {}).setdefault(exposure, []).append(name)
        n_direct = sum(len(v.get(_sm.DIRECT, [])) for v in grouped.values())
        table = Table(title=f"Skill Tree — {len(handlers)} tools ({n_direct} loaded each turn; rest via skill_search)", box=box.SIMPLE)
        table.add_column("Domain", style="cyan", no_wrap=True)
        table.add_column("Total", justify="right")
        table.add_column("Direct", justify="right", style="green")
        table.add_column("Deferred", justify="right", style="yellow")
        table.add_column("Hidden", justify="right", style="dim")
        table.add_column("Examples")
        for domain in sorted(grouped):
            tiers = grouped[domain]
            allnames = [n for names in tiers.values() for n in names]
            examples = ", ".join(sorted(allnames)[:6])
            extra = len(allnames) - 6
            if extra > 0:
                examples += f" … (+{extra})"
            table.add_row(
                domain, str(len(allnames)),
                str(len(tiers.get(_sm.DIRECT, []))),
                str(len(tiers.get(_sm.DEFERRED, []))),
                str(len(tiers.get(_sm.HIDDEN, []))),
                examples,
            )
        self.console.print(table)
        wgs = self._wgs_readiness_snapshot()
        wgs_table = Table(title="WGS Readiness", box=box.SIMPLE)
        wgs_table.add_column("Field", style="cyan")
        wgs_table.add_column("Value")
        wgs_table.add_row("VCF samples", str(wgs.get("n_samples", 0)))
        wgs_table.add_row("Indexed", f"{wgs.get('n_indexed', 0)}/{wgs.get('n_samples', 0)}")
        wgs_table.add_row("Workflow mode", str(wgs.get("workflow_mode", "unknown")))
        wgs_table.add_row("Existing VCF dirs", ", ".join(wgs.get("existing_vcf_dirs") or []) or "none")
        missing = ", ".join(str(x) for x in (wgs.get("missing_for_standard_workflow") or []) if x)
        wgs_table.add_row("Missing for standard", missing or "none")
        self.console.print(wgs_table)
        return {
            "status": "ok",
            "total_tools": len(handlers),
            "direct_tools": n_direct,
            "domains": {
                domain: {
                    "total": sum(len(names) for names in tiers.values()),
                    "direct": len(tiers.get(_sm.DIRECT, [])),
                    "deferred": len(tiers.get(_sm.DEFERRED, [])),
                    "hidden": len(tiers.get(_sm.HIDDEN, [])),
                    "examples": sorted([n for names in tiers.values() for n in names])[:12],
                }
                for domain, tiers in sorted(grouped.items())
            },
            "wgs": wgs,
        }

    def _cmd_model(self, arg: str = "") -> None:
        runtime = self._require_runtime()
        if arg:
            runtime.config.primary_model = arg.strip()
        self.console.print(f"Primary model: [cyan]{runtime.config.primary_model}[/]")
        return {"primary_model": runtime.config.primary_model}

    def _cmd_figures(self, *_args: Any) -> list[str]:
        session = self._require_session()
        figures = [str(p) for p in session.state.custom_data.get("figure_paths", []) or []]
        if not figures and self.legacy_agent is not None and hasattr(self.legacy_agent.state, "figures"):
            figures = [str(p) for p in self.legacy_agent.state.figures]
        if not figures and self.legacy_agent is not None and hasattr(self.legacy_agent.state, "records"):
            for record in self.legacy_agent.state.records:
                figures.extend(str(p) for p in getattr(record, "figure_paths", []) or [])
        table = Table(title="Generated Figures", box=box.SIMPLE)
        table.add_column("#", style="dim", width=3)
        table.add_column("Path", style="cyan")
        for i, fig in enumerate(figures, 1):
            table.add_row(str(i), fig)
        if figures:
            self.console.print(table)
        else:
            self.console.print("[dim]No figures generated yet.[/]")
        return figures

    def _cmd_cohorts(self, *_args: Any) -> dict[str, Any]:
        session = self._require_session()
        runtime = self._require_runtime()
        custom = dict(session.state.custom_data.get("cohort_summaries") or {})
        if not custom and hasattr(self._require_legacy_agent().state, "cohorts"):
            for name, df in self._require_legacy_agent().state.cohorts.items():
                n_total = len(df)
                n_cases = int(df["label"].sum()) if "label" in df.columns else None
                custom[name] = {"subjects": n_total, "cases": n_cases, "controls": (n_total - n_cases) if n_cases is not None else None}
        table = Table(title="Active Cohorts", box=box.SIMPLE)
        table.add_column("Name", style="cyan")
        table.add_column("Subjects", justify="right")
        table.add_column("Cases", justify="right")
        table.add_column("Controls", justify="right")
        for name, summary in custom.items():
            table.add_row(
                name,
                str(summary.get("subjects", "?")),
                str(summary.get("cases", "?")),
                str(summary.get("controls", "?")),
            )
        if custom:
            self.console.print(table)
        else:
            self.console.print("[dim]No active cohorts.[/]")
        return custom

    def _cmd_models(self, *_args: Any) -> dict[str, Any]:
        session = self._require_session()
        custom = dict(session.state.custom_data.get("model_summaries") or {})
        if not custom and self.legacy_agent is not None and hasattr(self.legacy_agent.state, "model_metadata"):
            custom = {
                key: {
                    "model_type": meta.get("model_type", ""),
                    "auc": meta.get("auc"),
                    "n_cases": meta.get("n_cases"),
                    "n_features": meta.get("n_features"),
                }
                for key, meta in self.legacy_agent.state.model_metadata.items()
            }
        table = Table(title="Trained Models", box=box.SIMPLE)
        table.add_column("Key", style="cyan")
        table.add_column("Type")
        table.add_column("AUC", justify="right")
        for key, meta in custom.items():
            auc = meta.get("auc")
            table.add_row(key, str(meta.get("model_type", "")), f"{float(auc):.4f}" if isinstance(auc, (int, float)) else "N/A")
        if custom:
            self.console.print(table)
        else:
            self.console.print("[dim]No trained models.[/]")
        return custom

    def _cmd_history(self, *_args: Any) -> list[dict[str, Any]]:
        session = self._require_session()
        rows = [
            {
                "timestamp": turn.started_at,
                "content": turn.content,
                "status": turn.status,
            }
            for turn in session.turns
        ]
        if not rows:
            self.console.print("[dim]No analyses performed yet.[/]")
            return []
        table = Table(title="Analysis History", box=box.SIMPLE)
        table.add_column("#", style="dim", width=3)
        table.add_column("Content", style="cyan")
        for i, row in enumerate(rows[-20:], 1):
            table.add_row(str(i), str(row["content"])[:80])
        self.console.print(table)
        return rows

    def _cmd_record(self, arg: str = "") -> dict[str, Any]:
        session = self._require_session()
        name = str(arg or "").strip()
        if not name:
            self.console.print("[yellow]Usage: /record <pipeline_name>[/]")
            return {"status": "missing_name"}
        steps = []
        for turn in session.turns:
            if turn.content:
                steps.append({"skill": "user_turn", "args": {"text": turn.content}})
            for msg in turn.assistant_messages:
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        steps.append({"skill": tc.name, "args": dict(tc.args)})
        if not steps and self.legacy_agent is not None and hasattr(self.legacy_agent.state, "records"):
            steps = [{"skill": r.skill, "args": dict(r.args)} for r in self.legacy_agent.state.records if r.skill not in {"think", "record_macro", "replay_pipeline", "list_pipelines"}]
        if not steps:
            self.console.print("[yellow]No recordable skill calls found.[/]")
            return {"status": "empty"}
        self._require_legacy_agent().memory.save_pipeline(name, steps)
        self.console.print(f"[green]Saved pipeline '{name}' with {len(steps)} steps.[/]")
        return {"name": name, "steps": len(steps)}

    def _cmd_pipelines(self, *_args: Any) -> list[dict[str, Any]]:
        memory = self._require_legacy_agent().memory
        rows = [{"name": name, "steps": len(memory.get_pipeline(name) or [])} for name in memory.list_pipelines()]
        if not rows:
            self.console.print("[dim]No saved pipelines.[/]")
            return []
        table = Table(title="Saved Pipelines", box=box.SIMPLE)
        table.add_column("Name", style="cyan")
        table.add_column("Steps", justify="right")
        for row in rows:
            table.add_row(row["name"], str(row["steps"]))
        self.console.print(table)
        return rows

    def _cmd_errors(self, *_args: Any) -> list[dict[str, Any]]:
        memory = self._require_legacy_agent().memory
        rows = memory.most_common_errors(10)
        if not rows:
            self.console.print("[dim]No errors recorded.[/]")
            return []
        table = Table(title="Error Catalog", box=box.SIMPLE)
        table.add_column("Error", style="red")
        table.add_column("Skill", style="cyan")
        table.add_column("Count", justify="right")
        for row in rows:
            table.add_row(row["error_type"], row["skill"], str(row["count"]))
        self.console.print(table)
        return rows

    def _cmd_memory(self, *_args: Any) -> dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session()
        memory = self._require_legacy_agent().memory
        summary = memory.summary()
        info = {
            "summary": summary,
            "session_summary": session.state.summary,
            "plan": session.state.plan.to_dict() if session.state.plan else None,
            "goal": session.state.goal.to_dict() if session.state.goal else None,
            "last_orchestration": session.state.last_orchestration,
            "custom_data_keys": sorted(session.state.custom_data.keys()),
            "approval_profile": runtime.config.approval_profile,
        }
        if summary:
            self.console.print(Panel(summary, title="Long-term Memory"))
        else:
            self.console.print("[dim]Long-term memory is empty.[/]")
        return info

    def _cmd_debate(self, arg: str = "") -> dict[str, Any]:
        if not arg:
            self.console.print("Usage: /debate <query>")
            return {"status": "missing_query"}
        session = self._require_session()
        session.state.custom_data["debate_query"] = arg
        self._require_runtime().save_session(session)
        return self.handle_message(arg)

    def _cmd_invoke_tool(self, skill_name: str, args: Any) -> dict[str, Any]:
        session = self._require_session()
        runtime = self._require_runtime()
        payload = dict(args or {}) if isinstance(args, dict) else {"value": args}
        session.state.custom_data.setdefault("tool_invocation_requests", []).append({"skill": skill_name, "args": payload})
        runtime.save_session(session)
        outcome = runtime.invoke_tool(
            session,
            skill_name,
            payload,
            turn_id=session.state.active_turn_id,
            call_id=f"tool:{skill_name}:{uuid.uuid4().hex[:8]}",
            context_factory=lambda req: self._build_tool_context_for_command(req, session=session, runtime=runtime),
        )
        result = outcome.result if isinstance(outcome.result, dict) else {"output": outcome.result}
        if outcome.error and "error" not in result:
            result = {**result, "error": outcome.error}
        self.console.print(Panel(json.dumps(result, indent=2, ensure_ascii=False, default=str), title=f"Tool: {skill_name}"))
        return {
            "skill": skill_name,
            "status": outcome.state.value,
            "result": result,
            "error": outcome.error,
            "approval_profile": runtime.config.approval_profile,
        }

    def _build_tool_context_for_command(self, request, *, session: AgentSession, runtime: AgentRuntime):
        turn = _CommandToolTurn(session.state.active_turn_id or f"command:{request.call_id}")
        return self._build_tool_context(request, session=session, turn=turn, runtime=runtime)

    def _cmd_replicate(self, arg: str = "") -> dict[str, Any]:
        if not arg:
            self.console.print("[yellow]Usage: /replicate <paper_path_or_text>[/]")
            return {"status": "missing_source"}
        result = self._cmd_invoke_tool("replicate_paper", {"source": arg, "source_type": "auto"})
        return result

    def _cmd_strategy(self, arg: str = "") -> dict[str, Any]:
        runtime = self._require_runtime()
        text = str(arg or "").strip().lower()
        if not text:
            current = "auto" if getattr(self.legacy_agent.settings, "multi_model_enabled", False) else "single"
            self.console.print(f"Current strategy: [cyan]{current}[/]")
            self.console.print("[dim]Options: auto, single, debate, ensemble[/]")
            return {"strategy": current}
        if text == "single":
            self.legacy_agent.settings.multi_model_enabled = False
        elif text in {"auto", "debate", "ensemble"}:
            self.legacy_agent.settings.multi_model_enabled = True
        else:
            self.console.print(f"[yellow]Unknown strategy: {_rich_escape(str(text))}. Use: auto, single, debate, ensemble[/]")
            return {"error": "unknown strategy", "value": text}
        session = self._require_session()
        session.state.last_orchestration["strategy"] = text
        runtime.save_session(session)
        self.console.print(f"[green]Strategy set to {_rich_escape(str(text))}[/]")
        return {"strategy": text}

    def _render_latest_assistant(self) -> None:
        session = self._require_session()
        if not session.turns:
            return
        messages = session.turns[-1].assistant_messages
        if not messages:
            return
        text = messages[-1].text.strip()
        if text:
            self.console.print(Markdown(text))

    def _update_token_usage(self) -> None:
        session = self._require_session()
        prompt = 0
        completion = 0
        for turn in session.turns:
            for msg in turn.assistant_messages:
                usage = msg.usage or {}
                prompt += int(usage.get("prompt_tokens", 0) or 0)
                completion += int(usage.get("completion_tokens", 0) or 0)
        self.token_usage["prompt_tokens"] = prompt
        self.token_usage["completion_tokens"] = completion

    def _record_event(self, event: AgentEvent) -> None:
        session = self._require_session()
        payload = event.to_dict()
        payload["schema_version"] = session.schema_version
        session.events.append(payload)
        session.touch()
        self._require_runtime().bus.publish_nowait(event)

    def _normalize_command_result(self, result: Any) -> Any:
        if result is None:
            return {}
        if isinstance(result, AgentEvent):
            return result.to_dict()
        if isinstance(result, (str, int, float, bool)):
            return result
        if isinstance(result, Path):
            return str(result)
        if isinstance(result, dict):
            return {str(k): self._normalize_command_result(v) for k, v in result.items()}
        if isinstance(result, (list, tuple, set)):
            return [self._normalize_command_result(v) for v in result]
        if hasattr(result, "to_dict") and callable(result.to_dict):
            return self._normalize_command_result(result.to_dict())
        return str(result)

    def _read_line(self) -> str:
        if self.prompt_session is not None:
            return self.prompt_session.prompt(self._prompt_prefix())
        return self.console.input("[bold cyan]biobank>[/] ")

    def _build_prompt_session(self) -> PromptSession:
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
            from prompt_toolkit.completion import Completer, Completion
            from prompt_toolkit.formatted_text import HTML
            from prompt_toolkit.history import FileHistory
            from prompt_toolkit.styles import Style as PTStyle
        except Exception:
            return None
        Path(self.settings.memory_dir).mkdir(parents=True, exist_ok=True)
        command_names = sorted(self.slash_registry.commands)

        class _CommandCompleter(Completer):
            def get_completions(self, document, complete_event):
                word = document.text_before_cursor
                if not word.startswith("/"):
                    return
                for command in command_names:
                    if command.startswith(word.lower()):
                        yield Completion(command, start_position=-len(word), display=command)

        style = PTStyle.from_dict(
            {
                "bottom-toolbar": "fg:#9aa4b2 bg:#1f2937",
                "prompt.main": "bold #56d4dd",
                "prompt.plan": "bold #f59e0b",
                "prompt.sep": "#768194",
            }
        )
        return PromptSession(
            history=FileHistory(str(Path(self.settings.memory_dir) / "cli_history.txt")),
            auto_suggest=AutoSuggestFromHistory(),
            completer=_CommandCompleter(),
            complete_while_typing=True,
            complete_in_thread=True,
            reserve_space_for_menu=8,
            style=style,
            bottom_toolbar=lambda: HTML(self._bottom_toolbar_text()),
        )

    def _prompt_prefix(self):
        try:
            from prompt_toolkit.formatted_text import FormattedText
        except Exception:
            return "biobank › "
        session = self._require_session()
        plan = session.state.plan
        if plan is not None and plan.status == PlanStatus.DRAFT:
            return FormattedText(
                [
                    ("class:prompt.plan", "plan:review"),
                    ("class:prompt.sep", " › "),
                ]
            )
        return FormattedText(
            [
                ("class:prompt.main", "biobank"),
                ("class:prompt.sep", " › "),
            ]
        )

    def _bottom_toolbar_text(self) -> str:
        runtime = self.runtime
        session = self.session
        elapsed = max(0, int(time.time() - self._activity_started_at))
        permission = runtime.config.approval_profile if runtime is not None else "unknown"
        session_short = session.session_id[:8] if session is not None else "no-session"
        plan_status = session.state.plan.status.value if session is not None and session.state.plan else "none"
        return (
            f"<b>{self._activity}</b> {elapsed}s  •  "
            f"session {session_short}  •  permissions {permission}  •  plan {plan_status}  •  "
            "Tab complete  /help commands"
        )

    def _set_activity(self, activity: str) -> None:
        text = str(activity or "idle")
        if text != self._activity:
            self._activity = text
            self._activity_started_at = time.time()

    @staticmethod
    def _interactive_terminal() -> bool:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())

    def _require_runtime(self) -> AgentRuntime:
        if self.runtime is None:
            raise RuntimeError("interactive shell is not initialized")
        return self.runtime

    def _require_legacy_agent(self) -> Agent:
        if self.legacy_agent is None:
            raise RuntimeError("interactive shell is not initialized")
        return self.legacy_agent

    def _require_session(self) -> AgentSession:
        if self.session is None:
            raise RuntimeError("interactive shell is not initialized")
        return self.session

    def _require_mcp_manager(self) -> McpManager:
        if self.mcp_manager is None:
            raise RuntimeError("interactive shell is not initialized")
        return self.mcp_manager

    def _update_active_role(self, arg: str) -> None:
        runtime = self._require_runtime()
        session = self._require_session()
        role = self._parse_provider_role(arg)
        if role is None:
            return
        runtime.config.active_role = role.value
        session.config.active_role = role.value
        node_key = self._record_action_graph_node(
            node_type="provider_role",
            node_id=role.value,
            payload={
                "session_id": session.session_id,
                "active_role": role.value,
                "primary": runtime.config.primary_model,
                "planner": runtime.config.planner_model,
                "critic": runtime.config.critic_model,
            },
            score=0.5,
        )
        self._record_event(
            AgentEvent.make(
                AgentEventType.PLAN_PHASE,
                session_id=session.session_id,
                phase="Provider routing",
                actor="biobank",
                status="success",
                message=f"active role switched to {role.value}",
                metadata={"active_role": role.value, "action_graph_node": node_key},
            )
        )
        self.console.print(f"[green]Active role:[/] {role.value}")
        self.runtime.save_session(session)

    def _parse_provider_role(self, arg: str) -> ProviderRole | None:
        text = str(arg or "").strip().lower()
        if not text:
            return None
        text = text.split(None, 1)[0]
        aliases = {
            "primary": ProviderRole.PRIMARY_EXECUTOR,
            "executor": ProviderRole.PRIMARY_EXECUTOR,
            "main": ProviderRole.PRIMARY_EXECUTOR,
            "plan": ProviderRole.PLANNER,
            "planner": ProviderRole.PLANNER,
            "critic": ProviderRole.CRITIC,
            "review": ProviderRole.SAFETY_REVIEWER,
            "reviewer": ProviderRole.SAFETY_REVIEWER,
            "safety": ProviderRole.SAFETY_REVIEWER,
            "safety_reviewer": ProviderRole.SAFETY_REVIEWER,
            "summarizer": ProviderRole.SUMMARIZER,
            "summary": ProviderRole.SUMMARIZER,
        }
        return aliases.get(text)

    def _parse_subagent_request(self, arg: str) -> tuple[SubagentRole, str]:
        from biobank_agent.orchestrator import SubagentRole
        tokens = shlex.split(str(arg or ""))
        if not tokens:
            return SubagentRole.EXPLORER, self._default_subagent_task()
        head = tokens[0].lower()
        if head in {"spawn", "run"} and len(tokens) >= 3:
            return self._parse_subagent_role(tokens[1]), " ".join(tokens[2:]).strip() or self._default_subagent_task()
        if head in {"review", "verifier", "verify", "critic"}:
            return SubagentRole.VERIFIER, " ".join(tokens[1:]).strip() or self._default_subagent_task()
        if head in {"explorer", "worker", "verifier"}:
            return self._parse_subagent_role(head), " ".join(tokens[1:]).strip() or self._default_subagent_task()
        return self._parse_subagent_role(head), " ".join(tokens[1:]).strip() or self._default_subagent_task()

    def _parse_subagent_role(self, token: str) -> SubagentRole:
        from biobank_agent.orchestrator import SubagentRole
        text = str(token or "").strip().lower()
        if text in {"worker", "execute", "exec", "run"}:
            return SubagentRole.WORKER
        if text in {"verifier", "review", "reviewer", "critic", "verify"}:
            return SubagentRole.VERIFIER
        return SubagentRole.EXPLORER

    def _default_subagent_task(self) -> str:
        session = self._require_session()
        if session.state.plan and session.state.plan.objective:
            return session.state.plan.objective
        if session.state.goal and session.state.goal.objective:
            return session.state.goal.objective
        if session.turns:
            return session.turns[-1].content
        return "Summarize the current session state and likely next step."

    def _run_subagent(self, role: SubagentRole, task: str, *, context_pack: dict[str, Any] | None = None) -> Any:
        from biobank_agent.orchestrator import SubagentCall
        legacy = self._require_legacy_agent()
        orchestrator = getattr(legacy, "orchestrator", None)
        if orchestrator is None or not hasattr(orchestrator, "dispatch_subagent"):
            raise RuntimeError("subagent orchestration is unavailable")
        model_id = self._model_for_subagent_role(role)
        call = SubagentCall(role=role, task=task)
        result = orchestrator.dispatch_subagent(call, context_pack=context_pack or {}, model_id=model_id)
        return {
            "response": result,
            "model": model_id,
            "role": role.value,
            "task": task,
        }

    def _model_for_subagent_role(self, role: SubagentRole) -> str:
        from biobank_agent.orchestrator import SubagentRole

        runtime = self._require_runtime()
        if role == SubagentRole.WORKER:
            return runtime.config.primary_model
        if role == SubagentRole.VERIFIER:
            return runtime.config.critic_model
        return runtime.config.planner_model

    def _normalize_subagent_summary(self, result: Any) -> dict[str, str]:
        text = ""
        model = ""
        run_id = ""
        if isinstance(result, dict):
            model = str(result.get("model") or "")
            run_id = str(result.get("run_id") or result.get("id") or "")
            result = result.get("response") or result
        if hasattr(result, "text"):
            text = str(getattr(result, "text", "") or "")
        elif isinstance(result, dict):
            text = str(result.get("text") or result.get("final_answer") or result.get("summary") or "")
        else:
            text = str(result or "")
        model = model or str(getattr(result, "model", "") or getattr(result, "provider", "") or "")
        return {"text": text[:1200], "model": model, "run_id": run_id}

    def _record_action_graph_node(self, *, node_type: str, node_id: str, payload: dict[str, Any], score: float = 1.0) -> str:
        runtime = self._require_runtime()
        session = self._require_session()
        node_key = runtime.action_graph.upsert_node(node_type, node_id, payload=payload, score=score)
        session.action_graph_refs.append(
            ActionGraphNode(node_type=node_type, node_id=node_id, payload=dict(payload), score=score, node_key=node_key)
        )
        self._record_event(
            AgentEvent.make(
                AgentEventType.ACTION_GRAPH_NODE_CREATED,
                session_id=session.session_id,
                node_key=node_key,
                node_type=node_type,
                node_id=node_id,
                payload=dict(payload),
            )
        )
        return node_key

    def _show_mcp_tables(self, manager: McpManager, *, verbose: bool = False) -> None:
        configs = manager.load_config()
        status_by_name = {row["name"]: row for row in manager.status()}
        table = Table(title=f"MCP Servers ({manager.config_path})", box=box.SIMPLE)
        table.add_column("Name", style="cyan")
        table.add_column("Transport")
        table.add_column("Enabled", justify="center")
        table.add_column("Running", justify="center")
        table.add_column("Tools", justify="right")
        table.add_column("Safety")
        table.add_column("Target", overflow="fold")
        table.add_column("Error", overflow="fold")
        if configs:
            for cfg in configs:
                transport = str(cfg.transport).strip().lower()
                target = cfg.url if transport in {"http", "http-sse", "sse", "streamable-http"} else " ".join([cfg.command, *cfg.args]).strip()
                status = status_by_name.get(cfg.name, {})
                safety = "remote"
                if cfg.enabled:
                    safety = "enabled"
                table.add_row(
                    cfg.name,
                    cfg.transport,
                    "yes" if cfg.enabled else "no",
                    "yes" if status.get("running") else "no",
                    str(status.get("loaded_tools", 0)),
                    safety,
                    target,
                    str(status.get("error", "")),
                )
        else:
            table.add_row("(none)", "", "", "", "", "", "Create ~/.biobank_agent/mcp_servers.json", "")
        self.console.print(table)

        remote_handlers = [handler for handler in manager.handlers if handler.name.startswith("mcp_")]
        tools = Table(title=f"Loaded MCP Tools ({len(remote_handlers)})", box=box.SIMPLE)
        tools.add_column("Tool", style="cyan")
        tools.add_column("Capabilities")
        tools.add_column("Safety")
        if verbose:
            tools.add_column("Schema", overflow="fold")
        if remote_handlers:
            for handler in remote_handlers:
                meta = handler.spec().metadata()
                caps = ", ".join(cap.value for cap in handler.required_capabilities()) or "-"
                safety = f"{meta.get('safety_class', '')}/{meta.get('approval_requirement', '')}/{meta.get('workspace_scope', '')}"
                row = [handler.name, caps, safety]
                if verbose:
                    row.append(json.dumps(handler.spec().to_openai_schema(), ensure_ascii=False, default=str))
                tools.add_row(*row)
        else:
            row = ["(none)", "Run /mcp-start after configuring an MCP server", "-"]
            if verbose:
                row.append("")
            tools.add_row(*row)
        self.console.print(tools)

        if verbose:
            summary = Table(title="MCP Safety Summary", box=box.SIMPLE)
            summary.add_column("Server", style="cyan")
            summary.add_column("Tools")
            summary.add_column("Approval")
            summary.add_column("Scope")
            summary.add_column("Error")
            for cfg in configs or []:
                tools_for_server = [handler for handler in remote_handlers if handler.name.startswith(f"mcp_{cfg.name.replace(' ', '_')}")]
                summary.add_row(
                    cfg.name,
                    str(len(tools_for_server)),
                    "loaded",
                    "remote",
                    str(status_by_name.get(cfg.name, {}).get("error", "")),
                )
            if not configs:
                summary.add_row("(none)", "0", "-", "-", "Create ~/.biobank_agent/mcp_servers.json")
            self.console.print(summary)

    def _render_mcp_health(self, rows: list[dict[str, Any]], *, repair: bool = False) -> None:
        table = Table(title=f"MCP Health ({'repair' if repair else 'probe'})", box=box.SIMPLE)
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
        self.console.print(table)

    def _mcp_status_payload(
        self,
        manager: McpManager,
        *,
        action: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "action": action,
            "status": "ok",
            "config_path": str(manager.config_path),
            "servers": manager.status(),
            "loaded_tools": [handler.name for handler in manager.handlers],
        }
        if extra:
            payload.update(extra)
        return payload

    def _parse_mcp_call_args(self, raw: str) -> tuple[str, dict[str, Any]]:
        text = (raw or "").strip()
        if not text:
            raise ValueError("Usage: /mcp call <tool> [json|key=value...]")
        parts = shlex.split(text)
        if not parts:
            raise ValueError("Usage: /mcp call <tool> [json|key=value...]")
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
                parsed = json.loads(candidate)
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

    async def _call_mcp_tool(self, manager: McpManager, server_name: str, remote_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session()
        result = await manager.call_tool(server_name, remote_name, tool_args)
        node_key = self._record_action_graph_node(
            node_type="mcp",
            node_id=f"{server_name}__{remote_name}",
            payload={
                "action": "call",
                "server": server_name,
                "remote_tool": remote_name,
                "args": tool_args,
                "result": result,
                "approval_profile": runtime.config.approval_profile,
                "session_id": session.session_id,
            },
            score=0.7,
        )
        self._record_event(
            AgentEvent.make(
                AgentEventType.PLAN_PHASE,
                session_id=session.session_id,
                phase="MCP",
                actor="biobank",
                status="success",
                message=f"called {server_name}__{remote_name}",
                metadata={
                    "action": "call",
                    "server": server_name,
                    "tool": remote_name,
                    "action_graph_node": node_key,
                },
            )
        )
        return result

    def _normalize_mcp_tool_name(self, tool_name: str) -> tuple[str, str]:
        text = str(tool_name or "").strip()
        if text.startswith("mcp_"):
            text = text[4:]
        if "__" not in text:
            raise ValueError("MCP tools must be addressed as <server>__<tool> or mcp_<server>__<tool>")
        server, remote = text.split("__", 1)
        return server, remote

    def _normalize_mcp_runtime_tool_name(self, tool_name: str) -> str:
        text = str(tool_name or "").strip()
        if text.startswith("mcp_"):
            return text
        if "__" not in text:
            raise ValueError("MCP tools must be addressed as <server>__<tool> or mcp_<server>__<tool>")
        return f"mcp_{text}"


def _path_state(path: Path) -> str:
    if path.exists():
        return f"ok: {path}"
    return f"missing: {path}"


def _title_from_task(task: str) -> str:
    text = " ".join(str(task or "").strip().split())
    if not text:
        return "Interactive session"
    return text[:80]


def _text_progress_bar(done: int, total: int, width: int = 24) -> str:
    total = max(0, int(total or 0))
    done = max(0, min(int(done or 0), total))
    if total <= 0:
        return "[dim]" + ("░" * width) + "[/dim]"
    filled = int(width * done / total)
    return f"[green]{'█' * filled}[/green][dim]{'░' * (width - filled)}[/dim]"


def run_interactive_shell(settings: Any, *, initial_task: str = "", console: Console | None = None,
                          workspace: str = "") -> None:
    shell = InteractiveShell(settings=settings, console=console or Console(), workspace=workspace or "")
    shell.run(initial_task=initial_task)


__all__ = ["InteractiveShell", "LLMProvider", "run_interactive_shell"]
