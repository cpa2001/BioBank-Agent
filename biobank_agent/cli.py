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
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import Agent
from .config import get_settings
from .planner import PlanMode

console = Console()

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
    ]),
    ("Planning", [
        ("/plan <task>", "Enter plan mode for complex tasks"),
        ("/plan-approve", "Approve plan and begin execution"),
        ("/plan-exit", "Exit plan mode"),
        ("/plans", "List all saved plans"),
    ]),
    ("Pipelines & Memory", [
        ("/record <name>", "Save current session as a replayable pipeline"),
        ("/pipelines", "List saved pipelines"),
        ("/memory", "Show long-term memory summary"),
        ("/errors", "Show error catalog from long-term memory"),
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


def _prompt_message(is_plan_mode: bool) -> FormattedText:
    """Prompt prefix with mode-aware visual style."""
    if is_plan_mode:
        return FormattedText(
            [
                ("class:prompt.plan", "plan"),
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
    if prompt_session is None:
        prompt_str = "[bold magenta]plan>[/] " if planner.is_active else "[bold cyan]biobank>[/] "
        return console.input(prompt_str)
    return prompt_session.prompt(_prompt_message(planner.is_active))


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
    left.add_row("[bold #56d4dd]Biobank Agent[/] [bold #94a3b8]v2.0[/]")
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


def eval_cmd() -> None:
    """Subcommand: run benchmark suites with reliability observability metrics.

    Usage:
      biobank eval --suite research_eval_v1 --mode baseline|mas_v2 [--enforce-gate]
    """
    from .eval.benchmarks import (
        BiomedQABenchmark,
        ResearchEvalV1,
        SkillCallBenchmark,
        SkillSchemaBenchmark,
    )
    from .eval.harness import EvalHarness

    suite = "research_eval_v1"
    mode = "baseline"
    enforce_gate = False
    ab_compare = False
    baseline_report = ""
    for arg in sys.argv[2:]:
        if arg.startswith("--suite="):
            suite = arg.split("=", 1)[1].strip().lower()
        elif arg.startswith("--mode="):
            mode = arg.split("=", 1)[1].strip().lower()
        elif arg.startswith("--baseline-report="):
            baseline_report = arg.split("=", 1)[1].strip()
        elif arg == "--ab":
            ab_compare = True
        elif arg == "--enforce-gate":
            enforce_gate = True

    settings = get_settings()
    settings.ensure_dirs()
    if mode == "baseline":
        settings.multi_model_enabled = False
    elif mode == "mas_v2":
        settings.multi_model_enabled = True
    else:
        console.print(f"[red]Unknown mode: {mode}. Use baseline or mas_v2.[/]")
        raise SystemExit(2)

    benchmarks = {
        "research_eval_v1": ResearchEvalV1,
        "skill_schemas": SkillSchemaBenchmark,
        "biomedical_qa": BiomedQABenchmark,
        "skill_calls": SkillCallBenchmark,
    }
    if suite not in benchmarks:
        console.print(f"[red]Unknown suite: {suite}.[/]")
        console.print(f"[dim]Available: {', '.join(sorted(benchmarks))}[/]")
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

    with console.status(f"[bold green]Running {suite} ({mode}) ..."):
        result = harness.run(
            benchmark=benchmark,
            agent=agent,
            mode=mode,
            enforce_gate=enforce_gate,
            baseline_observability=baseline_obs,
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
        "observability": obs,
        "baseline_observability": result.baseline_observability,
        "comparative": result.comparative,
        "gate_passed": result.gate_passed,
        "gate_failures": result.gate_failures,
        "n_total": result.n_total,
        "n_passed": result.n_passed,
        "timestamp": result.timestamp,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    console.print(f"[green]Saved report:[/] {out_path}")

    if enforce_gate and not result.gate_passed:
        raise SystemExit(3)


def main() -> None:
    """CLI entry point."""
    # Check for subcommands
    if len(sys.argv) > 1 and sys.argv[1] == "rebuild-parquet":
        rebuild_parquet_cmd()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        eval_cmd()
        return

    # Parse optional args
    model = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]

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

    # Create agent
    with console.status("[bold green]Loading data layer..."):
        agent = Agent(settings)

    # Warn about deprecated env var names
    import os as _os
    if _os.getenv("UKB_PARQUET_DIR") and not _os.getenv("DATA_DIR"):
        console.print(
            "[dim yellow]Note: UKB_PARQUET_DIR is deprecated; "
            "rename to DATA_DIR in .env (both work for now)[/]"
        )

    # Initialize plan mode
    planner = PlanMode(settings.plans_dir)
    prompt_session = _build_prompt_session(settings) if sys.stdin.isatty() else None

    # Token tracking
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_cost_usd": 0.0}

    n_subjects = None
    try:
        n_subjects = agent.dm.count_subjects()
    except Exception:
        pass

    n_fields = len(agent.catalog.fields)
    model_pool = [spec.model_id for spec in agent.orchestrator.model_pool]
    _render_startup_dashboard(
        settings=settings,
        n_skills=len(agent.registry),
        n_subjects=n_subjects,
        n_fields=n_fields,
        model_pool=model_pool,
        available_model_count=len(getattr(agent, "available_models", [])),
    )

    if n_subjects is None:
        console.print(f"[yellow]![/] Biomarker data not found at {settings.data_dir}")
        console.print("[dim]Set DATA_DIR in .env to your parquet directory[/]")
    if n_fields <= 0:
        console.print(f"[yellow]![/] Field catalogue empty (check {settings.field_txt})")
    console.print("[dim]Press Ctrl+C or type quit to exit.[/]")
    console.print()

    # REPL loop
    while True:
        try:
            query = _read_query(prompt_session, planner).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/]")
            break

        # ── Slash commands ──────────────────────────────────
        if query.startswith("/"):
            _handle_command(query, agent, planner, token_usage)
            continue

        # ── Run agent ───────────────────────────────────────
        try:
            # In plan mode, prepend plan context
            if planner.is_active:
                plan_context = (
                    f"[PLAN MODE - Status: {planner.status}]\n"
                    f"Current plan:\n{planner.get_plan_content()}\n\n"
                    f"User says: {query}"
                )
                effective_query = plan_context
            else:
                effective_query = query

            figs_before = len(agent.state.figures)

            with console.status("[bold green]Thinking..."):
                response = agent.run(effective_query)

            # Update cumulative token usage
            tu = agent.state.token_usage
            token_usage["prompt_tokens"] = tu.prompt_tokens
            token_usage["completion_tokens"] = tu.completion_tokens

            console.print()
            console.print(Markdown(response))
            console.print()

            # Show new figures from this turn only
            new_figs = agent.state.figures[figs_before:]
            if new_figs:
                for fig_path in new_figs[-5:]:
                    console.print(f"[dim]Figure saved: {fig_path}[/]")
                console.print()

            # Show token usage after each turn
            console.print(
                f"[dim]tokens: {tu.prompt_tokens:,} in + "
                f"{tu.completion_tokens:,} out[/]"
            )

        except KeyboardInterrupt:
            agent.state.interrupted = True
            console.print("\n[yellow]Interrupted.[/]")
            agent.state.interrupted = False
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")


def _handle_command(
    query: str,
    agent: Agent,
    planner: PlanMode,
    token_usage: dict,
) -> None:
    """Dispatch slash commands."""
    parts = query.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    # ── Session commands ────────────────────────────────
    if cmd == "/help":
        _show_help()

    elif cmd == "/status":
        _show_status(agent, planner, token_usage)

    elif cmd == "/routing-status":
        _show_routing_status(agent)

    elif cmd == "/cost":
        _show_cost(token_usage, agent)

    elif cmd == "/compact":
        _compact(agent)

    elif cmd == "/clear":
        _clear(agent)

    elif cmd == "/export":
        fmt = arg or "json"
        _export(agent, fmt)

    # ── Analysis commands ───────────────────────────────
    elif cmd == "/skills":
        _show_skills(agent)

    elif cmd == "/history":
        _show_history(agent)

    elif cmd == "/figures":
        _show_figures(agent)

    elif cmd == "/cohorts":
        _show_cohorts(agent)

    elif cmd == "/models":
        _show_models(agent)

    elif cmd == "/evidence":
        if not arg:
            console.print("[yellow]Usage: /evidence <claim_id>[/]")
        else:
            _show_evidence(agent, arg)

    # ── Plan commands ───────────────────────────────────
    elif cmd == "/plan":
        if not arg:
            console.print("[yellow]Usage: /plan <task description>[/]")
        else:
            result = planner.enter(arg)
            console.print(Markdown(result))

    elif cmd == "/plan-approve":
        result = planner.approve()
        console.print(Markdown(result))

    elif cmd == "/plan-exit":
        result = planner.exit()
        console.print(result)

    elif cmd == "/plans":
        plans = planner.list_plans()
        if not plans:
            console.print("[dim]No saved plans.[/]")
        else:
            for p in plans:
                status_color = {"DONE": "green", "EXECUTION": "yellow", "BLOCKED": "red"}.get(p["status"], "dim")
                console.print(f"  [{status_color}]{p['status']}[/] {p['file']}")

    # ── Pipeline & memory commands ──────────────────────
    elif cmd == "/record":
        if not arg:
            console.print("[yellow]Usage: /record <pipeline_name>[/]")
        else:
            _record_pipeline(agent, arg)

    elif cmd == "/pipelines":
        _show_pipelines(agent)

    elif cmd == "/errors":
        _show_errors(agent)

    elif cmd == "/memory":
        _show_memory(agent)

    # ── Configuration commands ──────────────────────────
    elif cmd == "/model":
        if not arg:
            console.print(f"Current model: [cyan]{agent.settings.llm_model}[/]")
        else:
            old = agent.settings.llm_model
            switch_fn = getattr(agent, "switch_model", None)
            if callable(switch_fn):
                switch_fn(arg)
            # Keep direct assignments for compatibility with mocks/tests.
            agent.settings.llm_model = arg
            agent.llm.model = arg
            console.print(f"Model switched: {old} → [cyan]{arg}[/]")

    elif cmd == "/models-pool":
        _show_model_pool(agent)

    elif cmd == "/models-available":
        _show_available_llm_models(agent)

    elif cmd == "/strategy":
        _set_strategy(agent, arg)

    elif cmd == "/debate":
        if not arg:
            console.print("[yellow]Usage: /debate <query>[/]")
        else:
            _force_debate(agent, arg, token_usage)

    else:
        console.print(f"[yellow]Unknown command: {cmd}. Type /help for available commands.[/]")


# ── Command implementations ─────────────────────────────────


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
        if planner.current_plan:
            lines.append(f"Plan File: {planner.current_plan.name}")

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

    table = Table(title="Routing Status")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("strategy", str(strategy))
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
