"""Rich-based CLI for Biobank Agent.

Entry point: `biobank` command (configured in pyproject.toml).
Subcommands: `biobank rebuild-parquet` for batch parquet rebuild.
Slash commands: /skills, /status, /history, /plan, /compact, /clear, /cost,
                /model, /export, /help, /figures, /cohorts, /models,
                /record, /pipelines, /errors, /memory
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import Agent
from .config import get_settings
from .planner import PlanMode

console = Console()

BANNER = r"""
[bold cyan]╔═══════════════════════════════════════════════════╗
║   Biobank Agent v2.0                              ║
║   Autonomous Scientific Discovery for UK Biobank  ║
║   502K subjects · 4,971 fields · 6.9M diagnoses   ║
╚═══════════════════════════════════════════════════╝[/]
"""

HELP_TEXT = """
[bold]Available Commands:[/]

[bold cyan]Session[/]
  /help              Show this help message
  /status            Session state, platform info, memory summary
  /cost              Show token usage and estimated cost
  /compact           Compress conversation history (keep last 10 turns)
  /clear             Reset session state (cohorts, models, figures)
  /export [format]   Export session as JSON or Markdown

[bold cyan]Analysis[/]
  /skills            List all available analysis tools
  /history           Show analysis history
  /figures           List all generated figures
  /cohorts           List active cohorts with summary stats
  /models            List trained models with AUC

[bold cyan]Planning[/]
  /plan <task>       Enter plan mode for complex tasks
  /plan-approve      Approve plan and begin execution
  /plan-exit         Exit plan mode
  /plans             List all saved plans

[bold cyan]Pipelines & Memory[/]
  /record <name>     Save current session as a replayable pipeline
  /pipelines         List saved pipelines
  /memory            Show long-term memory summary
  /errors            Show error catalog from long-term memory

[bold cyan]Configuration[/]
  /model <name>      Switch LLM model at runtime

[bold cyan]General[/]
  quit / exit / q    Exit the agent
"""


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


def main() -> None:
    """CLI entry point."""
    # Check for subcommands
    if len(sys.argv) > 1 and sys.argv[1] == "rebuild-parquet":
        rebuild_parquet_cmd()
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

    console.print(BANNER)

    # Load settings
    settings = get_settings()
    if model:
        settings.llm_model = model

    console.print(f"[dim]Model: {settings.llm_model}[/]")
    console.print(f"[dim]Data: {settings.ukb_parquet_dir}[/]")
    console.print()

    # Create agent
    with console.status("[bold green]Loading data layer..."):
        agent = Agent(settings)

    # Initialize plan mode
    planner = PlanMode(settings.plans_dir)

    # Token tracking
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_cost_usd": 0.0}

    console.print(f"[green]>[/] {len(agent.registry)} skills loaded")
    console.print(f"[green]>[/] {agent.dm.count_subjects():,} subjects available")
    console.print(f"[green]>[/] {len(agent.catalog.fields):,} field definitions")
    console.print()
    console.print("[dim]Type your query, or /help for commands. Ctrl+C to interrupt.[/]")
    console.print()

    # REPL loop
    while True:
        try:
            prompt_str = "[bold magenta]plan>[/] " if planner.is_active else "[bold cyan]biobank>[/] "
            query = console.input(prompt_str).strip()
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
        console.print(Markdown(HELP_TEXT))

    elif cmd == "/status":
        _show_status(agent, planner, token_usage)

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
            agent.settings.llm_model = arg
            agent.llm.model = arg
            console.print(f"Model switched: {old} → [cyan]{arg}[/]")

    else:
        console.print(f"[yellow]Unknown command: {cmd}. Type /help for available commands.[/]")


# ── Command implementations ─────────────────────────────────


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

    if planner.is_active:
        lines.append(f"\nPlan Mode: [magenta]{planner.status}[/]")
        if planner.current_plan:
            lines.append(f"Plan File: {planner.current_plan.name}")

    lines.extend([
        "",
        f"Platform: {platform_summary()}",
        f"Model: {agent.settings.llm_model}",
        f"Data: {agent.settings.ukb_parquet_dir}",
        f"Skills: {len(agent.registry)}",
        f"Tokens: {token_usage['prompt_tokens']:,} prompt + {token_usage['completion_tokens']:,} completion",
    ])

    mem_summary = agent.memory.summary()
    if mem_summary:
        lines.append(mem_summary)
    console.print(Panel("\n".join(lines), title="Session Status"))


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


if __name__ == "__main__":
    main()
