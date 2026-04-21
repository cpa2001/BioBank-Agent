"""Rich-based CLI for Biobank Agent.

Entry point: `bb` command (configured in pyproject.toml).
Subcommands: `bb rebuild-parquet` for batch parquet rebuild.
"""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .agent import Agent
from .config import get_settings

console = Console()

BANNER = r"""
[bold cyan]╔══════════════════════════════════════════╗
║   Biobank Agent (bb) v0.1.0              ║
║   UK Biobank Phenotype Analysis          ║
║   502K subjects · 8,868 fields · ICD10   ║
╚══════════════════════════════════════════╝[/]
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

    console.print(f"[green]✓[/] {len(agent.registry)} skills loaded")
    console.print(f"[green]✓[/] {agent.dm.count_subjects():,} subjects available")
    console.print(f"[green]✓[/] {len(agent.catalog.fields):,} field definitions")
    console.print()
    console.print("[dim]Type your query, or 'quit' to exit. Ctrl+C to interrupt.[/]")
    console.print()

    # REPL loop
    while True:
        try:
            query = console.input("[bold cyan]bb>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/]")
            break

        # Special commands
        if query == "/skills":
            _show_skills(agent)
            continue
        if query == "/status":
            _show_status(agent)
            continue
        if query == "/history":
            _show_history(agent)
            continue
        if query.startswith("/record "):
            name = query.split(" ", 1)[1].strip()
            _record_pipeline(agent, name)
            continue
        if query == "/pipelines":
            _show_pipelines(agent)
            continue
        if query == "/errors":
            _show_errors(agent)
            continue
        if query == "/memory":
            _show_memory(agent)
            continue

        # Run agent
        try:
            with console.status("[bold green]Thinking..."):
                response = agent.run(query)
            console.print()
            console.print(Markdown(response))
            console.print()

            # Show any new figures
            if agent.state.figures:
                last_figs = agent.state.figures[-3:]
                for fig_path in last_figs:
                    console.print(f"[dim]📊 Figure saved: {fig_path}[/]")
                console.print()

        except KeyboardInterrupt:
            agent.state.interrupted = True
            console.print("\n[yellow]Interrupted.[/]")
            agent.state.interrupted = False
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")


def _show_skills(agent: Agent) -> None:
    skills = agent.registry.list_skills()
    console.print(Panel(
        "\n".join(f"[cyan]{s['name']}[/]: {s['description']}" for s in skills),
        title="Available Skills",
    ))


def _show_status(agent: Agent) -> None:
    from .utils.platform import platform_summary
    status_lines = [
        agent.state.context_summary(),
        "",
        f"Platform: {platform_summary()}",
        f"Model: {agent.settings.llm_model}",
        f"Data: {agent.settings.ukb_parquet_dir}",
        f"Skills: {len(agent.registry)}",
    ]
    mem_summary = agent.memory.summary()
    if mem_summary:
        status_lines.append(mem_summary)
    console.print(Panel("\n".join(status_lines), title="Session Status"))


def _show_history(agent: Agent) -> None:
    if not agent.state.records:
        console.print("[dim]No analyses performed yet.[/]")
        return
    for r in agent.state.records:
        console.print(f"[dim]{r.timestamp}[/] [cyan]{r.skill}[/]({r.args}) → {r.key_results}")


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
    console.print(Panel(
        "\n".join(
            f"[red]{e['error_type']}[/] in [cyan]{e['skill']}[/] "
            f"({e['count']}x, last: {e['last_seen'][:10]})"
            for e in errors
        ),
        title="Error Catalog",
    ))


def _show_memory(agent: Agent) -> None:
    """Show long-term memory summary."""
    summary = agent.memory.summary()
    if not summary:
        console.print("[dim]Long-term memory is empty.[/]")
        return
    console.print(Panel(summary, title="Long-term Memory"))


if __name__ == "__main__":
    main()
