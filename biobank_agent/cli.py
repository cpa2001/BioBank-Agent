"""Rich-based CLI for Biobank Agent.

Entry point: `bb` command (configured in pyproject.toml).
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


def main() -> None:
    """CLI entry point."""
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
    console.print(Panel(
        agent.state.context_summary(),
        title="Session Status",
    ))


def _show_history(agent: Agent) -> None:
    if not agent.state.records:
        console.print("[dim]No analyses performed yet.[/]")
        return
    for r in agent.state.records:
        console.print(f"[dim]{r.timestamp}[/] [cyan]{r.skill}[/]({r.args}) → {r.key_results}")


if __name__ == "__main__":
    main()
