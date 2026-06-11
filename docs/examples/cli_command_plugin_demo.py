"""Minimal Biobank Agent slash-command plugin.

Run from the repository root:

    PYTHONPATH=docs/examples \
    BIOBANK_CLI_COMMAND_MODULES=cli_command_plugin_demo \
    biobank

Then type:

    /plugin-demo hello
"""

from __future__ import annotations

from biobank_agent.cli.commands.base import CommandContext, RegisteredCommand


def commands() -> list[RegisteredCommand]:
    def handle(ctx: CommandContext, arg: str) -> None:
        text = arg.strip() or "(empty)"
        ctx.console.print(f"[cyan]plugin-demo:[/] {text}")

    return [
        RegisteredCommand(
            name="/plugin-demo",
            usage="/plugin-demo <text>",
            description="Echo text through an external command plugin",
            handle=handle,
        )
    ]
