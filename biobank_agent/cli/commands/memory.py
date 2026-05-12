"""Pipeline, memory, and debate slash commands."""

from __future__ import annotations

from .base import CommandContext, RegisteredCommand, action


def _record(ctx: CommandContext, arg: str) -> None:
    if not arg:
        ctx.console.print("[yellow]Usage: /record <pipeline_name>[/]")
        return
    ctx.action("record")(arg)


def _debate(ctx: CommandContext, arg: str) -> None:
    if not arg:
        ctx.console.print("[yellow]Usage: /debate <query>[/]")
        return
    ctx.action("debate")(arg)


def commands() -> list[RegisteredCommand]:
    return [
        RegisteredCommand("/record", "/record <pipeline_name>", "Save current session as a replayable macro", _record),
        RegisteredCommand("/pipelines", "/pipelines", "List saved analysis pipelines", action("pipelines")),
        RegisteredCommand("/errors", "/errors", "Show common recorded tool errors", action("errors")),
        RegisteredCommand("/memory", "/memory", "Show long-term memory summary", action("memory")),
        RegisteredCommand("/debate", "/debate <query>", "Force a multi-model debate for one query", _debate),
    ]


__all__ = ["commands"]
