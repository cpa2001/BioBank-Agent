"""Research-mode slash command (multi-agent cited research brief)."""

from __future__ import annotations

from .base import CommandContext, RegisteredCommand


def _research(ctx: CommandContext, arg: str) -> None:
    return ctx.action("research")(arg)


def commands() -> list[RegisteredCommand]:
    return [
        RegisteredCommand(
            "/research",
            "/research <question>",
            "Run a multi-agent, multi-source cited research brief",
            _research,
        ),
    ]


__all__ = ["commands"]
