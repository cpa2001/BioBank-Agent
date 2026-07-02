"""Plugin marketplace slash command (Claude-Code-style plugin consumption)."""

from __future__ import annotations

from .base import CommandContext, RegisteredCommand


def _plugin(ctx: CommandContext, arg: str) -> None:
    return ctx.action("plugin")(arg)


def commands() -> list[RegisteredCommand]:
    return [
        RegisteredCommand(
            "/plugin",
            "/plugin marketplace add <repo> | /plugin install <name> | /plugin list",
            "Consume Claude-Code-style plugins: add a marketplace, install a plugin's skills (hooks gated)",
            _plugin,
        ),
    ]


__all__ = ["commands"]
