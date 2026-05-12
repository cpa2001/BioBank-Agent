"""Small slash-command protocol used by the v3 CLI package.

Command modules depend on this context and injected actions rather than
importing the legacy Rich CLI. That keeps commands reusable across the default
REPL, Textual TUI, and future embedded shells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


Action = Callable[..., Any]
Handler = Callable[["CommandContext", str], None]


@dataclass
class CommandContext:
    """Runtime objects available to a slash command."""

    agent: Any
    planner: Any
    token_usage: dict[str, Any]
    console: Any
    actions: dict[str, Action] = field(default_factory=dict)

    def action(self, name: str) -> Action:
        try:
            return self.actions[name]
        except KeyError as exc:
            raise RuntimeError(f"Slash command action {name!r} is not registered") from exc


@dataclass(frozen=True)
class RegisteredCommand:
    """Command metadata plus executable handler."""

    name: str
    usage: str
    description: str
    handle: Handler


class SlashCommand(Protocol):
    """Protocol implemented by command registry entries."""

    name: str
    usage: str
    description: str

    def handle(self, ctx: CommandContext, arg: str) -> None:
        """Execute the command."""


def action(name: str, *extra_args: Any) -> Handler:
    """Build a command handler that forwards to an injected legacy/TUI action."""

    def run(ctx: CommandContext, arg: str) -> None:
        ctx.action(name)(*extra_args)

    return run


__all__ = ["CommandContext", "Handler", "RegisteredCommand", "SlashCommand", "action"]
