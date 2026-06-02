"""Small slash-command protocol used by the v3 CLI package.

Command modules depend on this context and injected actions rather than
importing the legacy Rich CLI. That keeps commands reusable across the default
REPL, Textual TUI, and future embedded shells.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


Action = Callable[..., Any]
Handler = Callable[["CommandContext", str], Any]


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


@dataclass(frozen=True)
class ParsedSlashCommand:
    """Normalized slash-command input."""

    name: str
    arg: str = ""
    argv: tuple[str, ...] = ()
    raw: str = ""


def parse_slash_command(text: str) -> ParsedSlashCommand | None:
    """Parse a user-entered slash command.

    The command token is case-insensitive and may be surrounded by arbitrary
    whitespace. The remainder is preserved exactly as ``arg`` so natural
    language prompts and paths do not lose formatting.
    """
    raw = str(text or "")
    stripped = raw.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped.split(None, 1)
    name = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    try:
        argv = tuple(shlex.split(arg)) if arg else ()
    except ValueError:
        argv = tuple(arg.split()) if arg else ()
    return ParsedSlashCommand(name=name, arg=arg, argv=argv, raw=raw)


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
        return ctx.action(name)(*extra_args)

    return run


__all__ = [
    "CommandContext",
    "Handler",
    "ParsedSlashCommand",
    "RegisteredCommand",
    "SlashCommand",
    "action",
    "parse_slash_command",
]
