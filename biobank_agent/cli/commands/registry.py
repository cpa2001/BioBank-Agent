"""Built-in slash command registry for the v3 CLI package.

The public API remains ``build_core_registry()``. Individual command groups now
live in small modules so the default REPL, Textual TUI, and future plugin
commands can share the same protocol without importing the legacy Rich CLI.
"""

from __future__ import annotations

import os
from importlib import import_module
from types import ModuleType
from typing import Any

from .base import CommandContext, Handler, ParsedSlashCommand, RegisteredCommand, parse_slash_command


BUILTIN_COMMAND_MODULES = (
    "session",
    "runtime",
    "plan",
    "research",
    "mcp",
    "reproducibility",
    "memory",
)
EXTRA_COMMAND_MODULES_ENV = "BIOBANK_CLI_COMMAND_MODULES"


def _load_command_module(name: str) -> ModuleType:
    module_name = str(name or "").strip()
    if not module_name:
        raise RuntimeError("Empty slash command module name")
    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        if "." in module_name:
            raise
        return import_module(f"{__package__}.{module_name}")


def _extra_command_modules() -> tuple[str, ...]:
    raw = os.getenv(EXTRA_COMMAND_MODULES_ENV, "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def iter_registered_commands(*, include_external: bool = True) -> list[RegisteredCommand]:
    """Load built-in command modules and return their command entries."""
    rows: list[RegisteredCommand] = []
    module_names = [*BUILTIN_COMMAND_MODULES]
    if include_external:
        module_names.extend(_extra_command_modules())
    for module_name in module_names:
        module = _load_command_module(module_name)
        factory = getattr(module, "commands", None)
        if not callable(factory):
            raise RuntimeError(f"Slash command module {module.__name__} has no commands() factory")
        for command in factory():
            if not isinstance(command.name, str) or not command.name.startswith("/"):
                raise RuntimeError(f"Invalid slash command name from {module.__name__}: {command!r}")
            rows.append(command)
    return rows


def build_core_registry() -> dict[str, RegisteredCommand]:
    """Return commands that are safe to dispatch through the registry now."""
    registry: dict[str, RegisteredCommand] = {}
    for row in iter_registered_commands():
        if row.name in registry:
            raise RuntimeError(f"Duplicate slash command registered: {row.name}")
        registry[row.name] = row
    return registry


class SlashCommandRegistry:
    """Dispatch slash commands through a shared parser and command table."""

    def __init__(self, commands: dict[str, RegisteredCommand] | None = None) -> None:
        self.commands = dict(commands or build_core_registry())

    def parse(self, text: str) -> ParsedSlashCommand | None:
        return parse_slash_command(text)

    def get(self, name: str) -> RegisteredCommand | None:
        return self.commands.get(str(name or "").lower())

    def dispatch(self, text: str, ctx: CommandContext) -> Any:
        parsed = self.parse(text)
        if parsed is None:
            raise ValueError("not a slash command")
        command = self.get(parsed.name)
        if command is None:
            raise KeyError(parsed.name)
        return command.handle(ctx, parsed.arg)


__all__ = [
    "BUILTIN_COMMAND_MODULES",
    "CommandContext",
    "EXTRA_COMMAND_MODULES_ENV",
    "Handler",
    "ParsedSlashCommand",
    "RegisteredCommand",
    "SlashCommandRegistry",
    "build_core_registry",
    "iter_registered_commands",
    "parse_slash_command",
]
