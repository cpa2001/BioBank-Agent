"""Compatibility package for the v3 command registry.

New command implementations live under ``biobank_agent.cli.commands``. This
module keeps older imports working while the legacy Rich CLI continues to call
the same registry.
"""

from .base import CommandContext, SlashCommand
from .registry import build_core_registry

__all__ = ["CommandContext", "SlashCommand", "build_core_registry"]
