"""Compatibility re-export for the migrated command protocol types."""

from biobank_agent.cli.commands.base import CommandContext, SlashCommand

__all__ = ["CommandContext", "SlashCommand"]
