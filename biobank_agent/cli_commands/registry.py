"""Compatibility re-export for migrated built-in slash commands."""

from biobank_agent.cli.commands.registry import RegisteredCommand, build_core_registry

__all__ = ["RegisteredCommand", "build_core_registry"]
