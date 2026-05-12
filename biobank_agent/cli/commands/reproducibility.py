"""Reproducibility and self-evolution slash commands."""

from __future__ import annotations

from .base import CommandContext, RegisteredCommand, action


def _replay(ctx: CommandContext, arg: str) -> None:
    ctx.action("replay")(arg)


def _evolve(ctx: CommandContext, arg: str) -> None:
    ctx.action("evolve")(arg)


def commands() -> list[RegisteredCommand]:
    return [
        RegisteredCommand("/replicate", "/replicate <paper_path_or_text>", "Draft a review-only paper replication StudySpec and plan", action("replicate")),
        RegisteredCommand("/replay", "/replay <hash|prefix|prov_id> [--strict|--dry-run]", "Replay checkpointed skill execution", _replay),
        RegisteredCommand("/evolve", "/evolve [--dry-run|--pause|--resume|--write-history|--scheduled-run]", "Inspect repeated tool failures and proposed safe improvements", _evolve),
    ]


__all__ = ["commands"]
