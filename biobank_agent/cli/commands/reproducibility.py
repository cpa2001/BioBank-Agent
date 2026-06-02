"""Reproducibility and self-evolution slash commands."""

from __future__ import annotations

from .base import CommandContext, RegisteredCommand, action


def _replay(ctx: CommandContext, arg: str) -> None:
    return ctx.action("replay")(arg)


def _harness(ctx: CommandContext, arg: str) -> None:
    return ctx.action("harness")(arg)


def _learn(ctx: CommandContext, arg: str) -> None:
    return ctx.action("learn")(arg)


def commands() -> list[RegisteredCommand]:
    return [
        RegisteredCommand("/replicate", "/replicate <paper_path_or_text>", "Draft a review-only paper replication StudySpec and plan", action("replicate")),
        RegisteredCommand("/replay", "/replay [session-id|trajectory.jsonl]", "Replay and validate a runtime trajectory without model/tool execution", _replay),
    ]


__all__ = ["commands"]
