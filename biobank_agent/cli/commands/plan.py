"""Plan-mode slash commands."""

from __future__ import annotations

from .base import CommandContext, RegisteredCommand, action


def _plan(ctx: CommandContext, arg: str) -> None:
    return ctx.action("plan")(arg)


def _plan_approve(ctx: CommandContext, arg: str) -> None:
    return ctx.action("plan_approve")()


def _plan_edit(ctx: CommandContext, arg: str) -> None:
    return ctx.action("plan_edit")(arg)


def _plan_title(ctx: CommandContext, arg: str) -> None:
    return ctx.action("plan_title")(arg)


def _plan_reject(ctx: CommandContext, arg: str) -> None:
    return ctx.action("plan_reject")(arg)


def _plan_option(ctx: CommandContext, arg: str) -> None:
    return ctx.action("plan_option")(arg)


def _plan_skip(ctx: CommandContext, arg: str) -> None:
    return ctx.action("plan_skip")(arg)


def _plan_retry(ctx: CommandContext, arg: str) -> None:
    return ctx.action("plan_retry")(arg)


def _plan_use(ctx: CommandContext, arg: str) -> None:
    return ctx.action("plan_use")(arg)


def commands() -> list[RegisteredCommand]:
    return [
        RegisteredCommand("/plan", "/plan <task>", "Design and execute a structured plan", _plan),
        RegisteredCommand("/plan-approve", "/plan-approve", "Approve plan and begin execution", _plan_approve),
        RegisteredCommand("/plan-edit", "/plan-edit <feedback>", "Refine plan with natural language", _plan_edit),
        RegisteredCommand("/plan-title", "/plan-title <title>", "Rename the active plan/report title", _plan_title),
        RegisteredCommand("/plan-rename", "/plan-rename <title>", "Alias for /plan-title", _plan_title),
        RegisteredCommand("/plan-reject", "/plan-reject [reason]", "Reject the draft plan without executing it", _plan_reject),
        RegisteredCommand("/plan-pause", "/plan-pause", "Pause plan execution", action("plan_pause")),
        RegisteredCommand("/plan-resume", "/plan-resume", "Resume paused plan execution after repair", action("plan_resume")),
        RegisteredCommand("/plan-diagnose", "/plan-diagnose", "Explain why the active plan is blocked or failed", action("plan_diagnose")),
        RegisteredCommand("/plan-retry", "/plan-retry [step_id]", "Retry the failed or named plan step", _plan_retry),
        RegisteredCommand("/plan-use", "/plan-use key=value", "Set plan context such as vcf_dir or workflow_mode", _plan_use),
        RegisteredCommand("/plan-option", "/plan-option <A|B|C|N>", "Choose a suggested repair/review option after a block", _plan_option),
        RegisteredCommand("/plan-skip", "/plan-skip <step_id>", "Explicitly skip an optional/diagnostic step", _plan_skip),
        RegisteredCommand("/plan-exit", "/plan-exit", "Exit plan mode", action("plan_exit")),
        RegisteredCommand("/plans", "/plans", "List all saved plans", action("plans")),
    ]


__all__ = ["commands"]
