"""Runtime-oriented slash commands for the interactive shell."""

from __future__ import annotations

from .base import CommandContext, RegisteredCommand


def _call(ctx: CommandContext, name: str, arg: str = ""):
    return ctx.action(name)(arg)


def _goal(ctx: CommandContext, arg: str):
    return _call(ctx, "goal", arg)


def _resume(ctx: CommandContext, arg: str):
    return _call(ctx, "resume", arg)


def _new(ctx: CommandContext, arg: str):
    return _call(ctx, "new", arg)


def _fork(ctx: CommandContext, arg: str):
    return _call(ctx, "fork", arg)


def _diff(ctx: CommandContext, arg: str):
    return _call(ctx, "diff", arg)


def _cd(ctx: CommandContext, arg: str):
    return _call(ctx, "cd", arg)


def _jobs(ctx: CommandContext, arg: str):
    return _call(ctx, "jobs", arg)


def _job_tail(ctx: CommandContext, arg: str):
    return _call(ctx, "job_tail", arg)


def _permissions(ctx: CommandContext, arg: str):
    return _call(ctx, "permissions", arg)


def _doctor(ctx: CommandContext, arg: str):
    return _call(ctx, "doctor", arg)


def _agent(ctx: CommandContext, arg: str):
    return _call(ctx, "agent", arg)


def _subagents(ctx: CommandContext, arg: str):
    return _call(ctx, "subagents", arg)


def _review(ctx: CommandContext, arg: str):
    return _call(ctx, "review", arg)


def _audit(ctx: CommandContext, arg: str):
    return _call(ctx, "audit", arg)


def _harness(ctx: CommandContext, arg: str):
    return _call(ctx, "harness", arg)


def _learn(ctx: CommandContext, arg: str):
    return _call(ctx, "learn", arg)


def _verify(ctx: CommandContext, arg: str):
    return _call(ctx, "verify", arg)


def _quit(ctx: CommandContext, arg: str):
    return _call(ctx, "quit", arg)


def _evolve(ctx: CommandContext, arg: str):
    return _call(ctx, "evolve", arg)


def _mcp(ctx: CommandContext, arg: str):
    """Friendly aggregate alias for existing MCP subcommands."""
    if not arg:
        return ctx.action("mcp_list")()
    first = arg.split(None, 1)[0].strip().lower()
    rest = arg.split(None, 1)[1].strip() if len(arg.split(None, 1)) > 1 else ""
    if first in {"list", "status"}:
        return ctx.action("mcp_list")()
    if first == "start":
        return ctx.action("mcp_start")()
    if first == "stop":
        return ctx.action("mcp_stop")()
    if first in {"health", "doctor"}:
        return ctx.action("mcp_health")(rest)
    if first == "call":
        return ctx.action("mcp_call")(rest)
    ctx.console.print("[yellow]Usage: /mcp [list|start|health|call|stop][/]")
    return None


def commands() -> list[RegisteredCommand]:
    return [
        RegisteredCommand("/goal", "/goal [objective]", "Show or set the active long-running goal", _goal),
        RegisteredCommand("/resume", "/resume [session-id|--last]", "Resume or inspect saved sessions", _resume),
        RegisteredCommand("/new", "/new [title]", "Start a new interactive session", _new),
        RegisteredCommand("/fork", "/fork [session-id]", "Fork the current or selected session", _fork),
        RegisteredCommand("/cd", "/cd <path>", "Switch the active workspace directory (inputs/outputs go here)", _cd),
        RegisteredCommand("/workspace", "/workspace <path>", "Alias for /cd", _cd),
        RegisteredCommand("/jobs", "/jobs", "List background jobs (run_job) and their state", _jobs),
        RegisteredCommand("/job-tail", "/job-tail <job-id>", "Show the tail of a background job's log", _job_tail),
        RegisteredCommand("/diff", "/diff", "Show current workspace changes", _diff),
        RegisteredCommand("/permissions", "/permissions [profile]", "Show or change permission mode", _permissions),
        RegisteredCommand("/mcp", "/mcp [list|start|health|call|stop]", "Manage MCP servers and tools", _mcp),
        RegisteredCommand("/agent", "/agent", "Show active agent/runtime details", _agent),
        RegisteredCommand("/subagents", "/subagents", "Show configured subagent/reviewer routes", _subagents),
        RegisteredCommand("/review", "/review [focus]", "Run or inspect review hooks", _review),
        RegisteredCommand("/doctor", "/doctor", "Run read-only readiness diagnostics", _doctor),
        RegisteredCommand("/audit", "/audit [session-id]", "Summarize runtime trajectory and action graph evidence", _audit),
        RegisteredCommand("/harness", "/harness <task.json>", "Run a versioned runtime harness task through the interactive shell path", _harness),
        RegisteredCommand("/learn", "/learn [--write]", "Mine the active trajectory for review-only improvement proposals", _learn),
        RegisteredCommand("/evolve", "/evolve [--write|--apply]", "Review controlled self-evolution proposals; persistent apply is approval-gated", _evolve),
        RegisteredCommand("/verify", "/verify <command>", "Run a bounded verification command and record repair state on failure", _verify),
        RegisteredCommand("/quit", "/quit", "Exit interactive mode", _quit),
    ]


__all__ = ["commands"]
