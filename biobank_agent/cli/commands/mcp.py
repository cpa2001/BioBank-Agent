"""MCP slash commands."""

from __future__ import annotations

from .base import CommandContext, RegisteredCommand, action


def _mcp_call(ctx: CommandContext, arg: str) -> None:
    if not arg:
        ctx.console.print("[yellow]Usage: /mcp-call <mcp_server__tool> '{\"arg\":\"value\"}'[/]")
        return
    return ctx.action("mcp_call")(arg)


def _mcp_health(ctx: CommandContext, arg: str) -> None:
    return ctx.action("mcp_health")(arg)


def commands() -> list[RegisteredCommand]:
    return [
        RegisteredCommand("/mcp-list", "/mcp-list", "List configured MCP servers and loaded MCP tools", action("mcp_list")),
        RegisteredCommand("/mcp-start", "/mcp-start", "Start configured STDIO MCP servers and register remote tools", action("mcp_start")),
        RegisteredCommand("/mcp-health", "/mcp-health [--repair]", "Probe MCP servers and optionally reconnect unhealthy ones", _mcp_health),
        RegisteredCommand("/mcp-call", "/mcp-call <tool> [json|key=value...]", "Call a loaded MCP tool directly for audit/debugging", _mcp_call),
        RegisteredCommand("/mcp-stop", "/mcp-stop", "Stop active MCP clients and unregister remote tools", action("mcp_stop")),
    ]


__all__ = ["commands"]
