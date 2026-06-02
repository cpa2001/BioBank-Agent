"""Session, inspection, and model slash commands."""

from __future__ import annotations

from .base import CommandContext, RegisteredCommand, action


def _export(ctx: CommandContext, arg: str) -> None:
    return ctx.action("export")(arg or "json")


def _evidence(ctx: CommandContext, arg: str) -> None:
    if not arg:
        ctx.console.print("[yellow]Usage: /evidence <claim_id>[/]")
        return
    return ctx.action("show_evidence")(arg)


def _model(ctx: CommandContext, arg: str) -> None:
    return ctx.action("model")(arg)


def _strategy(ctx: CommandContext, arg: str) -> None:
    return ctx.action("strategy")(arg)


def commands() -> list[RegisteredCommand]:
    return [
        RegisteredCommand("/help", "/help", "Show this help message", action("help")),
        RegisteredCommand("/status", "/status", "Session state, platform info, memory summary", action("status")),
        RegisteredCommand("/graph", "/graph [limit]", "Print the current session Action Graph flowchart", action("graph")),
        RegisteredCommand("/routing-status", "/routing-status", "Show latest orchestration trace, claims, and safety status", action("routing_status")),
        RegisteredCommand("/cost", "/cost", "Show token usage and estimated cost", action("cost")),
        RegisteredCommand("/compact", "/compact", "Compress conversation history", action("compact")),
        RegisteredCommand("/clear", "/clear", "Reset session state", action("clear")),
        RegisteredCommand("/export", "/export [format]", "Export session as JSON or Markdown", _export),
        RegisteredCommand("/skills", "/skills", "List all available analysis tools", action("skills")),
        RegisteredCommand("/tools", "/tools", "Show available tools by category and health", action("tools")),
        RegisteredCommand("/history", "/history", "Show analysis history", action("history")),
        RegisteredCommand("/figures", "/figures", "List all generated figures", action("figures")),
        RegisteredCommand("/cohorts", "/cohorts", "List active cohorts with summary stats", action("cohorts")),
        RegisteredCommand("/models", "/models", "List trained models with AUC", action("models")),
        RegisteredCommand("/evidence", "/evidence <claim_id>", "Show evidence chain for a claim from Action Graph", _evidence),
        RegisteredCommand("/model", "/model <name>", "Switch LLM model at runtime", _model),
        RegisteredCommand("/models-pool", "/models-pool", "Show available models in the pool", action("models_pool")),
        RegisteredCommand("/models-available", "/models-available", "Fetch model list from current relay endpoint", action("models_available")),
        RegisteredCommand("/strategy", "/strategy <mode>", "Set routing: auto, single, debate, ensemble", _strategy),
    ]


__all__ = ["commands"]
