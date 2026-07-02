"""Parallel + plan-mode orchestration over external coding-agent CLIs.

Fans several external agents (codex / claude-code / gemini) out concurrently, reusing the council
thread-pool with its timeout, per-item cancel, and partial-result harvest (``council.map_parallel``),
and adds a prompt-level "plan mode" that asks an agent for a plan instead of executing. The process
runner and agent detector are injected, so the orchestration is unit-testable offline with no real CLI.

Everything an external agent returns is UNTRUSTED: a caller that acts on a produced diff must route it
through the review-branch apply gate (``runtime.agent_delegation`` / ``runtime.self_evolve``). This
module only launches processes and hands back captured text; it is invoked only behind default-OFF gates.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from .council import CouncilContext, map_parallel
from .external_agents import (
    ExternalAgentResult,
    ExternalAgentSpec,
    Runner,
    _subprocess_runner,
    detect_available,
    parse_agent_list,
    run_external_agent,
)

# Plan mode is realised at the prompt level (reliable across every CLI) rather than via a per-CLI flag:
# the agent is asked to PLAN, not act. A caller that also wants execution runs a normal (non-plan) pass.
PLAN_MODE_PREAMBLE = (
    "You are in PLAN MODE. Do NOT modify files, run commands, or execute the task. Produce a concise, "
    "numbered, step-by-step plan to accomplish the request below, calling out risks and the exact "
    "commands/edits you WOULD make. Output only the plan.\n\nRequest:\n"
)


def plan_mode_prompt(prompt: str) -> str:
    """Wrap a request so an external agent returns a plan instead of executing it."""
    return f"{PLAN_MODE_PREAMBLE}{prompt}"


def run_external_agents_parallel(
    specs: Sequence[ExternalAgentSpec],
    prompt: str,
    *,
    cwd: str = "",
    timeout_s: float = 180.0,
    max_workers: int = 4,
    runner: Runner = _subprocess_runner,
    emit: Optional[Callable[..., None]] = None,
    invoker: Callable[..., ExternalAgentResult] = run_external_agent,
) -> list[ExternalAgentResult]:
    """Run several external agents concurrently on the same prompt; return results in spec order.

    Reuses ``council.map_parallel`` for the thread-pool, overall timeout, per-item cancel, and
    partial-result harvest. A slow or hanging agent is bounded by ``timeout_s`` and reported
    ``ok=False`` rather than blocking the others. Never raises — a worker error becomes a failed result.
    """
    specs = list(specs)
    if not specs:
        return []
    # provider_router is unused by map_parallel (it only reads max_workers/timeout_s/emit); pass None.
    ctx = CouncilContext(provider_router=None, max_workers=max(1, int(max_workers)), timeout_s=float(timeout_s), emit=emit)

    def _run(spec: ExternalAgentSpec) -> ExternalAgentResult:
        return invoker(spec, prompt, cwd=cwd, timeout_s=timeout_s, runner=runner)

    mapped = map_parallel(
        ctx, specs, _run, stage="external_agents",
        label_fn=lambda i, s: getattr(s, "name", f"agent-{i + 1}"),
    )
    results: list[ExternalAgentResult] = []
    for item in mapped:
        if item.ok and isinstance(item.value, ExternalAgentResult):
            results.append(item.value)
        else:
            spec = specs[item.index]
            results.append(ExternalAgentResult(getattr(spec, "name", item.label), ok=False, error=item.error or "no result"))
    return results


def consult_external_agents(
    agent_names: Any,
    prompt: str,
    *,
    plan_mode: bool = False,
    cwd: str = "",
    timeout_s: float = 180.0,
    max_workers: int = 4,
    detector: Callable[..., list[ExternalAgentSpec]] = detect_available,
    overrides: Optional[dict] = None,
    runner: Runner = _subprocess_runner,
    emit: Optional[Callable[..., None]] = None,
    invoker: Callable[..., ExternalAgentResult] = run_external_agent,
) -> list[ExternalAgentResult]:
    """Detect the requested agents that are installed and consult them in parallel.

    ``agent_names`` is a config-style list/csv; only agents whose CLI resolves on PATH are consulted
    (via the injected ``detector``). With ``plan_mode`` the prompt is wrapped so agents return plans.
    """
    names = parse_agent_list(agent_names)
    specs = detector(names, overrides=overrides)
    text = plan_mode_prompt(prompt) if plan_mode else prompt
    return run_external_agents_parallel(
        specs, text, cwd=cwd, timeout_s=timeout_s, max_workers=max_workers, runner=runner, emit=emit, invoker=invoker,
    )


__all__ = [
    "PLAN_MODE_PREAMBLE",
    "plan_mode_prompt",
    "run_external_agents_parallel",
    "consult_external_agents",
]
