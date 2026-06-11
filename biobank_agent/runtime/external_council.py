"""Plan-review council: consult external coding-agent CLIs to critique a drafted plan.

Pure orchestration over the ``external_agents`` seam, with injected detector/invoker so it is
unit-testable offline. The planner calls :func:`collect_plan_reviews` after drafting a plan; the
returned notes are advisory — folded into ``open_questions``, never blocking execution.

codex/claude/gemini are autonomous coding agents, so each review runs in a throwaway directory,
NEVER the live repo — a review must not operate on or mutate the user's working tree. They critique
the plan from the prompt text and only their stdout is consumed (true OS-level confinement of a
subprocess still requires an external sandbox; this keeps the agent out of the repo by default).
"""

from __future__ import annotations

import shutil
import tempfile
from typing import Callable, Sequence

from biobank_agent.runtime.external_agents import detect_available, parse_agent_list, run_external_agent

_REQUEST_KEYWORDS = (
    "external review",
    "external council",
    "second opinion",
    "consult",
    "ask codex",
    "ask claude",
    "ask gemini",
)
_MAX_NOTE_CHARS = 600


def review_requested(*texts: str) -> bool:
    """True when the objective/refinement text asks for an outside opinion (policy ``requested``)."""
    blob = " ".join(str(t or "") for t in texts).lower()
    return any(k in blob for k in _REQUEST_KEYWORDS)


def policy_permits(policy: str, *, requested: bool) -> bool:
    p = str(policy or "requested").strip().lower()
    if p == "never":
        return False
    if p == "always":
        return True
    return bool(requested)  # "requested" (the default)


def build_review_prompt(objective: str, step_descriptions: Sequence[str]) -> str:
    steps = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(step_descriptions) if str(d).strip())
    return (
        "Review this research analysis plan and critique it briefly. Do not rewrite it.\n\n"
        f"Objective:\n{objective}\n\n"
        f"Plan steps:\n{steps or '(none)'}\n\n"
        "In 3-5 concise bullets, flag any missing steps, methodology risks (confounding, data "
        "leakage, uncorrected multiple testing, selection bias), and data-readiness gaps. "
        "Be specific; if the plan looks sound, say so."
    )


def collect_plan_reviews(
    objective: str,
    step_descriptions: Sequence[str],
    *,
    agents,
    policy: str,
    requested: bool,
    timeout_s: float = 180.0,
    detector: Callable = detect_available,
    invoker: Callable = run_external_agent,
) -> list[str]:
    """Return advisory critique notes from each available external agent, honoring ``policy``.

    Each agent runs in a throwaway directory (never the live repo), so a review cannot operate on or
    mutate the user's working tree. ``detector`` / ``invoker`` are injected so this is fully testable
    offline. Defensive: a single agent failing (non-zero exit, timeout) is simply skipped, never raised.
    """
    if not policy_permits(policy, requested=requested):
        return []
    specs = detector(parse_agent_list(agents))
    if not specs:
        return []
    prompt = build_review_prompt(objective, step_descriptions)
    notes: list[str] = []
    for spec in specs:
        # A FRESH throwaway dir per agent: keeps coding agents out of the live repo AND out of each
        # other's working dir, so reviews stay independent and one agent cannot seed files for the next.
        sandbox = tempfile.mkdtemp(prefix="biobank-review-")
        try:
            res = invoker(spec, prompt, cwd=sandbox, timeout_s=float(timeout_s))
            if getattr(res, "ok", False) and (getattr(res, "text", "") or "").strip():
                notes.append(f"[{spec.name}] {res.text.strip()[:_MAX_NOTE_CHARS]}")
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)
    return notes


__all__ = ["review_requested", "policy_permits", "build_review_prompt", "collect_plan_reviews"]
