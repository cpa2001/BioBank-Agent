"""Live skill: delegate a coding subtask to an external coding agent (codex / claude-code).

Gated behind ``external_agent_delegation_enabled`` (default off). The agent runs in an isolated git
worktree; its diff is applied to a REVIEW BRANCH ONLY via the trust kernel (``force_review_branch``),
never auto-merged, and verified against the caller's ``test_commands``. See
``runtime/agent_delegation.py`` for the worktree + apply-gate mechanics.
"""

from __future__ import annotations

import os

from biobank_agent.registry import skill


@skill(
    name="delegate_to_coding_agent",
    description=(
        "Delegate a focused coding task to an external coding agent (codex or claude-code). The agent "
        "runs in an ISOLATED git worktree; its diff is applied to a REVIEW BRANCH ONLY (never "
        "auto-merged) and verified against the supplied test_commands. Off unless "
        "external_agent_delegation_enabled is set."
    ),
    parameters={
        "task": {"type": "string", "description": "The coding task for the external agent"},
        "agent": {"type": "string", "description": "codex | claude | gemini", "default": "codex"},
        "test_commands": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Commands that verify the change before it lands on the review branch",
            "default": [],
        },
    },
    required=["task"],
)
def delegate_to_coding_agent(task, agent="codex", test_commands=None, *, ctx=None) -> dict:
    from biobank_agent.runtime.agent_delegation import delegate_and_apply

    settings = getattr(ctx, "settings", None)
    repo_root = getattr(ctx, "workspace_root", None) or os.getcwd()
    return delegate_and_apply(
        task,
        agent=agent,
        repo_root=repo_root,
        test_commands=test_commands or [],
        enabled=bool(getattr(settings, "external_agent_delegation_enabled", False)),
    )
