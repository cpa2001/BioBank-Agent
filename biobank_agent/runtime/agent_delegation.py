"""Delegate a coding subtask to an external coding agent, then land its diff through the gate.

The external agent runs with a throwaway ``git worktree`` as its working directory and is UNTRUSTED.
We capture ONLY the worktree's net diff and apply it ONLY via
``self_evolve.apply_patch_transactionally(force_review_branch=True)`` — re-verified in an isolated
worktree against the caller's ``test_commands`` and kept on a review branch, never auto-merged. That
apply gate is the decisive safety control: an undetected agent side-effect is never propagated by us.

``cwd`` is NOT an OS sandbox — a subprocess can write anywhere the user can. A best-effort tripwire
flags the common escape (the agent editing *tracked* files in the live tree) and aborts, but it
cannot see edits to gitignored files (``.env``, ``data/``) or paths outside the repo. For hard
confinement run the agent under an OS sandbox (bwrap / firejail / container); the feature is therefore
default-off and opt-in.

The diff producer and the apply function are injected, so the orchestration is unit-testable offline.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from biobank_agent.runtime.external_agents import (
    ExternalAgentResult,
    ExternalAgentSpec,
    run_external_agent,
    spec_for,
)

_DELEGATION_PROMPT = (
    "You are a coding agent working inside a git repository. Make the change described below by "
    "editing files directly in the working tree. Do NOT commit, push, or run destructive commands.\n\n"
    "Task:\n{task}\n"
)


@dataclass
class DelegationResult:
    agent: str
    ok: bool
    diff: str = ""
    text: str = ""
    error: str = ""


def _git(repo: Path, *args: str, timeout: int = 180) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], text=True, capture_output=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args=("git", *args), returncode=124, stdout="", stderr="git timed out")


def _first_diff_path(diff: str) -> str:
    """The first file a unified diff touches — used as a label for the apply gate's branch/summary."""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            return line[6:].strip()
        if line.startswith("diff --git a/"):
            parts = line.split()
            if len(parts) >= 4 and parts[3].startswith("b/"):
                return parts[3][2:]
    return ""


def _worktree_diff_producer(
    task: str, spec: ExternalAgentSpec, repo_root: str, timeout_s: float, *, agent_invoke: Callable = run_external_agent
):
    """Default producer: run the agent in a throwaway ``git worktree``, capture its net diff, clean up.

    Returns ``(ok, diff, error)``. The diff is taken vs the base commit (``add -A`` +
    ``diff --cached <base>``) so new files and any in-worktree commits are included. A BEST-EFFORT
    tripwire snapshots the live tree before/after and aborts if the agent edited *tracked* files
    there; it does NOT (cannot, in-process) detect edits to gitignored files or paths outside the
    repo — the decisive control is the review-branch apply gate. ``agent_invoke`` is injected for
    tests."""
    repo = Path(repo_root).resolve()
    base = _git(repo, "rev-parse", "HEAD").stdout.strip()
    if not base:
        return False, "", "repo_root is not a git repository"
    holder = Path(tempfile.mkdtemp(prefix="biobank-delegate-"))
    worktree = holder / "tree"
    try:
        add = _git(repo, "worktree", "add", "--detach", str(worktree), base)
        if add.returncode != 0:
            return False, "", f"worktree add failed: {add.stderr.strip()}"
        # Best-effort tripwire: cwd is a working directory, not a sandbox. `git status` catches the
        # common escape (the agent editing tracked files in the live tree); it cannot see gitignored
        # files or out-of-repo paths, so this is a courtesy alarm, not a boundary. We only ever apply
        # the worktree diff (below) through the review-branch gate, so an undetected escape is the
        # agent's own side-effect, never something this pipeline propagates.
        before = _git(repo, "status", "--porcelain").stdout
        try:
            res = agent_invoke(spec, _DELEGATION_PROMPT.format(task=task), cwd=str(worktree), timeout_s=timeout_s)
        except Exception as exc:  # a raising invoker must still face the tripwire below
            res = ExternalAgentResult(spec.name, ok=False, error=str(exc))
        # Tripwire BEFORE acting on the agent's result: a FAILED or crashing agent can still have
        # escaped its worktree and edited tracked files in the live tree, so check the live tree first.
        if _git(repo, "status", "--porcelain").stdout != before:
            return False, "", ("isolation tripwire: the external agent modified tracked files in the "
                               "live tree outside its worktree; refusing to apply its output")
        if not getattr(res, "ok", False):
            return False, "", getattr(res, "error", "") or "external agent invocation failed"
        _git(worktree, "add", "-A")
        diff = _git(worktree, "diff", "--cached", base)
        return True, diff.stdout or "", ""
    finally:
        _git(repo, "worktree", "remove", "--force", str(worktree))
        shutil.rmtree(holder, ignore_errors=True)


def delegate_coding_task(
    task: str,
    *,
    agent: str = "codex",
    repo_root: str,
    timeout_s: float = 600.0,
    producer: Callable = _worktree_diff_producer,
    spec_overrides: Optional[dict] = None,
) -> DelegationResult:
    """Run an external coding agent on ``task`` in isolation and return the diff it produced.

    The diff is UNTRUSTED — callers must apply it through the review-branch gate, not directly."""
    spec = spec_for(agent, overrides=spec_overrides)
    if spec is None:
        return DelegationResult(agent, ok=False, error=f"unknown external agent '{agent}'")
    if not str(task or "").strip():
        return DelegationResult(agent, ok=False, error="empty task")
    try:
        ok, diff, err = producer(task, spec, repo_root, float(timeout_s))
    except Exception as exc:  # pragma: no cover - defensive
        return DelegationResult(agent, ok=False, error=str(exc))
    if not ok:
        return DelegationResult(agent, ok=False, error=err)
    if not (diff or "").strip():
        return DelegationResult(agent, ok=True, diff="", text="agent produced no changes")
    return DelegationResult(agent, ok=True, diff=diff)


def delegate_and_apply(
    task: str,
    *,
    agent: str = "codex",
    repo_root: str,
    test_commands,
    enabled: bool,
    timeout_s: float = 600.0,
    producer: Callable = _worktree_diff_producer,
    apply_fn: Optional[Callable] = None,
    spec_overrides: Optional[dict] = None,
) -> dict:
    """Delegate ``task`` to an external coding agent and apply its diff to a REVIEW BRANCH ONLY.

    ``enabled`` is the caller's gate (e.g. ``settings.external_agent_delegation_enabled``); when
    false this is a no-op. The diff is applied via ``apply_patch_transactionally(force_review_branch=
    True)`` so it is verified against ``test_commands`` in an isolated worktree and never auto-merged.
    A change with no ``test_commands`` is refused (the gate cannot verify it).
    """
    if not enabled:
        return {"status": "disabled",
                "message": "Set external_agent_delegation_enabled to delegate to an external coding agent."}
    # Validate the verification step BEFORE spawning the agent: a change we cannot verify can never be
    # applied, so there is no reason to pay to run the external CLI. Accept a single string or a list,
    # and reject anything that yields no real command (None, "", a bare int, all-whitespace) — a bare
    # string must NOT be iterated into per-character "commands".
    if isinstance(test_commands, str):
        test_commands = [test_commands]
    if not isinstance(test_commands, (list, tuple)):
        test_commands = []
    tests = [str(c).strip() for c in test_commands if str(c).strip()]
    if not tests:
        return {"status": "needs_tests", "agent": agent,
                "message": "supply test_commands (a list of shell commands) to verify the change before apply"}
    result = delegate_coding_task(
        task, agent=agent, repo_root=repo_root, timeout_s=timeout_s, producer=producer, spec_overrides=spec_overrides
    )
    if not result.ok:
        return {"status": "failed", "agent": agent, "error": result.error}
    if not (result.diff or "").strip():
        return {"status": "no_changes", "agent": agent, "message": result.text or "agent produced no changes"}
    if apply_fn is None:
        from biobank_agent.runtime.self_evolve import apply_patch_transactionally

        apply_fn = apply_patch_transactionally
    target = _first_diff_path(result.diff) or "delegated_change"
    applied = apply_fn(
        repo_root=repo_root,
        target_path=target,
        diff=result.diff,
        test_commands=tests,
        summary=f"delegated to {agent}: {str(task)[:80]}",
        force_review_branch=True,
    )
    return {
        "status": getattr(applied, "status", "applied"),
        "agent": agent,
        "target_path": getattr(applied, "target_path", target),
        "branch": getattr(applied, "branch", "") or getattr(applied, "review_branch", ""),
        "error": getattr(applied, "error", ""),
    }


__all__ = ["DelegationResult", "delegate_coding_task", "delegate_and_apply"]
