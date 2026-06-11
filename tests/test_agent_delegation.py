"""Task delegation to an external coding agent: capture diff -> review-branch apply (never auto-merge)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from biobank_agent.runtime.agent_delegation import (
    _first_diff_path,
    _worktree_diff_producer,
    delegate_and_apply,
    delegate_coding_task,
)
from biobank_agent.runtime.external_agents import ExternalAgentResult, spec_for

_DIFF = (
    "diff --git a/biobank_agent/skills/foo.py b/biobank_agent/skills/foo.py\n"
    "--- a/biobank_agent/skills/foo.py\n"
    "+++ b/biobank_agent/skills/foo.py\n"
    "@@ -1 +1,2 @@\n line\n+added\n"
)


def _with_diff(task, spec, repo_root, timeout_s):
    return True, _DIFF, ""


def _empty(task, spec, repo_root, timeout_s):
    return True, "", ""


def _fail(task, spec, repo_root, timeout_s):
    return False, "", "agent crashed"


def _must_not_apply(**kwargs):
    raise AssertionError("apply_fn must not be called on this path")


# ── delegate_coding_task ─────────────────────────────────────────────────────

def test_unknown_agent_is_rejected():
    r = delegate_coding_task("do x", agent="nope", repo_root=".", producer=_with_diff)
    assert r.ok is False and "unknown" in r.error


def test_empty_task_is_rejected():
    r = delegate_coding_task("   ", agent="codex", repo_root=".", producer=_with_diff)
    assert r.ok is False and "empty" in r.error


def test_captures_agent_diff():
    r = delegate_coding_task("do x", agent="codex", repo_root=".", producer=_with_diff)
    assert r.ok is True and "added" in r.diff


def test_no_changes_is_ok_with_empty_diff():
    r = delegate_coding_task("do x", agent="codex", repo_root=".", producer=_empty)
    assert r.ok is True and r.diff == "" and "no changes" in r.text


def test_producer_failure_propagates():
    r = delegate_coding_task("do x", agent="codex", repo_root=".", producer=_fail)
    assert r.ok is False and "crashed" in r.error


def test_first_diff_path_extraction():
    assert _first_diff_path(_DIFF) == "biobank_agent/skills/foo.py"
    assert _first_diff_path("no diff here") == ""


# ── delegate_and_apply (the gated full path) ─────────────────────────────────

def test_disabled_is_noop():
    out = delegate_and_apply("do x", agent="codex", repo_root=".", test_commands=["pytest -q"],
                             enabled=False, producer=_with_diff, apply_fn=_must_not_apply)
    assert out["status"] == "disabled"


def test_applies_to_review_branch_only():
    seen = {}

    def fake_apply(*, repo_root, target_path, diff, test_commands, summary, force_review_branch):
        seen.update(target_path=target_path, diff=diff, force_review_branch=force_review_branch,
                    test_commands=test_commands, summary=summary)
        return type("R", (), {"status": "review_branch", "target_path": target_path,
                              "branch": "evolve/delegated", "error": ""})()

    out = delegate_and_apply("add a docstring", agent="codex", repo_root=".",
                             test_commands=["pytest tests/test_x.py -q"], enabled=True,
                             producer=_with_diff, apply_fn=fake_apply)
    assert out["status"] == "review_branch" and out["branch"] == "evolve/delegated"
    assert seen["force_review_branch"] is True  # untrusted output is NEVER auto-merged
    assert seen["target_path"] == "biobank_agent/skills/foo.py"
    assert seen["test_commands"] == ["pytest tests/test_x.py -q"]
    assert "added" in seen["diff"]


def test_requires_test_commands_to_apply():
    out = delegate_and_apply("do x", agent="codex", repo_root=".", test_commands=[], enabled=True,
                             producer=_with_diff, apply_fn=_must_not_apply)
    assert out["status"] == "needs_tests"


def test_no_changes_short_circuits_before_apply():
    out = delegate_and_apply("do x", agent="codex", repo_root=".", test_commands=["pytest -q"],
                             enabled=True, producer=_empty, apply_fn=_must_not_apply)
    assert out["status"] == "no_changes"


def test_failure_short_circuits_before_apply():
    out = delegate_and_apply("do x", agent="codex", repo_root=".", test_commands=["pytest -q"],
                             enabled=True, producer=_fail, apply_fn=_must_not_apply)
    assert out["status"] == "failed" and "crashed" in out["error"]


# ── isolation invariant: real temp git repo + an in-process fake agent ───────

def _init_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def g(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)

    g("init", "-q")
    g("config", "user.email", "t@example.com")
    g("config", "user.name", "t")
    (repo / "a.txt").write_text("hello\n")
    g("add", "-A")
    g("commit", "-q", "-m", "init")
    return repo


def test_isolation_well_behaved_agent_diff_captured(tmp_path):
    repo = _init_repo(tmp_path)

    def good(spec, prompt, *, cwd, timeout_s):
        (Path(cwd) / "a.txt").write_text("hello\nfrom agent\n")  # edits inside its worktree
        return ExternalAgentResult(spec.name, ok=True, text="done")

    ok, diff, err = _worktree_diff_producer("edit a.txt", spec_for("codex"), str(repo), 60.0, agent_invoke=good)
    assert ok is True and err == "" and "from agent" in diff


def test_isolation_new_file_in_worktree_is_captured(tmp_path):
    repo = _init_repo(tmp_path)

    def good(spec, prompt, *, cwd, timeout_s):
        (Path(cwd) / "new_file.py").write_text("print('hi')\n")
        return ExternalAgentResult(spec.name, ok=True, text="done")

    ok, diff, err = _worktree_diff_producer("add file", spec_for("codex"), str(repo), 60.0, agent_invoke=good)
    assert ok is True and "new_file.py" in diff


def test_tripwire_catches_tracked_file_escape(tmp_path):
    repo = _init_repo(tmp_path)

    def escaper(spec, prompt, *, cwd, timeout_s):
        # Escape the worktree and edit a TRACKED file in the live repo — the best-effort tripwire
        # catches this common case via `git status` and refuses to apply.
        (repo / "a.txt").write_text("hello\nTAMPERED\n")
        return ExternalAgentResult(spec.name, ok=True, text="done")

    ok, diff, err = _worktree_diff_producer("x", spec_for("codex"), str(repo), 60.0, agent_invoke=escaper)
    assert ok is False and "isolation tripwire" in err


def test_tripwire_catches_escape_even_when_agent_fails(tmp_path):
    repo = _init_repo(tmp_path)

    def failing_escaper(spec, prompt, *, cwd, timeout_s):
        (repo / "a.txt").write_text("hello\nTAMPERED\n")  # escape the worktree...
        return ExternalAgentResult(spec.name, ok=False, error="agent crashed")  # ...then report failure

    ok, diff, err = _worktree_diff_producer("x", spec_for("codex"), str(repo), 60.0, agent_invoke=failing_escaper)
    # The escape must be surfaced even though the agent failed (tripwire runs before the ok/fail branch).
    assert ok is False and "isolation tripwire" in err


def test_tripwire_is_best_effort_and_misses_gitignored_live_edits(tmp_path):
    """Documented limitation: the tripwire cannot see gitignored files, so an agent that edits one in
    the live tree is NOT flagged. The decisive control is the review-branch apply gate — we still
    apply only the (here empty) worktree diff, never the agent's out-of-band side effect."""
    repo = _init_repo(tmp_path)
    (repo / ".gitignore").write_text("secret.txt\n")
    (repo / "secret.txt").write_text("orig\n")
    for args in (("add", ".gitignore"), ("commit", "-q", "-m", "ignore")):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)

    def escaper(spec, prompt, *, cwd, timeout_s):
        (repo / "secret.txt").write_text("tampered by agent\n")  # gitignored live edit -> not seen
        return ExternalAgentResult(spec.name, ok=True, text="done")

    ok, diff, err = _worktree_diff_producer("x", spec_for("codex"), str(repo), 60.0, agent_invoke=escaper)
    assert ok is True and diff == ""  # not flagged (known limit); only the empty worktree diff is kept
