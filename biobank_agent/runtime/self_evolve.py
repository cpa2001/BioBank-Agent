"""Transactional, test-gated self-evolution apply loop.

Closes the self-evolution loop safely: a proposed patch becomes a REAL applied
change ONLY if it passes its required tests in an ISOLATED git worktree, with
automatic rollback on any failure.

Security invariants (negotiated with Codex):
  * Patch is applied to a throwaway ``git worktree`` at HEAD — never the live
    working tree.
  * EVERY file the patch touches is path-validated: no absolute paths, no ``..``
    escape, must resolve inside the worktree. Touching core/runtime/tests (or any
    path outside the skill allow-list) downgrades the result to a review branch —
    it is NEVER auto-merged. This blocks ``custom_skills/../biobank_agent/...``
    traversal bypasses.
  * ``test_commands`` are allow-listed (pytest/python/ruff/mypy/…), rejected if
    they contain shell metacharacters, and run with a SECRET-STRIPPED env. An
    empty test set is refused (no unverified self-edit; no test laundering).
  * A failing test, dirty live tree, or any error removes the worktree and the
    scratch branch (only if this run created it), leaving zero residue.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_APPLY_ALLOW_PATHS = (
    "reports/generated_skills/",
    "domain/skills/",
    "custom_skills/",
    "biobank_agent/skills/custom",
)
PROTECTED_PREFIXES = ("biobank_agent/core", "biobank_agent/runtime", "tests/")

# Test commands that gate a code change must be trustworthy + simple: only known
# test/lint RUNNERS, never a bare `python -c "..."` / `python file.py` (which is a
# direct arbitrary-code channel), and no shell metacharacters (the command runs
# verbatim, never via a pipe/redirect/chain). `python -m pytest` is allowed.
_RUNNER_LEADERS = {"pytest", "ruff", "mypy", "flake8", "black"}
_PY_LEADERS = {"python", "python3"}
# `$` (covers `$(`, `${VAR}`, `$VAR`) is banned outright: the command runs via
# `bash -c`, so any unescaped `$` is a shell-expansion channel. A token like
# `${OPTS:---collect-only}` passes a literal-flag check but bash expands it into
# `--collect-only`, laundering a body-skipping flag past the gate — reproduced and
# now closed. Legitimate test/lint runners never need `$`.
_TEST_METACHARS = (";", "&", "|", "$", "`", ">", "<", "\n", "\r")
# Collect-only flags import the suite + run conftest/collection hooks but skip the
# test BODIES — accepting them as a gate proves nothing about behavior, so they are
# banned. pytest accepts THREE spellings (verified on pytest 9.x): --collect-only,
# its no-hyphen alias --collectonly, and the short --co. Tokens are normalized on
# '=' before the check, so `--cov`/`--cov=...` (a different, legitimate flag) stay
# allowed. Defense-in-depth shared with the generation layer (patch_generation.py).
_NO_EXECUTION_FLAGS = ("--collect-only", "--collectonly", "--co")
# Exact strip set for _safe_env(): env keys containing any of these tokens are
# removed before a test command runs, so attacker-influenced test code cannot read
# the LLM API key etc.
_SECRET_ENV_TOKENS = ("secret", "token", "key", "auth", "api", "password", "passwd", "credential")
# pytest reads extra args from these env vars. `PYTEST_ADDOPTS=--collect-only` skips
# every test body while exiting 0 — and `-o addopts=` on the command line does not
# override the env channel (verified), so these must be stripped from the test env.
_PYTEST_ENV_STRIP = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_DEBUG_TEMPROOT")
# pytest config / hook files can subvert the gate from inside the repo: addopts in
# pytest.ini/tox.ini/setup.cfg/pyproject.toml inject collect-only, and a conftest.py
# can mark every collected item skipped (exit 0, zero bodies run) in ways `-o addopts=`
# cannot stop. A patch that creates/edits any of these is never auto-merged — it is
# routed to a review branch for human inspection.
_TEST_CONFIG_BASENAMES = frozenset({
    "pytest.ini", ".pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml",
    "conftest.py", "pytest.cfg",
})


def _is_test_config_path(rel: str) -> bool:
    """True if ``rel`` is a pytest config or hook file (gate-subversion channel)."""
    return str(rel).rsplit("/", 1)[-1] in _TEST_CONFIG_BASENAMES


def _is_pytest_invocation(toks: list[str]) -> bool:
    if not toks:
        return False
    leader = toks[0].rsplit("/", 1)[-1]
    if leader == "pytest":
        return True
    return leader in _PY_LEADERS and len(toks) >= 3 and toks[1] == "-m" and toks[2] == "pytest"


def _harden_test_command(cmd: str) -> str:
    """Neutralize pytest's addopts file/command channel by appending ``-o addopts=``
    LAST (last assignment wins), so a ``pytest.ini``/``pyproject.toml`` addopts — or an
    attacker-supplied ``-o addopts=--collect-only`` token that slips past the literal
    collect-only check — cannot inject a body-skipping flag. No-op for non-pytest
    runners (ruff/mypy/black/flake8), which have no such option. The env channel is
    handled separately by stripping ``PYTEST_ADDOPTS`` in :func:`_safe_env`."""
    try:
        import shlex
        toks = shlex.split(cmd)
    except Exception:
        toks = cmd.split()
    if _is_pytest_invocation(toks):
        return f"{cmd} -o addopts="
    return cmd


@dataclass
class EvolutionApplyResult:
    status: str  # applied | tests_failed | review_branch | rejected | error
    target_path: str = ""
    tests_passed: bool = False
    test_output: str = ""
    branch: str = ""
    error: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "target_path": self.target_path,
            "tests_passed": self.tests_passed,
            "test_output": self.test_output[-4000:],
            "branch": self.branch,
            "error": self.error,
            "summary": self.summary,
        }


def _git(repo: Path, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess:
    # Bounded + hooks disabled so a stuck lock or a repo hook cannot hang the
    # apply loop (which must always reach cleanup).
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", *args],
            input=input_text, text=True, capture_output=True, check=False, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args=("git", *args), returncode=124, stdout="", stderr="git command timed out")


def _changed_paths(worktree: Path) -> list[str]:
    """Files git actually changed in the worktree (authoritative — catches
    rename/copy/binary/mode headers the diff text may hide). Rename lines
    ('R  old -> new') resolve to the new path."""
    out = _git(worktree, "status", "--porcelain").stdout
    paths: list[str] = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        rest = line[3:]
        if " -> " in rest:
            rest = rest.split(" -> ", 1)[1]
        rest = rest.strip().strip('"')
        if rest:
            paths.append(rest)
    return paths


def _in_allow(rel: str, allow_paths: tuple[str, ...]) -> bool:
    for a in allow_paths:
        a = a.rstrip("/")
        if rel == a or rel.startswith(a + "/"):
            return True
    return False


def _safe_env() -> dict[str, str]:
    """A copy of the environment with secret-bearing keys removed, so a test
    command cannot read the LLM API key etc. Keeps PATH/PYTHONPATH/conda vars so
    pytest still runs. Also strips PYTEST_ADDOPTS/PYTEST_PLUGINS so an inherited
    env var cannot inject a body-skipping flag (--collect-only) into the gate run —
    the `-o addopts=` command hardening does NOT override the env channel."""
    return {
        k: v for k, v in os.environ.items()
        if k not in _PYTEST_ENV_STRIP
        and not any(tok in k.lower() for tok in _SECRET_ENV_TOKENS)
    }


def _validate_test_command(cmd: str) -> str:
    c = str(cmd or "").strip()
    if not c:
        return "empty test command"
    if any(m in c for m in _TEST_METACHARS):
        return f"test command contains a disallowed shell metacharacter: {cmd!r}"
    try:
        import shlex
        toks = shlex.split(c)
    except Exception:
        toks = c.split()
    if not toks:
        return "empty test command"
    if any(t.split("=", 1)[0] in _NO_EXECUTION_FLAGS for t in toks):
        return f"test command uses a collect-only flag that skips test bodies: {cmd!r}"
    leader = toks[0].rsplit("/", 1)[-1]
    if leader in _RUNNER_LEADERS:
        return ""
    # Allow `python -m pytest/ruff/...` but NOT `python -c` / `python file.py`.
    if leader in _PY_LEADERS and len(toks) >= 3 and toks[1] == "-m" and toks[2] in _RUNNER_LEADERS:
        return ""
    return (
        f"test command must be a test/lint runner ({sorted(_RUNNER_LEADERS)} or 'python -m pytest'); "
        f"got {toks[0]!r}. Bare 'python -c'/'python file.py' is not allowed as a gate."
    )


def _normalize_under(worktree: Path, path: str) -> tuple[str | None, str]:
    """Return the worktree-relative normalized path, or (None, reason) if the
    path is absolute, contains ``..``, or escapes the worktree."""
    p = str(path or "").strip().replace("\\", "/")
    if not p:
        return None, "empty path"
    if p.startswith("/") or (len(p) > 1 and p[1] == ":"):
        return None, f"absolute path not allowed: {path!r}"
    if ".." in Path(p).parts:
        return None, f"'..' not allowed in path: {path!r}"
    resolved = (worktree / p).resolve()
    try:
        rel = resolved.relative_to(worktree.resolve())
    except ValueError:
        return None, f"path escapes the worktree: {path!r}"
    return str(rel).replace("\\", "/"), ""


def _diff_paths(diff: str) -> list[str]:
    """Best-effort extraction of every file path a unified diff touches."""
    paths: set[str] = set()
    for line in diff.splitlines():
        if line.startswith(("+++ ", "--- ")):
            p = line[4:].split("\t")[0].strip()
            if p in ("", "/dev/null"):
                continue
            if p.startswith(("a/", "b/")):
                p = p[2:]
            paths.add(p)
        elif line.startswith("diff --git "):
            for tok in line.split()[2:]:
                paths.add(tok[2:] if tok.startswith(("a/", "b/")) else tok)
    return sorted(paths)


def _is_protected(rel: str) -> bool:
    return any(rel.startswith(p) for p in PROTECTED_PREFIXES)


def mutation_check(diff: str, target_path: str = "") -> list[str]:
    """Static red flags in a proposed self-edit, evaluated BEFORE the worktree apply
    (mutation gating). Returns blocking reasons (empty ⇒ no static objection).

    Catches the classic self-evolution gaming vectors that a passing test run would
    not: deleting test functions or assertions to make the gate trivially green, and
    disabling tests via skip/xfail. These are never legitimate for an AUTONOMOUS
    self-edit (a human can still do them on a review branch)."""
    reasons: list[str] = []
    lines = (diff or "").splitlines()
    removed = [ln[1:] for ln in lines if ln.startswith("-") and not ln.startswith("---")]
    added = [ln[1:] for ln in lines if ln.startswith("+") and not ln.startswith("+++")]
    removed_test_defs = sum(1 for ln in removed if re.search(r"\bdef\s+test_\w+", ln))
    removed_asserts = sum(1 for ln in removed if re.match(r"\s*assert\b", ln))
    added_skips = sum(1 for ln in added if re.search(r"@pytest\.mark\.(skip|xfail)|pytest\.skip\(", ln))
    if removed_test_defs:
        reasons.append(f"removes {removed_test_defs} test function(s) — a self-edit must not delete tests to pass the gate")
    if removed_asserts > 2:
        reasons.append(f"removes {removed_asserts} assertion(s) — weakening test checks is not an allowed autonomous mutation")
    if added_skips:
        reasons.append(f"adds {added_skips} skip/xfail marker(s) — disabling tests is not an allowed autonomous mutation")
    return reasons


def apply_patch_transactionally(
    *,
    repo_root: str | Path,
    target_path: str,
    diff: str,
    test_commands: list[str],
    summary: str = "",
    branch_prefix: str = "evolve",
    allow_paths: tuple[str, ...] = DEFAULT_APPLY_ALLOW_PATHS,
    test_timeout_s: int = 600,
    force_review_branch: bool = False,
) -> EvolutionApplyResult:
    repo = Path(repo_root).resolve()
    if not diff.strip():
        return EvolutionApplyResult(status="rejected", target_path=target_path, error="empty diff")
    if not str(target_path).strip():
        return EvolutionApplyResult(status="rejected", error="missing target_path")
    if not test_commands:
        return EvolutionApplyResult(
            status="rejected", target_path=target_path,
            error="refusing to apply a self-edit with no test_commands to verify it",
        )
    for cmd in test_commands:
        reason = _validate_test_command(cmd)
        if reason:
            return EvolutionApplyResult(status="rejected", target_path=target_path, error=reason)
    static_flags = mutation_check(diff, target_path)
    if static_flags:
        return EvolutionApplyResult(
            status="rejected", target_path=target_path,
            error="mutation rejected (static check): " + "; ".join(static_flags),
        )
    if not (repo / ".git").exists():
        return EvolutionApplyResult(status="error", target_path=target_path, error=f"{repo} is not a git repository")

    head = _git(repo, "rev-parse", "HEAD")
    if head.returncode != 0:
        return EvolutionApplyResult(status="error", target_path=target_path, error="cannot resolve HEAD")
    base = head.stdout.strip()

    token = hashlib.sha256(f"{target_path}\0{diff}".encode("utf-8")).hexdigest()[:16]
    branch = f"{branch_prefix}/{token}"

    worktree_parent = Path(tempfile.mkdtemp(prefix="bb_evolve_"))
    worktree = worktree_parent / "wt"
    branch_created = False

    def _cleanup() -> None:
        _git(repo, "worktree", "remove", "--force", str(worktree))
        if branch_created:
            _git(repo, "branch", "-D", branch)
        shutil.rmtree(worktree_parent, ignore_errors=True)

    add = _git(repo, "worktree", "add", "--detach", str(worktree), base)
    if add.returncode != 0:
        shutil.rmtree(worktree_parent, ignore_errors=True)
        return EvolutionApplyResult(status="error", target_path=target_path, error=f"worktree add failed: {add.stderr.strip()}")
    co = _git(worktree, "checkout", "-b", branch)
    if co.returncode != 0:
        _cleanup()
        return EvolutionApplyResult(status="error", target_path=target_path, error=f"branch checkout failed (possible name collision): {co.stderr.strip()}")
    branch_created = True

    try:
        is_diff = diff.lstrip().startswith(("diff --git", "--- ", "+++ ", "Index:"))
        # First-line validation of the declared target(s): inside worktree, no
        # '..', no absolute. (Authoritative check is the actual changed set below.)
        for raw in (_diff_paths(diff) if is_diff else [target_path]):
            rel, reason = _normalize_under(worktree, raw)
            if rel is None:
                _cleanup()
                return EvolutionApplyResult(status="rejected", target_path=target_path, error=f"unsafe patch path: {reason}")

        if is_diff:
            applied = _git(worktree, "apply", "--whitespace=nowarn", "-", input_text=diff)
            if applied.returncode != 0:
                _cleanup()
                return EvolutionApplyResult(status="error", target_path=target_path, error=f"git apply failed: {applied.stderr.strip()}")
        else:
            rel0, _r = _normalize_under(worktree, target_path)
            tgt = (worktree / rel0)
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_text(diff, encoding="utf-8")

        check = _git(worktree, "diff", "--check")
        if check.returncode != 0:
            _cleanup()
            return EvolutionApplyResult(status="error", target_path=target_path, error=f"diff --check failed: {check.stdout.strip() or check.stderr.strip()}")

        # Authoritative path check: inspect the files git actually changed (this
        # catches rename/copy/binary headers, mode changes, and any diff that
        # writes outside its advertised target). Re-validate every one and decide
        # auto-merge from the real set.
        actual = _changed_paths(worktree)
        norm_actual: list[str] = []
        for raw in actual:
            rel, reason = _normalize_under(worktree, raw)
            if rel is None:
                _cleanup()
                return EvolutionApplyResult(status="rejected", target_path=target_path, error=f"unsafe changed path: {reason}")
            norm_actual.append(rel)
        if not norm_actual:
            _cleanup()
            return EvolutionApplyResult(status="rejected", target_path=target_path, error="patch changed no files")
        # A patch is auto-merge eligible only if EVERY changed path is allow-listed,
        # unprotected, AND not a pytest config/hook file. The last clause closes the
        # addopts/conftest gate-subversion channels: such a patch is still verified,
        # but it lands on a review branch for human inspection instead of auto-merging.
        # force_review_branch (agent-synthesized/ingested skills) keeps a verified change on a
        # review branch even when its paths are allow-listed — it only ever strengthens the gate.
        auto_mergeable = (not force_review_branch) and all(
            (not _is_protected(rel)) and _in_allow(rel, allow_paths) and (not _is_test_config_path(rel))
            for rel in norm_actual
        )

        # Run required tests INSIDE the worktree with a secret-stripped env.
        # SECURITY: these test_commands may be LLM-authored (see
        # runtime/patch_generation.py). pytest imports and runs the patch's
        # conftest.py module body + collection hooks here, with full network
        # access, BEFORE the auto-merge decision below — repo-content exfiltration
        # is technically possible (env is secret-stripped, output is captured for
        # review). The change is still gated: it only auto-merges if every test
        # passes AND every changed path is allow-listed AND the live tree is clean.
        env = _safe_env()
        outputs: list[str] = []
        for cmd in test_commands:
            run_cmd = _harden_test_command(cmd)  # append `-o addopts=` for pytest
            proc = subprocess.run(
                ["bash", "-c", run_cmd], cwd=str(worktree), env=env, text=True,
                capture_output=True, check=False, timeout=test_timeout_s,
            )
            outputs.append(f"$ {run_cmd}\n{proc.stdout}\n{proc.stderr}")
            if proc.returncode != 0:
                _cleanup()
                return EvolutionApplyResult(
                    status="tests_failed", target_path=target_path, tests_passed=False,
                    test_output="\n".join(outputs), summary=summary,
                    error=f"test command failed (exit {proc.returncode}): {cmd}",
                )

        _git(worktree, "add", "-A")
        commit = _git(worktree, "commit", "-m", f"self-evolve: {summary or target_path}\n\nVerified by: {'; '.join(test_commands)}")
        if commit.returncode != 0:
            _cleanup()
            return EvolutionApplyResult(status="error", target_path=target_path, error=f"commit failed: {commit.stderr.strip()}")

        test_output = "\n".join(outputs)
        if not auto_mergeable:
            # Protected / outside allow-list: keep the verified branch for review.
            _git(repo, "worktree", "remove", "--force", str(worktree))
            shutil.rmtree(worktree_parent, ignore_errors=True)
            return EvolutionApplyResult(
                status="review_branch", target_path=target_path, tests_passed=True,
                test_output=test_output, branch=branch, summary=summary,
                error="protected path (core/runtime/tests) or outside allow-list: verified on a review branch, not auto-merged",
            )

        # Only fast-forward onto the live branch if the live tree is CLEAN — never
        # ff-merge over uncommitted user changes.
        dirty = _git(repo, "status", "--porcelain").stdout.strip()
        if dirty:
            _git(repo, "worktree", "remove", "--force", str(worktree))
            shutil.rmtree(worktree_parent, ignore_errors=True)
            return EvolutionApplyResult(
                status="review_branch", target_path=target_path, tests_passed=True,
                test_output=test_output, branch=branch, summary=summary,
                error="live working tree is dirty; verified on a review branch instead of auto-merging",
            )
        new_commit = _git(worktree, "rev-parse", "HEAD").stdout.strip()
        merge = _git(repo, "merge", "--ff-only", new_commit)
        if merge.returncode != 0:
            _git(repo, "worktree", "remove", "--force", str(worktree))
            shutil.rmtree(worktree_parent, ignore_errors=True)
            return EvolutionApplyResult(
                status="review_branch", target_path=target_path, tests_passed=True,
                test_output=test_output, branch=branch, summary=summary,
                error=f"verified but could not fast-forward: {merge.stderr.strip()}",
            )
        _cleanup()  # merged into the live branch; scratch branch no longer needed
        return EvolutionApplyResult(
            status="applied", target_path=target_path, tests_passed=True,
            test_output=test_output, summary=summary,
        )
    except subprocess.TimeoutExpired as exc:
        _cleanup()
        return EvolutionApplyResult(status="tests_failed", target_path=target_path, error=f"test timeout: {exc}", summary=summary)
    except Exception as exc:  # pragma: no cover - defensive
        _cleanup()
        return EvolutionApplyResult(status="error", target_path=target_path, error=str(exc), summary=summary)


def apply_proposal(proposal, *, repo_root: str | Path, allow_paths: tuple[str, ...] = DEFAULT_APPLY_ALLOW_PATHS, force_review_branch: bool = False) -> EvolutionApplyResult:
    """Apply an EvolutionProposal that carries a concrete patch + tests.

    ``force_review_branch=True`` (agent-synthesized/ingested skills) keeps the verified change on a
    review branch instead of auto-merging, regardless of allow-listing."""
    return apply_patch_transactionally(
        repo_root=repo_root,
        target_path=getattr(proposal, "target_path", "") or "",
        diff=getattr(proposal, "diff", "") or "",
        test_commands=list(getattr(proposal, "test_commands", []) or []),
        summary=getattr(proposal, "summary", "") or getattr(proposal, "proposal_id", ""),
        allow_paths=allow_paths,
        force_review_branch=force_review_branch,
    )


__all__ = [
    "EvolutionApplyResult",
    "apply_patch_transactionally",
    "apply_proposal",
    "mutation_check",
    "DEFAULT_APPLY_ALLOW_PATHS",
    "PROTECTED_PREFIXES",
]
