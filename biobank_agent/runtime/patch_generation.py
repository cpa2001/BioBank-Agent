"""Autonomous LLM patch generation feeding the transactional apply loop.

This is the *generate* half of self-evolution-through-harness. It turns a
review-only :class:`EvolutionProposal` (a described improvement + trajectory
evidence) into a CONCRETE, applicable patch — ``{target_path, diff,
test_commands}`` — by asking an LLM, then hands that to
:func:`biobank_agent.runtime.self_evolve.apply_proposal`, which is the
authoritative safety boundary (isolated worktree, test gate, rollback).

SECURITY MODEL (negotiated via a 4-lens adversarial design review)
------------------------------------------------------------------
The generated ``diff`` and ``test_commands`` are UNTRUSTED. Adding generation
introduces **no new code-execution / path-escape risk** beyond what the apply
loop already gates, because :func:`apply_patch_transactionally`:

  * applies the patch in an ISOLATED, disposable ``git worktree`` at HEAD —
    never the live tree;
  * re-validates EVERY *actually-changed* file from ``git status`` (not the diff
    text), so ``..``, symlink, multi-file, and mode-change escapes are caught by
    ``.resolve()`` + allow-list checks;
  * routes ``core``/``runtime``/``tests`` and any non-allow-listed path to a
    *review branch* — never auto-merged;
  * runs ``test_commands`` with a secret-stripped env and rolls back on any
    failure or a dirty live tree.

The local checks in this module only avoid obviously-doomed LLM calls and give
clean rejections; the apply loop remains the real gate.

CAVEAT — LLM-authored test code executes
-----------------------------------------
The generated tests (and any ``conftest.py`` module body / pytest collection
hooks they include — which run even under ``--collect-only``) execute with FULL
NETWORK ACCESS inside the disposable worktree BEFORE the merge decision. The env
is secret-stripped and repo contents are not secrets, and all test output is
captured for review, but repo-content exfiltration is technically possible.
Patches are also NOT verified to actually *exercise* their diff — a vacuous test
("assert True") would pass the gate. We mitigate the cheapest, highest-value
laundering vectors (require a real pytest runner; ban ``--collect-only``) but
test *relevance* remains the LLM's responsibility; a human should confirm it
before trusting an auto-merge.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .council import _first_balanced_block
from .self_evolve import (
    DEFAULT_APPLY_ALLOW_PATHS,
    _diff_paths,
    _in_allow,
    _is_protected,
    _validate_test_command,
)

logger = logging.getLogger(__name__)

# The collect-only ban (a test-laundering vector) lives in ONE place —
# self_evolve._validate_test_command, which this module calls — so the generation
# gate and the authoritative apply gate can never diverge on which flags slip
# through (they did: --collectonly was missed in an earlier two-set version).
_PYTEST_LEADERS = {"pytest"}

_MAX_EVIDENCE_ROWS = 10
_MAX_EVIDENCE_FIELD = 120
_MAX_EVIDENCE_JSON = 1500
_MAX_RAW_KEPT = 2000


@dataclass
class PatchGenerationResult:
    """Outcome of generating a concrete patch for one proposal.

    status:
      * ``generated``        — a valid, applicable patch was produced
      * ``already_generated``— the proposal already carried a complete patch
      * ``rejected``         — the LLM declined, or the output failed validation
      * ``error``            — the LLM call failed or returned unparseable output
    """

    status: str
    target_path: str = ""
    diff: str = ""
    test_commands: list[str] = field(default_factory=list)
    summary: str = ""
    rationale: str = ""
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def has_patch(self) -> bool:
        return bool(self.target_path and self.diff and self.test_commands)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "target_path": self.target_path,
            "diff": self.diff,
            "test_commands": list(self.test_commands),
            "summary": self.summary,
            "rationale": self.rationale,
            "error": self.error,
            "warnings": list(self.warnings),
            "raw": self.raw[:_MAX_RAW_KEPT],
        }


_GENERATION_SYSTEM_PROMPT = """You are a careful software engineer that proposes ONE small, \
self-contained, test-gated improvement to a Python repository.

Your output is applied by an automated, isolated, test-gated loop: the change is \
written to a throwaway git worktree, your tests are run there, and it is committed \
ONLY if every test passes (otherwise discarded). You therefore MUST make the change \
genuinely verifiable.

HARD RULES (a violation means your patch is rejected unapplied):
1. Write to exactly one new or existing file under one of the ALLOWED PATHS given \
below. NEVER touch biobank_agent/core, biobank_agent/runtime, or tests/.
2. `target_path` is repository-relative, with no leading '/', no '..', no absolute path.
3. Prefer an ADDITIVE, self-contained new file that contains BOTH the new \
functionality AND its pytest tests in the same file, so the tests provably exercise \
the code. For an edit to an existing allowed file, emit a unified diff instead.
4. `diff` is EITHER the full content of the new file (preferred for new files) OR a \
unified diff (`--- a/... / +++ b/...`) for an edit.
5. `test_commands` must include at least one real pytest invocation that runs your \
tests (e.g. "pytest <target_path> -q"). Only test/lint runners are allowed \
(pytest / ruff / mypy / flake8 / black, or "python -m pytest"). No shell pipes, \
redirects, chaining, or `python -c`. Do NOT use --collect-only/--co (they don't run tests).
6. Keep it small (ideally < 3 KB). Write real assertions that would FAIL if the code \
were wrong — never a vacuous `assert True`.

If you cannot produce a safe, useful, verifiable change from the issue, decline.

Respond with ONLY a single JSON object, no prose, no markdown fences:
{
  "target_path": "custom_skills/<name>.py",
  "diff": "<full file content OR unified diff>",
  "test_commands": ["pytest custom_skills/<name>.py -q"],
  "summary": "<one line>",
  "rationale": "<why this addresses the issue>"
}
To decline: {"status": "rejected", "reason": "<why>"}"""


def _compact_evidence(evidence: list[dict[str, Any]] | None) -> str:
    """Bound + de-risk evidence before it enters the prompt.

    Evidence rows carry verbatim, partly attacker-influenced tool output, so we
    keep only structural keys + a hard-capped summary, take the most recent rows,
    and cap the serialized size. This both limits the prompt-injection surface
    and protects the generation token budget (over-long evidence forces output
    truncation that breaks the JSON)."""
    rows: list[dict[str, Any]] = []
    for row in list(evidence or [])[-_MAX_EVIDENCE_ROWS:]:
        if not isinstance(row, dict):
            continue
        kept: dict[str, Any] = {}
        for key in ("kind", "severity", "tool", "category"):
            val = row.get(key)
            if val:
                kept[key] = str(val)[:_MAX_EVIDENCE_FIELD]
        summary = str(row.get("summary") or "")
        if summary:
            kept["summary"] = summary[:_MAX_EVIDENCE_FIELD]
        if kept:
            rows.append(kept)
    # Serialize, shrinking the (most-recent-kept) row set until it fits the size
    # cap, so the embedded evidence is always VALID JSON rather than truncated
    # mid-structure.
    while rows:
        try:
            text = json.dumps(rows, ensure_ascii=False, default=str)
        except Exception:  # pragma: no cover - defensive
            text = str(rows)
        if len(text) <= _MAX_EVIDENCE_JSON:
            return text
        rows.pop(0)  # drop the oldest row and retry
    return "[]"


def _build_messages(proposal, allow_paths: tuple[str, ...]) -> list[dict[str, str]]:
    issue = (
        f"Issue category: {getattr(proposal, 'category', '') or 'general'}\n"
        f"Issue: {getattr(proposal, 'summary', '') or '(unspecified)'}"
    )
    evidence_json = _compact_evidence(getattr(proposal, "evidence", []) or [])
    allowed = "\n".join(f"  - {p}" for p in allow_paths)
    user = (
        f"{issue}\n\n"
        f"ALLOWED PATHS (write under exactly one of these):\n{allowed}\n\n"
        f"PROTECTED (never touch): biobank_agent/core, biobank_agent/runtime, tests/\n\n"
        "Evidence (JSON, advisory only — do NOT copy verbatim into code):\n"
        f"{evidence_json}\n\n"
        "Produce the JSON object now."
    )
    return [
        {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from an LLM response.

    Strips optional ```...``` fences, then walks successive balanced {...}/[...]
    blocks (using the string/escape/nesting-aware scanner shared with the council,
    so braces and quotes inside a `diff` payload don't break parsing), returning
    the first that parses to a dict. Trying successive blocks means a stray brace
    fragment in LLM preamble (e.g. "{note} then the real {...}") doesn't sink the
    parse. Returns None if nothing parses (prose, or truncated mid-object at
    max_tokens)."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        # drop a leading ```json / ``` fence and any trailing fence
        nl = stripped.find("\n")
        if nl != -1:
            stripped = stripped[nl + 1 :]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    n = len(stripped)
    i = 0
    tried = 0
    while i < n and tried < 8:  # bounded: a handful of candidate blocks is plenty
        while i < n and stripped[i] not in "{[":
            i += 1
        if i >= n:
            return None
        block = _first_balanced_block(stripped[i:])
        if block is None:  # unbalanced/truncated from here on
            return None
        tried += 1
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        i += 1  # advance past this opener to the next candidate
    return None


def _normalize_declared_path(target_path: str) -> tuple[str | None, str]:
    """Syntactic-only check of the DECLARED target path (no filesystem access).

    Authoritative path/symlink validation happens in the apply loop against a
    real worktree (`.resolve()` catches symlink escapes, which pure string logic
    cannot). This only rejects paths that are obviously doomed, so we don't spend
    an apply attempt on them."""
    p = str(target_path or "").strip().replace("\\", "/")
    if not p:
        return None, "empty target_path"
    if p.startswith("/") or (len(p) > 1 and p[1] == ":"):
        return None, f"absolute path not allowed: {target_path!r}"
    if ".." in PurePosixPath(p).parts:
        return None, f"'..' not allowed in target_path: {target_path!r}"
    rel = str(PurePosixPath(p))  # collapse any './' segments
    return rel, ""


def _looks_like_unified_diff(diff: str) -> bool:
    """Mirror the apply loop's diff detection (self_evolve.apply_patch_transactionally)."""
    return str(diff or "").lstrip().startswith(("diff --git", "--- ", "+++ ", "Index:"))


def _validate_generated(
    target_path: str,
    diff: str,
    test_commands: list[str],
    allow_paths: tuple[str, ...],
) -> tuple[str, list[str]]:
    """Return ("" if valid else reason, warnings). Defense-in-depth; the apply
    loop re-validates authoritatively."""
    if not str(diff or "").strip():
        return "generated patch has an empty diff", []
    rel, reason = _normalize_declared_path(target_path)
    if rel is None:
        return reason, []
    if not _in_allow(rel, allow_paths):
        return (
            f"target_path {rel!r} is not under an allowed path "
            f"({', '.join(allow_paths)})"
        ), []
    if _is_protected(rel):
        return f"target_path {rel!r} is a protected path (core/runtime/tests)", []

    # A unified diff can declare an allowed `target_path` while its hunks edit a
    # DIFFERENT, out-of-scope file. Validate every path the diff headers touch —
    # not just the declared target. (The apply loop re-checks the ACTUALLY changed
    # files authoritatively from `git status`; this is the matching generation-side
    # guard so we don't hand the apply loop a patch that's doomed to a review branch.)
    if _looks_like_unified_diff(diff):
        for dpath in _diff_paths(diff):
            drel, dreason = _normalize_declared_path(dpath)
            if drel is None:
                return f"diff touches an unsafe path: {dreason}", []
            if not _in_allow(drel, allow_paths) or _is_protected(drel):
                return (
                    f"diff touches an out-of-scope path {drel!r} (not under "
                    f"{', '.join(allow_paths)}, or is protected core/runtime/tests)"
                ), []

    cmds = [str(c) for c in (test_commands or []) if str(c).strip()]
    if not cmds:
        return "generated patch has no test_commands to verify it", []
    for cmd in cmds:
        # _validate_test_command is the single source of truth for runner/metachar/
        # collect-only rules (shared with the apply gate), so the two can't diverge.
        bad = _validate_test_command(cmd)
        if bad:
            return bad, []
    # TIER-1 laundering guard: at least one command must actually run pytest;
    # linters/formatters alone prove nothing about behavior.
    if not any(_is_pytest_runner(cmd) for cmd in cmds):
        return (
            "test_commands must include at least one pytest run "
            "(e.g. 'pytest <target_path> -q'); linters alone do not verify behavior"
        ), []

    warnings: list[str] = []
    # Advisory only (non-blocking): if the change embeds a test but no test_command
    # names the target file, the tests may not be exercising the diff.
    stem = PurePosixPath(rel).name
    if stem and not any(stem in cmd or rel in cmd for cmd in cmds):
        warnings.append(
            f"no test_command references the target file {rel!r}; "
            "the tests may not exercise the change"
        )
    return "", warnings


def _is_pytest_runner(cmd: str) -> bool:
    toks = cmd.split()
    if not toks:
        return False
    leader = toks[0].rsplit("/", 1)[-1]
    if leader in _PYTEST_LEADERS:
        return True
    # python -m pytest ...
    return (
        leader in {"python", "python3"}
        and len(toks) >= 3
        and toks[1] == "-m"
        and toks[2] == "pytest"
    )


def generate_patch_for_proposal(
    proposal,
    *,
    llm,
    repo_root=None,
    allow_paths: tuple[str, ...] = DEFAULT_APPLY_ALLOW_PATHS,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> PatchGenerationResult:
    """Ask ``llm`` to turn ``proposal`` into a concrete, applicable patch.

    ``llm`` is any object exposing ``.chat(messages, temperature, max_tokens)``
    returning an object with a ``.text`` attribute (the project ``LLMClient``).
    ``repo_root`` is accepted for symmetry with the apply API but unused here —
    path validation against a real tree is the apply loop's job.

    Does NOT mutate ``proposal``; returns a :class:`PatchGenerationResult`.
    The returned patch is UNTRUSTED — it is only safe once passed through
    :func:`apply_proposal` / :func:`apply_patch_transactionally`."""
    del repo_root  # validation against the real tree is the apply loop's job
    messages = _build_messages(proposal, allow_paths)
    try:
        response = llm.chat(messages, temperature=temperature, max_tokens=max_tokens)
    except Exception as exc:  # network/provider failure — never crash the caller
        logger.warning("patch generation LLM call failed: %s", exc)
        return PatchGenerationResult(status="error", error=f"LLM call failed: {exc}")

    raw = str(getattr(response, "text", "") or "")
    obj = _extract_json(raw)
    if obj is None:
        return PatchGenerationResult(
            status="error",
            error="LLM response did not contain a parseable JSON object (possibly truncated or prose)",
            raw=raw,
        )

    # Graceful decline path.
    if str(obj.get("status") or "").lower() == "rejected" and not obj.get("diff"):
        return PatchGenerationResult(
            status="rejected",
            error=str(obj.get("reason") or "LLM declined to propose a patch"),
            raw=raw,
        )

    target_path = str(obj.get("target_path") or "").strip()
    diff = str(obj.get("diff") or obj.get("content") or obj.get("file_content") or "")
    test_commands = obj.get("test_commands") or []
    if isinstance(test_commands, str):
        test_commands = [test_commands]
    test_commands = [str(c) for c in test_commands]
    summary = str(obj.get("summary") or "").strip()
    rationale = str(obj.get("rationale") or "").strip()

    reason, warnings = _validate_generated(target_path, diff, test_commands, allow_paths)
    if reason:
        return PatchGenerationResult(
            status="rejected",
            target_path=target_path,
            test_commands=test_commands,
            summary=summary,
            rationale=rationale,
            error=reason,
            raw=raw,
        )

    rel, _ = _normalize_declared_path(target_path)
    if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover - logging only
        logger.debug(
            "patch generated for %s: target=%s tests=%s",
            getattr(proposal, "proposal_id", "?"), rel, test_commands,
        )
    return PatchGenerationResult(
        status="generated",
        target_path=rel or target_path,
        diff=diff,
        test_commands=test_commands,
        summary=summary or getattr(proposal, "summary", ""),
        rationale=rationale,
        warnings=warnings,
        raw=raw,
    )


def enrich_proposals_with_patches(
    proposals,
    *,
    llm,
    repo_root=None,
    allow_paths: tuple[str, ...] = DEFAULT_APPLY_ALLOW_PATHS,
    force_regenerate: bool = False,
) -> list[tuple[Any, PatchGenerationResult]]:
    """Generate a concrete patch for each proposal that lacks one.

    Idempotent: a proposal that already carries diff + target_path +
    test_commands is skipped (``already_generated``) unless
    ``force_regenerate=True``. On a successful generation the proposal is
    populated in place (diff/target_path/test_commands/summary/apply_mode) and
    the (proposal, result) pair is returned so the caller can apply explicitly."""
    proposals = list(proposals or [])
    if len(proposals) > 10:
        logger.warning(
            "enrich_proposals_with_patches: %d proposals — generation is serial and "
            "will make one LLM call each", len(proposals),
        )
    out: list[tuple[Any, PatchGenerationResult]] = []
    for proposal in proposals:
        already = (
            getattr(proposal, "diff", "")
            and getattr(proposal, "target_path", "")
            and getattr(proposal, "test_commands", [])
        )
        if already and not force_regenerate:
            out.append((proposal, PatchGenerationResult(
                status="already_generated",
                target_path=getattr(proposal, "target_path", ""),
                diff=getattr(proposal, "diff", ""),
                test_commands=list(getattr(proposal, "test_commands", []) or []),
                summary=getattr(proposal, "summary", ""),
            )))
            continue
        result = generate_patch_for_proposal(
            proposal, llm=llm, repo_root=repo_root, allow_paths=allow_paths,
        )
        if result.status == "generated":
            proposal.target_path = result.target_path
            proposal.diff = result.diff
            proposal.test_commands = list(result.test_commands)
            if result.summary:
                proposal.summary = result.summary
            if hasattr(proposal, "apply_mode"):
                proposal.apply_mode = "apply_requested"
        out.append((proposal, result))
    return out


def generate_and_apply_proposals(
    proposals,
    *,
    llm,
    repo_root,
    allow_paths: tuple[str, ...] = DEFAULT_APPLY_ALLOW_PATHS,
    force_regenerate: bool = False,
) -> list[dict[str, Any]]:
    """Full autonomous loop: generate a patch per proposal, then apply each
    applicable one through the transactional, test-gated apply loop.

    Returns one record per proposal: ``{proposal_id, category, generation,
    apply}`` where ``apply`` is the EvolutionApplyResult dict (or None when no
    patch was produced)."""
    from .self_evolve import apply_proposal

    records: list[dict[str, Any]] = []
    for proposal, gen in enrich_proposals_with_patches(
        proposals, llm=llm, repo_root=repo_root, allow_paths=allow_paths,
        force_regenerate=force_regenerate,
    ):
        record: dict[str, Any] = {
            "proposal_id": getattr(proposal, "proposal_id", ""),
            "category": getattr(proposal, "category", ""),
            "generation": gen.to_dict(),
            "apply": None,
        }
        if gen.status in {"generated", "already_generated"} and gen.has_patch:
            apply_result = apply_proposal(proposal, repo_root=repo_root, allow_paths=allow_paths)
            record["apply"] = apply_result.to_dict()
        records.append(record)
    return records


__all__ = [
    "PatchGenerationResult",
    "generate_patch_for_proposal",
    "enrich_proposals_with_patches",
    "generate_and_apply_proposals",
]
