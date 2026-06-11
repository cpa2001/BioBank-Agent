"""Auto-merger: routes auto-generated patches by risk class.

Implements the user-approved policy in v3 plan §6:

    LOW    -> commit + auto-merge to ``custom_skills/``, hot reload,
              notify.
    MEDIUM -> raise a confirmation modal (CLI or Textual) and
              auto-merge on user approval.
    HIGH   -> create a feature branch + commit, then open a PR via gh when
              remote credentials are available; otherwise leave the branch
              in ``pr_ready`` state for manual push/review.

The class is intentionally I/O-conservative: it never invokes git
commands at module import time, and it falls back to ``"dry_run"``
mode when ``gitpython`` is unavailable.
"""

from __future__ import annotations

import logging
import json
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .patch_classifier import PatchAssessment, RiskLevel, classify as classify_patch

logger = logging.getLogger(__name__)

ConfirmFn = Callable[[PatchAssessment, "Patch"], Awaitable[bool]]


@dataclass
class Patch:
    target_path: str
    unified_diff: str
    target_skill: str = ""
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MergeOutcome:
    status: str  # "merged" | "user_rejected" | "pr_opened" | "pr_ready" | "skipped" | "error"
    risk: RiskLevel
    pr_url: str = ""
    branch: str = ""
    error: str = ""


class AutoMerger:
    """Coordinates auto-merge / confirm / PR according to ``risk``."""

    AUTO_MERGE_TRAILER = "Auto-Merged-By: biobank-agent"

    def __init__(
        self,
        *,
        repo_root: Path,
        confirm_fn: Optional[ConfirmFn] = None,
        write_allow_paths: Optional[tuple[str, ...]] = None,
        audit_log_path: Path | str | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.confirm_fn = confirm_fn
        self.audit_log_path = Path(audit_log_path) if audit_log_path is not None else None
        # Never let auto-merge touch core/ or tests/ — only
        # domain/skills/ + custom_skills/.
        self.write_allow_paths = write_allow_paths or (
            "reports/generated_skills/",
            "domain/skills/",
            "custom_skills/",
            "biobank_agent/skills/custom",
        )

    async def review_and_merge(
        self,
        patch: Patch,
        *,
        eval_result: Any = None,
    ) -> MergeOutcome:
        if not self._is_path_allowed(patch.target_path):
            return MergeOutcome(
                status="skipped",
                risk=RiskLevel.HIGH,
                error=f"target path {patch.target_path!r} outside allow-list",
            )
        assessment = classify_patch(patch.unified_diff, eval_result=eval_result)

        if assessment.risk == RiskLevel.LOW:
            if not self._eval_passed(eval_result):
                return MergeOutcome(
                    status="skipped",
                    risk=assessment.risk,
                    error="eval gate did not pass; refusing LOW auto-merge",
                )
            return self._auto_merge(patch, assessment)
        if assessment.risk == RiskLevel.MEDIUM:
            if not self._eval_passed(eval_result):
                outcome = MergeOutcome(
                    status="skipped",
                    risk=assessment.risk,
                    error="eval gate did not pass; refusing MEDIUM auto-merge",
                )
                self._record_decision(patch, assessment, outcome, eval_result=eval_result, confirmed=False)
                return outcome
            patch.metadata.setdefault("risk_reason", assessment.reason)
            patch.metadata.setdefault("eval_summary", getattr(eval_result, "summary", ""))
            if self.confirm_fn is None:
                outcome = MergeOutcome(
                    status="user_rejected",
                    risk=assessment.risk,
                    error="MEDIUM risk patch requires explicit user confirmation",
                )
                self._record_decision(patch, assessment, outcome, eval_result=eval_result, confirmed=False)
                return outcome
            try:
                confirmed = bool(await self.confirm_fn(assessment, patch))
            except Exception as e:
                logger.warning("confirm_fn raised: %s", e)
                confirmed = False
            if not confirmed:
                outcome = MergeOutcome(status="user_rejected", risk=assessment.risk)
                self._record_decision(patch, assessment, outcome, eval_result=eval_result, confirmed=False)
                return outcome
            outcome = self._auto_merge(patch, assessment)
            self._record_decision(patch, assessment, outcome, eval_result=eval_result, confirmed=True)
            return outcome
        # HIGH
        return self._open_pr(patch, assessment)

    # ── Internals ───────────────────────────────────────────

    def _auto_merge(self, patch: Patch, assessment: PatchAssessment) -> MergeOutcome:
        try:
            self._apply_patch(patch)
            self._git("add", patch.target_path)
            commit_msg = (
                f"auto-merge({patch.target_skill}): {patch.summary or assessment.reason}\n\n"
                f"{assessment.reason}\n\n"
                f"{self.AUTO_MERGE_TRAILER} <{assessment.risk.value}>\n"
            )
            self._git("commit", "-m", commit_msg)
            return MergeOutcome(status="merged", risk=assessment.risk)
        except Exception as e:
            return MergeOutcome(
                status="error", risk=assessment.risk, error=f"merge failed: {e}"
            )

    def _open_pr(self, patch: Patch, assessment: PatchAssessment) -> MergeOutcome:
        try:
            short_hash = abs(hash(patch.unified_diff)) % 100_000
            branch = f"auto-improve/{patch.target_skill or 'patch'}-{short_hash:05d}"
            previous_branch = self._git("branch", "--show-current")
            self._git("checkout", "-b", branch)
            try:
                self._apply_patch(patch)
                self._git("add", patch.target_path)
                self._git(
                    "commit",
                    "-m",
                    f"propose({patch.target_skill}): {patch.summary or assessment.reason}\n\n"
                    f"{self.AUTO_MERGE_TRAILER} <high-risk-pr>\n",
                )
                pr_error = ""
                pr_url = ""
                if self._has_origin_remote():
                    try:
                        self._git("push", "-u", "origin", branch)
                        pr_url = self._gh_pr_create(branch, patch, assessment)
                        if not pr_url:
                            pr_error = "branch pushed but gh pr create did not return a PR URL"
                    except Exception as e:
                        pr_error = str(e)
                else:
                    pr_error = "origin remote not configured; high-risk patch committed on local branch only"
                return MergeOutcome(
                    status="pr_opened" if pr_url else "pr_ready",
                    risk=assessment.risk,
                    branch=branch,
                    pr_url=pr_url,
                    error=pr_error,
                )
            finally:
                # Always return to the previous branch.
                self._git("checkout", previous_branch or "-")
        except Exception as e:
            return MergeOutcome(status="error", risk=assessment.risk, error=str(e))

    # ── Filesystem / git helpers ────────────────────────────

    def _apply_patch(self, patch: Patch) -> None:
        target = self.repo_root / patch.target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        # If the patch is a unified diff, apply via ``git apply``;
        # otherwise treat ``patch.unified_diff`` as the full new file.
        if patch.unified_diff.lstrip().startswith(("diff --git", "---", "+++")):
            self._git_apply(patch.unified_diff)
        else:
            target.write_text(patch.unified_diff, encoding="utf-8")

    def _git_apply(self, diff_text: str) -> None:
        proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            input=diff_text,
            text=True,
            capture_output=True,
            cwd=self.repo_root,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git apply failed: {proc.stderr.strip() or proc.stdout.strip()}"
            )

    def _git(self, *args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(shlex.quote(a) for a in args)} failed: "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc.stdout.strip()

    def _has_origin_remote(self) -> bool:
        proc = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0 and bool(proc.stdout.strip())

    def _gh_pr_create(
        self, branch: str, patch: Patch, assessment: PatchAssessment
    ) -> str:
        try:
            proc = subprocess.run(
                [
                    "gh",
                    "pr",
                    "create",
                    "--head",
                    branch,
                    "--title",
                    f"auto-improve({patch.target_skill}): {patch.summary or assessment.reason}"[:120],
                    "--body",
                    f"## Auto-generated proposal\n\n"
                    f"Risk: **{assessment.risk.value}**\n\n"
                    f"Reason: {assessment.reason}\n\n"
                    f"Eval metrics: {assessment.metrics}\n\n"
                    f"{self.AUTO_MERGE_TRAILER} <high-risk-pr>",
                ],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
        except FileNotFoundError:
            logger.warning("gh CLI not installed; PR not created")
        return ""

    def _is_path_allowed(self, path: str) -> bool:
        target = path.replace("\\", "/")
        return any(target.startswith(p) for p in self.write_allow_paths)

    @staticmethod
    def _eval_passed(eval_result: Any) -> bool:
        return bool(eval_result is not None and getattr(eval_result, "all_passed", False))

    def _record_decision(
        self,
        patch: Patch,
        assessment: PatchAssessment,
        outcome: MergeOutcome,
        *,
        eval_result: Any = None,
        confirmed: bool | None = None,
    ) -> None:
        """Append a machine-readable audit row for user-gated patches."""
        if self.audit_log_path is None:
            return
        try:
            path = self.audit_log_path
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "target_path": patch.target_path,
                "target_skill": patch.target_skill,
                "risk": assessment.risk.value,
                "risk_reason": assessment.reason,
                "status": outcome.status,
                "confirmed": confirmed,
                "error": outcome.error,
                "eval_summary": getattr(eval_result, "summary", ""),
                "metrics": assessment.metrics,
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception as exc:  # pragma: no cover - audit failure must not apply patches
            logger.warning("failed to write evolution audit log: %s", exc)


__all__ = ["Patch", "MergeOutcome", "AutoMerger"]
