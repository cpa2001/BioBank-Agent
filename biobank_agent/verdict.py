"""Structured verdict system — PASS / FAIL / PARTIAL judgement framework.

Ported from heathcliff233/my_codex verifier.toml patterns:
  - PASS:    All checks passed, no issues beyond minor suggestions
  - FAIL:    At least one blocking issue or failing check
  - PARTIAL: Could not complete due to environment blockers (NOT for ambiguity)

Used by:
  - Verifier role in orchestrator.py (adversarial review after implementation)
  - Agent post-run verification (4-phase pipeline Verify step)
  - Safety checks (k-anonymity, minimum cell count)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMClient

logger = logging.getLogger(__name__)


class VerdictStatus(Enum):
    """Final verdict — exactly one of three states."""
    PASS = "PASS"        # all checks passed, no blocking issues
    FAIL = "FAIL"        # at least one blocking issue
    PARTIAL = "PARTIAL"  # could not complete (environment blocker ONLY)


class Severity(Enum):
    """Issue severity levels."""
    BLOCKER = "BLOCKER"         # must fix before shipping
    WARNING = "WARNING"         # should address
    SUGGESTION = "SUGGESTION"   # optional improvement


@dataclass
class Issue:
    """A single issue found during verification."""
    file: str                   # file path or "general"
    line: int = 0               # 0 if not line-specific
    severity: Severity = Severity.WARNING
    description: str = ""
    suggested_fix: str = ""


@dataclass
class Check:
    """A single verification check with its result."""
    name: str                   # e.g. "null safety", "output schema"
    command: str = ""           # exact command or check description
    output: str = ""            # verbatim output (truncated)
    passed: bool = True

    @property
    def result_str(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class VerdictResult:
    """Complete verdict from a verification run."""
    status: VerdictStatus
    checks: list[Check] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    rationale: str = ""

    @property
    def n_blockers(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.BLOCKER)

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def summary(self) -> str:
        """One-line summary for display."""
        parts = [f"**VERDICT: {self.status.value}**"]
        if self.checks:
            passed = sum(1 for c in self.checks if c.passed)
            parts.append(f"({passed}/{len(self.checks)} checks passed)")
        if self.n_blockers:
            parts.append(f"{self.n_blockers} blockers")
        if self.n_warnings:
            parts.append(f"{self.n_warnings} warnings")
        return " — ".join(parts)

    def report(self) -> str:
        """Structured report for LLM context or display."""
        lines = [f"## Verdict: {self.status.value}\n"]
        if self.rationale:
            lines.append(f"{self.rationale}\n")

        if self.checks:
            lines.append("### Checks")
            for c in self.checks:
                lines.append(f"- **{c.name}**: {c.result_str}")
                if c.command:
                    lines.append(f"  Command: `{c.command}`")
                if not c.passed and c.output:
                    lines.append(f"  Output: {c.output[:200]}")

        if self.issues:
            lines.append("\n### Issues")
            for i in self.issues:
                lines.append(
                    f"- [{i.severity.value}] {i.file}"
                    + (f":{i.line}" if i.line else "")
                    + f" — {i.description}"
                )
                if i.suggested_fix:
                    lines.append(f"  Fix: {i.suggested_fix}")

        return "\n".join(lines)


def determine_verdict(checks: list[Check], issues: list[Issue]) -> VerdictStatus:
    """Compute verdict from checks and issues.

    Rules (from my_codex verifier.toml):
    - PASS:    all checks passed AND no BLOCKER issues
    - FAIL:    any check failed OR any BLOCKER issue
    - PARTIAL: only for environment failures (check had to be skipped)
    """
    has_failed_check = any(not c.passed for c in checks)
    has_blocker = any(i.severity == Severity.BLOCKER for i in issues)

    if has_blocker or has_failed_check:
        return VerdictStatus.FAIL
    return VerdictStatus.PASS


def build_verdict(
    checks: list[Check],
    issues: list[Issue],
    rationale: str = "",
) -> VerdictResult:
    """Convenience: compute status and build VerdictResult."""
    status = determine_verdict(checks, issues)
    return VerdictResult(
        status=status,
        checks=checks,
        issues=issues,
        rationale=rationale,
    )


class VerdictEngine:
    """LLM-powered adversarial verification engine.

    Mirrors the verifier.toml adversarial mindset:
    "Assume every change has a bug until proven otherwise."
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def verify_skill_result(
        self,
        skill_name: str,
        args: dict,
        result: dict,
        context: str = "",
    ) -> VerdictResult:
        """Verify a skill execution result for correctness."""
        checks = []
        issues = []

        # ── Check 1: Output schema ────────────
        if isinstance(result, dict):
            if "error" in result:
                checks.append(Check("no_error", passed=False, output=str(result["error"])[:200]))
                issues.append(Issue(
                    file=f"skills/{skill_name}.py",
                    severity=Severity.BLOCKER,
                    description=f"Skill returned error: {result['error']}",
                ))
            else:
                checks.append(Check("no_error", passed=True))
        else:
            checks.append(Check("output_is_dict", passed=False, output=str(type(result))))
            issues.append(Issue(
                file=f"skills/{skill_name}.py",
                severity=Severity.BLOCKER,
                description=f"Skill returned {type(result).__name__} instead of dict",
            ))

        # ── Check 2: Numeric sanity ───────────
        for key, val in (result if isinstance(result, dict) else {}).items():
            if isinstance(val, (int, float)):
                if key.startswith("n_") and val < 0:
                    checks.append(Check(f"{key}_non_negative", passed=False, output=f"{key}={val}"))
                    issues.append(Issue(
                        file=f"skills/{skill_name}.py",
                        severity=Severity.BLOCKER,
                        description=f"Count field {key} is negative: {val}",
                    ))
                elif key in ("auc", "auc_mean", "prevalence_pct") and not (0 <= val <= 100):
                    checks.append(Check(f"{key}_range", passed=False, output=f"{key}={val}"))
                    issues.append(Issue(
                        file=f"skills/{skill_name}.py",
                        severity=Severity.WARNING,
                        description=f"Metric {key}={val} outside expected range",
                    ))

        # ── Check 3: Non-empty results ────────
        if isinstance(result, dict) and not result:
            checks.append(Check("non_empty_result", passed=False))
            issues.append(Issue(
                file=f"skills/{skill_name}.py",
                severity=Severity.WARNING,
                description="Skill returned empty dict",
            ))
        elif isinstance(result, dict):
            checks.append(Check("non_empty_result", passed=True))

        return build_verdict(checks, issues)
