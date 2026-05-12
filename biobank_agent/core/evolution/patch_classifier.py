"""Risk classifier for auto-generated skill patches (M4 §6.1).

Three risk classes drive the auto-merge policy:

    LOW    - whitespace/comments/docstring/parameter tweaks; auto-merge
             after eval pass.
    MEDIUM - small logic fix matching a known failure pattern; surface
             a user confirmation modal before auto-merging.
    HIGH   - new skill, deletes, dependency changes, cross-module - go
             through git PR for human review.

The classifier is pure-Python and deliberately conservative: when in
doubt we escalate. ``classify`` accepts the patch as a unified diff
string (what reflexion produces) plus the eval result; both are
already pre-computed by the time auto_merger calls us.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PatchAssessment:
    risk: RiskLevel
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)


_HIGH_TRIGGERS = (
    re.compile(r"^\s*-\s*from\s+", re.M),       # removal of an import
    re.compile(r"^\s*\+\s*import\s+", re.M),    # new top-level import
    re.compile(r"^\+\+\+\s+(?:[ab]/)?pyproject\.toml", re.M),
    re.compile(
        r"^\+\+\+\s+(?:[ab]/)?biobank_agent/(?:agent|cli|registry)\.py", re.M
    ),
    re.compile(r"^\+\+\+\s+(?:[ab]/)?biobank_agent/core/", re.M),
    re.compile(r"^\+\+\+\s+(?:[ab]/)?tests/", re.M),
    re.compile(r"^\s*\+\s*def\s+\w+\(", re.M),  # net new top-level function
    re.compile(r"^\s*\+\s*class\s+\w+", re.M),
)

_MEDIUM_TRIGGERS = (
    re.compile(r"^\s*[-+]\s*(if|elif|while|for|return|raise|assert)\b", re.M),
)


def _count_diff_lines(diff: str) -> tuple[int, int, int]:
    added = 0
    removed = 0
    files: set[str] = set()
    for line in diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            path = line[4:].strip()
            # Drop common ``a/`` ``b/`` prefixes git uses so a single
            # file edit doesn't look like two distinct files.
            if path.startswith(("a/", "b/")):
                path = path[2:]
            if path and path != "/dev/null":
                files.add(path)
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed, len(files)


def classify(diff: str, *, eval_result: Any = None) -> PatchAssessment:
    """Classify a unified-diff patch for auto-merge eligibility."""
    if not diff:
        return PatchAssessment(
            risk=RiskLevel.HIGH,
            reason="empty diff is unusable",
            metrics={"added": 0, "removed": 0, "files": 0},
        )

    added, removed, files_touched = _count_diff_lines(diff)
    metrics = {"added": added, "removed": removed, "files": files_touched}

    for pattern in _HIGH_TRIGGERS:
        if pattern.search(diff):
            return PatchAssessment(
                risk=RiskLevel.HIGH,
                reason=f"matches high-risk pattern {pattern.pattern!r}",
                metrics=metrics,
            )

    if files_touched > 1 or added + removed > 30:
        return PatchAssessment(
            risk=RiskLevel.HIGH,
            reason=f"changeset spans {files_touched} file(s) and {added + removed} line(s)",
            metrics=metrics,
        )

    if eval_result is not None and not getattr(eval_result, "all_passed", True):
        return PatchAssessment(
            risk=RiskLevel.HIGH,
            reason="eval did not pass; cannot auto-merge",
            metrics=metrics,
        )

    for pattern in _MEDIUM_TRIGGERS:
        if pattern.search(diff):
            return PatchAssessment(
                risk=RiskLevel.MEDIUM,
                reason="touches control flow; needs user confirmation",
                metrics=metrics,
            )

    return PatchAssessment(
        risk=RiskLevel.LOW,
        reason="small comment/parameter tweak with passing eval",
        metrics=metrics,
    )


__all__ = ["RiskLevel", "PatchAssessment", "classify"]
