"""Disclosure-control policy engine (NHS SDC + UKB DUA aligned).

Implements the rules a biobank skill output must respect before it
can be exported externally:

1. **Minimum cell count** — any aggregate count must be >= 5
   (configurable). Counts of 1-4 leak individual identity.
2. **Minimum cohort size** — any model / regression result must be
   based on a cohort of at least ``min_cohort`` (default 100) cases.
3. **Rounding** — counts are rounded to the nearest 5 before
   disclosure (NHS SDC standard).
4. **PHI fields** — any output containing UKB PHI fields must be
   redacted before export.

Important: the local project data is already de-identified. Internal
research execution should preserve analytical data and exact counts; this
engine is for explicit external-disclosure modes, not a default in-memory
data reduction step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from biobank_agent.core.events import SENSITIVE_FIELD_IDS, scrub_pii


@dataclass
class DisclosurePolicy:
    min_cell_count: int = 5
    min_cohort_size: int = 100
    round_counts_to: int = 5
    drop_phi_fields: bool = True

    @classmethod
    def strict(cls) -> "DisclosurePolicy":
        return cls(min_cell_count=10, min_cohort_size=200, round_counts_to=10)

    @classmethod
    def relaxed(cls) -> "DisclosurePolicy":
        return cls(min_cell_count=3, min_cohort_size=50, round_counts_to=5)


@dataclass
class DisclosureViolation:
    code: str
    message: str
    field: str = ""


@dataclass
class DisclosureCheck:
    passed: bool
    violations: list[DisclosureViolation] = field(default_factory=list)
    sanitised: dict[str, Any] = field(default_factory=dict)


# ── Internal helpers ────────────────────────────────────────


def _round(n: int, base: int) -> int:
    if base <= 1:
        return n
    return int(base * round(n / base))


def _maybe_round(value: Any, base: int) -> Any:
    if isinstance(value, int) and value >= 0:
        return _round(value, base)
    return value


def _check_counts(payload: dict, policy: DisclosurePolicy) -> list[DisclosureViolation]:
    out: list[DisclosureViolation] = []
    for key in ("n_cases", "n_controls", "n_total", "n_subjects", "n"):
        if key not in payload:
            continue
        try:
            n = int(payload[key])
        except (TypeError, ValueError):
            continue
        if 0 < n < policy.min_cell_count:
            out.append(DisclosureViolation(
                code="MIN_CELL",
                field=key,
                message=(
                    f"{key}={n} below minimum cell count {policy.min_cell_count}; "
                    f"refusing to disclose."
                ),
            ))
    return out


def _check_cohort(payload: dict, policy: DisclosurePolicy) -> list[DisclosureViolation]:
    n_cases = payload.get("n_cases")
    if isinstance(n_cases, int) and 0 < n_cases < policy.min_cohort_size:
        return [DisclosureViolation(
            code="MIN_COHORT",
            field="n_cases",
            message=(
                f"n_cases={n_cases} below minimum cohort size "
                f"{policy.min_cohort_size}; results unstable."
            ),
        )]
    return []


def _strip_phi(payload: dict) -> dict:
    return scrub_pii(payload)


# ── Public API ──────────────────────────────────────────────


def enforce(
    *,
    payload: dict[str, Any],
    policy: Optional[DisclosurePolicy] = None,
    strict: bool = True,
) -> DisclosureCheck:
    """Apply disclosure controls to a tool result before disclosure.

    Returns a ``DisclosureCheck``:
        - ``passed=True`` => safe to surface; ``sanitised`` is the
          rounded / PHI-stripped payload.
        - ``passed=False`` => violations explain why.

    When ``strict=False`` the function still produces a sanitised
    payload but does not flag below-minimum counts as failures (used
    in research-only mode where the user takes responsibility).
    """
    policy = policy or DisclosurePolicy()
    violations: list[DisclosureViolation] = []
    violations.extend(_check_counts(payload, policy))
    violations.extend(_check_cohort(payload, policy))

    sanitised = _strip_phi(payload) if policy.drop_phi_fields else dict(payload)
    if policy.round_counts_to > 1:
        for key in ("n_cases", "n_controls", "n_total", "n_subjects", "n"):
            if key in sanitised:
                sanitised[key] = _maybe_round(sanitised[key], policy.round_counts_to)

    passed = not violations or not strict
    return DisclosureCheck(passed=passed, violations=violations, sanitised=sanitised)


__all__ = [
    "DisclosurePolicy",
    "DisclosureCheck",
    "DisclosureViolation",
    "enforce",
]
