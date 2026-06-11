"""Formal constraint verification for biobank analysis outputs.

Inspired by VERGE (arXiv:2601.06181): semantic routing identifies verifiable
claims, Z3 checks logical consistency, Minimal Correction Subsets (MCS)
identify which claims to relax if verification fails.

Scope: numerical bounds, cohort constraints, statistical validity.
NOT: prose quality, clinical interpretation (those stay in verdict.py).

Architecture decision: standalone module producing
Check/Issue/VerdictResult types from verdict.py — merges cleanly with
LLM-based verdicts without coupling to LLMClient.

Usage
-----
    from biobank_agent.verification import BiobankConstraintVerifier

    verifier = BiobankConstraintVerifier()
    result = verifier.verify_cohort_claims(n_cases=300_000, n_controls=300_000)
    assert result.status == VerdictStatus.FAIL
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .verdict import Check, Issue, Severity, VerdictResult, VerdictStatus, build_verdict

logger = logging.getLogger(__name__)

# ── Z3 availability guard ──────────────────────────────────────
try:
    from z3 import (
        Solver, Int, Real, And, Or, Not, Implies,
        sat, unsat, unknown,
        IntVal, RealVal,
    )
    _Z3_AVAILABLE = True
except ImportError:
    _Z3_AVAILABLE = False
    logger.debug("z3-solver not installed; formal verification disabled (fallback to bounds checks)")


# ── Claim types ────────────────────────────────────────────────

class ClaimType(Enum):
    """Types of verifiable claims extracted from agent output."""
    COHORT_SIZE = "cohort_size"
    METRIC_BOUND = "metric_bound"
    CI_ORDERING = "ci_ordering"
    FILTER_LOGIC = "filter_logic"
    SAMPLE_RATIO = "sample_ratio"
    STATISTICAL_POWER = "statistical_power"


@dataclass
class MinimalCorrection:
    """One element of a Minimal Correction Subset (MCS).

    Identifies the smallest set of claims that must change to restore
    logical consistency.
    """
    claim: str
    current_value: Any
    constraint_violated: str
    suggested_action: str


@dataclass
class VerificationResult:
    """Output of formal verification pass (extends into VerdictResult)."""
    status: str  # "verified" | "violated" | "undecidable"
    claims_checked: int = 0
    checks: list[Check] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    corrections: list[MinimalCorrection] = field(default_factory=list)

    def to_verdict(self, rationale: str = "") -> VerdictResult:
        """Convert to VerdictResult for merging with LLM verdict."""
        return build_verdict(self.checks, self.issues, rationale=rationale)


# ── UK Biobank hard constraints ────────────────────────────────

class UKBiobankConstraints:
    """Immutable constraints from UK Biobank dataset.

    Sources values from biobank_agent.constants (single source of truth).
    """
    from .constants import (
        UKB_TOTAL_PARTICIPANTS as _total,
        UKB_MIN_AGE_RECRUITMENT as _min_age,
        UKB_MAX_AGE_RECRUITMENT as _max_age,
        UKB_ASSESSMENT_CENTRES as _centres,
        UKB_MAX_FOLLOW_UP_YEARS as _follow_up,
    )
    TOTAL_PARTICIPANTS = _total
    MIN_AGE_AT_RECRUITMENT = _min_age
    MAX_AGE_AT_RECRUITMENT = _max_age
    MIN_FOLLOW_UP_YEARS = 0
    MAX_FOLLOW_UP_YEARS = _follow_up
    N_ASSESSMENT_CENTRES = _centres
    del _total, _min_age, _max_age, _centres, _follow_up  # cleanup namespace


# ── Valid metric ranges ────────────────────────────────────────

METRIC_RANGES: dict[str, tuple[float, float]] = {
    "auc": (0.0, 1.0),
    "auc_mean": (0.0, 1.0),
    "roc_auc": (0.0, 1.0),
    "accuracy": (0.0, 1.0),
    "precision": (0.0, 1.0),
    "recall": (0.0, 1.0),
    "sensitivity": (0.0, 1.0),
    "specificity": (0.0, 1.0),
    "f1": (0.0, 1.0),
    "f1_score": (0.0, 1.0),
    "r_squared": (0.0, 1.0),
    "r2": (0.0, 1.0),
    "p_value": (0.0, 1.0),
    "prevalence": (0.0, 1.0),  # as proportion
    "prevalence_pct": (0.0, 100.0),  # as percentage
    "odds_ratio": (0.0, float("inf")),
    "hazard_ratio": (0.0, float("inf")),
    "relative_risk": (0.0, float("inf")),
    "concordance_index": (0.0, 1.0),
    "c_index": (0.0, 1.0),
    "brier_score": (0.0, 1.0),
}


class BiobankConstraintVerifier:
    """Z3-backed verifier for biobank analysis claims.

    Falls back to simple bounds checking when z3-solver is not installed.
    Produces Check/Issue types compatible with verdict.py.
    """

    def __init__(
        self,
        total_participants: int = UKBiobankConstraints.TOTAL_PARTICIPANTS,
        custom_constraints: Optional[dict[str, tuple[float, float]]] = None,
    ) -> None:
        self.total_participants = total_participants
        self.metric_ranges = {**METRIC_RANGES, **(custom_constraints or {})}

    @property
    def z3_available(self) -> bool:
        return _Z3_AVAILABLE

    # ── Cohort verification ────────────────────────────────────

    def verify_cohort_claims(
        self,
        n_cases: Optional[int] = None,
        n_controls: Optional[int] = None,
        total: Optional[int] = None,
        n_excluded: Optional[int] = None,
    ) -> VerificationResult:
        """Verify cohort size claims are logically consistent.

        Checks:
          - n_cases + n_controls ≤ total_participants
          - Each count ≥ 0
          - If total provided: n_cases + n_controls ≤ total ≤ total_participants
          - n_excluded + total ≤ total_participants (if both provided)

        Strategy: ALWAYS run the descriptive bounds checks — they produce the
        human-readable, per-constraint issues and corrections the rest of the
        system relies on ("n_excluded=… exceeds …"). When z3 is installed,
        additionally run the formal Z3 consistency proof + Minimal Correction
        Subset extraction and merge its findings on top. A solver-only path
        silently degraded output to a single opaque "contradictory" verdict;
        merging keeps the rich diagnostics regardless of whether z3 is present.
        """
        if n_cases is None and n_controls is None and total is None and n_excluded is None:
            return VerificationResult(status="verified", claims_checked=0)

        bounds = self._verify_cohort_bounds(n_cases, n_controls, total, n_excluded)
        if not _Z3_AVAILABLE:
            return bounds

        z3_result = self._verify_cohort_z3(n_cases, n_controls, total, n_excluded)
        return self._merge_cohort_results(bounds, z3_result)

    def _verify_cohort_bounds(
        self,
        n_cases: Optional[int] = None,
        n_controls: Optional[int] = None,
        total: Optional[int] = None,
        n_excluded: Optional[int] = None,
    ) -> VerificationResult:
        """Descriptive arithmetic bounds checking (no solver required).

        Emits a Check + human-readable Issue per violated constraint so callers
        get actionable, specific diagnostics rather than one opaque verdict.
        """
        checks: list[Check] = []
        issues: list[Issue] = []
        corrections: list[MinimalCorrection] = []

        if n_cases is not None:
            if n_cases < 0:
                checks.append(Check("n_cases_non_negative", passed=False, output=f"n_cases={n_cases}"))
                issues.append(Issue(
                    file="general", severity=Severity.BLOCKER,
                    description=f"n_cases is negative: {n_cases}",
                ))
            elif n_cases > self.total_participants:
                checks.append(Check("n_cases_within_ukb", passed=False, output=f"n_cases={n_cases} > {self.total_participants}"))
                issues.append(Issue(
                    file="general", severity=Severity.BLOCKER,
                    description=f"n_cases={n_cases} exceeds total UK Biobank participants ({self.total_participants})",
                ))
            else:
                checks.append(Check("n_cases_valid", passed=True))

        if n_controls is not None:
            if n_controls < 0:
                checks.append(Check("n_controls_non_negative", passed=False, output=f"n_controls={n_controls}"))
                issues.append(Issue(
                    file="general", severity=Severity.BLOCKER,
                    description=f"n_controls is negative: {n_controls}",
                ))
            elif n_controls > self.total_participants:
                checks.append(Check("n_controls_within_ukb", passed=False, output=f"n_controls={n_controls} > {self.total_participants}"))
                issues.append(Issue(
                    file="general", severity=Severity.BLOCKER,
                    description=f"n_controls={n_controls} exceeds total participants ({self.total_participants})",
                ))
            else:
                checks.append(Check("n_controls_valid", passed=True))

        # Sum check
        if n_cases is not None and n_controls is not None:
            combined = n_cases + n_controls
            if combined > self.total_participants:
                checks.append(Check("combined_within_ukb", passed=False,
                                    output=f"{n_cases}+{n_controls}={combined} > {self.total_participants}"))
                issues.append(Issue(
                    file="general", severity=Severity.BLOCKER,
                    description=f"n_cases({n_cases}) + n_controls({n_controls}) = {combined} exceeds total participants ({self.total_participants})",
                ))
                corrections.append(MinimalCorrection(
                    claim="n_cases + n_controls",
                    current_value=combined,
                    constraint_violated=f"sum ≤ {self.total_participants}",
                    suggested_action=f"Reduce one or both counts. Maximum combined: {self.total_participants}",
                ))
            else:
                checks.append(Check("combined_within_ukb", passed=True))

        # Total check
        if total is not None:
            if total > self.total_participants:
                checks.append(Check("total_within_ukb", passed=False, output=f"total={total}"))
                issues.append(Issue(
                    file="general", severity=Severity.BLOCKER,
                    description=f"Reported total={total} exceeds UK Biobank ({self.total_participants})",
                ))
            else:
                checks.append(Check("total_within_ukb", passed=True))

        # n_excluded checks
        if n_excluded is not None:
            if n_excluded < 0:
                checks.append(Check("n_excluded_non_negative", passed=False, output=f"n_excluded={n_excluded}"))
                issues.append(Issue(
                    file="general", severity=Severity.BLOCKER,
                    description=f"n_excluded is negative: {n_excluded}",
                ))
            elif n_excluded > self.total_participants:
                checks.append(Check("n_excluded_within_ukb", passed=False, output=f"n_excluded={n_excluded} > {self.total_participants}"))
                issues.append(Issue(
                    file="general", severity=Severity.BLOCKER,
                    description=f"n_excluded={n_excluded} exceeds total participants ({self.total_participants})",
                ))
            else:
                checks.append(Check("n_excluded_valid", passed=True))

        # Cross-check: total + n_excluded ≤ total_participants
        if total is not None and n_excluded is not None:
            combined_with_excluded = total + n_excluded
            if combined_with_excluded > self.total_participants:
                checks.append(Check("total_plus_excluded_within_ukb", passed=False,
                                    output=f"total({total}) + n_excluded({n_excluded}) = {combined_with_excluded} > {self.total_participants}"))
                issues.append(Issue(
                    file="general", severity=Severity.BLOCKER,
                    description=(
                        f"total({total}) + n_excluded({n_excluded}) = {combined_with_excluded} "
                        f"exceeds total participants ({self.total_participants})"
                    ),
                ))
                corrections.append(MinimalCorrection(
                    claim="total + n_excluded",
                    current_value=combined_with_excluded,
                    constraint_violated=f"total + n_excluded ≤ {self.total_participants}",
                    suggested_action=f"Reduce total or n_excluded. Maximum combined: {self.total_participants}",
                ))
            else:
                checks.append(Check("total_plus_excluded_within_ukb", passed=True))

        # Cross-check: n_cases + n_controls + n_excluded ≤ total_participants
        if n_cases is not None and n_controls is not None and n_excluded is not None:
            full_sum = n_cases + n_controls + n_excluded
            if full_sum > self.total_participants:
                checks.append(Check("all_counts_within_ukb", passed=False,
                                    output=f"cases({n_cases}) + controls({n_controls}) + excluded({n_excluded}) = {full_sum} > {self.total_participants}"))
                issues.append(Issue(
                    file="general", severity=Severity.BLOCKER,
                    description=(
                        f"n_cases({n_cases}) + n_controls({n_controls}) + n_excluded({n_excluded}) = {full_sum} "
                        f"exceeds total participants ({self.total_participants})"
                    ),
                ))
            else:
                checks.append(Check("all_counts_within_ukb", passed=True))

        # Cross-check: any sum of provided counts ≤ total_participants
        # Catches cases where n_controls is missing but n_cases + n_excluded still exceeds
        provided_counts = []
        if n_cases is not None:
            provided_counts.append(("n_cases", n_cases))
        if n_controls is not None:
            provided_counts.append(("n_controls", n_controls))
        if n_excluded is not None:
            provided_counts.append(("n_excluded", n_excluded))
        if len(provided_counts) >= 2:
            sum_all = sum(v for _, v in provided_counts)
            if sum_all > self.total_participants:
                names = " + ".join(f"{name}({val})" for name, val in provided_counts)
                checks.append(Check("sum_all_provided_within_ukb", passed=False,
                                    output=f"{names} = {sum_all} > {self.total_participants}"))
                issues.append(Issue(
                    file="general", severity=Severity.BLOCKER,
                    description=(
                        f"Sum of all provided counts ({names} = {sum_all}) "
                        f"exceeds total participants ({self.total_participants})"
                    ),
                ))
            else:
                checks.append(Check("sum_all_provided_within_ukb", passed=True))

        status = "violated" if any(not c.passed for c in checks) else "verified"
        return VerificationResult(
            status=status,
            claims_checked=len(checks),
            checks=checks,
            issues=issues,
            corrections=corrections,
        )

    @staticmethod
    def _merge_cohort_results(
        bounds: VerificationResult,
        z3_result: VerificationResult,
    ) -> VerificationResult:
        """Combine descriptive bounds output with the formal Z3 proof.

        - checks/issues: union (bounds first → specific descriptions lead, the
          Z3 consistency verdict follows).
        - corrections: Z3 MCS first (it pinpoints the minimal relaxation),
          then the bounds-derived corrections.
        - status: violated if either path failed; undecidable only if Z3 could
          not decide and nothing else flagged a violation; otherwise verified.
        """
        checks = [*bounds.checks, *z3_result.checks]
        issues = [*bounds.issues, *z3_result.issues]
        corrections = [*z3_result.corrections, *bounds.corrections]

        if any(not c.passed for c in checks):
            status = "violated"
        elif z3_result.status == "undecidable":
            status = "undecidable"
        else:
            status = "verified"

        return VerificationResult(
            status=status,
            claims_checked=len(checks),
            checks=checks,
            issues=issues,
            corrections=corrections,
        )

    def _verify_cohort_z3(
        self,
        n_cases: Optional[int],
        n_controls: Optional[int],
        total: Optional[int],
        n_excluded: Optional[int],
    ) -> VerificationResult:
        """Z3-powered cohort verification with MCS extraction."""
        checks: list[Check] = []
        issues: list[Issue] = []
        corrections: list[MinimalCorrection] = []

        # Symbolic variables
        cases = Int("n_cases")
        controls = Int("n_controls")
        tot = Int("total")
        excluded = Int("n_excluded")
        ukb_total = IntVal(self.total_participants)

        # Add constraints
        constraints = []

        if n_cases is not None:
            constraints.append(("n_cases_value", cases == IntVal(n_cases)))
            constraints.append(("n_cases_non_negative", cases >= 0))
            constraints.append(("n_cases_within_ukb", cases <= ukb_total))
        if n_controls is not None:
            constraints.append(("n_controls_value", controls == IntVal(n_controls)))
            constraints.append(("n_controls_non_negative", controls >= 0))
            constraints.append(("n_controls_within_ukb", controls <= ukb_total))
        if total is not None:
            constraints.append(("total_value", tot == IntVal(total)))
            constraints.append(("total_non_negative", tot >= 0))
        if n_excluded is not None:
            constraints.append(("excluded_value", excluded == IntVal(n_excluded)))
            constraints.append(("excluded_non_negative", excluded >= 0))

        # Hard UKB constraints
        if n_cases is not None and n_controls is not None:
            constraints.append(("combined_within_ukb", cases + controls <= ukb_total))
        if total is not None:
            constraints.append(("total_within_ukb", tot <= ukb_total))
        if n_cases is not None and n_controls is not None and total is not None:
            constraints.append(("sum_equals_total", cases + controls <= tot))
        if n_excluded is not None:
            constraints.append(("excluded_within_ukb", excluded <= ukb_total))
        if total is not None and n_excluded is not None:
            constraints.append(("excluded_plus_total", tot + excluded <= ukb_total))
        if n_cases is not None and n_excluded is not None:
            constraints.append(("cases_plus_excluded_within_ukb", cases + excluded <= ukb_total))
        if n_controls is not None and n_excluded is not None:
            constraints.append(("controls_plus_excluded_within_ukb", controls + excluded <= ukb_total))
        if n_cases is not None and n_controls is not None and n_excluded is not None:
            constraints.append(("all_counts_within_ukb", cases + controls + excluded <= ukb_total))

        # Check satisfiability of all constraints together
        s_full = Solver()
        for name, constraint in constraints:
            s_full.add(constraint)

        result = s_full.check()

        if result == sat:
            checks.append(Check("z3_cohort_consistency", passed=True,
                                output="All cohort constraints satisfiable"))
            status = "verified"
        elif result == unsat:
            checks.append(Check("z3_cohort_consistency", passed=False,
                                output="Cohort claims are logically contradictory"))
            issues.append(Issue(
                file="general", severity=Severity.BLOCKER,
                description="Cohort size claims are mutually contradictory (Z3 proved UNSAT)",
            ))
            # Extract MCS by removing constraints one at a time
            corrections = self._extract_mcs_cohort(constraints, n_cases, n_controls, total, n_excluded)
            status = "violated"
        else:
            checks.append(Check("z3_cohort_consistency", passed=True,
                                output="Z3 returned unknown (treating as pass)"))
            status = "undecidable"

        return VerificationResult(
            status=status,
            claims_checked=len(constraints),
            checks=checks,
            issues=issues,
            corrections=corrections,
        )

    def _extract_mcs_cohort(
        self,
        constraints: list[tuple[str, Any]],
        n_cases: Optional[int],
        n_controls: Optional[int],
        total: Optional[int],
        n_excluded: Optional[int],
    ) -> list[MinimalCorrection]:
        """Extract Minimal Correction Subsets using greedy O(m×SAT) approach."""
        corrections = []
        # Try removing each constraint to find which ones restore satisfiability
        for i, (name, _) in enumerate(constraints):
            s = Solver()
            for j, (_, c) in enumerate(constraints):
                if i != j:
                    s.add(c)
            if s.check() == sat:
                corrections.append(MinimalCorrection(
                    claim=name,
                    current_value=self._get_claim_value(name, n_cases, n_controls, total, n_excluded),
                    constraint_violated=name,
                    suggested_action=f"Relaxing constraint '{name}' restores consistency",
                ))
        return corrections

    @staticmethod
    def _get_claim_value(name: str, n_cases, n_controls, total, n_excluded) -> Any:
        mapping = {
            "n_cases_value": n_cases,
            "n_controls_value": n_controls,
            "total_value": total,
            "excluded_value": n_excluded,
        }
        return mapping.get(name, "constraint")

    # ── Metric verification ────────────────────────────────────

    def verify_metric_bounds(
        self,
        metric_name: str,
        value: float,
        ci_low: Optional[float] = None,
        ci_high: Optional[float] = None,
    ) -> VerificationResult:
        """Verify a reported metric is within valid bounds.

        Checks:
          - Value within known range for metric type
          - CI ordering: ci_low ≤ value ≤ ci_high
          - CI within valid range
        """
        checks: list[Check] = []
        issues: list[Issue] = []
        corrections: list[MinimalCorrection] = []

        # Normalize metric name
        metric_key = metric_name.lower().replace(" ", "_").replace("-", "_")

        # Range check
        if metric_key in self.metric_ranges:
            lo, hi = self.metric_ranges[metric_key]
            if value < lo or (hi != float("inf") and value > hi):
                checks.append(Check(
                    f"{metric_key}_in_range", passed=False,
                    output=f"{metric_name}={value} outside [{lo}, {hi}]",
                ))
                issues.append(Issue(
                    file="general", severity=Severity.BLOCKER,
                    description=f"{metric_name}={value} is outside valid range [{lo}, {hi}]",
                ))
                corrections.append(MinimalCorrection(
                    claim=metric_name, current_value=value,
                    constraint_violated=f"{metric_name} ∈ [{lo}, {hi}]",
                    suggested_action=f"Value must be between {lo} and {hi}",
                ))
            else:
                checks.append(Check(f"{metric_key}_in_range", passed=True))
        else:
            checks.append(Check(f"{metric_key}_in_range", passed=True,
                                output=f"No known range for '{metric_key}' — skipping"))

        # CI ordering check
        if ci_low is not None and ci_high is not None:
            if ci_low > ci_high:
                checks.append(Check("ci_ordering", passed=False,
                                    output=f"CI: [{ci_low}, {ci_high}] — lower > upper"))
                issues.append(Issue(
                    file="general", severity=Severity.BLOCKER,
                    description=f"Confidence interval inverted: lower={ci_low} > upper={ci_high}",
                ))
            elif ci_low > value or value > ci_high:
                checks.append(Check("ci_contains_point", passed=False,
                                    output=f"Point estimate {value} outside CI [{ci_low}, {ci_high}]"))
                issues.append(Issue(
                    file="general", severity=Severity.WARNING,
                    description=f"Point estimate {value} outside CI [{ci_low}, {ci_high}]",
                ))
            else:
                checks.append(Check("ci_ordering", passed=True))
                checks.append(Check("ci_contains_point", passed=True))

            # CI bounds in valid range
            if metric_key in self.metric_ranges:
                lo, hi = self.metric_ranges[metric_key]
                if ci_low < lo or (hi != float("inf") and ci_high > hi):
                    checks.append(Check("ci_within_metric_range", passed=False,
                                        output=f"CI [{ci_low}, {ci_high}] extends outside [{lo}, {hi}]"))
                    issues.append(Issue(
                        file="general", severity=Severity.WARNING,
                        description=f"CI [{ci_low}, {ci_high}] extends beyond valid range [{lo}, {hi}]",
                    ))
                else:
                    checks.append(Check("ci_within_metric_range", passed=True))

        status = "violated" if any(not c.passed for c in checks) else "verified"
        return VerificationResult(
            status=status,
            claims_checked=len(checks),
            checks=checks,
            issues=issues,
            corrections=corrections,
        )

    # ── Filter consistency ─────────────────────────────────────

    def verify_filter_consistency(
        self,
        inclusions: list[str],
        exclusions: list[str],
    ) -> VerificationResult:
        """Check inclusion/exclusion criteria don't contradict each other.

        Detects: same condition in both lists (direct contradiction).
        """
        checks: list[Check] = []
        issues: list[Issue] = []

        # Normalize and find overlaps
        incl_normalized = {s.strip().lower() for s in inclusions}
        excl_normalized = {s.strip().lower() for s in exclusions}
        overlap = incl_normalized & excl_normalized

        if overlap:
            checks.append(Check("no_contradictory_filters", passed=False,
                                output=f"Found in both inclusion and exclusion: {overlap}"))
            for item in overlap:
                issues.append(Issue(
                    file="general", severity=Severity.BLOCKER,
                    description=f"Contradictory filter: '{item}' appears in both inclusion and exclusion criteria",
                ))
        else:
            checks.append(Check("no_contradictory_filters", passed=True))

        status = "violated" if overlap else "verified"
        return VerificationResult(
            status=status,
            claims_checked=1,
            checks=checks,
            issues=issues,
        )

    # ── Batch verification ─────────────────────────────────────

    def verify_skill_output(self, skill_name: str, result: dict) -> VerificationResult:
        """Verify a complete skill output dict — extracts claims and verifies all.

        This is the main entry point for integration with verdict.py.
        """
        all_checks: list[Check] = []
        all_issues: list[Issue] = []
        all_corrections: list[MinimalCorrection] = []

        # Extract cohort claims
        cohort_keys = {"n_cases", "n_controls", "total", "n_total", "n_participants", "cohort_size", "n_excluded", "excluded"}
        cohort_vals = {k: v for k, v in result.items() if k in cohort_keys and isinstance(v, (int, float))}
        if cohort_vals:
            cr = self.verify_cohort_claims(
                n_cases=cohort_vals.get("n_cases"),
                n_controls=cohort_vals.get("n_controls"),
                total=cohort_vals.get("total") or cohort_vals.get("n_total") or cohort_vals.get("n_participants") or cohort_vals.get("cohort_size"),
                n_excluded=cohort_vals.get("n_excluded") or cohort_vals.get("excluded"),
            )
            all_checks.extend(cr.checks)
            all_issues.extend(cr.issues)
            all_corrections.extend(cr.corrections)

        # Extract metric claims
        for key, val in result.items():
            if not isinstance(val, (int, float)):
                continue
            metric_key = key.lower().replace(" ", "_").replace("-", "_")
            if metric_key in self.metric_ranges:
                # Look for associated CI fields
                ci_low = result.get(f"{key}_lower") or result.get(f"{key}_ci_low") or result.get(f"{key}_lo")
                ci_high = result.get(f"{key}_upper") or result.get(f"{key}_ci_high") or result.get(f"{key}_hi")
                mr = self.verify_metric_bounds(
                    metric_name=key,
                    value=float(val),
                    ci_low=float(ci_low) if ci_low is not None else None,
                    ci_high=float(ci_high) if ci_high is not None else None,
                )
                all_checks.extend(mr.checks)
                all_issues.extend(mr.issues)
                all_corrections.extend(mr.corrections)

        status = "violated" if any(not c.passed for c in all_checks) else "verified"
        return VerificationResult(
            status=status,
            claims_checked=len(all_checks),
            checks=all_checks,
            issues=all_issues,
            corrections=all_corrections,
        )
