"""Tests for formal constraint verification (Stream C).

Tests both the Z3-backed path (when installed) and the fallback bounds checking.
"""

import importlib
import sys
from types import SimpleNamespace

import pytest
from biobank_agent import verification as ver_mod
from biobank_agent.verification import (
    BiobankConstraintVerifier,
    ClaimType,
    METRIC_RANGES,
    UKBiobankConstraints,
    _Z3_AVAILABLE,
)
from biobank_agent.verdict import VerdictStatus


def test_verification_import_marks_z3_available_with_fake_module():
    sentinel = object()
    original_z3 = sys.modules.get("z3", sentinel)
    fake_z3 = SimpleNamespace(
        Solver=object,
        Int=lambda name: name,
        Real=lambda name: name,
        And=lambda *args: args,
        Or=lambda *args: args,
        Not=lambda arg: arg,
        Implies=lambda *args: args,
        sat="sat",
        unsat="unsat",
        unknown="unknown",
        IntVal=lambda value: value,
        RealVal=lambda value: value,
    )

    sys.modules["z3"] = fake_z3
    try:
        reloaded = importlib.reload(ver_mod)
        assert reloaded._Z3_AVAILABLE is True
    finally:
        if original_z3 is sentinel:
            sys.modules.pop("z3", None)
        else:
            sys.modules["z3"] = original_z3
        importlib.reload(ver_mod)


@pytest.fixture
def verifier():
    return BiobankConstraintVerifier()


# ── Cohort verification tests ──────────────────────────────────


class TestCohortVerification:
    """Test cohort size constraint verification."""

    def test_valid_cohort(self, verifier):
        """Normal cohort sizes should pass."""
        result = verifier.verify_cohort_claims(n_cases=5000, n_controls=20000)
        assert result.status == "verified"
        assert all(c.passed for c in result.checks)

    def test_impossible_cohort_exceeds_total(self, verifier):
        """n_cases + n_controls > 502,411 should fail."""
        result = verifier.verify_cohort_claims(n_cases=300_000, n_controls=300_000)
        assert result.status == "violated"
        assert any(not c.passed for c in result.checks)
        assert any("exceeds" in i.description for i in result.issues)

    def test_negative_cases(self, verifier):
        """Negative case count should fail."""
        result = verifier.verify_cohort_claims(n_cases=-100)
        assert result.status == "violated"
        assert any("negative" in i.description for i in result.issues)

    def test_single_count_exceeds_total(self, verifier):
        """Single count > total participants should fail."""
        result = verifier.verify_cohort_claims(n_cases=600_000)
        assert result.status == "violated"

    def test_valid_with_total(self, verifier):
        """Valid with explicit total."""
        result = verifier.verify_cohort_claims(n_cases=1000, n_controls=4000, total=5000)
        assert result.status == "verified"

    def test_total_exceeds_ukb(self, verifier):
        """Reported total > UKB should fail."""
        result = verifier.verify_cohort_claims(total=600_000)
        assert result.status == "violated"

    def test_none_inputs_pass(self, verifier):
        """All None inputs should pass (nothing to verify)."""
        result = verifier.verify_cohort_claims()
        assert result.status == "verified"
        assert result.claims_checked == 0

    def test_controls_only_cohort_claim(self, verifier):
        """Controls-only claims skip case-specific constraints."""
        result = verifier.verify_cohort_claims(n_controls=1000)

        assert result.status == "verified"
        if ver_mod._Z3_AVAILABLE:
            direct = verifier._verify_cohort_z3(None, 1000, None, None)
            assert direct.status == "verified"

    def test_mcs_suggestions_on_violation(self, verifier):
        """Violated cohort should provide correction suggestions."""
        result = verifier.verify_cohort_claims(n_cases=300_000, n_controls=300_000)
        assert result.status == "violated"
        assert len(result.corrections) > 0
        assert "Reduce" in result.corrections[0].suggested_action or "relax" in result.corrections[0].suggested_action.lower()

    def test_excluded_plus_total_exceeds_ukb(self, verifier):
        """total + n_excluded > 502,411 should fail."""
        result = verifier.verify_cohort_claims(total=400_000, n_excluded=200_000)
        assert result.status == "violated"
        assert any("n_excluded" in i.description for i in result.issues)

    def test_excluded_alone_exceeds_ukb(self, verifier):
        """n_excluded > 502,411 should fail."""
        result = verifier.verify_cohort_claims(n_excluded=600_000)
        assert result.status == "violated"

    def test_all_counts_exceed_ukb(self, verifier):
        """n_cases + n_controls + n_excluded > 502,411 should fail."""
        result = verifier.verify_cohort_claims(n_cases=5_000, n_controls=5_000, n_excluded=500_000)
        assert result.status == "violated"

    def test_valid_excluded(self, verifier):
        """Valid n_excluded within bounds should pass."""
        result = verifier.verify_cohort_claims(total=300_000, n_excluded=100_000)
        assert result.status == "verified"


# ── Metric verification tests ──────────────────────────────────


class TestMetricVerification:
    """Test metric bounds verification."""

    def test_valid_auc(self, verifier):
        """AUC within [0,1] should pass."""
        result = verifier.verify_metric_bounds("auc", 0.78, ci_low=0.75, ci_high=0.81)
        assert result.status == "verified"

    def test_impossible_auc_above_1(self, verifier):
        """AUC > 1.0 should fail."""
        result = verifier.verify_metric_bounds("auc", 1.5)
        assert result.status == "violated"
        assert any("outside valid range" in i.description for i in result.issues)

    def test_negative_auc(self, verifier):
        """AUC < 0 should fail."""
        result = verifier.verify_metric_bounds("auc", -0.1)
        assert result.status == "violated"

    def test_inverted_ci(self, verifier):
        """CI where lower > upper should fail."""
        result = verifier.verify_metric_bounds("auc", 0.80, ci_low=0.85, ci_high=0.75)
        assert result.status == "violated"
        assert any("inverted" in i.description.lower() for i in result.issues)

    def test_point_outside_ci(self, verifier):
        """Point estimate outside CI should warn."""
        result = verifier.verify_metric_bounds("auc", 0.90, ci_low=0.70, ci_high=0.80)
        assert result.status == "violated"
        assert any("outside CI" in i.description for i in result.issues)

    def test_valid_odds_ratio(self, verifier):
        """Positive OR should pass."""
        result = verifier.verify_metric_bounds("odds_ratio", 2.5)
        assert result.status == "verified"

    def test_negative_odds_ratio(self, verifier):
        """Negative OR should fail."""
        result = verifier.verify_metric_bounds("odds_ratio", -0.5)
        assert result.status == "violated"

    def test_unknown_metric_passes(self, verifier):
        """Unknown metric names should pass (no range to check)."""
        result = verifier.verify_metric_bounds("some_custom_metric", 999.0)
        assert result.status == "verified"

    def test_p_value_above_1(self, verifier):
        """p-value > 1 should fail."""
        result = verifier.verify_metric_bounds("p_value", 1.5)
        assert result.status == "violated"


# ── Filter consistency tests ───────────────────────────────────


class TestFilterConsistency:
    """Test filter logic verification."""

    def test_no_contradiction(self, verifier):
        """Non-overlapping filters should pass."""
        result = verifier.verify_filter_consistency(
            inclusions=["diabetes", "age > 50"],
            exclusions=["cancer", "pregnant"],
        )
        assert result.status == "verified"

    def test_contradictory_filter(self, verifier):
        """Same condition in both should fail."""
        result = verifier.verify_filter_consistency(
            inclusions=["diabetes", "hypertension"],
            exclusions=["diabetes", "pregnant"],
        )
        assert result.status == "violated"
        assert any("diabetes" in i.description.lower() for i in result.issues)


# ── Batch verification tests ──────────────────────────────────


class TestBatchVerification:
    """Test verify_skill_output batch processing."""

    def test_valid_result(self, verifier):
        """Normal result dict should pass."""
        result = verifier.verify_skill_output("train_model", {
            "auc": 0.82,
            "n_cases": 5000,
            "n_controls": 20000,
            "precision": 0.75,
        })
        assert result.status == "verified"

    def test_impossible_result(self, verifier):
        """Result with impossible values should fail."""
        result = verifier.verify_skill_output("train_model", {
            "auc": 1.5,
            "n_cases": 600_000,
        })
        assert result.status == "violated"
        assert len(result.issues) >= 2  # Both AUC and n_cases fail

    def test_empty_result(self, verifier):
        """Empty dict has nothing to verify."""
        result = verifier.verify_skill_output("some_skill", {})
        assert result.status == "verified"

    def test_n_excluded_caught_in_skill_output(self, verifier):
        """verify_skill_output must pass n_excluded through to verify_cohort_claims."""
        result = verifier.verify_skill_output("cohort_summary", {
            "n_cases": 5000,
            "n_controls": 5000,
            "n_excluded": 500_000,
        })
        assert result.status == "violated"
        assert any("n_excluded" in i.description or "excluded" in i.description for i in result.issues)

    def test_total_plus_excluded_caught_in_skill_output(self, verifier):
        """verify_skill_output must catch total + n_excluded > UKB."""
        result = verifier.verify_skill_output("cohort_summary", {
            "total": 400_000,
            "n_excluded": 200_000,
        })
        assert result.status == "violated"

    def test_verdict_conversion(self, verifier):
        """VerificationResult should convert to VerdictResult."""
        result = verifier.verify_cohort_claims(n_cases=600_000)
        verdict = result.to_verdict(rationale="Z3 verification")
        assert verdict.status == VerdictStatus.FAIL
        assert verdict.rationale == "Z3 verification"


# ── Z3-specific tests ─────────────────────────────────────────


@pytest.mark.skipif(not _Z3_AVAILABLE, reason="z3-solver not installed")
class TestZ3Verification:
    """Tests that only run when Z3 is available."""

    def test_z3_cohort_valid(self):
        v = BiobankConstraintVerifier()
        result = v.verify_cohort_claims(n_cases=1000, n_controls=5000)
        assert result.status == "verified"

    def test_z3_cohort_impossible(self):
        v = BiobankConstraintVerifier()
        result = v.verify_cohort_claims(n_cases=300_000, n_controls=300_000)
        assert result.status == "violated"
        # Z3 should provide MCS corrections
        assert len(result.corrections) > 0


class TestGracefulWithoutZ3:
    """Verify the module works without Z3 installed."""

    def test_fallback_still_catches_impossible(self, verifier):
        """Even without Z3, basic bounds checking works."""
        result = verifier.verify_cohort_claims(n_cases=600_000)
        assert result.status == "violated"

    def test_z3_available_property(self, verifier):
        """z3_available property should be boolean."""
        assert isinstance(verifier.z3_available, bool)


class TestFallbackCoverage:
    """Force fallback bounds checking even when Z3 is installed."""

    def test_fallback_negative_controls_excluded_and_sum_paths(self, monkeypatch):
        monkeypatch.setattr(ver_mod, "_Z3_AVAILABLE", False)
        v = BiobankConstraintVerifier(total_participants=100)

        result = v.verify_cohort_claims(
            n_cases=60,
            n_controls=-1,
            total=80,
            n_excluded=50,
        )

        assert result.status == "violated"
        descriptions = "\n".join(issue.description for issue in result.issues)
        assert "n_controls is negative" in descriptions
        assert "total(80) + n_excluded(50)" in descriptions
        assert "Sum of all provided counts" in descriptions

    def test_fallback_controls_and_excluded_exceed_total(self, monkeypatch):
        monkeypatch.setattr(ver_mod, "_Z3_AVAILABLE", False)
        v = BiobankConstraintVerifier(total_participants=100)

        controls = v.verify_cohort_claims(n_controls=120)
        excluded = v.verify_cohort_claims(n_excluded=-2)
        all_counts_ok = v.verify_cohort_claims(n_cases=20, n_controls=30, n_excluded=40)

        assert controls.status == "violated"
        assert any("n_controls=120 exceeds" in issue.description for issue in controls.issues)
        assert excluded.status == "violated"
        assert any("n_excluded is negative" in issue.description for issue in excluded.issues)
        assert all_counts_ok.status == "verified"
        assert any(check.name == "all_counts_within_ukb" for check in all_counts_ok.checks)

    def test_metric_ci_extends_outside_range_and_unknown_metric_ci(self):
        v = BiobankConstraintVerifier(custom_constraints={"custom_metric": (10.0, 20.0)})

        outside = v.verify_metric_bounds("auc", 0.5, ci_low=-0.1, ci_high=0.9)
        custom = v.verify_metric_bounds("custom metric", 15.0, ci_low=10.0, ci_high=20.0)
        unknown = v.verify_metric_bounds("unbounded score", 999.0, ci_low=1.0, ci_high=2.0)

        assert outside.status == "violated"
        assert any("extends beyond valid range" in issue.description for issue in outside.issues)
        assert custom.status == "verified"
        assert unknown.status == "violated"
        assert any("outside CI" in issue.description for issue in unknown.issues)

    def test_verify_skill_output_skips_non_numeric_and_collects_ci(self):
        v = BiobankConstraintVerifier()

        result = v.verify_skill_output("model", {
            "auc": 0.5,
            "auc_ci_low": 0.6,
            "auc_ci_high": 0.7,
            "note": "not numeric",
        })

        assert result.status == "violated"
        assert any(check.name == "ci_contains_point" and not check.passed for check in result.checks)


class TestFakeZ3Path:
    """Exercise Z3-specific branches with a lightweight fake solver."""

    class Expr:
        def __init__(self, name):
            self.name = name

        def __eq__(self, other):
            return ("eq", self.name, other)

        def __ge__(self, other):
            return ("ge", self.name, other)

        def __le__(self, other):
            return ("le", self.name, other)

        def __add__(self, other):
            return TestFakeZ3Path.Expr(f"({self.name}+{getattr(other, 'name', other)})")

    def _install_fake_z3(self, monkeypatch, checks):
        checks = list(checks)

        class FakeSolver:
            def __init__(self):
                self.constraints = []

            def add(self, constraint):
                self.constraints.append(constraint)

            def check(self):
                return checks.pop(0) if checks else "sat"

        monkeypatch.setattr(ver_mod, "_Z3_AVAILABLE", True)
        monkeypatch.setattr(ver_mod, "Solver", FakeSolver, raising=False)
        monkeypatch.setattr(ver_mod, "Int", lambda name: self.Expr(name), raising=False)
        monkeypatch.setattr(ver_mod, "IntVal", lambda value: value, raising=False)
        monkeypatch.setattr(ver_mod, "sat", "sat", raising=False)
        monkeypatch.setattr(ver_mod, "unsat", "unsat", raising=False)

    def test_fake_z3_sat_unknown_and_claim_value(self, monkeypatch):
        self._install_fake_z3(monkeypatch, ["sat", "unknown"])
        v = BiobankConstraintVerifier(total_participants=100)

        sat_result = v.verify_cohort_claims(n_cases=10, n_controls=20)
        unknown_result = v._verify_cohort_z3(n_cases=10, n_controls=None, total=None, n_excluded=None)
        controls_only = v._verify_cohort_z3(n_cases=None, n_controls=20, total=None, n_excluded=None)

        assert sat_result.status == "verified"
        assert unknown_result.status == "undecidable"
        assert controls_only.status == "verified"
        assert v._get_claim_value("n_cases_value", 1, 2, 3, 4) == 1
        assert v._get_claim_value("other", 1, 2, 3, 4) == "constraint"

    def test_fake_z3_unsat_extracts_mcs(self, monkeypatch):
        # First check is the full solver. Subsequent checks are per-constraint
        # MCS attempts, with the first removal restoring satisfiability.
        self._install_fake_z3(monkeypatch, ["unsat", "sat", "unsat", "unsat", "unsat", "unsat"])
        v = BiobankConstraintVerifier(total_participants=100)

        result = v.verify_cohort_claims(n_cases=80, n_controls=80, total=120, n_excluded=10)

        assert result.status == "violated"
        assert any("mutually contradictory" in issue.description for issue in result.issues)
        assert result.corrections
        assert result.corrections[0].suggested_action.startswith("Relaxing constraint")
