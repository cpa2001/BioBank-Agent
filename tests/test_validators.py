"""Tests for domain-aware functional validators (Stream C).

Tests ModelValidator and LeakageDetector for biological plausibility checks.
"""

import pytest
from biobank_agent.validators import (
    ModelValidator,
    LeakageDetector,
    ValidationCheck,
    EXPECTED_DIRECTIONS,
    LEAKAGE_SUSPECT_FEATURES,
)


@pytest.fixture
def validator():
    return ModelValidator()


@pytest.fixture
def leakage_detector():
    return LeakageDetector()


def test_validation_check_string_formats_pass_and_fail_icons():
    passed = ValidationCheck("ok", True, "all clear", severity="info")
    failed = ValidationCheck("bad", False, "needs review", severity="error")

    assert str(passed) == "[INFO] ✓ ok: all clear"
    assert str(failed) == "[ERROR] ✗ bad: needs review"


# ── AUC Plausibility Tests ─────────────────────────────────────


class TestAUCPlausibility:
    """Test data leakage detection via AUC plausibility."""

    def test_reasonable_auc_passes(self, validator):
        """Normal AUC should pass."""
        checks = validator.validate_auc_plausibility("E11", auc=0.82, n_features=50)
        assert all(c.passed for c in checks)

    def test_perfect_auc_fails(self, validator):
        """AUC = 1.0 always indicates leakage."""
        checks = validator.validate_auc_plausibility("E11", auc=1.0, n_features=100)
        assert any(not c.passed for c in checks)
        assert any("leakage" in c.message.lower() or "contamination" in c.message.lower() for c in checks)

    def test_extreme_auc_few_features(self, validator):
        """AUC > 0.98 with < 50 features → leakage."""
        checks = validator.validate_auc_plausibility("I21", auc=0.99, n_features=10)
        assert any(not c.passed for c in checks)
        assert any(c.severity == "error" for c in checks)

    def test_high_auc_very_few_features(self, validator):
        """AUC > 0.95 with < 20 features → likely leakage."""
        checks = validator.validate_auc_plausibility("E11", auc=0.96, n_features=5)
        assert any(not c.passed for c in checks)

    def test_disease_specific_ceiling(self, validator):
        """AUC above disease-specific ceiling should warn."""
        # Depression max is 0.75
        checks = validator.validate_auc_plausibility("F32", auc=0.85, n_features=100)
        assert any(not c.passed for c in checks)
        assert any("exceeds maximum plausible" in c.message for c in checks)

    def test_small_sample_high_auc(self, validator):
        """High AUC with tiny sample → overfitting."""
        checks = validator.validate_auc_plausibility("E11", auc=0.92, n_features=50, n_samples=100)
        assert any(not c.passed for c in checks)
        assert any("overfitting" in c.message.lower() for c in checks)


# ── Feature Importance Tests ───────────────────────────────────


class TestFeatureImportance:
    """Test biological plausibility of feature rankings."""

    def test_expected_biomarkers_present(self, validator):
        """Top features matching expected biomarkers should pass."""
        checks = validator.validate_feature_importance("E11", [
            ("hba1c", 0.35),
            ("glucose", 0.20),
            ("bmi", 0.15),
            ("age", 0.10),
            ("triglyceride", 0.08),
        ])
        assert any(c.passed and "expected biomarkers" in c.message.lower() for c in checks)

    def test_leakage_suspect_feature_detected(self, validator):
        """Leakage-suspect features should be flagged."""
        checks = validator.validate_feature_importance("E11", [
            ("date_of_death", 0.50),
            ("hba1c", 0.20),
            ("glucose", 0.15),
        ])
        assert any(not c.passed and "leakage" in c.message.lower() for c in checks)

    def test_no_expected_biomarkers(self, validator):
        """Missing all expected biomarkers should warn."""
        checks = validator.validate_feature_importance("E11", [
            ("shoe_size", 0.35),
            ("hair_color", 0.20),
            ("pet_ownership", 0.15),
            ("music_preference", 0.10),
            ("handedness", 0.08),
        ])
        assert any(not c.passed and "expected biomarkers" in c.message.lower() for c in checks)

    def test_unknown_icd_and_short_feature_list_skip_expected_biomarker_checks(self, validator):
        unknown = validator.validate_feature_importance("Z99", [("shoe_size", 0.3)])
        short = validator.validate_feature_importance("E11", [("shoe_size", 0.3)])

        assert unknown == []
        assert short == []


# ── Direction Consistency Tests ────────────────────────────────


class TestDirectionConsistency:
    """Test biomarker direction validation."""

    def test_correct_direction(self, validator):
        """HbA1c↑ for T2D should pass."""
        check = validator.validate_direction_consistency("E11", "hba1c", "+")
        assert check.passed

    def test_wrong_direction(self, validator):
        """HDL↑ for MI should fail (HDL is protective)."""
        check = validator.validate_direction_consistency("I21", "hdl", "+")
        assert not check.passed
        assert "direction" in check.message.lower()

    def test_unknown_disease(self, validator):
        """Unknown ICD-10 should pass (no reference)."""
        check = validator.validate_direction_consistency("Z99", "anything", "+")
        assert check.passed

    def test_unknown_feature(self, validator):
        """Unknown feature for known disease should pass."""
        check = validator.validate_direction_consistency("E11", "exotic_biomarker", "+")
        assert check.passed


# ── Leakage Detector Tests ─────────────────────────────────────


class TestLeakageDetector:
    """Test data leakage detection."""

    def test_temporal_leakage_detected(self, leakage_detector):
        """Features with temporal info should be flagged."""
        checks = leakage_detector.check_temporal_leakage(
            features=["age", "bmi", "date_of_death", "glucose"]
        )
        assert any(not c.passed and "temporal" in c.message.lower() for c in checks)

    def test_no_temporal_leakage(self, leakage_detector):
        """Normal features should pass."""
        checks = leakage_detector.check_temporal_leakage(
            features=["age", "bmi", "glucose", "cholesterol"]
        )
        assert all(c.passed for c in checks)

    def test_target_leakage_detected(self, leakage_detector):
        """Features derived from outcome should be flagged."""
        checks = leakage_detector.check_target_leakage(
            features=["age", "diabetes_diagnosed", "bmi"],
            outcome_icd10="E11",
        )
        assert any(not c.passed for c in checks)

    def test_icd_code_in_features(self, leakage_detector):
        """Feature containing outcome ICD code should flag."""
        checks = leakage_detector.check_target_leakage(
            features=["age", "e11_status", "bmi"],
            outcome_icd10="E11",
        )
        assert any(not c.passed for c in checks)

    def test_unknown_outcome_without_code_in_features_has_no_target_leakage(self, leakage_detector):
        checks = leakage_detector.check_target_leakage(
            features=["age", "bmi"],
            outcome_icd10="Z99",
        )

        assert len(checks) == 1
        assert checks[0].passed is True

    def test_full_scan(self, leakage_detector):
        """Full scan catches multiple leakage types."""
        checks = leakage_detector.full_leakage_scan(
            features=["eid", "date_of_death", "age", "bmi"],
            outcome_icd10="E11",
        )
        failed = [c for c in checks if not c.passed]
        assert len(failed) >= 2  # ID + temporal

    def test_clean_features_pass(self, leakage_detector):
        """Clean feature set should pass all checks."""
        checks = leakage_detector.full_leakage_scan(
            features=["age", "bmi", "glucose", "cholesterol", "systolic_bp"],
            outcome_icd10="I21",
        )
        assert all(c.passed for c in checks)
