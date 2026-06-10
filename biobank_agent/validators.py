"""Domain-aware functional validators for biobank ML pipelines.

Each validator answers: "Is this result biologically plausible?"
Not "is it statistically correct?" (that's verification.py's job).

Inspired by Agentomics-ML (arXiv:2506.05542) functional validator pattern.
Each ML step has an associated validator that checks domain knowledge
BEFORE presenting results to users.

Usage
-----
    from biobank_agent.validators import ModelValidator, LeakageDetector

    validator = ModelValidator()
    checks = validator.validate_auc_plausibility("E11", auc=0.99, n_features=5)
    # → [ValidationCheck(name="leakage_suspect", passed=False, ...)]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationCheck:
    """A single domain-aware validation check result."""
    name: str
    passed: bool
    message: str
    severity: str = "warning"  # "info" | "warning" | "error"
    evidence: str = ""         # supporting evidence for the check

    def __str__(self) -> str:
        icon = "✓" if self.passed else "✗"
        return f"[{self.severity.upper()}] {icon} {self.name}: {self.message}"


# ── Known biomarker–disease associations ───────────────────────
# ICD-10 code → {biomarker_keyword: expected_direction}
# "+" = higher value → higher risk; "-" = higher value → lower risk

EXPECTED_DIRECTIONS: dict[str, dict[str, str]] = {
    # Diabetes mellitus type 2
    "E11": {
        "hba1c": "+", "glucose": "+", "bmi": "+", "waist": "+",
        "triglyceride": "+", "hdl": "-", "insulin": "+",
        "c_reactive_protein": "+", "crp": "+",
    },
    # Myocardial infarction
    "I21": {
        "ldl": "+", "hdl": "-", "cholesterol": "+",
        "systolic": "+", "diastolic": "+", "smoking": "+",
        "bmi": "+", "lipoprotein_a": "+", "apolipoprotein_b": "+",
    },
    # Ischaemic heart disease
    "I25": {
        "ldl": "+", "hdl": "-", "cholesterol": "+",
        "systolic": "+", "bmi": "+", "crp": "+",
    },
    # Breast cancer
    "C50": {
        "age": "+", "bmi": "+", "estradiol": "+",
        "igf1": "+", "alcohol": "+",
    },
    # Stroke
    "I63": {
        "systolic": "+", "ldl": "+", "bmi": "+",
        "atrial_fibrillation": "+", "smoking": "+",
    },
    # Liver disease
    "K70": {
        "alt": "+", "ast": "+", "ggt": "+", "bilirubin": "+",
        "albumin": "-", "alcohol": "+",
    },
    # Chronic kidney disease
    "N18": {
        "creatinine": "+", "cystatin_c": "+", "egfr": "-",
        "urea": "+", "phosphate": "+", "potassium": "+",
    },
    # COPD
    "J44": {
        "fev1": "-", "smoking": "+", "pack_years": "+",
        "age": "+", "bmi": "-",
    },
    # Depression
    "F32": {
        "sleep_duration": "-", "physical_activity": "-",
        "neuroticism": "+", "loneliness": "+",
    },
    # Alzheimer's
    "G30": {
        "age": "+", "apoe4": "+", "education": "-",
        "physical_activity": "-",
    },
}

# Biomarkers that are never legitimate top predictors for most diseases
# (likely data leakage if they appear as #1 feature)
LEAKAGE_SUSPECT_FEATURES: set[str] = {
    "date_of_death", "death_date", "cause_of_death",
    "date_of_diagnosis", "diagnosis_date",
    "hospital_admission_date", "date_lost_to_followup",
    "eid",  # participant ID
}

# Maximum plausible AUC by disease complexity
# (empirical: simple metabolic diseases can reach 0.85; complex diseases rarely exceed 0.75)
MAX_PLAUSIBLE_AUC: dict[str, float] = {
    "E11": 0.90,  # T2D — well-predicted by HbA1c/glucose
    "I21": 0.85,  # MI
    "C50": 0.80,  # Breast cancer
    "G30": 0.80,  # Alzheimer's
    "F32": 0.75,  # Depression — inherently hard
}
DEFAULT_MAX_PLAUSIBLE_AUC = 0.92


class ModelValidator:
    """Validates ML model outputs against biomedical domain knowledge.

    Catches:
      - Biologically implausible feature rankings
      - Suspiciously high AUC (data leakage indicator)
      - Direction inconsistency (protective factor showing as risk)
    """

    def __init__(
        self,
        expected_directions: Optional[dict] = None,
        max_plausible_auc: Optional[dict] = None,
    ) -> None:
        self.expected_directions = expected_directions or EXPECTED_DIRECTIONS
        self.max_plausible_auc = max_plausible_auc or MAX_PLAUSIBLE_AUC

    def validate_feature_importance(
        self,
        icd10: str,
        top_features: list[tuple[str, float]],
        top_n: int = 5,
    ) -> list[ValidationCheck]:
        """Check if top features make biological sense for the disease.

        Parameters
        ----------
        icd10 : str
            ICD-10 code (e.g. "E11" for T2D)
        top_features : list of (feature_name, importance_score)
            Ranked list of features by importance
        top_n : int
            How many top features to check
        """
        checks: list[ValidationCheck] = []
        icd10_upper = icd10.upper().strip()

        # Check for leakage-suspect features
        for feat_name, score in top_features[:top_n]:
            feat_lower = feat_name.lower().replace(" ", "_")
            if feat_lower in LEAKAGE_SUSPECT_FEATURES:
                checks.append(ValidationCheck(
                    name="leakage_suspect_feature",
                    passed=False,
                    message=f"Feature '{feat_name}' (importance={score:.4f}) is a known data leakage indicator",
                    severity="error",
                    evidence=f"'{feat_name}' contains outcome-related information that would not be available at prediction time",
                ))

        # Check biological plausibility of top features
        if icd10_upper in self.expected_directions:
            expected = self.expected_directions[icd10_upper]
            top_feat_names = {name.lower().replace(" ", "_") for name, _ in top_features[:top_n]}

            # Check if ANY expected biomarkers appear in top features
            expected_keys = set(expected.keys())
            overlap = top_feat_names & expected_keys
            if not overlap and len(top_features) >= top_n:
                checks.append(ValidationCheck(
                    name="expected_biomarkers_absent",
                    passed=False,
                    message=f"None of the expected biomarkers for {icd10_upper} appear in top-{top_n} features",
                    severity="warning",
                    evidence=f"Expected: {sorted(expected_keys)[:5]}; Got: {[f for f, _ in top_features[:top_n]]}",
                ))
            elif overlap:
                checks.append(ValidationCheck(
                    name="expected_biomarkers_present",
                    passed=True,
                    message=f"Found expected biomarkers for {icd10_upper}: {sorted(overlap)}",
                    severity="info",
                ))

        return checks

    def validate_auc_plausibility(
        self,
        icd10: str,
        auc: float,
        n_features: int,
        n_samples: Optional[int] = None,
    ) -> list[ValidationCheck]:
        """Flag suspiciously high AUC (possible data leakage).

        Rules:
          - AUC > 0.98 with < 50 features → almost certainly leakage
          - AUC > max_plausible for disease type → suspicious
          - AUC > 0.95 with < 20 features → likely leakage
          - AUC = 1.0 → definite leakage or test-set contamination
        """
        checks: list[ValidationCheck] = []
        icd10_upper = icd10.upper().strip()

        # Perfect AUC is always wrong
        if auc >= 1.0:
            checks.append(ValidationCheck(
                name="perfect_auc",
                passed=False,
                message=f"AUC={auc:.4f} — perfect discrimination indicates data leakage or test-set contamination",
                severity="error",
                evidence="No real-world biomedical model achieves AUC=1.0",
            ))
            return checks

        # Suspiciously high with few features
        if auc > 0.98 and n_features < 50:
            checks.append(ValidationCheck(
                name="extreme_auc_few_features",
                passed=False,
                message=f"AUC={auc:.3f} with only {n_features} features — strongly suggests data leakage",
                severity="error",
                evidence="Models with <50 features rarely achieve AUC>0.98 without leakage",
            ))
        elif auc > 0.95 and n_features < 20:
            checks.append(ValidationCheck(
                name="high_auc_very_few_features",
                passed=False,
                message=f"AUC={auc:.3f} with only {n_features} features — likely data leakage",
                severity="error",
                evidence="Models with <20 features achieving AUC>0.95 require careful leakage inspection",
            ))

        # Disease-specific plausibility
        max_auc = self.max_plausible_auc.get(icd10_upper, DEFAULT_MAX_PLAUSIBLE_AUC)
        if auc > max_auc:
            checks.append(ValidationCheck(
                name="disease_specific_auc_ceiling",
                passed=False,
                message=f"AUC={auc:.3f} exceeds maximum plausible for {icd10_upper} (≤{max_auc})",
                severity="warning",
                evidence=f"Literature suggests maximum realistic AUC for {icd10_upper} is ~{max_auc}",
            ))

        # Small sample concern
        if n_samples is not None and n_samples < 200 and auc > 0.90:
            checks.append(ValidationCheck(
                name="small_sample_high_auc",
                passed=False,
                message=f"AUC={auc:.3f} with only {n_samples} samples — likely overfitting",
                severity="warning",
                evidence="Small samples (<200) combined with high AUC suggest overfitting to noise",
            ))

        if not checks:
            checks.append(ValidationCheck(
                name="auc_plausibility",
                passed=True,
                message=f"AUC={auc:.3f} is plausible for {icd10_upper} with {n_features} features",
                severity="info",
            ))

        return checks

    def validate_direction_consistency(
        self,
        icd10: str,
        feature: str,
        direction: str,
    ) -> ValidationCheck:
        """Check if a feature's effect direction matches biomedical expectation.

        Parameters
        ----------
        direction : str
            "+" for positive association (risk factor), "-" for protective
        """
        icd10_upper = icd10.upper().strip()
        feat_lower = feature.lower().replace(" ", "_")

        if icd10_upper not in self.expected_directions:
            return ValidationCheck(
                name="direction_check",
                passed=True,
                message=f"No reference data for {icd10_upper} — direction not validated",
                severity="info",
            )

        expected = self.expected_directions[icd10_upper]

        # Find matching biomarker (fuzzy)
        matched_key = None
        for key in expected:
            if key in feat_lower or feat_lower in key:
                matched_key = key
                break

        if matched_key is None:
            return ValidationCheck(
                name="direction_check",
                passed=True,
                message=f"Feature '{feature}' not in reference set for {icd10_upper}",
                severity="info",
            )

        expected_dir = expected[matched_key]
        if direction != expected_dir:
            return ValidationCheck(
                name="direction_inconsistency",
                passed=False,
                message=(
                    f"Feature '{feature}' shows direction='{direction}' for {icd10_upper}, "
                    f"but expected '{expected_dir}' based on biomedical literature"
                ),
                severity="warning",
                evidence=f"Reference: {matched_key} is expected to be '{expected_dir}' for {icd10_upper}",
            )

        return ValidationCheck(
            name="direction_consistent",
            passed=True,
            message=f"Feature '{feature}' direction='{direction}' matches expectation for {icd10_upper}",
            severity="info",
        )


class LeakageDetector:
    """Detect data leakage in biobank ML pipelines.

    Types of leakage detected:
      - Temporal leakage: using future information to predict past
      - Target leakage: features directly derived from the outcome
      - Feature contamination: outcome-related fields in feature set
    """

    # Fields that encode temporal information about outcomes
    TEMPORAL_FIELDS: set[str] = {
        "date_of_death", "death_date", "date_lost_to_followup",
        "hospital_admission_date", "date_of_diagnosis",
        "diagnosis_date", "discharge_date",
    }

    # ICD-10 to directly related features (would be circular)
    CIRCULAR_FEATURES: dict[str, set[str]] = {
        "E11": {"diabetes_diagnosed", "diabetes_medication", "insulin_use", "hba1c_diagnosis"},
        "I21": {"mi_diagnosed", "statin_use", "cardiac_medication", "troponin"},
        "C50": {"mammography_result", "breast_cancer_screening", "tamoxifen"},
    }

    def check_temporal_leakage(
        self,
        features: list[str],
        outcome_date_field: Optional[str] = None,
    ) -> list[ValidationCheck]:
        """Detect features that encode future temporal information."""
        checks: list[ValidationCheck] = []
        for feat in features:
            feat_lower = feat.lower().replace(" ", "_")
            if feat_lower in self.TEMPORAL_FIELDS:
                checks.append(ValidationCheck(
                    name="temporal_leakage",
                    passed=False,
                    message=f"Feature '{feat}' contains temporal information about outcomes",
                    severity="error",
                    evidence="This field would not be available at prediction time",
                ))
        if not checks:
            checks.append(ValidationCheck(
                name="temporal_leakage_check",
                passed=True,
                message="No temporal leakage detected in feature set",
                severity="info",
            ))
        return checks

    def check_target_leakage(
        self,
        features: list[str],
        outcome_icd10: str,
    ) -> list[ValidationCheck]:
        """Detect features directly derived from the prediction target."""
        checks: list[ValidationCheck] = []
        icd10_upper = outcome_icd10.upper().strip()

        # Check known circular features
        if icd10_upper in self.CIRCULAR_FEATURES:
            circular = self.CIRCULAR_FEATURES[icd10_upper]
            for feat in features:
                feat_lower = feat.lower().replace(" ", "_")
                if feat_lower in circular:
                    checks.append(ValidationCheck(
                        name="target_leakage",
                        passed=False,
                        message=f"Feature '{feat}' is directly derived from outcome {icd10_upper}",
                        severity="error",
                        evidence=f"Using diagnosis-related features to predict diagnosis is circular",
                    ))

        # Check for outcome code in feature names
        for feat in features:
            feat_lower = feat.lower()
            if icd10_upper.lower() in feat_lower:
                checks.append(ValidationCheck(
                    name="target_code_in_features",
                    passed=False,
                    message=f"Feature '{feat}' contains the outcome ICD-10 code {icd10_upper}",
                    severity="error",
                    evidence="Feature name suggests direct relationship to prediction target",
                ))

        if not checks:
            checks.append(ValidationCheck(
                name="target_leakage_check",
                passed=True,
                message=f"No target leakage detected for {icd10_upper}",
                severity="info",
            ))
        return checks

    def full_leakage_scan(
        self,
        features: list[str],
        outcome_icd10: str,
        outcome_date_field: Optional[str] = None,
    ) -> list[ValidationCheck]:
        """Run all leakage detection checks."""
        checks = []
        checks.extend(self.check_temporal_leakage(features, outcome_date_field))
        checks.extend(self.check_target_leakage(features, outcome_icd10))

        # Check for participant ID
        for feat in features:
            if feat.lower() in ("eid", "participant_id", "subject_id", "id"):
                checks.append(ValidationCheck(
                    name="id_in_features",
                    passed=False,
                    message=f"Participant identifier '{feat}' in feature set — will cause leakage",
                    severity="error",
                    evidence="IDs can memorize individual outcomes during training",
                ))

        return checks
