"""Shared constants for the Biobank Agent.

Central repository for hard-coded domain values that are used across
multiple modules. Prevents constant mismatches between verifiers.

Source: UK Biobank documentation
    - 502,411: genotyped participants (Bycroft et al. 2018, Nature 562:203-209)
    - Recruitment age: 37-73 years at baseline (2006-2010)
    - Assessment centres: 22 across England, Scotland, Wales
"""

# ── UK Biobank Hard Bounds ────────────────────────────────────────────────────

# Total genotyped participants (authoritative count from Bycroft 2018)
UKB_TOTAL_PARTICIPANTS: int = 502_411

# Recruitment age range at baseline assessment
UKB_MIN_AGE_RECRUITMENT: int = 37
UKB_MAX_AGE_RECRUITMENT: int = 73

# Number of assessment centres
UKB_ASSESSMENT_CENTRES: int = 22

# Maximum follow-up years (recruitment 2006-2010, approx to 2026)
UKB_MAX_FOLLOW_UP_YEARS: int = 20

# Valid chromosome range (1-22 autosomal + X=23)
UKB_VALID_CHROMOSOMES: set = set(range(1, 24))


# ── Statistical Bounds ────────────────────────────────────────────────────────

# Metrics that must be in [0, 1]
BOUNDED_01_METRICS: set = {
    "auc", "auroc", "auprc", "accuracy", "f1",
    "precision", "recall", "sensitivity", "specificity", "r2",
}

# Metrics that must be > 0
POSITIVE_ONLY_METRICS: set = {
    "hazard_ratio", "hr", "odds_ratio", "or_value",
    "relative_risk", "rr",
}

# Keys that represent sample counts
SAMPLE_SIZE_KEYS: set = {
    "n_cases", "n_controls", "total", "n_total", "n_participants",
    "cohort_size", "sample_size", "n_samples",
}
