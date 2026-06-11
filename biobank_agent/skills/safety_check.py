"""Safety layer — privacy and data protection checks for biobank analyses.

Enforces:
- k-anonymity: no output group with fewer than k individuals
- Minimum cell count: cross-tabulations must have n >= 5 per cell
- Re-identification risk: warnings for queries returning very few subjects
- IRB compliance: reminders about data access agreements
"""

import logging

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)

# Default privacy thresholds
K_ANONYMITY_THRESHOLD = 5
MIN_CELL_COUNT = 5
REIDENTIFICATION_THRESHOLD = 10


@skill(
    name="safety_check",
    description="Check analysis results for privacy and safety compliance. "
                "Verifies k-anonymity, minimum cell counts, re-identification risk, "
                "and IRB compliance. Run automatically after patient-level analyses "
                "or manually before sharing results.",
    parameters={
        "scope": {
            "type": "string",
            "description": "'last' (most recent), 'session' (all), or 'cohort:<name>' (specific cohort)",
            "default": "last",
        },
        "k": {
            "type": "integer",
            "description": "Minimum group size for k-anonymity (default: 5)",
            "default": 5,
        },
    },
    required=[],
)
def safety_check(scope: str = "last", k: int = 5, *, ctx=None) -> dict:
    """Check analyses for privacy and data protection compliance."""
    issues = []
    records = ctx.state.records
    settings = getattr(ctx, "settings", None)
    data_deidentified = bool(getattr(settings, "data_deidentified", False))
    disclosure_control_mode = str(getattr(settings, "disclosure_control_mode", "external")).lower()
    deidentified_internal = (
        data_deidentified
        and disclosure_control_mode in {"internal", "research", "off"}
    )

    if not records:
        return {"issues": [], "overall": "NO DATA", "message": "No analyses to check."}

    if scope == "last":
        records = records[-1:]

    for r in records:
        # ── k-Anonymity Check ─────────────────────────────
        # Flag any result that reports individual-level data or small groups
        if r.skill == "predict":
            predictions = r.key_results.get("predictions", [])
            if predictions and len(predictions) < k:
                issues.append({
                    "severity": "WARNING" if deidentified_internal else "CRITICAL",
                    "type": "k_anonymity_violation",
                    "skill": r.skill,
                    "message": (
                        f"Prediction results contain only {len(predictions)} individuals. "
                        f"k-anonymity requires >= {k} individuals per group."
                    ),
                    "recommendation": "Aggregate results or suppress individual-level output.",
                })

        # ── Minimum Cell Count ────────────────────────────
        if r.skill in ("cohort_summary", "prevalence"):
            n_cases = r.key_results.get("n_cases", 0)
            if isinstance(n_cases, (int, float)) and 0 < n_cases < MIN_CELL_COUNT:
                issues.append({
                    "severity": "INFO" if deidentified_internal else "CRITICAL",
                    "type": "min_cell_count",
                    "skill": r.skill,
                    "message": (
                        f"Cohort has only {int(n_cases)} cases. "
                        f"Minimum cell count is {MIN_CELL_COUNT}."
                    ),
                    "recommendation": "Do not publish case counts below minimum threshold. "
                                      "Consider broadening the diagnosis code.",
                })

        # ── Re-identification Risk ────────────────────────
        if r.skill in ("predict", "cohort_summary", "biomarker_dist"):
            n_subjects = r.key_results.get("n_total", r.key_results.get("n_patients", 0))
            if isinstance(n_subjects, (int, float)) and 0 < n_subjects < REIDENTIFICATION_THRESHOLD:
                issues.append({
                    "severity": "WARNING",
                    "type": "reidentification_risk",
                    "skill": r.skill,
                    "message": (
                        f"Analysis involves only {int(n_subjects)} subjects. "
                        f"Small populations may enable re-identification."
                    ),
                    "recommendation": "Consider whether this analysis could identify individuals "
                                      "when combined with external data.",
                })

        # ── IRB Compliance ────────────────────────────────
        if r.skill in ("cohort_summary",) and scope == "session":
            # First cohort build in session → remind about data access
            cohort_records = [rec for rec in ctx.state.records if rec.skill == "cohort_summary"]
            if len(cohort_records) == 1:
                bank_name = ctx.settings.biobank_name if ctx and hasattr(ctx, "settings") else "the biobank"
                issues.append({
                    "severity": "INFO",
                    "type": "irb_reminder",
                    "skill": r.skill,
                    "message": (
                        f"New cohort constructed from {bank_name} data. "
                        "Ensure your analysis is covered by your data access agreement."
                    ),
                    "recommendation": "Verify IRB approval covers this analysis scope. "
                                      "Document the approval number in your report.",
                })

        # ── Cross-cohort cohort card safety ───────────────
        if r.skill == "cohort_card":
            n_cases = r.key_results.get("n_cases")
            if isinstance(n_cases, (int, float)) and 0 < n_cases < MIN_CELL_COUNT:
                issues.append({
                    "severity": "INFO" if deidentified_internal else "CRITICAL",
                    "type": "min_cell_count",
                    "skill": r.skill,
                    "message": f"Cohort card reports only {int(n_cases)} cases.",
                    "recommendation": "Suppress small counts or broaden phenotype before sharing externally.",
                })
            if len(r.key_results.get("banks", []) or []) > 1:
                issues.append({
                    "severity": "INFO",
                    "type": "cross_cohort_release_review",
                    "skill": r.skill,
                    "message": "Cross-cohort cohort card requires per-bank release and governance review.",
                    "recommendation": "Confirm each cohort permits harmonised aggregate reporting for this endpoint.",
                })

        # ── World-model claim safety ──────────────────────
        if r.skill == "world_model_audit":
            safety = str(r.key_results.get("safety_status", "PARTIAL")).upper()
            if safety == "FAIL":
                issues.append({
                    "severity": "CRITICAL",
                    "type": "blocked_world_model_claim",
                    "skill": r.skill,
                    "message": "World-model audit blocked unsupported causal or out-of-distribution claim.",
                    "recommendation": "Do not release as a final scientific conclusion without execution-grounded validation.",
                })

        # ── Genetic target hypothesis release review ──────
        if r.skill == "genetic_target_hypothesis":
            if r.key_results.get("source_data_type") == "rare_variant_burden_summary_statistics":
                issues.append({
                    "severity": "INFO",
                    "type": "summary_statistic_release_review",
                    "skill": r.skill,
                    "message": (
                        "Genetic target hypotheses are derived from aggregate burden summary statistics "
                        "and must not be framed as treatment recommendations."
                    ),
                    "recommendation": (
                        "Confirm redistribution terms for supplied summary statistics, avoid individual-level "
                        "variant disclosure, and label outputs as therapeutic hypotheses requiring validation."
                    ),
                })

        # ── External target annotation release review ─────
        if r.skill == "target_annotation_context":
            issues.append({
                "severity": "INFO",
                "type": "external_annotation_license_review",
                "skill": r.skill,
                "message": (
                    "Target annotations come from external aggregate databases and should be "
                    "reported with source provenance and redistribution terms."
                ),
                "recommendation": (
                    "Archive source names, query dates and cache files; avoid presenting "
                    "trial or tractability context as treatment recommendations."
                ),
            })
            if any(t.get("cell_type_context") for t in r.key_results.get("targets", []) or []):
                issues.append({
                    "severity": "INFO",
                    "type": "single_cell_context_release_review",
                    "skill": r.skill,
                    "message": "CELLxGENE context is aggregate reference expression and not participant-level evidence.",
                    "recommendation": "Keep single-cell context separate from individual-level biobank inference.",
                })

        # ── Enrichment release review ─────────────────────
        if r.skill == "target_enrichment":
            issues.append({
                "severity": "INFO",
                "type": "enrichment_context_release_review",
                "skill": r.skill,
                "message": "Gene-set enrichment is aggregate pathway context and does not establish biological mechanism.",
                "recommendation": "Report the GMT source, universe, overlap genes and FDR method with any released table.",
            })

    # ── Sort by severity ──────────────────────────────────
    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    issues.sort(key=lambda x: severity_order.get(x["severity"], 9))

    critical = sum(1 for i in issues if i["severity"] == "CRITICAL")
    if critical > 0:
        overall = "BLOCK — critical privacy issues detected"
    elif issues:
        overall = "REVIEW — minor issues noted"
    else:
        overall = "PASS — no privacy issues detected"

    return {
        "overall": overall,
        "n_issues": len(issues),
        "issues": issues,
        "k_threshold": k,
        "scope": scope,
        "data_deidentified": data_deidentified,
        "internal_data_preserved": deidentified_internal,
        "disclosure_control_mode": disclosure_control_mode,
    }
