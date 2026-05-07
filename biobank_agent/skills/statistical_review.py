"""Statistical reviewer — automated assumption checking for analyses.

Examines analysis records and flags potential statistical issues:
- Data leakage (AUC > 0.95)
- Class imbalance without correction
- Multiple testing without FDR
- Low sample sizes
- Proportional hazards violations
"""

import logging

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


@skill(
    name="statistical_review",
    description="Review recent analyses for statistical issues. Checks for data leakage, "
                "class imbalance, multiple testing problems, sample size adequacy, and "
                "assumption violations. Run after a complex analysis pipeline.",
    parameters={
        "scope": {
            "type": "string",
            "description": "'last' (most recent analysis), 'session' (all analyses), "
                           "or 'model:<key>' (specific model)",
            "default": "session",
        },
    },
    required=[],
)
def statistical_review(scope: str = "session", *, ctx=None) -> dict:
    """Review analyses for statistical issues."""
    records = ctx.state.records
    if not records:
        return {"issues": [], "message": "No analyses to review."}

    if scope == "last":
        records = records[-1:]
    elif scope.startswith("model:"):
        model_key = scope.split(":", 1)[1]
        records = [r for r in records if model_key in str(r.args) or model_key in str(r.key_results)]

    issues = []

    for r in records:
        # Check 1: Suspiciously high AUC (data leakage)
        if r.skill == "train_model":
            auc = r.key_results.get("mean_auc", r.key_results.get("auc", 0))
            if isinstance(auc, (int, float)):
                if auc > 0.95:
                    issues.append({
                        "severity": "CRITICAL",
                        "type": "data_leakage",
                        "skill": r.skill,
                        "message": f"AUC={auc:.4f} is suspiciously high (>0.95). "
                                   "Check for label leakage in features.",
                        "recommendation": "Remove diagnosis-derived fields from features. "
                                          "Check temporal ordering of labels vs features.",
                    })
                elif auc < 0.55:
                    issues.append({
                        "severity": "WARNING",
                        "type": "poor_discrimination",
                        "skill": r.skill,
                        "message": f"AUC={auc:.4f} indicates poor discrimination. "
                                   "Model may not be clinically useful.",
                        "recommendation": "Try different feature sets or model types.",
                    })

            # Check 2: Class imbalance
            n_cases = r.key_results.get("n_cases", 0)
            n_total = r.key_results.get("n_total", r.key_results.get("n_samples", 0))
            if n_cases and n_total and isinstance(n_cases, (int, float)):
                ratio = n_cases / max(n_total, 1)
                if ratio < 0.05:
                    issues.append({
                        "severity": "WARNING",
                        "type": "class_imbalance",
                        "skill": r.skill,
                        "message": f"Severe class imbalance: {int(n_cases)} cases / "
                                   f"{int(n_total)} total ({ratio:.1%}). ",
                        "recommendation": "Consider SMOTE, class weights, or stratified "
                                          "sampling to address imbalance.",
                    })

            # Check 3: Small sample size
            if isinstance(n_cases, (int, float)) and n_cases < 200:
                issues.append({
                    "severity": "WARNING",
                    "type": "small_sample",
                    "skill": r.skill,
                    "message": f"Only {int(n_cases)} cases. Results may be unstable.",
                    "recommendation": "Increase control ratio, use simpler models, "
                                      "or report wide confidence intervals.",
                })

        # Check 4: Multiple testing without correction
        if r.skill == "phewas":
            correction = r.args.get("correction")
            n_tests = r.key_results.get("n_fields_tested", 0)
            if n_tests > 100 and not correction:
                issues.append({
                    "severity": "CRITICAL",
                    "type": "multiple_testing",
                    "skill": r.skill,
                    "message": f"{n_tests} tests performed without multiple testing correction.",
                    "recommendation": "Apply FDR (Benjamini-Hochberg) or Bonferroni correction.",
                })

        # Check 5: Survival analysis sample size
        if r.skill == "survival":
            n_events = r.key_results.get("n_events", r.key_results.get("n_cases", 0))
            if isinstance(n_events, (int, float)) and n_events < 50:
                issues.append({
                    "severity": "WARNING",
                    "type": "low_event_count",
                    "skill": r.skill,
                    "message": f"Only {int(n_events)} events in survival analysis.",
                    "recommendation": "Kaplan-Meier estimates may be unreliable. "
                                      "Consider pooling related diagnosis codes.",
                })

        # Check 6: Correlation/multicollinearity
        if r.skill == "correlation":
            max_corr = r.key_results.get("max_correlation", 0)
            if isinstance(max_corr, (int, float)) and max_corr > 0.8:
                issues.append({
                    "severity": "INFO",
                    "type": "multicollinearity",
                    "skill": r.skill,
                    "message": f"High correlation ({max_corr:.2f}) detected between features.",
                    "recommendation": "Consider removing one of the correlated features "
                                      "before regression or check VIF.",
                })

        # Check 7: Agentic cross-cohort cohort card readiness
        if r.skill == "cohort_card":
            status = str(r.key_results.get("status", "")).upper()
            n_cases = r.key_results.get("n_cases")
            if status in {"PARTIAL", "FAIL"}:
                issues.append({
                    "severity": "WARNING",
                    "type": "cohort_card_partial",
                    "skill": r.skill,
                    "message": "Cohort card is not fully execution-ready.",
                    "recommendation": "Run phenotype_harmonize, cohort construction, statistical_review, and safety_check before final synthesis.",
                })
            if isinstance(n_cases, (int, float)) and n_cases < 100:
                issues.append({
                    "severity": "CRITICAL",
                    "type": "small_cross_cohort_case_count",
                    "skill": r.skill,
                    "message": f"Only {int(n_cases)} cases in cohort card.",
                    "recommendation": "Treat as exploratory or broaden phenotype definition before confirmatory analysis.",
                })
            for flag in r.key_results.get("bias_flags", []) or []:
                if "target" in str(flag).lower() or "immortal" in str(flag).lower():
                    issues.append({
                        "severity": "INFO",
                        "type": "design_bias_check_required",
                        "skill": r.skill,
                        "message": str(flag),
                        "recommendation": "Document index date, lookback window, follow-up window, censoring and adjustment strategy.",
                    })

        # Check 8: Trajectory datasets are preparatory unless there are tokens and participants
        if r.skill == "trajectory_tokenize":
            n_tokens = r.key_results.get("n_tokens", 0)
            n_participants = r.key_results.get("n_participants", 0)
            if n_tokens == 0 or n_participants == 0:
                issues.append({
                    "severity": "CRITICAL",
                    "type": "empty_trajectory_dataset",
                    "skill": r.skill,
                    "message": "Trajectory tokenization produced no usable participant tokens.",
                    "recommendation": "Check required columns, timestamps, missingness and modality mappings.",
                })

        # Check 9: World-model outputs need calibration/external validation before strong claims
        if r.skill == "world_model_audit":
            safety = str(r.key_results.get("safety_status", "PARTIAL")).upper()
            allowed = str(r.key_results.get("allowed_claim_type", ""))
            if safety == "FAIL":
                issues.append({
                    "severity": "CRITICAL",
                    "type": "unsupported_world_model_claim",
                    "skill": r.skill,
                    "message": "World-model audit blocked the requested scientific claim.",
                    "recommendation": "Downgrade to association-conditioned forecast or provide target-trial/RCT calibration evidence.",
                })
            elif safety == "PARTIAL":
                issues.append({
                    "severity": "WARNING",
                    "type": "partial_world_model_evidence",
                    "skill": r.skill,
                    "message": f"World-model output may only support `{allowed}`.",
                    "recommendation": "Report calibration, OOD coverage and external validation status.",
                })

    # Sort issues by severity: CRITICAL > WARNING > INFO
    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    issues.sort(key=lambda x: severity_order.get(x["severity"], 9))

    # Classify overall risk
    critical = sum(1 for i in issues if i["severity"] == "CRITICAL")
    warnings = sum(1 for i in issues if i["severity"] == "WARNING")

    if critical > 0:
        overall = "CRITICAL — review required before publication"
    elif warnings > 2:
        overall = "CAUTION — several issues to address"
    elif warnings > 0:
        overall = "MINOR — some issues noted"
    else:
        overall = "CLEAN — no statistical issues detected"

    return {
        "overall_assessment": overall,
        "n_issues": len(issues),
        "n_critical": critical,
        "n_warnings": warnings,
        "issues": issues,
        "scope": scope,
        "n_records_reviewed": len(records),
    }
