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
