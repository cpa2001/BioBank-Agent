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

    # Methodology reviewer: structured consensus-statistical-sin flags
    # (uncorrected multiple testing, underpowered arms, missing CIs, unaddressed
    # stratification) layered on the bespoke per-skill checks below.
    try:
        from biobank_agent.runtime.methodology import review_methodology

        for r in records:
            for flag in review_methodology(getattr(r, "key_results", None) or {}):
                issues.append({
                    "severity": "CRITICAL" if flag.get("severity") == "block" else "WARNING",
                    "skill": getattr(r, "skill", ""),
                    "type": flag.get("issue"),
                    "message": flag.get("detail"),
                    "recommendation": flag.get("detail"),
                })
    except Exception:
        pass

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
            if not n_total:
                n_controls = r.key_results.get("n_controls", 0)
                if isinstance(n_cases, (int, float)) and isinstance(n_controls, (int, float)):
                    n_total = n_cases + n_controls
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

            analysis_design = str(r.key_results.get("analysis_design", "") or "").lower()
            incident_supported = bool(r.key_results.get("incident_risk_supported", True))
            prediction_target = str(r.key_results.get("prediction_target", "") or "")
            if "prevalent" in analysis_design or incident_supported is False:
                issues.append({
                    "severity": "WARNING",
                    "type": "prevalent_case_control_not_incident_risk",
                    "skill": r.skill,
                    "message": (
                        f"Model target `{prediction_target or 'diagnosis discrimination'}` is a prevalent/ever-diagnosed "
                        "case-control endpoint, not an incident risk-prediction design."
                    ),
                    "recommendation": (
                        "Report this as internal EHR/linked-diagnosis discrimination unless index date, baseline exclusion, "
                        "washout and follow-up windows are constructed."
                    ),
                })

            leakage_features = r.key_results.get("diagnostic_biomarker_leakage_features") or []
            leakage_risk = bool(r.key_results.get("diagnostic_biomarker_leakage_risk"))
            if leakage_risk or leakage_features:
                severity = "CRITICAL" if isinstance(auc, (int, float)) and auc >= 0.90 else "WARNING"
                preview = ", ".join(str(x) for x in list(leakage_features)[:5]) or "endpoint-adjacent biomarkers"
                issues.append({
                    "severity": severity,
                    "type": "diagnostic_biomarker_temporal_leakage_risk",
                    "skill": r.skill,
                    "message": (
                        f"Endpoint-adjacent diabetes biomarkers were included in an E11 discrimination model ({preview}). "
                        "Without proof that measurements precede diagnosis, high AUC may reflect temporal or clinical-label leakage."
                    ),
                    "recommendation": (
                        "For an incident-risk study, rebuild the cohort with pre-index predictors only. For the current run, "
                        "label the result as prevalent E11 discrimination and include this limitation in the report."
                    ),
                })

            evaluation_strategy = str(r.key_results.get("evaluation_strategy", "") or "").lower()
            requested_folds = ((r.key_results.get("model_selection") or {}).get("requested_n_folds")
                               if isinstance(r.key_results.get("model_selection"), dict) else None)
            if evaluation_strategy == "stratified_holdout" and requested_folds:
                issues.append({
                    "severity": "INFO",
                    "type": "holdout_selection_not_kfold_cv",
                    "skill": r.skill,
                    "message": (
                        f"Auto model selection used a stratified holdout although n_folds={requested_folds} was requested."
                    ),
                    "recommendation": (
                        "Distinguish holdout model selection from any later cross-validation diagnostics in report methods."
                    ),
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
            support = str(r.key_results.get("longitudinal_support", "") or "").lower()
            time_source = str(r.key_results.get("trajectory_time_source", "") or "").lower()
            if n_tokens == 0 or n_participants == 0:
                issues.append({
                    "severity": "CRITICAL",
                    "type": "empty_trajectory_dataset",
                    "skill": r.skill,
                    "message": "Trajectory tokenization produced no usable participant tokens.",
                    "recommendation": "Check required columns, timestamps, missingness and modality mappings.",
                })
            elif support == "single_timepoint":
                issues.append({
                    "severity": "WARNING",
                    "type": "single_timepoint_trajectory",
                    "skill": r.skill,
                    "message": "Trajectory tokenization found only single-timepoint biomarker support.",
                    "recommendation": "Report this as cross-sectional tokenization or add repeated assessment rows before forecast claims.",
                })
            if "synthetic" in time_source:
                issues.append({
                    "severity": "INFO",
                    "type": "synthetic_trajectory_time",
                    "skill": r.skill,
                    "message": "Trajectory timestamps were derived from assessment instances rather than exact visit dates.",
                    "recommendation": "Treat time ordering as approximate and avoid claims requiring exact elapsed time.",
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

        # Check 10: Genetic target hypotheses must be treated as discovery signals
        if r.skill == "genetic_target_hypothesis":
            status = str(r.key_results.get("status", "")).upper()
            n_family = r.key_results.get("n_family_tests", 0)
            n_rows = r.key_results.get("n_matched_rows", 0)
            targets = r.key_results.get("targets", []) or []
            scope = str(r.key_results.get("multiple_testing_scope", "provided_rows_only"))
            if status == "NEEDS_INPUT":
                issues.append({
                    "severity": "CRITICAL",
                    "type": "genetic_target_missing_burden_stats",
                    "skill": r.skill,
                    "message": "Genetic target hypothesis run did not include burden summary statistics.",
                    "recommendation": "Provide GeneBass-like rows with gene, phenotype, annotation, beta and P-value columns.",
                })
            if status == "INVALID_INPUT":
                issues.append({
                    "severity": "CRITICAL",
                    "type": "genetic_target_invalid_input",
                    "skill": r.skill,
                    "message": "No valid rare-variant burden rows were available for target prioritization.",
                    "recommendation": "Fix row parsing errors before interpreting or reporting any target hypotheses.",
                })
            if status == "NO_MATCH":
                issues.append({
                    "severity": "CRITICAL",
                    "type": "genetic_target_phenotype_no_match",
                    "skill": r.skill,
                    "message": "Requested phenotype did not match any labelled burden rows.",
                    "recommendation": "Correct the phenotype query or explicitly provide a pre-filtered table with prefiltered=true.",
                })
            if status == "NEEDS_PREFILTERED_DECLARATION":
                issues.append({
                    "severity": "CRITICAL",
                    "type": "genetic_target_prefilter_required",
                    "skill": r.skill,
                    "message": "Burden rows lacked phenotype labels and were not explicitly declared pre-filtered.",
                    "recommendation": "Set prefiltered=true only after confirming every supplied row belongs to the requested phenotype.",
                })
            if scope == "filtered_with_family_size":
                issues.append({
                    "severity": "WARNING",
                    "type": "filtered_burden_family",
                    "skill": r.skill,
                    "message": f"FDR/Bonferroni tiers use family_size={int(n_family)} with {int(n_rows)} matched rows supplied.",
                    "recommendation": "Treat BY-FDR q-values as conservative approximations unless the full phenotype-family table is supplied.",
                })
            elif scope == "provided_rows_only":
                issues.append({
                    "severity": "WARNING",
                    "type": "provided_rows_multiple_testing_scope",
                    "skill": r.skill,
                    "message": "Multiple-testing tiers were computed only over supplied rows.",
                    "recommendation": "Provide family_size or the full phenotype-family table before using tiers in publication-grade prioritization.",
                })
            if targets and all(str(t.get("tier", "")).lower() == "exploratory" for t in targets):
                issues.append({
                    "severity": "WARNING",
                    "type": "exploratory_genetic_targets_only",
                    "skill": r.skill,
                    "message": "All genetic target hypotheses are exploratory-tier signals.",
                    "recommendation": "Do not prioritize for therapeutic programs without replication or orthogonal validation.",
                })
            if targets and not any("concordant" in str(t.get("variant_support", "")).lower() for t in targets):
                issues.append({
                    "severity": "INFO",
                    "type": "single_variant_class_support",
                    "skill": r.skill,
                    "message": "No target has pLoF plus missense|LC concordant support.",
                    "recommendation": "Review variant-class specificity and seek independent support before strong claims.",
                })
            if targets and any(str(t.get("therapeutic_direction", "")).lower() == "uncertain" for t in targets):
                issues.append({
                    "severity": "WARNING",
                    "type": "uncertain_therapeutic_direction",
                    "skill": r.skill,
                    "message": "At least one genetic target has no inferable inhibit/activate direction.",
                    "recommendation": "Do not use direction-uncertain targets for therapeutic strategy without additional functional evidence.",
                })

        # Check 11: External target annotations are context only
        if r.skill == "target_annotation_context":
            status = str(r.key_results.get("status", "")).upper()
            targets = r.key_results.get("targets", []) or []
            source_counts = r.key_results.get("source_status_counts", {}) or {}
            if status == "NEEDS_INPUT":
                issues.append({
                    "severity": "CRITICAL",
                    "type": "target_annotation_missing_targets",
                    "skill": r.skill,
                    "message": "Target annotation context was requested without usable target genes.",
                    "recommendation": "Provide target rows from a biobank GWAS, burden, or target-hypothesis result.",
                })
            if status == "INVALID_INPUT":
                issues.append({
                    "severity": "CRITICAL",
                    "type": "target_annotation_invalid_input",
                    "skill": r.skill,
                    "message": "Target annotation context received unsupported sources or malformed inputs.",
                    "recommendation": "Use supported sources: opentargets, uniprot, gtex, clinicaltrials, or cellxgene.",
                })
            if status == "PARTIAL" or any(
                any(key in counts for key in {"ERROR", "PARTIAL", "SKIPPED"})
                for counts in source_counts.values()
                if isinstance(counts, dict)
            ):
                issues.append({
                    "severity": "WARNING",
                    "type": "target_annotation_partial_sources",
                    "skill": r.skill,
                    "message": "One or more annotation sources were unavailable, skipped, or returned errors.",
                    "recommendation": "Inspect per-source status before using annotation context in a report.",
                })
            if any(not t.get("ensembl_id") for t in targets):
                issues.append({
                    "severity": "INFO",
                    "type": "target_annotation_missing_gene_id",
                    "skill": r.skill,
                    "message": "At least one target lacks an Ensembl gene ID, limiting GTEx/Open Targets context.",
                    "recommendation": "Provide Ensembl IDs from the upstream target table or a validated gene resolver.",
                })
            if targets:
                issues.append({
                    "severity": "INFO",
                    "type": "annotation_context_not_rank_evidence",
                    "skill": r.skill,
                    "message": "External annotations are translational context and should not replace direct biobank evidence.",
                    "recommendation": "Keep target ranking tied to GWAS or rare-variant burden evidence.",
                })

        # Check 12: Enrichment depends on gene-set universe and provenance
        if r.skill == "target_enrichment":
            status = str(r.key_results.get("status", "")).upper()
            method = str(r.key_results.get("method", ""))
            if status in {"NEEDS_INPUT", "INVALID_INPUT"}:
                issues.append({
                    "severity": "CRITICAL",
                    "type": "target_enrichment_missing_gene_sets",
                    "skill": r.skill,
                    "message": "Target enrichment did not have a usable target list and local gene-set file.",
                    "recommendation": "Provide gene_list or ranked_genes plus a provenance-tracked GMT file.",
                })
            if not r.key_results.get("explicit_universe", False) and method == "local_ora":
                issues.append({
                    "severity": "WARNING",
                    "type": "target_enrichment_missing_universe",
                    "skill": r.skill,
                    "message": "ORA used the union of GMT genes rather than an explicit test universe.",
                    "recommendation": "Use the assayed or tested gene universe from the upstream biobank analysis.",
                })
            if "gseapy" in method or "enrichr" in method:
                issues.append({
                    "severity": "INFO",
                    "type": "target_enrichment_online_context",
                    "skill": r.skill,
                    "message": "GSEApy/Enrichr-style enrichment should be treated as exploratory context.",
                    "recommendation": "Archive gene-set version, query date, and result table with the analysis.",
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
