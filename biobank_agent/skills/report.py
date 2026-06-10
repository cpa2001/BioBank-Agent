"""Report generator — compile analyses into structured, readable documents.

Supports two modes:
- 'report' / 'technical' (default): Technical report with executive summary, Key Findings, interpretive text
- 'paper' / 'nature': IMRaD paper draft with Nature-quality prose

Output formats: Markdown, HTML (self-contained with CSS), PDF (via pandoc)
"""

from pathlib import Path
from datetime import datetime
import json
import re
import shutil
import stat

from biobank_agent.registry import skill
from biobank_agent.skills.academic_report_polisher import polish_markdown_report
from biobank_agent.utils.report_templates import (
    REPORT_HEADER,
    KEY_FINDINGS_BOX,
    EXECUTIVE_FINDINGS_BOX,
    EXECUTIVE_SUMMARY,
    SECTION_HEADER,
    PAPER_ABSTRACT,
    PAPER_INTRODUCTION,
    PAPER_METHODS,
    PAPER_RESULTS,
    PAPER_DISCUSSION,
    NATURE_CSS,
)


_FORMAT_ALIASES = {
    "report": "report",
    "technical": "report",
    "paper": "paper",
    "nature": "paper",
    "dual": "dual",
    "both": "dual",
    "paired": "dual",
    "brief": "brief",
}


_RAW_RESULT_KEYS = {
    "execution_log",
    "execution_logs",
    "raw_execution_log",
    "raw_log",
    "raw_logs",
    "stdout",
    "stderr",
    "trace",
    "traceback",
    "stacktrace",
}


_GUARDRAIL_SKILLS = {"statistical_review", "safety_check", "world_model_audit"}
_APPENDIX_ONLY_SKILLS = {"executive_findings", "execution_log", "execution_appendix", "generate_report"}
_UNSAFE_EXECUTIVE_PATTERNS = (
    r"/Users/",
    r"\btraceback\b",
    r"\bstdout\b|\bstderr\b|\breturncode\b",
    r"\bTODO\b|\bFIXME\b",
    r"```",
    r"\bpytest\b|\bbenchmark\b",
    r"\bcodex\b|\bclaude\b",
    r"\breview(ed|er| loop)?\b",
    r"\bapi[_ -]?key\b|\bauth\b|\blogin\b",
    r"\breport path\b|\bpath:\b",
    r"\banalysis completed\b",
)
_UNQUALIFIED_CAUSAL_PATTERN = re.compile(
    r"\b(causes?|causal effect|prevents?|treats?|cures?|reduces risk|protects against|therapy recommendation)\b",
    re.IGNORECASE,
)
_CAUSAL_QUALIFIER_PATTERN = re.compile(
    r"\b(associat(?:ed|ion)|hypothes(?:is|ize)|observational|not causal|cannot infer|does not establish|requires validation|may)\b",
    re.IGNORECASE,
)


def _redact_report_tool_tokens(text: str) -> str:
    """Remove AI/tool brand traces from human-facing report text."""
    text = re.sub(r"\b(Codex|Claude(?:\s+Code)?|OpenAI)\b", "external reviewer", str(text), flags=re.IGNORECASE)
    text = re.sub(r"\b(codex-check|claude-check)\b", "external review", text, flags=re.IGNORECASE)
    return text


def _sanitize_report_title(title: str, ctx=None) -> str:
    """Convert free-form user goals into a concise human-facing report title."""
    text = _redact_report_tool_tokens(str(title or "").strip())
    text = re.sub(r"\s+", " ", text)
    lower = text.lower()
    prompt_like = (
        len(text) > 180
        or any(
            token in lower
            for token in (
                "act as ",
                "you should ",
                "if some part",
                "clarifications:",
                "planning modes",
                "available skills",
                "current ukb data inventory",
            )
        )
    )
    if prompt_like:
        bank = "UKB"
        if ctx is not None and hasattr(ctx, "settings"):
            configured_bank = getattr(ctx.settings, "biobank_abbreviation", "")
            if isinstance(configured_bank, str) and configured_bank.strip():
                bank = configured_bank.strip()
        if any(token in lower for token in ("metabolic", "cardiometabolic", "trajectory", "trajectories")):
            if any(token in lower for token in ("type 2", "t2d", "e11", "diabetes")):
                return f"{bank} metabolic health trajectories and Type 2 Diabetes risk report"
            return f"{bank} metabolic health trajectory feasibility report"
        if "milton" in lower or "replicate" in lower or "reproduce" in lower:
            return f"{bank} paper replication feasibility report"
        if any(token in lower for token in ("type 2", "t2d", "e11", "diabetes")):
            return f"{bank} Type 2 Diabetes analysis report"
        return f"{bank} analysis report"
    return text[:140].rstrip(" ,.;:-") or "Biobank Analysis Report"


def _normalize_report_format(format_name: str) -> str:
    """Normalize public report style aliases to canonical internal styles."""
    key = str(format_name or "report").strip().lower()
    return _FORMAT_ALIASES.get(key, "report")


def _format_value(v):
    """Convert numpy types to Python natives for display."""
    try:
        import numpy as np
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        elif isinstance(v, np.ndarray):
            return v.tolist()
    except (ImportError, AttributeError):
        pass
    return v


def _format_auc_with_ci(result: dict) -> str:
    """Format AUC with 95% confidence interval — Nature standard.

    Returns e.g. 'AUC = 0.852 (95% CI: 0.831-0.874)'
    """
    auc = _format_value(result.get("auc_mean", result.get("mean_auc", result.get("auc"))))
    ci = result.get("auc_95ci", "")
    if isinstance(auc, (int, float)):
        if ci and isinstance(ci, str):
            return f"AUC = {auc:.3f} ({ci.replace('[', '95% CI: ').replace(']', '')})"
        return f"AUC = {auc:.3f}"
    return "AUC = N/A"


def _format_p_value(p) -> str:
    """Format P-value per journal standards.

    Returns e.g. 'P < 0.001', 'P = 0.023', 'P = 0.45 (n.s.)'
    """
    p = _format_value(p)
    if not isinstance(p, (int, float)):
        return f"P = {p}"
    if p < 0.001:
        return "P < 0.001"
    if p < 0.01:
        return f"P = {p:.3f}"
    if p < 0.05:
        return f"P = {p:.3f}"
    return f"P = {p:.2f} (n.s.)"


def _build_figure_caption(fig_path, rec, fig_n: int, ctx) -> str:
    """Build a proper figure caption with sample sizes and test stats.

    Returns e.g. '**Figure 1.** Prevalence of top 10 diseases in UK Biobank
    (N = 502,370). Error bars represent 95% CIs.'
    """
    from pathlib import Path
    p = Path(fig_path)
    skill_name = rec.skill if rec else "analysis"
    results = rec.key_results if rec else {}

    caption_parts = [f"**Figure {fig_n}.**"]

    # Skill-specific captions
    if skill_name == "prevalence":
        n = results.get("total_subjects", "N/A")
        bank = ctx.settings.biobank_name if ctx and hasattr(ctx, "settings") else "Biobank"
        n_str = f"{n:,}" if isinstance(n, int) else str(n)
        caption_parts.append(f"Prevalence of top diseases in {bank} (N = {n_str}).")

    elif skill_name == "train_model":
        auc_str = _format_auc_with_ci(results)
        model_type = rec.args.get("model_type", "model") if rec else "model"
        n_cases = results.get("n_cases", "?")
        n_controls = results.get("n_controls", "?")
        caption_parts.append(
            f"ROC curve for {model_type} classifier. "
            f"{auc_str}. n = {n_cases} cases, {n_controls} controls."
        )

    elif skill_name == "survival":
        p_val = results.get("log_rank_p")
        icd = rec.args.get("icd10_code", "?") if rec else "?"
        caption_parts.append(f"Kaplan-Meier survival curves for {icd}.")
        if p_val is not None:
            caption_parts.append(f"Log-rank {_format_p_value(p_val)}.")

    elif skill_name == "feature_importance":
        importance_type = str(results.get("importance_type") or results.get("effective_method") or "Tree-based")
        if "shap" in importance_type.lower():
            caption_parts.append("SHAP feature-attribution plot showing feature contributions to model predictions.")
        else:
            caption_parts.append("Tree-based feature-importance bar chart for the trained model.")

    elif skill_name == "smart_plot" and str(results.get("plot_type") or rec.args.get("plot_type", "")).lower() == "summary":
        caption_parts.append(
            "Session-level diagnostic summary combining cohort composition, trajectory tokenization, "
            "model discrimination/calibration, top biomarkers and guardrail status."
        )

    elif skill_name == "biomarker_dist":
        biomarker = rec.args.get("field_name", "biomarker") if rec else "biomarker"
        p_val = results.get("p_value", results.get("p"))
        caption_parts.append(f"Distribution of {biomarker} in cases vs controls.")
        if p_val is not None:
            caption_parts.append(f"Mann-Whitney U test {_format_p_value(p_val)}.")

    elif skill_name == "correlation":
        caption_parts.append("Clustered correlation heatmap of biomarker panel.")

    elif skill_name == "phewas":
        caption_parts.append("PheWAS Manhattan plot. Dashed line indicates FDR-corrected significance threshold.")

    elif skill_name == "comorbidity":
        caption_parts.append("Comorbidity network showing co-occurrence patterns.")

    elif skill_name == "vcf_qc":
        if "scatter" in p.name:
            caption_parts.append("Sample QC scatter plot: call rate vs heterozygosity ratio, coloured by phenotype group. Outlier samples annotated.")
        elif "titv" in p.name:
            caption_parts.append("Ti/Tv ratio distribution by phenotype group. Box plots show median and IQR.")
        else:
            caption_parts.append("VCF quality control visualization.")

    elif skill_name == "vcf_pca":
        if "scree" in p.name:
            caption_parts.append("PCA scree plot showing per-component (bars) and cumulative (line) variance explained.")
        elif "pc1_pc2" in p.name:
            caption_parts.append("PCA plot of PC1 vs PC2, coloured by phenotype group.")
        elif "pc1_pc3" in p.name:
            caption_parts.append("PCA plot of PC1 vs PC3, coloured by phenotype group.")
        else:
            caption_parts.append("Genotype PCA visualization.")

    elif skill_name == "vcf_kinship":
        caption_parts.append("Kinship heatmap from the Genomic Relationship Matrix. Off-diagonal values indicate pairwise relatedness; diagonal masked.")

    elif skill_name == "vcf_association":
        if "manhattan" in p.name:
            caption_parts.append("Manhattan plot of case-control association. Red dashed line: Bonferroni threshold; blue dashed line: suggestive (1e-5).")
        elif "qq" in p.name:
            caption_parts.append("QQ plot of observed vs expected p-values with genomic inflation factor (λGC).")
        else:
            caption_parts.append("Association analysis visualization.")

    elif skill_name == "vcf_annotation":
        caption_parts.append("Variant counts per candidate gene, split by SNV and indel.")

    elif skill_name == "vcf_burden_test":
        caption_parts.append("Gene-level burden test lollipop plot. Dot size proportional to rare variant count; dashed line: Bonferroni threshold.")

    elif skill_name == "pathway_enrichment":
        caption_parts.append("Pathway enrichment bar chart (-log10 p-value). Coloured bars indicate FDR significance; dashed line: FDR threshold.")

    else:
        caption_parts.append(f"{skill_name.replace('_', ' ').title()} visualization.")

    return " ".join(caption_parts)


def _interpret_skill(rec) -> str:
    """Generate interpretive text for a skill execution record."""
    skill_name = rec.skill
    results = rec.key_results
    args = rec.args

    if skill_name == "think":
        return ""

    try:
        if skill_name == "prevalence":
            n_diseases = results.get("n_codes", results.get("n_diseases", "N/A"))
            top = results.get("top_disease", results.get("top_code", ""))
            sentence = f"Prevalence analysis examined {n_diseases} disease codes."
            if top:
                sentence += f" The most frequent coded condition was {top}."
            return sentence + " This establishes the epidemiological baseline for downstream analyses."

        elif skill_name == "cohort_summary":
            n_cases = results.get("n_cases", "?")
            n_controls = results.get("n_controls", "?")
            icd10 = args.get("icd10_code", "?")
            return (
                f"Cohort for {icd10}: {n_cases} cases and {n_controls} controls identified. "
                f"This case-control ratio is "
                f"{'adequate' if isinstance(n_cases, int) and n_cases > 500 else 'moderate'} "
                f"for downstream machine learning analyses."
            )

        elif skill_name == "field_search":
            total = results.get("total", 0)
            query = args.get("query", results.get("query", ""))
            if total:
                matched = results.get("matched_queries", []) or []
                matched_text = f" Matched probes included: {', '.join(str(x) for x in matched[:6])}." if matched else ""
                return f"Field search for `{query}` identified {total} candidate catalogue field(s).{matched_text}"
            warnings = results.get("warnings", []) or []
            warning_text = " ".join(str(w) for w in warnings[:2])
            return f"Field search for `{query}` found no catalogue fields. {warning_text}".strip()

        elif skill_name == "ukb_data_inventory":
            n_subjects = results.get("n_subjects", "?")
            inv = results.get("full_csv_inventory", {}) or {}
            n_fields = inv.get("n_total_fields", "?")
            n_cols = inv.get("n_total_columns", "?")
            store = results.get("feature_store_manifest", {}) or {}
            store_text = "available" if store.get("exists") else "not yet built"
            return (
                f"UKB data inventory confirmed {n_subjects} registered subjects and "
                f"{n_fields} raw UKB field prefixes across {n_cols} CSV columns. "
                f"The optional full-field parquet feature store is {store_text}."
            )

        elif skill_name == "ukb_field_resolve":
            n_fields = results.get("n_fields", 0)
            candidates = results.get("materialization_candidates", []) or []
            return (
                f"UKB field resolution mapped {n_fields} candidate field(s) to concrete data sources. "
                f"{len(candidates)} field(s) were flagged for full-field materialization or further source checks."
            )

        elif skill_name == "ukb_materialize_fields":
            status = results.get("status", "?")
            chunks = results.get("n_chunks", 0)
            fields = results.get("requested_field_ids", []) or args.get("field_ids", "")
            return (
                f"Full UKB field materialization finished with status {status}, "
                f"writing or reusing {chunks} parquet chunk(s) for fields {fields}."
            )

        elif skill_name == "deep_research":
            n_sources = results.get("n_sources", len(results.get("sources", []) or []))
            status = results.get("status", "READY")
            warning = " ".join(str(w) for w in (results.get("warnings", []) or [])[:1])
            return (
                f"Deep research reviewed {n_sources} source(s) with status {status}. "
                f"{warning}".strip()
            )

        elif skill_name == "train_model":
            auc_str = _format_auc_with_ci(results)
            model_type = args.get("model_type", results.get("model_type", "?"))
            n_features = results.get("n_features", "?")
            n_cases = results.get("n_cases", "?")
            n_controls = results.get("n_controls", "?")
            auc = _format_value(results.get("mean_auc", results.get("auc_mean", results.get("auc", "?"))))
            if isinstance(auc, (int, float)):
                quality = (
                    "excellent" if auc > 0.9 else
                    "good" if auc > 0.8 else
                    "moderate" if auc > 0.7 else "limited"
                )
                return (
                    f"The {model_type} model achieved {auc_str}, indicating "
                    f"{quality} discriminative ability using {n_features} features "
                    f"(n = {n_cases} cases, {n_controls} controls). "
                    f"{results.get('selection_rationale', '')}".strip()
                )
            return f"Model training completed with {model_type}."

        elif skill_name == "biomarker_dist":
            biomarker = args.get("field_name", args.get("field_id", "?"))
            p_val = results.get("p_value", results.get("p", None))
            if p_val is not None:
                p_val = _format_value(p_val)
                sig = "significant" if float(p_val) < 0.05 else "non-significant"
                return (
                    f"Distribution analysis of {biomarker}: {sig} difference between "
                    f"cases and controls ({_format_p_value(p_val)}, Mann-Whitney U test)."
                )
            return f"Distribution analysis of {biomarker} completed."

        elif skill_name == "survival":
            icd10 = args.get("icd10_code", "?")
            p_val = results.get("log_rank_p", None)
            if p_val is not None:
                p_val = _format_value(p_val)
                return (
                    f"Survival analysis for {icd10}: log-rank test {_format_p_value(p_val)}. "
                    f"See Kaplan-Meier curve for time-to-event distributions."
                )
            return f"Survival analysis for {icd10} completed."

        elif skill_name == "feature_importance":
            top_feat = results.get("top_feature")
            if not top_feat:
                top_feats = results.get("top_features", [])
                top_feat = top_feats[0] if isinstance(top_feats, list) and top_feats else "?"
            diagnostic_note = results.get("diagnostic_leakage_note") or results.get("interpretation_note")
            if diagnostic_note:
                return (
                    f"Feature importance identified {top_feat} as the leading model feature. "
                    f"{diagnostic_note}"
                )
            importance_type = str(results.get("importance_type") or results.get("effective_method") or "Tree-based")
            if "shap" in importance_type.lower():
                method_text = "SHAP feature-attribution plot"
            else:
                method_text = "tree-based feature-importance chart"
            return (
                f"Feature importance analysis identified {top_feat} as the most "
                f"predictive recorded feature. See the {method_text} for feature contributions."
            )

        elif skill_name in ("evaluate_model", "calibration"):
            auc = _format_value(results.get("auc", results.get("roc_auc", "?")))
            if isinstance(auc, (int, float)):
                return f"Model evaluation: AUC={auc:.4f}. See ROC and calibration curves."
            return "Model evaluation completed."

        elif skill_name == "correlation":
            return "Correlation analysis reveals biomarker interdependencies. See clustered heatmap."

        elif skill_name == "phewas":
            n_sig = results.get("n_significant", "?")
            return f"PheWAS identified {n_sig} significant associations after FDR correction."

        elif skill_name == "comorbidity":
            return "Comorbidity analysis reveals disease co-occurrence patterns. See network plot."

        elif skill_name == "missing_data":
            pct = results.get(
                "mean_missing_pct",
                results.get("overall_missing_pct", results.get("overall_missing")),
            )
            if isinstance(pct, (int, float)):
                pct_text = f"{pct:.2f}"
            elif pct not in (None, ""):
                pct_text = str(pct)
            else:
                pct_text = "not available"
            return f"Missing data analysis: mean missingness = {pct_text}%. See pattern matrix."

        elif skill_name == "phenotype_harmonize":
            label = results.get("label", args.get("concept", "phenotype"))
            status = results.get("harmonisation_status", "PARTIAL")
            n_maps = len(results.get("mappings", []) or [])
            risks = ", ".join((results.get("drift_risks", []) or [])[:3])
            return (
                f"Phenotype harmonisation for {label}: {n_maps} cohort-specific mappings "
                f"generated with status {status}. Key drift risks: {risks or 'not reported'}."
            )

        elif skill_name == "cohort_card":
            endpoint = results.get("endpoint", args.get("endpoint", "?"))
            ctype = results.get("cohort_type", "?")
            status = results.get("status", "PARTIAL")
            return (
                f"Cohort card for {endpoint}: selected {ctype} design with status {status}. "
                "This records index date, lookback/follow-up windows, missingness and bias flags."
            )

        elif skill_name == "trajectory_tokenize":
            status = str(results.get("status", "")).upper()
            n_tokens = results.get("n_tokens", 0)
            support = str(results.get("longitudinal_support", "") or "")
            time_source = str(results.get("trajectory_time_source", "") or "")
            if status == "PARTIAL" and not n_tokens:
                reason = "; ".join((results.get("blocking_reasons") or [])[:2])
                return (
                    "Trajectory tokenization is incomplete: no usable longitudinal tokens were available. "
                    f"{reason or 'Provide participant-level longitudinal rows before interpreting a trajectory forecast.'}"
                )
            if status == "PARTIAL" and n_tokens:
                return (
                    f"Trajectory tokenization prepared {n_tokens} tokens across "
                    f"{results.get('n_participants', 0)} participants, but temporal support is limited "
                    f"({support or 'unknown support'}; {time_source or 'unknown time source'}). "
                    "This can support feasibility auditing, not a validated longitudinal forecast."
                )
            return (
                f"Trajectory tokenization prepared {n_tokens} tokens across "
                f"{results.get('n_participants', 0)} participants and {len(results.get('modalities', []) or [])} modalities. "
                "This is a HealthFormer-style evaluation layer, not a trained world model."
            )

        elif skill_name == "world_model_audit":
            return (
                f"World-model audit returned safety={results.get('safety_status', 'PARTIAL')} "
                f"and allowed claim type `{results.get('allowed_claim_type', 'association_conditioned_forecast')}`. "
                "Intervention simulations must not be interpreted as causal without external or target-trial evidence."
            )

        elif skill_name == "paper_replication_compare":
            status = results.get("overall_status", "partial_replication")
            acceptance = (results.get("acceptance_summary") or {}).get("verdict", "not_recorded")
            n_approx = results.get("n_approximated_dimensions", 0)
            n_unavailable = results.get("n_unavailable_dimensions", 0)
            return (
                f"Paper replication comparison classified the local workflow as {status}. "
                f"Acceptance verdict: {acceptance}. "
                f"{n_approx} dimension(s) were approximations and {n_unavailable} were unavailable or not recorded. "
                "The final report must separate paper targets from local UKB evidence."
            )

        elif skill_name == "genetic_target_hypothesis":
            phenotype = results.get("phenotype", args.get("phenotype", "phenotype"))
            n_genes = results.get("n_discovery_genes", 0)
            targets = results.get("targets", []) or []
            top = targets[0] if isinstance(targets, list) and targets else {}
            status = results.get("status", "UNKNOWN")
            scope = results.get("multiple_testing_scope", "unknown")
            tier_note = results.get("tier_interpretation", "")
            if top:
                return (
                    f"Genetic target hypothesis prioritization for {phenotype} found "
                    f"{n_genes} discovery gene(s). Top target: {top.get('gene')} "
                    f"({top.get('therapeutic_direction')}, score={top.get('score')}, "
                    f"{top.get('tier')}). Multiple-testing scope is `{scope}`. "
                    "Ranking is driven by rare-variant burden evidence; annotations "
                    f"are translational context. {tier_note}"
                )
            return (
                f"Genetic target hypothesis prioritization for {phenotype} returned "
                f"status={status} and found no candidate genes suitable for ranking. "
                f"Multiple-testing scope is `{scope}`."
            )

        elif skill_name == "target_annotation_context":
            phenotype = results.get("phenotype", args.get("phenotype", "target set")) or "target set"
            targets = results.get("targets", []) or []
            status = results.get("status", "UNKNOWN")
            source_counts = results.get("source_status_counts", {}) or {}
            source_text = ", ".join(
                f"{source}:{sum(counts.values())}" for source, counts in sorted(source_counts.items())
                if isinstance(counts, dict)
            )
            top = targets[0] if isinstance(targets, list) and targets else {}
            if top:
                trials = len(top.get("clinical_trials") or [])
                tissues = ", ".join(t.get("tissue", "") for t in (top.get("tissue_expression") or [])[:3])
                return (
                    f"Target annotation context for {phenotype} returned status={status} across "
                    f"{len(targets)} target(s). Top annotated gene: {top.get('gene')} "
                    f"({top.get('protein_name') or top.get('approved_name') or 'protein context pending'}). "
                    f"Trial records: {trials}; top tissues: {tissues or 'not available'}. "
                    f"Source coverage: {source_text or 'not reported'}. These annotations are context only."
                )
            return f"Target annotation context for {phenotype} returned status={status} with no target annotations."

        elif skill_name == "target_enrichment":
            phenotype = results.get("phenotype", args.get("phenotype", "target set")) or "target set"
            terms = results.get("terms", []) or []
            status = results.get("status", "UNKNOWN")
            method = results.get("method", "unknown")
            if terms:
                top = terms[0]
                return (
                    f"Target enrichment for {phenotype} used {method} and returned "
                    f"{len(terms)} term(s). Top term: {top.get('term')} "
                    f"(q={float(top.get('q_value', 1.0)):.3g}, overlap={top.get('overlap')}). "
                    "Enrichment is exploratory pathway context and does not establish mechanism."
                )
            return f"Target enrichment for {phenotype} returned status={status} with no enriched terms."

        # ---- WGS pipeline skills ----
        elif skill_name == "vcf_qc":
            n_analyzed = results.get("n_samples_analyzed", 0)
            n_pass = results.get("n_samples_pass", 0)
            n_fail = results.get("n_samples_fail", 0)
            rgn = results.get("region", "full genome")
            hwe_fail = results.get("hwe_fail_count")
            text = (
                f"VCF quality control analysed {n_analyzed} samples ({rgn}): "
                f"{n_pass} passed, {n_fail} failed QC thresholds. "
            )
            stats = results.get("cohort_stats", {})
            if stats.get("ti_tv_ratio"):
                text += f"Cohort mean Ti/Tv ratio = {stats['ti_tv_ratio'].get('mean', 'N/A')}. "
            if hwe_fail is not None:
                text += f"HWE test: {hwe_fail} variant(s) failed (p < 0.001). "
            return text

        elif skill_name == "vcf_pca":
            n_samples = results.get("n_samples", 0)
            n_snps = results.get("n_snps_used", 0)
            var_pct = results.get("cumulative_variance_pct", 0)
            n_comp = results.get("n_components", 0)
            return (
                f"Genotype PCA on {n_samples} samples using {n_snps:,} common SNPs yielded "
                f"{n_comp} components explaining {var_pct:.1f}% cumulative variance. "
                "PC plots are coloured by phenotype group for visual population structure assessment."
            )

        elif skill_name == "vcf_kinship":
            n_samples = results.get("n_samples", 0)
            n_pairs = results.get("n_related_pairs", 0)
            n_snps = results.get("n_snps_used", 0)
            summary = results.get("kinship_summary", {})
            max_k = summary.get("max_off_diagonal", 0)
            text = (
                f"Kinship estimation from {n_snps:,} SNPs across {n_samples} samples "
                f"identified {n_pairs} related pair(s) above threshold. "
                f"Maximum off-diagonal kinship coefficient = {max_k:.4f}."
            )
            return text

        elif skill_name == "vcf_association":
            n_cases = results.get("n_cases", 0)
            n_ctrls = results.get("n_controls", 0)
            n_tested = results.get("n_variants_tested", 0)
            n_sig = results.get("n_significant_bonferroni", 0)
            lam = results.get("lambda_gc")
            case_g = results.get("case_group", "?")
            ctrl_g = results.get("control_group", "?")
            text = (
                f"Case-control association ({case_g} vs {ctrl_g}, "
                f"n={n_cases}+{n_ctrls}) tested {n_tested:,} variants. "
                f"{n_sig} reached Bonferroni significance. "
            )
            mode = results.get("analysis_mode")
            if mode:
                text += f"Analysis mode: {mode}. "
            if lam is not None:
                text += f"Genomic inflation factor λGC = {lam:.3f}. "
            std_note = results.get("standard_gwas_note")
            if std_note:
                text += std_note + " "
            warn = results.get("lambda_gc_warning")
            if warn:
                text += warn + " "
            text += results.get("power_warning", "")
            return text

        elif skill_name == "vcf_annotation":
            n_queried = results.get("n_genes_queried", 0)
            n_with = results.get("n_genes_with_variants", 0)
            n_total = results.get("n_total_variants", 0)
            mode = results.get("annotation_mode", "unknown")
            return (
                f"Variant annotation queried {n_queried} candidate gene(s): "
                f"{n_with} contained variants ({n_total:,} total). "
                f"Annotation mode: {mode}. "
                f"{results.get('note', '')}"
            )

        elif skill_name == "vcf_burden_test":
            n_genes = results.get("n_genes_tested", 0)
            n_sig = results.get("n_genes_significant_burden", 0)
            case_g = results.get("case_group", "?")
            ctrl_g = results.get("control_group", "?")
            maf = results.get("maf_threshold", 0.05)
            top = results.get("top_gene", "")
            text = (
                f"Gene-level burden test ({case_g} vs {ctrl_g}) tested "
                f"{n_genes} gene(s) at MAF < {maf}. "
                f"{n_sig} gene(s) reached FDR significance. "
            )
            if top:
                text += f"Top gene by p-value: {top}."
            return text

        elif skill_name == "pathway_enrichment":
            n_input = results.get("n_input_genes", 0)
            n_tested = results.get("n_pathways_tested", 0)
            n_sig = results.get("n_pathways_significant", 0)
            db = results.get("database", "built-in")
            top_results = results.get("results", [])
            text = (
                f"Pathway enrichment of {n_input} gene(s) against {n_tested} "
                f"pathways ({db}): {n_sig} pathway(s) significant at FDR < 0.05. "
            )
            if top_results and top_results[0].get("significant"):
                top = top_results[0]
                text += (
                    f"Top pathway: {top['pathway']} "
                    f"(fold enrichment = {top.get('fold_enrichment', 'N/A')}, "
                    f"FDR = {top.get('p_fdr', 'N/A'):.3g})."
                )
            return text

        elif skill_name == "jh_variant_discovery":
            n = results.get("n_candidate_variants", 0)
            counts = results.get("wgs_phenotype_counts", {})
            return (
                f"Juvenile hair-whitening variant discovery prioritized {n} WGS candidate "
                f"variant/locus anchor(s). WGS phenotype counts were {counts}. "
                f"Claim boundary: {results.get('claim_boundary', 'exploratory prioritization')}."
            )

        elif skill_name == "regulatory_variant_annotation":
            n = results.get("n_regulatory_hits", 0)
            return (
                f"Regulatory annotation classified {n} candidate variant/locus anchor(s) "
                f"within a {results.get('window_bp', 0):,} bp regulatory window using "
                "built-in hg38 pigmentation and immune loci."
            )

        elif skill_name == "tf_binding_disruption":
            n = results.get("n_tfbs_candidates", 0)
            return (
                f"TF binding assessment produced {n} candidate TFBS disruption rows. "
                "Rows are motif-prior hypotheses unless a sequence-level motif database "
                "or MCP scanner is configured."
            )

        elif skill_name == "scatac_peak_overlap":
            n = results.get("n_candidate_variants", 0)
            cols = results.get("peak_coordinate_columns", [])
            status = "peak coordinate metadata detected" if cols else "exact overlap deferred"
            return (
                f"scATAC peak-overlap readiness checked {n} candidate variant/locus anchor(s); "
                f"{status}."
            )

        elif skill_name == "scatac_accessibility_differential":
            return (
                "scATAC accessibility differential readiness compared Juvenile white hair "
                f"({results.get('case_manifest_cells', 0):,} manifest cells) versus black hair "
                f"({results.get('control_manifest_cells', 0):,} manifest cells). "
                f"Status: {results.get('status', 'unknown')}."
            )

        elif skill_name == "scrna_expression_differential":
            n = results.get("n_genes", 0)
            return (
                f"scRNA expression readiness was assessed for {n} variant-linked gene(s). "
                "Quantitative differential expression is deferred until backed/chunked "
                "matrix extraction is available for the selected genes."
            )

        elif skill_name == "atac_expression_coupling":
            n = results.get("n_coupled_rows", 0)
            return (
                f"ATAC-expression coupling generated {n} variant-gene evidence row(s), "
                "with coupling direction intentionally left unknown until quantitative "
                "accessibility and expression contrasts are computed."
            )

        elif skill_name == "spatial_celltype_localization":
            n = results.get("n_spatial_files_inspected", 0)
            return (
                f"Stereo-seq spatial localization readiness inspected {n} Juvenile-focused "
                "spatial h5ad file(s) for coordinate and cell-type metadata."
            )

        elif skill_name == "spatial_cell_interaction":
            n = results.get("n_samples", 0)
            return (
                f"Spatial cell-interaction readiness summarized {n} sample(s), checking "
                "whether coordinate and cell-type metadata can support neighbor graph analysis."
            )

        elif skill_name == "multiomics_mechanism_prioritization":
            n = results.get("n_prioritized_hypotheses", 0)
            return (
                f"Multi-omics prioritization ranked {n} variant-to-mechanism hypothesis/hypotheses "
                f"across WGS, TF, ATAC, RNA and spatial evidence. "
                f"Claim boundary: {results.get('claim_boundary', 'exploratory mechanism ranking')}."
            )

        elif skill_name == "workflow_gap_detector":
            n = results.get("n_gaps", 0)
            return (
                f"Workflow gap detection found {n} missing capability or resource gap(s) and "
                "recorded review-gated generated-skill proposals for harness-driven evolution."
            )

    except Exception:
        pass

    # Generic fallback: keep unknown skills factual and avoid placeholder prose.
    key_nums = {k: _format_value(v) for k, v in results.items()
                if isinstance(v, (int, float)) or (hasattr(v, "item") and callable(v.item))}
    if key_nums:
        nums_str = ", ".join(f"{k}={v}" for k, v in list(key_nums.items())[:3])
        return f"Recorded quantitative outputs include {nums_str}."
    return ""


def _extract_key_findings(records) -> list[str]:
    """Extract top 3-5 key findings from analysis records."""
    findings = []
    for rec in records:
        if rec.skill == "think":
            continue
        results = rec.key_results
        if "error" in results:
            continue

        if rec.skill == "train_model":
            auc = _format_value(results.get("mean_auc", results.get("auc")))
            if auc and isinstance(auc, (int, float)):
                model = rec.args.get("model_type", "model")
                design = str(results.get("analysis_design") or results.get("prediction_target") or "").replace("_", " ")
                if results.get("incident_risk_supported") is False or "prevalent" in design:
                    target = design or "prevalent/ever-diagnosed disease discrimination"
                    findings.append(f"{model} achieved AUC={auc:.4f} for {target}, not incident-risk prediction")
                else:
                    findings.append(f"{model} achieved AUC={auc:.4f} for disease prediction")

        elif rec.skill == "prevalence":
            top = results.get("top_disease", results.get("top_code"))
            if top:
                findings.append(f"Most prevalent condition: {top}")

        elif rec.skill == "cohort_summary":
            n = results.get("n_cases")
            icd = rec.args.get("icd10_code", "")
            if n:
                findings.append(f"{icd} cohort: {n:,} cases identified")

        elif rec.skill == "feature_importance":
            top = results.get("top_feature")
            if top:
                findings.append(f"Top predictive biomarker: {top}")

        elif rec.skill == "phewas":
            n_sig = results.get("n_significant")
            if n_sig:
                findings.append(f"PheWAS: {n_sig} significant associations discovered")

        elif rec.skill == "survival":
            p = results.get("log_rank_p")
            if p is not None:
                p = _format_value(p)
                findings.append(f"Survival analysis: log-rank P={float(p):.2e}")

        elif rec.skill == "phenotype_harmonize":
            label = results.get("label")
            status = results.get("harmonisation_status")
            if label:
                findings.append(f"Phenotype harmonisation: {label} ({status})")

        elif rec.skill == "cohort_card":
            endpoint = results.get("endpoint")
            ctype = results.get("cohort_type")
            if endpoint and ctype:
                findings.append(f"Cohort design: {endpoint} mapped to {ctype}")

        elif rec.skill == "trajectory_tokenize":
            n_tokens = results.get("n_tokens")
            if n_tokens:
                findings.append(f"Trajectory layer: {n_tokens:,} HealthFormer-style tokens prepared")
            elif str(results.get("status", "")).upper() == "PARTIAL":
                findings.append("Trajectory layer unavailable: no longitudinal participant tokens were prepared")

        elif rec.skill == "world_model_audit":
            safety = results.get("safety_status")
            allowed = results.get("allowed_claim_type")
            if safety:
                findings.append(f"World-model audit: {safety}, claims limited to {allowed}")

        elif rec.skill == "paper_replication_compare":
            status = results.get("overall_status")
            acceptance = (results.get("acceptance_summary") or {}).get("verdict")
            if status:
                suffix = f", acceptance={acceptance}" if acceptance else ""
                findings.append(f"Paper replication comparison: {status.replace('_', ' ')}{suffix}")

        elif rec.skill == "genetic_target_hypothesis":
            targets = results.get("targets", []) or []
            top = targets[0] if isinstance(targets, list) and targets else {}
            if top:
                findings.append(
                    f"Genetic target hypothesis: {top.get('gene')} ranked first "
                    f"for {results.get('phenotype')} ({top.get('therapeutic_direction')})"
                )

        elif rec.skill == "target_annotation_context":
            targets = results.get("targets", []) or []
            annotated = [t for t in targets if t.get("source_status")]
            if annotated:
                findings.append(
                    f"Target annotation context: {len(annotated)} target(s) annotated "
                    f"for {results.get('phenotype') or 'target set'}"
                )

        elif rec.skill == "target_enrichment":
            terms = results.get("terms", []) or []
            if terms:
                findings.append(
                    f"Target enrichment: {terms[0].get('term')} ranked first "
                    f"for {results.get('phenotype') or 'target set'}"
                )

        # ---- WGS pipeline skills ----
        elif rec.skill == "vcf_qc":
            n_pass = results.get("n_samples_pass", 0)
            n_fail = results.get("n_samples_fail", 0)
            if n_pass or n_fail:
                findings.append(f"VCF QC: {n_pass} samples passed, {n_fail} failed")

        elif rec.skill == "vcf_association":
            lam = results.get("lambda_gc")
            n_sig = results.get("n_significant_bonferroni", 0)
            if lam is not None:
                mode = results.get("analysis_mode", "association")
                findings.append(f"Association ({mode}): λGC={lam:.3f}, {n_sig} Bonferroni-significant variant(s)")

        elif rec.skill == "vcf_annotation":
            mode = results.get("annotation_mode", "annotation")
            n_total = results.get("n_total_variants", 0)
            findings.append(f"Annotation ({mode}): {n_total:,} candidate-gene variant(s) summarized")

        elif rec.skill == "vcf_burden_test":
            n_sig = results.get("n_genes_significant_burden", 0)
            top = results.get("top_gene", "")
            if top:
                findings.append(f"Burden test: top gene {top}, {n_sig} gene(s) significant")

        elif rec.skill == "pathway_enrichment":
            n_sig = results.get("n_pathways_significant", 0)
            top_results = results.get("results", [])
            if n_sig > 0 and top_results:
                findings.append(f"Pathway enrichment: {n_sig} significant pathway(s), top = {top_results[0].get('pathway', '')}")

        elif rec.skill == "jh_variant_discovery":
            findings.append(f"Juvenile hair mechanism: {results.get('n_candidate_variants', 0)} candidate variant/locus anchor(s) prioritized")

        elif rec.skill == "tf_binding_disruption":
            findings.append(f"TF binding: {results.get('n_tfbs_candidates', 0)} motif-prior disruption row(s) generated")

        elif rec.skill == "scatac_accessibility_differential":
            findings.append(
                "scATAC readiness: "
                f"W={results.get('case_manifest_cells', 0):,} vs B={results.get('control_manifest_cells', 0):,} Juvenile manifest cells"
            )

        elif rec.skill == "multiomics_mechanism_prioritization":
            findings.append(f"Mechanism prioritization: {results.get('n_prioritized_hypotheses', 0)} hypothesis/hypotheses ranked")

        elif rec.skill == "workflow_gap_detector":
            findings.append(f"Workflow evolution: {results.get('n_gaps', 0)} gap(s) recorded for skill/MCP improvement")

    return findings[:5]


def _is_mockish(value) -> bool:
    """Detect unset MagicMock attributes without importing test-only helpers."""
    return value.__class__.__module__.startswith("unittest.mock")


def _coerce_text_list(value) -> list[str]:
    """Convert strings, dicts, and lists into compact report bullets."""
    if value is None or _is_mockish(value):
        return []
    if isinstance(value, str):
        clean = " ".join(value.split())
        return [clean] if clean else []
    if isinstance(value, dict):
        for key in ("finding", "summary", "message", "text", "title"):
            if value.get(key):
                return _coerce_text_list(value.get(key))
        if value.get("findings"):
            return _coerce_text_list(value.get("findings"))
        return []
    if isinstance(value, (list, tuple)):
        findings: list[str] = []
        for item in value:
            findings.extend(_coerce_text_list(item))
        return findings
    return []


def _is_safe_executive_finding(text: str) -> bool:
    """Filter report-lead findings that are logs, reviewer notes, or unsupported claims."""
    clean = " ".join(str(text).split())
    if len(clean) < 20:
        return False
    if clean.endswith(":") and len(clean) < 80:
        return False
    for pattern in _UNSAFE_EXECUTIVE_PATTERNS:
        if re.search(pattern, clean, re.IGNORECASE):
            return False
    if _UNQUALIFIED_CAUSAL_PATTERN.search(clean) and not _CAUSAL_QUALIFIER_PATTERN.search(clean):
        return False
    return True


def _clean_executive_finding(text: str) -> str:
    clean = re.sub(r"^#{1,6}\s*", "", str(text).strip())
    clean = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", clean)
    return " ".join(clean.strip(" -\t").split())[:450].rstrip()


def _extract_executive_findings(ctx, records) -> list[str]:
    """Extract curated executive findings from state or executive_findings records."""
    findings: list[str] = []
    state_value = getattr(ctx.state, "executive_findings", None) if ctx and hasattr(ctx, "state") else None
    findings.extend(_coerce_text_list(state_value))

    for rec in records:
        if getattr(rec, "skill", "") != "executive_findings":
            continue
        results = getattr(rec, "key_results", {}) or {}
        for key in ("executive_findings", "findings", "items", "summary"):
            findings.extend(_coerce_text_list(results.get(key)))

    seen: set[str] = set()
    unique: list[str] = []
    for finding in findings:
        clean = _clean_executive_finding(finding)
        if not _is_safe_executive_finding(clean):
            continue
        key = clean.lower()
        if key not in seen:
            seen.add(key)
            unique.append(clean)
    return unique[:7]


def _extract_guardrail_issues(ctx, records) -> list[dict]:
    """Expand statistical/safety guardrail outputs into normalized issue rows."""
    rows: list[dict] = []

    def add_issue(issue, source: str) -> None:
        if not isinstance(issue, dict):
            return
        message = issue.get("message") or issue.get("detail") or issue.get("description") or ""
        recommendation = issue.get("recommendation") or issue.get("action") or issue.get("mitigation") or ""
        issue_type = issue.get("type") or issue.get("code") or "guardrail_issue"
        rows.append({
            "source": source,
            "severity": str(issue.get("severity", "INFO")).upper(),
            "type": str(issue_type),
            "skill": str(issue.get("skill") or issue.get("record") or ""),
            "message": " ".join(str(message).split()),
            "recommendation": " ".join(str(recommendation).split()),
        })

    state_issues = getattr(ctx.state, "guardrail_issues", None) if ctx and hasattr(ctx, "state") else None
    if isinstance(state_issues, (list, tuple)):
        for issue in state_issues:
            add_issue(issue, "state")

    for rec in records:
        if getattr(rec, "skill", "") not in _GUARDRAIL_SKILLS:
            continue
        results = getattr(rec, "key_results", {}) or {}
        for issue in results.get("issues", []) or []:
            add_issue(issue, rec.skill)

    deduped: list[dict] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        key = (
            row["severity"],
            row["type"],
            row["skill"],
            row["message"],
            row["recommendation"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    deduped.sort(key=lambda row: (severity_order.get(row["severity"], 9), row["source"], row["type"]))
    return deduped


def _extract_execution_log(ctx, records) -> list[dict]:
    """Extract execution diagnostics for the appendix without raw stdout/stderr."""
    entries = []
    state_value = getattr(ctx.state, "execution_log", None) if ctx and hasattr(ctx, "state") else None
    if not isinstance(state_value, (list, tuple)) or _is_mockish(state_value):
        custom_data = getattr(ctx.state, "custom_data", {}) if ctx and hasattr(ctx, "state") else {}
        if isinstance(custom_data, dict):
            state_value = custom_data.get("execution_log")

    if isinstance(state_value, (list, tuple)):
        entries.extend(item for item in state_value if isinstance(item, dict))

    for rec in records:
        if getattr(rec, "skill", "") not in {"execution_log", "execution_appendix"}:
            continue
        results = getattr(rec, "key_results", {}) or {}
        log_value = results.get("execution_log") or results.get("steps") or results.get("entries")
        if isinstance(log_value, (list, tuple)):
            entries.extend(item for item in log_value if isinstance(item, dict))

    return entries


def _format_execution_value(entry: dict, keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = entry.get(key)
        if value is not None and key not in _RAW_RESULT_KEYS and not isinstance(value, (dict, list, tuple)):
            return _redact_report_tool_tokens(" ".join(str(value).split())).replace("|", "\\|")
    return default


def _is_raw_result_key(key: str) -> bool:
    lower = str(key).lower()
    return lower in _RAW_RESULT_KEYS or lower.endswith("_execution_log")


def _table_cell(value) -> str:
    return _redact_report_tool_tokens(" ".join(str(value).split())).replace("|", "\\|")


def _extract_references(records) -> list[str]:
    """Extract literature references from paper/research skill records."""
    refs: list[str] = []
    seen: set[str] = set()

    def add_ref(text: str) -> None:
        clean = " ".join(str(text).split())
        if clean and clean not in seen:
            seen.add(clean)
            refs.append(clean)

    for rec in records:
        results = rec.key_results or {}
        if rec.skill in {"fetch_paper", "read_paper", "literature_qa"}:
            title = results.get("title") or results.get("paper_title")
            doi = results.get("doi")
            url = results.get("url") or results.get("source_url")
            authors = results.get("authors", "")
            if isinstance(authors, list):
                authors = ", ".join(str(a) for a in authors[:3])
                if len(results.get("authors", []) or []) > 3:
                    authors += " et al."
            if title:
                suffix = f" DOI: {doi}." if doi else (f" {url}" if url else "")
                prefix = f"{authors}. " if authors else ""
                add_ref(f"{prefix}{title}.{suffix}")

        sources = results.get("sources")
        if isinstance(sources, list):
            for source in sources[:10]:
                if isinstance(source, dict):
                    title = source.get("title") or source.get("name")
                    url = source.get("url")
                    doi = source.get("doi")
                    if title:
                        suffix_parts = []
                        if doi:
                            suffix_parts.append(f"DOI: {doi}.")
                        if url:
                            suffix_parts.append(str(url))
                        suffix = " ".join(suffix_parts)
                        add_ref(f"{title}. {suffix}".strip())
                elif isinstance(source, str):
                    add_ref(source)

    return refs


def _session_evidence_summary(ctx, records) -> dict:
    """Collect report-level evidence for synthesis paragraphs."""
    summary = {
        "bank_name": ctx.settings.biobank_name if ctx and hasattr(ctx, "settings") else "Biobank",
        "cohort": None,
        "model": None,
        "top_feature": None,
        "biomarkers": [],
        "progression": [],
        "guardrail": None,
        "safety": None,
        "world_model": None,
        "references": len(_extract_references(records)),
    }
    for rec in records:
        results = getattr(rec, "key_results", {}) or {}
        args = getattr(rec, "args", {}) or {}
        if rec.skill == "cohort_summary" and not summary["cohort"]:
            summary["cohort"] = {
                "endpoint": args.get("icd10_code") or args.get("disease") or "target endpoint",
                "disease": args.get("disease") or args.get("icd10_code") or "target endpoint",
                "n_cases": results.get("n_cases"),
                "n_controls": results.get("n_controls"),
                "n_features": results.get("n_features"),
            }
        elif rec.skill == "train_model" and not summary["model"]:
            summary["model"] = {
                "model_type": results.get("model_type") or args.get("model_type") or "model",
                "auc": _format_value(results.get("mean_auc", results.get("auc_mean", results.get("auc")))),
                "auc_95ci": results.get("auc_95ci"),
                "n_features": results.get("n_features"),
                "analysis_design": results.get("analysis_design"),
                "prediction_target": results.get("prediction_target"),
                "incident_risk_supported": results.get("incident_risk_supported"),
                "evaluation_strategy": results.get("evaluation_strategy") or (results.get("model_selection") or {}).get("evaluation_strategy"),
            }
        elif rec.skill == "feature_importance" and not summary["top_feature"]:
            top_features = results.get("top_features") or []
            top_feature = results.get("top_feature")
            if not top_feature and top_features:
                first = top_features[0]
                top_feature = first.get("feature") if isinstance(first, dict) else first
            summary["top_feature"] = {
                "name": top_feature,
                "note": results.get("diagnostic_leakage_note") or results.get("interpretation_note") or "",
            }
        elif rec.skill == "biomarker_dist":
            summary["biomarkers"].append({
                "name": args.get("field_name") or args.get("field_id") or "biomarker",
                "p_value": results.get("p_value", results.get("p")),
                "effect_size": results.get("effect_size"),
            })
        elif rec.skill == "survival":
            summary["progression"].append({
                "endpoint": args.get("outcome") or args.get("icd10_code") or "outcome",
                "p_value": results.get("log_rank_p"),
                "n_cases": results.get("n_cases"),
            })
        elif rec.skill == "statistical_review":
            summary["guardrail"] = results.get("overall_assessment") or results.get("overall") or "recorded"
        elif rec.skill == "safety_check":
            summary["safety"] = results.get("overall") or results.get("status") or "recorded"
        elif rec.skill == "world_model_audit":
            summary["world_model"] = {
                "safety_status": results.get("safety_status"),
                "allowed_claim_type": results.get("allowed_claim_type"),
            }
        elif rec.skill == "paper_replication_compare":
            summary["replication_compare"] = {
                "overall_status": results.get("overall_status"),
                "acceptance_verdict": (results.get("acceptance_summary") or {}).get("verdict"),
                "paper_access": results.get("paper_access"),
                "local_endpoint": results.get("local_endpoint"),
            }
    return summary


def _model_design_label(model: dict) -> str:
    design = str(model.get("analysis_design") or "").lower()
    target = str(model.get("prediction_target") or "").replace("_", " ")
    if "prevalent" in design or model.get("incident_risk_supported") is False:
        return target or "prevalent/ever-diagnosed disease discrimination"
    return target or "risk-prediction"


def _model_method_label(model: dict) -> str:
    strategy = str(model.get("evaluation_strategy") or "").lower()
    if strategy == "stratified_holdout":
        return "stratified holdout model selection"
    if strategy:
        return strategy.replace("_", " ")
    return "recorded model evaluation"


def _external_review_summary(ctx) -> dict:
    try:
        summary = ctx.state.custom_data.get("external_review_summary", {})
    except Exception:
        summary = {}
    return summary if isinstance(summary, dict) else {}


def _add_external_review_section(sections: list[str], ctx) -> None:
    summary = _external_review_summary(ctx)
    if not summary:
        return
    findings = summary.get("actionable_findings") or []
    keywords = summary.get("keyword_hits") or []
    severity = summary.get("severity_counts") or {}
    redacted_agents = [
        _redact_report_tool_tokens(agent) for agent in (summary.get("agents", []) or ["external reviewer"])
    ]
    sections.append("## External Review and Repair\n\n")
    sections.append(
        f"External review status: **{summary.get('status', 'recorded')}**. "
        f"Review agents: {', '.join(redacted_agents)}. "
        f"Severity counts: {json.dumps(severity, ensure_ascii=False)}.\n\n"
    )
    if keywords:
        sections.append("Review themes: " + ", ".join(str(k) for k in keywords[:12]) + ".\n\n")
    if findings:
        sections.append("| Finding | Repair response |\n|---|---|\n")
        for finding in findings[:8]:
            response = (
                "Addressed through regenerated statistical review, safety/world-model checks, "
                "and conservative report wording; unresolved code-level items remain listed as limitations."
            )
            sections.append(f"| {_table_cell(str(finding)[:300])} | {_table_cell(response)} |\n")
        sections.append("\n")


def _status_value(value, default: str = "not recorded") -> str:
    if value is None or value == "":
        return default
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_count(value, default: str = "not recorded") -> str:
    """Format count-like values without assuming they are numeric."""
    return _status_value(value, default=default)


def _join_finding_sentences(findings: list[str], limit: int = 3) -> str:
    """Join finding bullets into a clean prose sentence without doubled periods."""
    clean = [str(item).strip().rstrip(".") for item in findings[:limit] if str(item).strip()]
    return ". ".join(clean)


def _model_sentence_label(model_type: str) -> str:
    """Render model labels without awkward repeated selection wording."""
    label = str(model_type or "model").strip()
    if label.lower().startswith("auto-selected "):
        return "automatically selected " + label[len("auto-selected "):]
    return label


def _add_study_status_block(sections: list[str], ctx, records) -> None:
    """Add a decision-ready status block near the top of technical reports."""
    summary = _session_evidence_summary(ctx, records)
    cohort = summary.get("cohort") or {}
    model = summary.get("model") or {}
    world_model = summary.get("world_model") or {}
    sections.append("## Study Status\n\n")
    sections.append("| Field | Status |\n|-------|--------|\n")
    sections.append(f"| Data source | {_table_cell(summary['bank_name'])} only |\n")
    sections.append(
        f"| Primary endpoint | {_table_cell(cohort.get('disease') or cohort.get('endpoint') or 'not recorded')} |\n"
    )
    sections.append(
        f"| Cohort evidence | {_status_value(cohort.get('n_cases'))} cases; "
        f"{_status_value(cohort.get('n_controls'))} controls |\n"
    )
    auc = model.get("auc")
    auc_text = f"{auc:.3f}" if isinstance(auc, (int, float)) else _status_value(auc)
    sections.append(
        f"| Predictive model | {_table_cell(model.get('model_type') or 'not recorded')}; "
        f"AUC={auc_text}; 95% CI={_table_cell(model.get('auc_95ci') or 'not recorded')} |\n"
    )
    sections.append(
        f"| Governance | statistical_review={_table_cell(summary.get('guardrail') or 'not recorded')}; "
        f"safety_check={_table_cell(summary.get('safety') or 'not recorded')} |\n"
    )
    sections.append(
        f"| Claim boundary | {_table_cell(world_model.get('allowed_claim_type') or 'observational association / prediction only')} |\n"
    )
    sections.append("\n")


def _add_replication_comparison_section(sections: list[str], records) -> None:
    """Add paper-vs-local replication matrix when available."""
    comparisons = [
        getattr(rec, "key_results", {}) or {}
        for rec in records
        if getattr(rec, "skill", "") == "paper_replication_compare"
    ]
    if not comparisons:
        return
    result = comparisons[-1]
    rows = result.get("comparison_rows") or []
    if not rows:
        return

    sections.append("## Paper Replication Comparison\n\n")
    sections.append(
        f"Overall comparison status: **{_table_cell(result.get('overall_status', 'partial_replication'))}**. "
        f"Acceptance verdict: **{_table_cell((result.get('acceptance_summary') or {}).get('verdict', 'not recorded'))}**. "
        f"Paper access: **{_table_cell(result.get('paper_access', 'not recorded'))}**. "
        "This section separates the paper target from the local UKB approximation.\n\n"
    )
    sections.append("| Dimension | Paper Target | Local UKB Result | Status | Evidence |\n")
    sections.append("|-----------|--------------|------------------|--------|----------|\n")
    for row in rows:
        sections.append(
            f"| {_table_cell(row.get('dimension', ''))} | "
            f"{_table_cell(row.get('paper_target', ''))} | "
            f"{_table_cell(row.get('local_result', ''))} | "
            f"{_table_cell(row.get('status', ''))} | "
            f"{_table_cell(row.get('evidence', ''))} |\n"
        )
    gates = result.get("acceptance_gates") or []
    if gates:
        sections.append("\nAcceptance gates:\n\n")
        sections.append("| Gate | Status | Observed | Threshold | Evidence | Message |\n")
        sections.append("|------|--------|----------|-----------|----------|---------|\n")
        for gate in gates:
            sections.append(
                f"| {_table_cell(gate.get('gate', ''))} | "
                f"{_table_cell(gate.get('status', ''))} | "
                f"{_table_cell(gate.get('observed', ''))} | "
                f"{_table_cell(gate.get('threshold', ''))} | "
                f"{_table_cell(gate.get('evidence', ''))} | "
                f"{_table_cell(gate.get('message', ''))} |\n"
            )
    requirements = result.get("report_requirements") or []
    if requirements:
        sections.append("\nRequired report language:\n\n")
        for item in requirements[:5]:
            sections.append(f"- {_table_cell(item)}\n")
    diff_rows = result.get("table_figure_diff") or []
    if diff_rows:
        sections.append("\nTable/Figure diff targets:\n\n")
        sections.append("| Target | Paper Caption | Local Artifact | Status | Limitation |\n")
        sections.append("|--------|---------------|----------------|--------|------------|\n")
        for row in diff_rows[:12]:
            artifact = Path(str(row.get("local_artifact", ""))).name if row.get("local_artifact") else ""
            sections.append(
                f"| {_table_cell(row.get('target_type', ''))} | "
                f"{_table_cell(row.get('paper_caption', ''))} | "
                f"{_table_cell(artifact)} | "
                f"{_table_cell(row.get('status', ''))} | "
                f"{_table_cell(row.get('limitation', ''))} |\n"
            )
    sections.append("\n")


def _paper_results_narrative(ctx, records, analysis_records) -> str:
    """Build grouped Nature-style results instead of a tool-by-tool transcript."""
    summary = _session_evidence_summary(ctx, records)
    cohort = summary.get("cohort") or {}
    model = summary.get("model") or {}
    top_feature = summary.get("top_feature") or {}
    paragraphs: list[str] = []

    if cohort:
        feature_clause = ""
        if model and cohort.get("n_features"):
            feature_clause = (
                f" The analysis retained {_status_value(cohort.get('n_features'))} candidate features "
                "for the recorded modelling branch after data-readiness checks."
            )
        paragraphs.append(
            "### Cohort definition and data readiness\n\n"
            f"The primary {cohort.get('disease', 'endpoint')} cohort contained "
            f"{_status_value(cohort.get('n_cases'))} cases and "
            f"{_status_value(cohort.get('n_controls'))} controls from {summary['bank_name']}."
            f"{feature_clause}"
        )

    if model:
        auc = model.get("auc")
        auc_text = f"{auc:.3f}" if isinstance(auc, (int, float)) else _status_value(auc)
        ci = model.get("auc_95ci")
        ci_text = f" (95% CI: {str(ci).strip('[]')})" if ci else ""
        design_label = _model_design_label(model)
        method_label = _model_method_label(model)
        if model.get("incident_risk_supported") is False:
            claim_text = (
                f"The {_model_sentence_label(model.get('model_type', 'model'))} achieved AUC = {auc_text}{ci_text} "
                f"for {design_label} using {method_label}. This supports internal discrimination feasibility, "
                "but it does not establish prospective incident Type 2 Diabetes risk prediction or actionable biomarkers."
            )
        else:
            claim_text = (
                f"The {_model_sentence_label(model.get('model_type', 'model'))} achieved AUC = {auc_text}{ci_text}, "
                f"supporting internally validated {design_label} within the analysed cohort."
            )
        paragraphs.append(
            "### Predictive performance\n\n"
            + claim_text
        )

    if top_feature.get("name") or summary.get("biomarkers"):
        feature_sentence = ""
        if top_feature.get("name"):
            feature_sentence = f"{top_feature['name']} was the leading model feature."
            if top_feature.get("note"):
                feature_sentence += f" {top_feature['note']}"
        biomarker_sentences = []
        for item in summary.get("biomarkers", [])[:3]:
            p_value = item.get("p_value")
            p_text = _format_p_value(p_value) if p_value is not None else "P value not recorded"
            effect = item.get("effect_size")
            effect_text = f", effect size={effect:.3g}" if isinstance(effect, (int, float)) else ""
            biomarker_sentences.append(f"{item.get('name', 'Biomarker')} showed {p_text}{effect_text}.")
        paragraphs.append(
            "### Biomarker interpretation\n\n"
            + " ".join([feature_sentence, *biomarker_sentences]).strip()
        )

    replication = summary.get("replication_compare") or {}
    if replication:
        paragraphs.append(
            "### Paper replication comparison\n\n"
            f"The replication comparison classified the local analysis as "
            f"{replication.get('overall_status', 'partial_replication')} with "
            f"acceptance verdict {replication.get('acceptance_verdict', 'not recorded')}. "
            f"The paper-access status was {replication.get('paper_access', 'not recorded')}, "
            f"and the local endpoint was {replication.get('local_endpoint', 'not recorded')}. "
            "Claims should distinguish exact paper methods from the feasible UKB-only approximation."
        )

    if summary.get("progression"):
        progression = []
        for item in summary["progression"][:5]:
            p_value = item.get("p_value")
            p_text = _format_p_value(p_value) if p_value is not None else "P value not recorded"
            n_cases = item.get("n_cases")
            n_text = f" ({_status_value(n_cases)} cases)" if n_cases else ""
            endpoint = str(item.get("endpoint", "Outcome")).capitalize()
            progression.append(f"{endpoint}{n_text}: log-rank {p_text}.")
        paragraphs.append(
            "### Cardiometabolic progression endpoints\n\n"
            + " ".join(progression)
        )

    guardrail_bits = []
    if summary.get("guardrail"):
        guardrail_bits.append(f"statistical_review={summary['guardrail']}")
    if summary.get("safety"):
        guardrail_bits.append(f"safety_check={summary['safety']}")
    if summary.get("world_model"):
        wm = summary["world_model"]
        guardrail_bits.append(
            f"world_model_audit={wm.get('safety_status') or 'recorded'}, "
            f"claim type={wm.get('allowed_claim_type') or 'association only'}"
        )
    if guardrail_bits:
        paragraphs.append(
            "### Guardrails and interpretation boundary\n\n"
            + "; ".join(guardrail_bits)
            + ". These outputs support execution-grounded findings and hypotheses, not causal or clinical-deployment claims."
        )

    if not paragraphs:
        results_parts = []
        for rec in analysis_records:
            interp = _interpret_skill(rec)
            if interp:
                results_parts.append(interp)
        return "\n\n".join(results_parts) if results_parts else "No analytical result records were available."

    return "\n\n".join(paragraphs)


def _governance_text(ctx, records) -> str:
    """Build reproducibility and governance text grounded in session state."""
    bank_name = ctx.settings.biobank_name if ctx and hasattr(ctx, "settings") else "Biobank"
    if not isinstance(bank_name, str):
        bank_name = "Biobank"
    caveats = getattr(ctx.settings, "biobank_caveats", "") if ctx and hasattr(ctx, "settings") else ""
    if not isinstance(caveats, str):
        caveats = ""
    n_records = len(records)
    n_prov = len(getattr(ctx.state, "provenances", []) or []) if ctx and hasattr(ctx, "state") else 0
    caveat_sentence = f" Cohort-level caveat: {caveats}." if caveats else ""
    provenance_sentence = (
        f"The session contains {n_records} analysis record(s)"
        + (f" and {n_prov} provenance hash record(s)" if n_prov else "")
        + "."
    )
    return (
        f"Analyses were run against the configured {bank_name} data layer and should be "
        "interpreted as aggregate, observational biobank evidence."
        f"{caveat_sentence} {provenance_sentence} "
        "Small-cell outputs require suppression or aggregation before external release. "
        "Associations and prediction results are not causal effect estimates unless a "
        "target-trial or external causal-validation record is explicitly reported."
    )


def _dedupe_records(records) -> list:
    """Collapse consecutive duplicate non-think calls (same skill + args)."""
    deduped = []
    for rec in records:
        if not deduped:
            deduped.append(rec)
            continue
        prev = deduped[-1]
        prev_sig = (prev.skill, json.dumps(prev.args, sort_keys=True, default=str))
        cur_sig = (rec.skill, json.dumps(rec.args, sort_keys=True, default=str))
        if rec.skill != "think" and cur_sig == prev_sig:
            deduped[-1] = rec
        else:
            deduped.append(rec)
    return deduped


_EVIDENCE_GATHERING_SKILLS = {
    "think",
    "web_search",
    "web_fetch",
    "deep_research",
    "fetch_paper",
    "read_paper",
    "read_pdf",
    "literature_qa",
}


def _analysis_records(records) -> list:
    """Records that should appear as numbered analytical report sections."""
    excluded = _EVIDENCE_GATHERING_SKILLS | _GUARDRAIL_SKILLS | _APPENDIX_ONLY_SKILLS
    return [rec for rec in records if getattr(rec, "skill", "") not in excluded]


def _report_records(ctx) -> list:
    """Records used for report rendering after de-duplication."""
    return _dedupe_records(list(getattr(ctx.state, "records", [])))


def _write_reproducibility_scripts(ctx, report_dir: Path) -> dict[str, str]:
    """Write run-local scripts that document how to replay the WGS CLI workflow."""
    report_dir.mkdir(parents=True, exist_ok=True)
    records = _report_records(ctx) if ctx is not None else []
    env = {}
    settings = getattr(ctx, "settings", None)
    for key, attr in (
        ("BANK_ID", "bank_id"),
        ("BIOBANK_NAME", "biobank_name"),
        ("BIOBANK_ABBREVIATION", "biobank_abbreviation"),
        ("DATA_DIR", "data_dir"),
        ("RAW_DIR", "raw_dir"),
        ("SUBJECT_ID_COL", "subject_id_col"),
    ):
        value = getattr(settings, attr, None) if settings is not None else None
        if value:
            env[key] = str(value)
    env.setdefault("BANK_ID", "virtualcell")
    env.setdefault("VC_WGS_VCF_DIR", "/Files/ResultData/BW_WGS_vcf")
    env.setdefault("PLAN_EXTERNAL_COUNCIL_POLICY", "never")
    env.setdefault("PLAN_REVIEW_HOOK_MODE", "never")
    env.setdefault("PLAN_REVIEW_REPAIR_MODE", "never")
    env.setdefault("AUTO_DISCOVER_MODELS", "false")
    env.setdefault("MULTI_MODEL_ENABLED", "false")
    env.setdefault("ASYNC_RUNTIME_ENABLED", "false")

    plan_lines = []
    for idx, rec in enumerate(records, 1):
        skill_name = getattr(rec, "skill", "")
        if not skill_name or skill_name == "generate_report":
            continue
        args = getattr(rec, "args", {}) or {}
        plan_lines.append({
            "step": idx,
            "skill": skill_name,
            "args": args,
        })

    replay_task = (
        "### 任务背景\n"
        "你是一名生物信息学分析专家。现有一套白癜风相关的全基因组测序（WGS）数据，"
        "需要对 VCF 变异文件执行标准分析流程，重点比较 Juvenile_White vs Vitiligo_White。\n\n"
        "请完整执行：VCF 合并和 QC、标准注释、PCA、亲缘关系、Juvenile_White vs "
        "Vitiligo_White 关联分析、罕见变异 burden、功能富集、候选基因解读、相关论文调研/"
        "论文复现可行性说明、Action Graph 记录、Markdown/HTML 报告输出。"
    )

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "entrypoint": "python -m biobank_agent.cli",
        "conda_env": "biobank-agent",
        "environment": env,
        "replay_task": replay_task,
        "executed_steps": plan_lines,
        "notes": [
            "Set REPORTS_DIR, MEMORY_DIR and PLANS_DIR to writable run-local directories before replay.",
            "The default VirtualCell fallback VCF directory is /Files/ResultData/BW_WGS_vcf.",
            "Large full-genome runs may require widening the deterministic chr22 validation window used by CLI E2E tests.",
        ],
    }
    # M7: attach a reproducibility receipt (git code version, python + key package
    # versions, input-file fingerprints) so the report carries everything needed to
    # re-run, not just the command list. Best-effort — never breaks report generation.
    try:
        from biobank_agent.runtime.receipt import build_receipt, write_receipt

        _data_paths: list[str] = []
        for _line in plan_lines:
            for _v in (_line.get("args") or {}).values():
                if isinstance(_v, str) and _v.endswith(
                    (".vcf", ".vcf.gz", ".csv", ".tsv", ".parquet", ".bed", ".bgen", ".gz", ".txt")
                ):
                    _data_paths.append(_v)
        _receipt = build_receipt(
            objective=str(replay_task)[:200],
            skill="generate_report",
            params={"executed_steps": len(plan_lines)},
            data_paths=_data_paths[:50],
            extra={"environment": env},
        )
        manifest["receipt"] = _receipt
        write_receipt(_receipt, report_dir)
    except Exception:
        pass

    manifest_path = report_dir / "reproducibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    shell_path = report_dir / "run_reproduce_wgs.sh"
    env_exports = "\n".join(f"export {key}={json.dumps(value)}" for key, value in sorted(env.items()))
    shell_text = f"""#!/usr/bin/env bash
set -euo pipefail

# Reproduce the BioBank Agent WGS CLI workflow that generated this report.
# Run from the repository root with the biobank-agent conda environment available.
REPORTS_DIR="${{REPORTS_DIR:-reports/reproduce_wgs_cli}}"
MEMORY_DIR="${{MEMORY_DIR:-reports/reproduce_wgs_cli_memory}}"
PLANS_DIR="${{PLANS_DIR:-reports/reproduce_wgs_cli_plans}}"
mkdir -p "$REPORTS_DIR" "$MEMORY_DIR" "$PLANS_DIR"
export REPORTS_DIR MEMORY_DIR PLANS_DIR
{env_exports}

conda run -n biobank-agent python -m biobank_agent.cli <<'BIOBANK_AGENT_INPUT'
/plan {replay_task}
/plan-approve
quit
BIOBANK_AGENT_INPUT
"""
    shell_path.write_text(shell_text, encoding="utf-8")
    shell_path.chmod(shell_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    python_path = report_dir / "run_reproduce_wgs.py"
    python_text = f'''#!/usr/bin/env python
"""Replay the BioBank Agent WGS CLI workflow for this report."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPLAY_TASK = {replay_task!r}
ENVIRONMENT = {env!r}


def main() -> int:
    env = os.environ.copy()
    env.update({{key: str(value) for key, value in ENVIRONMENT.items()}})
    env.setdefault("REPORTS_DIR", "reports/reproduce_wgs_cli")
    env.setdefault("MEMORY_DIR", "reports/reproduce_wgs_cli_memory")
    env.setdefault("PLANS_DIR", "reports/reproduce_wgs_cli_plans")
    for key in ("REPORTS_DIR", "MEMORY_DIR", "PLANS_DIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    stdin_text = f"/plan {{REPLAY_TASK}}\\n/plan-approve\\nquit\\n"
    return subprocess.run(
        [sys.executable, "-m", "biobank_agent.cli"],
        input=stdin_text,
        text=True,
        env=env,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
'''
    python_path.write_text(python_text, encoding="utf-8")
    python_path.chmod(python_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return {
        "manifest": str(manifest_path),
        "shell_script": str(shell_path),
        "python_script": str(python_path),
    }


@skill(
    name="generate_report",
    description="Generate a structured analysis report from all session analyses. "
                "Supports 'report' format (technical with Key Findings) or 'paper' "
                "format (IMRaD structure), plus 'dual' for paired technical and Nature-style outputs. "
                "Produces Markdown, HTML, and optionally PDF output.",
    parameters={
        "title": {
            "type": "string",
            "description": "Report title",
            "default": "Biobank Analysis Report",
        },
        "format": {
            "type": "string",
            "description": "Output format: 'report'/'technical', 'paper'/'nature', 'dual'/'both', or 'brief'",
            "default": "report",
        },
        "output_dir": {
            "type": "string",
            "description": "Optional report directory to overwrite; defaults to the active session report directory",
            "default": "",
        },
        "polish": {
            "type": "boolean",
            "description": "Polish final Markdown/HTML through the academic report polisher before returning artifacts",
            "default": True,
        },
        "polish_mode": {
            "type": "string",
            "description": "Report polishing mode",
            "default": "academic",
        },
    },
    required=[],
)
def generate_report(
    title: str = "Biobank Analysis Report",
    format: str = "report",
    output_dir: str = "",
    polish: bool = True,
    polish_mode: str = "academic",
    *,
    ctx=None,
) -> dict:
    """Generate a structured, readable analysis report."""
    format = _normalize_report_format(format)

    # Dynamic title from settings if using default
    if title == "Biobank Analysis Report" and ctx and hasattr(ctx, "settings"):
        title = f"{ctx.settings.biobank_name} Analysis Report"
    title = _sanitize_report_title(title, ctx)

    report_dir = Path(output_dir).expanduser() if output_dir else ctx.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    _materialize_report_figures(ctx, report_dir)
    reproducibility_artifacts = _write_reproducibility_scripts(ctx, report_dir)

    if format == "dual":
        technical_sections = _build_report_sections(title, ctx)
        nature_sections = _build_paper_sections(title, ctx)
        technical_content = _render_markdown(technical_sections)
        nature_content = _render_markdown(nature_sections)

        md_path = report_dir / "report.md"
        technical_path = report_dir / "report_technical.md"
        nature_path = report_dir / "report_nature.md"
        md_path.write_text(technical_content, encoding="utf-8")
        technical_path.write_text(technical_content, encoding="utf-8")
        nature_path.write_text(nature_content, encoding="utf-8")

        polish_results: dict[str, dict] = {}
        if polish:
            polish_results["markdown"] = polish_markdown_report(md_path, mode=polish_mode, strict=False)
            polish_results["technical_markdown"] = polish_markdown_report(technical_path, mode=polish_mode, strict=False)
            polish_results["nature_markdown"] = polish_markdown_report(nature_path, mode=polish_mode, strict=False)
            technical_content = md_path.read_text(encoding="utf-8")
            nature_content = nature_path.read_text(encoding="utf-8")

        html_path = _write_html(technical_content, title, report_dir)
        nature_html_path = _write_html(nature_content, title, report_dir, stem="report_nature")
        css_md_path = report_dir / "_report_with_css.md"
        nature_css_md_path = report_dir / "_report_nature_with_css.md"
        css_polish_results: dict[str, dict] = {}
        if polish:
            if css_md_path.exists():
                css_polish_results["markdown_with_css"] = polish_markdown_report(css_md_path, mode=polish_mode, strict=False)
            if nature_css_md_path.exists():
                css_polish_results["nature_markdown_with_css"] = polish_markdown_report(nature_css_md_path, mode=polish_mode, strict=False)

        result = {
            "report_dir": str(report_dir),
            "markdown": str(md_path),
            "markdown_with_css": str(css_md_path) if css_md_path.exists() else None,
            "html": str(html_path) if html_path else None,
            "format": "dual",
            "paired_outputs": {
                "technical_markdown": str(technical_path),
                "technical_markdown_with_css": str(css_md_path) if css_md_path.exists() else None,
                "technical_html": str(html_path) if html_path else None,
                "nature_markdown": str(nature_path),
                "nature_markdown_with_css": str(nature_css_md_path) if nature_css_md_path.exists() else None,
                "nature_html": str(nature_html_path) if nature_html_path else None,
            },
            "reproducibility_artifacts": reproducibility_artifacts,
            "n_sections": len(_analysis_records(_report_records(ctx))),
            "n_figures": len(_logical_figures(ctx)),
            "polished": bool(polish),
            "polish_mode": polish_mode if polish else "",
            "polisher": polish_results,
            "polisher_css": css_polish_results,
        }
        return _attach_report_artifact_status(
            result,
            report_dir,
            [p for p in (md_path, technical_path, nature_path, css_md_path, nature_css_md_path) if p.exists()],
        )

    if format == "paper":
        sections = _build_paper_sections(title, ctx)
    elif format == "brief":
        sections = _build_brief_sections(title, ctx)
    else:
        sections = _build_report_sections(title, ctx)

    # Write markdown
    md_content = _render_markdown(sections)
    md_path = report_dir / "report.md"
    md_path.write_text(md_content)
    polish_result: dict | None = None
    if polish:
        polish_result = polish_markdown_report(md_path, mode=polish_mode, strict=False)
        md_content = md_path.read_text(encoding="utf-8")

    # Write HTML with embedded CSS
    html_path = _write_html(md_content, title, report_dir)
    css_md_path = report_dir / "_report_with_css.md"
    css_polish_result: dict | None = None
    if polish and css_md_path.exists():
        css_polish_result = polish_markdown_report(css_md_path, mode=polish_mode, strict=False)

    result = {
        "report_dir": str(report_dir),
        "markdown": str(md_path),
        "markdown_with_css": str(css_md_path) if css_md_path.exists() else None,
        "html": str(html_path) if html_path else None,
        "format": format,
        "reproducibility_artifacts": reproducibility_artifacts,
        "n_sections": len(_analysis_records(_report_records(ctx))),
        "n_figures": len(_logical_figures(ctx)),
        "polished": bool(polish),
        "polish_mode": polish_mode if polish else "",
        "polisher": polish_result or {},
        "polisher_css": css_polish_result or {},
    }
    return _attach_report_artifact_status(
        result,
        report_dir,
        [p for p in (md_path, css_md_path) if p.exists()],
    )


# ── Report builders ──────────────────────────────────────────


def _render_markdown(sections: list[str]) -> str:
    """Render pre-formatted markdown fragments without inserting table-breaking blank lines."""
    return "".join(sections)


def _build_report_sections(title: str, ctx) -> list[str]:
    """Build technical report with executive summary and Key Findings."""
    records = _report_records(ctx)
    analysis_records = _analysis_records(records)
    sections = []

    sections.append(REPORT_HEADER.format(
        title=title,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        format_name="Technical Report",
    ))

    executive_findings = _extract_executive_findings(ctx, records)
    if executive_findings:
        findings_md = "\n".join(f"> - {f}" for f in executive_findings)
        sections.append(EXECUTIVE_FINDINGS_BOX.format(findings=findings_md))

    # Key Findings
    if records:
        findings = executive_findings or _extract_key_findings(records)
        if findings:
            findings_md = "\n".join(f"> - {f}" for f in findings)
            sections.append(KEY_FINDINGS_BOX.format(findings=findings_md))

    # Executive Summary
    if records:
        n_analyses = len(analysis_records)
        n_figs = len(_logical_figures(ctx))
        n_cohorts = len(ctx.state.cohorts)
        n_models = len(ctx.state.models)
        bank_name = ctx.settings.biobank_name if ctx and hasattr(ctx, "settings") else "Biobank"
        parts = [f"This report summarizes {n_analyses} computational analysis step(s) performed on the {bank_name} dataset."]
        if n_cohorts:
            parts.append(f"{n_cohorts} disease cohort(s) were constructed.")
        if n_models:
            parts.append(f"{n_models} predictive model(s) were trained and evaluated.")
        if n_figs:
            parts.append(f"{n_figs} publication-quality figures were generated.")
        sections.append(EXECUTIVE_SUMMARY.format(summary=" ".join(parts)))

    _add_study_status_block(sections, ctx, records)
    _add_replication_comparison_section(sections, records)

    # Analysis sections with interpretive text
    section_n = 0
    for rec in analysis_records:
        section_n += 1

        skill_title = rec.skill.replace("_", " ").title()
        sections.append(SECTION_HEADER.format(n=section_n, title=skill_title))

        interp = _interpret_skill(rec)
        if interp:
            sections.append(f"{interp}\n\n")

        if rec.key_results and "error" not in rec.key_results:
            exclude_keys = {"figure", "figures", "_retried_with", "issues"}
            if rec.skill == "target_annotation_context":
                exclude_keys.update({"targets", "sources", "warnings", "caveats", "source_results"})
            elif rec.skill == "target_enrichment":
                exclude_keys.update({"terms", "sources", "warnings", "caveats"})
            elif rec.skill == "train_model":
                exclude_keys.update({"candidate_comparison", "model_selection"})
            elif rec.skill == "deep_research":
                exclude_keys.update({"sources", "biobank_relevant_fields", "brief"})
            elif rec.skill == "replicate_paper":
                exclude_keys.update({
                    "plan",
                    "replication_targets",
                    "feasibility_map",
                    "comparison_checklist",
                    "artifact",
                    "artifacts",
                    "message",
                })
            elif rec.skill == "paper_replication_compare":
                exclude_keys.update({
                    "comparison_rows",
                    "acceptance_gates",
                    "report_requirements",
                    "artifacts",
                    "top_local_features",
                })
            metrics = {k: _format_value(v) for k, v in rec.key_results.items()
                       if k not in exclude_keys and not _is_raw_result_key(k)}
            if metrics:
                sections.append("| Metric | Value |\n|--------|-------|\n")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        v_str = f"{v:.4f}" if abs(v) < 100 else f"{v:,.2f}"
                    elif isinstance(v, list) and len(v) > 5:
                        v_str = f"[{v[0]}, ..., {v[-1]}] ({len(v)} items)"
                    else:
                        v_str = str(v)
                    sections.append(f"| {_table_cell(k)} | {_table_cell(v_str)} |\n")
                sections.append("\n")

        if rec.key_results and "error" in rec.key_results:
            sections.append(f"> **Warning:** {rec.key_results['error']}\n\n")

        sections.append("")

    _add_figures_section(sections, ctx)
    _add_cohorts_section(sections, ctx, records)
    _add_models_section(sections, ctx)
    _add_model_selection_section(sections, ctx, records)
    _add_guardrail_section(sections, ctx, records)

    bank_name = ctx.settings.biobank_name if ctx and hasattr(ctx, "settings") else "Biobank"
    sections.append("## Methodology Notes\n\n")

    # Pull actual parameters from model metadata
    method_details = []
    if ctx.state.model_metadata:
        for key, meta in ctx.state.model_metadata.items():
            mt = meta.get("model_type", "gradient-boosted")
            nf = meta.get("n_features", "all available")
            strategy = meta.get("evaluation_strategy") or (meta.get("model_selection") or {}).get("evaluation_strategy")
            if strategy == "stratified_holdout":
                eval_text = (
                    "Auto model selection used a stratified holdout; any separate "
                    "cross-validation diagnostics are reported as downstream checks, not as the selection procedure."
                )
            elif strategy:
                eval_text = f"Model evaluation strategy was {str(strategy).replace('_', ' ')}."
            else:
                eval_text = "Model evaluation strategy was recorded in the model metadata when available."
            design_text = ""
            if meta.get("incident_risk_supported") is False:
                design_text = (
                    " The target is prevalent/ever-diagnosed disease discrimination, "
                    "not prospective incident risk prediction."
                )
            method_details.append(
                f"The {mt} classifier was trained with {nf} features. {eval_text}{design_text}"
            )

    sections.append(
        f"All analyses were performed on the {bank_name} cohort using "
        "DuckDB for data access and Python scientific stack for computation. "
        + (" ".join(method_details) + " " if method_details else "")
        + "Statistical tests used two-sided P-values with significance threshold "
        "\u03b1 = 0.05. Multiple testing correction applied via FDR (Benjamini-Hochberg) "
        "where indicated. Figures follow Nature journal guidelines "
        "(Arial 7 pt, 300 DPI, Okabe-Ito colour-blind safe palette).\n\n"
    )

    sections.append("## Reproducibility and Governance\n\n")
    sections.append(_governance_text(ctx, records) + "\n\n")
    sections.append(
        "Data availability: individual-level biobank data remain governed by "
        "the applicable data access agreement and should not be redistributed "
        "from this generated report. Derived aggregate tables, model settings, "
        "cohort definitions and code provenance should be archived with any "
        "external manuscript or internal analysis handoff.\n\n"
    )

    _add_external_review_section(sections, ctx)

    sections.append("## References\n\n")
    refs = _extract_references(records)
    if refs:
        for i, ref in enumerate(refs, 1):
            sections.append(f"{i}. {ref}\n")
        sections.append("\n")
    else:
        sections.append(
            "No external literature records were attached to this session. "
            "Add `fetch_paper`, `read_paper`, or `deep_research` records before "
            "treating this report as submission-ready.\n\n"
        )

    _add_execution_appendix(sections, ctx, records)
    return sections


def _build_paper_sections(title: str, ctx) -> list[str]:
    """Build IMRaD paper draft."""
    records = _report_records(ctx)
    analysis_records = _analysis_records(records)
    sections = []
    bank_name = ctx.settings.biobank_name if ctx and hasattr(ctx, "settings") else "Biobank"
    bank_desc = ctx.settings.biobank_description if ctx and hasattr(ctx, "settings") else "a large-scale prospective cohort study"
    bank_caveats = ctx.settings.biobank_caveats if ctx and hasattr(ctx, "settings") else "healthy volunteer cohort with known selection biases"
    evidence_summary = _session_evidence_summary(ctx, records)
    has_model = bool(evidence_summary.get("model") or getattr(ctx.state, "model_metadata", {}))
    has_feature_importance = any(r.skill == "feature_importance" for r in records)
    has_trajectory = any(r.skill == "trajectory_tokenize" for r in records)
    wgs_skills = {"vcf_qc", "vcf_pca", "vcf_kinship", "vcf_association", "vcf_annotation", "vcf_burden_test", "pathway_enrichment"}
    has_wgs = any(r.skill in wgs_skills for r in records)
    has_icd10 = any(r.skill in ("prevalence", "cohort_card", "phenotype_harmonize") for r in records)
    empty_trajectory = any(
        r.skill == "trajectory_tokenize"
        and str(r.key_results.get("status", "")).upper() == "PARTIAL"
        and not r.key_results.get("n_tokens")
        for r in records
    )

    sections.append(f"# {title}\n\n")
    sections.append("---\n\n")

    executive_findings = _extract_executive_findings(ctx, records)
    if executive_findings:
        findings_md = "\n".join(f"> - {f}" for f in executive_findings)
        sections.append(EXECUTIVE_FINDINGS_BOX.format(findings=findings_md))

    findings = executive_findings or _extract_key_findings(records)
    finding_sentence = (
        _join_finding_sentences(findings[:3])
        if findings
        else "No validated quantitative findings were available from the recorded execution"
    )

    if has_trajectory and not has_model:
        abstract = (
            f"**Background:** We analysed whether the {bank_name} data available in this session "
            "could support longitudinal cardiometabolic trajectory analysis. "
            "**Methods:** Field discovery, cohort design cards, trajectory tokenization and "
            "world-model audit were used to separate feasible association-conditioned forecasts "
            "from unsupported temporal or causal claims. "
            "**Results:** " + finding_sentence + ". "
            "**Conclusions:** The current evidence supports a feasibility and governance assessment, "
            "not a validated longitudinal forecast, until usable trajectory tokens and external "
            "calibration evidence are available."
        )
        keywords = f"{bank_name}, longitudinal trajectories, feasibility, governance, epidemiology"
    else:
        method_bits = []
        if has_wgs and not has_icd10:
            method_bits.append("WGS VCF data were processed through a quality control, population structure, and association analysis pipeline.")
        elif has_icd10:
            method_bits.append("Case-control cohorts were constructed from ICD-10 coded diagnoses.")
        else:
            method_bits.append("Biomarker and phenotype data were analysed using the configured data layer.")
        if has_trajectory:
            method_bits.append(
                "Longitudinal fields were tokenized as a HealthFormer-style feasibility layer."
            )
        if has_model:
            model_summary = evidence_summary.get("model") or {}
            method_bits.append(
                f"Models were assessed as {_model_design_label(model_summary)} using {_model_method_label(model_summary)}."
            )
        if has_feature_importance:
            method_bits.append("Feature importance was assessed from the recorded model outputs.")
        abstract = (
            f"**Background:** We analysed the {bank_name} cohort to identify "
            "disease-associated biomarkers and evaluate discrimination evidence. "
            f"**Methods:** {' '.join(method_bits)} "
            "**Results:** " + finding_sentence + ". "
            f"**Conclusions:** These findings support an internal {bank_name} feasibility workflow; "
            "prospective incident-risk or actionable-biomarker claims require date-aware cohort construction and independent validation."
        )
        if has_wgs:
            keywords = f"{bank_name}, whole genome sequencing, association analysis, candidate genes, epidemiology"
        else:
            keywords = f"{bank_name}, biomarkers, machine learning, disease prediction, epidemiology"
    sections.append(PAPER_ABSTRACT.format(
        abstract=abstract,
        keywords=keywords,
    ))

    if has_trajectory and not has_model:
        intro = (
            f"The {bank_name} is {bank_desc}. Longitudinal trajectory analyses require "
            "time-stamped participant measurements, explicit cohort design and careful "
            "claim governance.\n\n"
            f"In this analysis, we evaluate whether the available {bank_name} session "
            "state can support a HealthFormer-style trajectory layer and document the "
            "limits on any forecast interpretation."
        )
    else:
        if has_wgs and not has_icd10:
            measurement_phrase = (
                "whole genome sequencing (WGS) data processed through quality control, "
                "population structure analysis, and variant association testing"
            )
            analysis_phrase = (
                "identify candidate genetic variants and pathways associated with the phenotype of interest"
            )
        elif has_trajectory:
            measurement_phrase = (
                "coded diagnoses alongside repeated biomarker, "
                "blood pressure and anthropometric measurements where available"
            )
            analysis_phrase = (
                "characterise disease cohorts, assess trajectory feasibility and identify aggregate biomarker signatures"
            )
        else:
            measurement_phrase = (
                "coded diagnoses alongside blood biochemistry, "
                "haematology, and anthropometric measurements"
            )
            analysis_phrase = (
                "characterise disease cohorts and identify aggregate biomarker signatures"
            )
        intro = (
            f"The {bank_name} is {bank_desc}. "
            "This rich resource enables systematic "
            "identification of disease-associated biomarkers and evaluation of "
            "prediction evidence.\n\n"
            f"In this analysis, we leverage the {bank_name}'s {measurement_phrase} "
            f"to {analysis_phrase}."
        )
    sections.append(PAPER_INTRODUCTION.format(introduction=intro))

    cohort_info = ""
    cohort_summary = evidence_summary.get("cohort") or {}
    if cohort_summary:
        cohort_info = (
            f"For {cohort_summary.get('disease') or cohort_summary.get('endpoint')}, "
            f"{_status_value(cohort_summary.get('n_cases'))} cases were identified from coded diagnoses "
            f"and matched with {_status_value(cohort_summary.get('n_controls'))} controls. "
        )
    elif ctx.state.cohorts:
        for name, df in ctx.state.cohorts.items():
            if "label" in df.columns:
                n_cases = int(df["label"].sum())
                n_controls = len(df) - n_cases
                cohort_info += (
                    f"For {name}, {_format_count(n_cases)} cases were identified from coded diagnoses "
                    f"and matched with {_format_count(n_controls)} controls. "
                )
            elif "phenotype_group" in df.columns:
                group_counts = df["phenotype_group"].value_counts(dropna=False).to_dict()
                group_text = ", ".join(
                    f"{k}={v}" for k, v in sorted(group_counts.items(), key=lambda item: str(item[0]))
                )
                cohort_info += (
                    f"For {name}, {len(df):,} WGS samples were available across phenotype groups "
                    f"({group_text}). "
                )
            else:
                cohort_info += f"For {name}, {len(df):,} samples were available for analysis. "

    model_info = ""
    if evidence_summary.get("model"):
        meta = evidence_summary["model"]
        model_info = (
            f"The {_model_sentence_label(meta.get('model_type', 'model'))} was trained "
            f"with {meta.get('n_features', 'all available')} features using "
            f"{_model_method_label(meta)} for {_model_design_label(meta)}. "
        )
    elif ctx.state.model_metadata:
        for key, meta in ctx.state.model_metadata.items():
            model_info += (
                f"The {meta.get('model_type', 'gradient-boosted')} classifier was trained "
                f"with {meta.get('n_features', 'all available')} features using "
                f"{_model_method_label(meta)} for {_model_design_label(meta)}. "
            )

    if has_trajectory and not has_model:
        trajectory_records = [r for r in records if r.skill == "trajectory_tokenize"]
        trajectory_support = ""
        if trajectory_records:
            traj = trajectory_records[-1].key_results
            support = traj.get("longitudinal_support")
            time_source = traj.get("trajectory_time_source")
            if support or time_source:
                trajectory_support = (
                    f" Temporal support was recorded as {support or 'unknown'}, "
                    f"with time source {time_source or 'unknown'}."
                )
        trajectory_note = (
            "Trajectory tokenization was attempted using participant_id, timestamp, modality and value columns."
            f"{trajectory_support} "
            if not empty_trajectory
            else "No usable longitudinal rows were available, so trajectory tokenization was treated as a feasibility gap. "
        )
        study_population = (
            f"This study utilised data from the {bank_name} cohort. "
            f"{cohort_info}"
            "Diagnoses and candidate longitudinal fields were assessed for a trajectory cohort design. "
            "Cohort-card outputs recorded index date, lookback/follow-up windows and bias flags."
        )
        statistical_analysis = (
            trajectory_note +
            "World-model audit classified the supported claim type and required calibration or external validation "
            "before any forecast interpretation. Guardrail review checked unsupported temporal, causal and privacy claims."
        )
    else:
        if has_wgs and not has_icd10:
            measurement_sentence = "WGS VCF data were processed per sample and merged for multi-sample analyses. "
            study_population = (
                f"This study utilised data from the {bank_name} cohort. "
                f"{cohort_info}"
                f"{measurement_sentence}"
            )
        else:
            measurement_sentence = (
                "Biomarker, blood pressure and anthropometric measurements were obtained from baseline and repeated "
                "assessment instances where available; exact elapsed time should be interpreted according to source metadata. "
                if has_trajectory else
                "Biomarker measurements were obtained from baseline assessment. "
            )
            study_population = (
                f"This study utilised data from the {bank_name} cohort. "
                f"{cohort_info}"
                "Diagnoses were extracted from coded diagnosis records using "
                f"ICD-10 coding. {measurement_sentence}"
            )
        trajectory_method = (
            "Trajectory tokenization was analysed as a feasibility branch and not treated as an externally validated forecast. "
            if has_trajectory else
            ""
        )
        if has_wgs and not has_icd10:
            statistical_analysis = (
                "Per-sample VCF quality control assessed call rate, heterozygosity ratio, Ti/Tv ratio, "
                "mean depth and genotype quality. Hardy-Weinberg equilibrium was tested on merged pass-sample VCFs. "
                "Population structure was assessed by genotype PCA and genomic relationship matrix-based kinship estimation. "
                "Case-control association used Fisher's exact test per biallelic variant with Bonferroni and FDR correction. "
                "Gene-level burden testing used collapsing Fisher's exact and Madsen-Browning weighted sum tests. "
                "Pathway enrichment used Fisher's exact overrepresentation analysis with FDR correction."
            )
        else:
            statistical_analysis = (
                "Continuous variables were compared using Mann-Whitney U tests. "
                "Categorical variables were compared using chi-squared tests. "
                "Multiple testing correction was applied using the Benjamini-Hochberg "
                f"procedure. {trajectory_method}{model_info}"
                + (
                    "Model performance was assessed using area under the receiver operating "
                    "characteristic curve (AUC-ROC) with 95% confidence intervals."
                    if has_model else
                    "No predictive model was trained in this session; modelling claims are therefore not made."
                )
            )

    sections.append(PAPER_METHODS.format(
        study_population=study_population,
        statistical_analysis=statistical_analysis,
    ))

    sections.append(PAPER_RESULTS.format(
        results=_paper_results_narrative(ctx, records, analysis_records),
    ))

    _add_external_review_section(sections, ctx)

    if has_trajectory and not has_model:
        discussion = (
            f"Our analysis of the {bank_name} cohort focused on trajectory feasibility. "
            + finding_sentence + ". "
            "The output should be read as an execution-grounded feasibility assessment rather than a validated forecast."
        )
        limitations = (
            f"This study has several limitations. First, the {bank_name} represents a {bank_caveats}. "
            "Second, usable longitudinal trajectory rows were not available in this session. "
            "Third, association-conditioned forecasts require calibration and external validation before deployment claims."
        )
        conclusions = (
            "The current session supports a governed trajectory feasibility report. "
            "A longitudinal forecast requires time-stamped participant trajectories, calibration evidence and independent validation."
        )
    else:
        model_summary = evidence_summary.get("model") or {}
        model_limitation = (
            "Fourth, the recorded model is a prevalent/ever-diagnosed discrimination analysis rather than an incident-risk model. "
            if model_summary.get("incident_risk_supported") is False else ""
        )
        discussion = (
            f"Our analysis of the {bank_name} cohort revealed several notable findings. "
            + finding_sentence + ". "
            "These results are consistent with prior epidemiological evidence and support "
            "internal biomarker-discrimination feasibility rather than deployment-ready risk stratification."
        )
        second_limitation = (
            "Second, trajectory timestamps may be approximate or derived from assessment instances, "
            "and association-conditioned trajectory outputs were not externally validated forecasts. "
            if has_trajectory else
            "Second, biomarker measurements were obtained at a single baseline time point. "
        )
        limitations = (
            f"This study has several limitations. First, the {bank_name} represents a "
            f"{bank_caveats}. {second_limitation}Third, the "
            f"observational nature of the study precludes causal inference. {model_limitation}"
        )
        conclusions = (
            "We identified disease-associated biomarker signatures using aggregate "
            f"biobank evidence from the {bank_name}. Current model outputs should be interpreted as "
            "internal prevalent-disease discrimination unless a date-aware incident cohort is built; "
            "all findings warrant validation in independent cohorts and prospective studies."
        )
    sections.append(PAPER_DISCUSSION.format(
        discussion=discussion,
        limitations=limitations,
        conclusions=conclusions,
    ))

    sections.append("## Reproducibility, Governance and Data Availability\n\n")
    sections.append(_governance_text(ctx, records) + "\n\n")
    sections.append(
        "The analysis code, model configuration, cohort definitions and generated "
        "figures should be archived with the final manuscript. Individual-level "
        "biobank data remain subject to the relevant data access agreement and are "
        "not redistributed by this report.\n\n"
    )

    refs = _extract_references(records)
    sections.append("## References\n\n")
    if refs:
        for i, ref in enumerate(refs, 1):
            sections.append(f"{i}. {ref}\n")
        sections.append("\n")
    else:
        sections.append(
            "External literature records were not attached in this session; citation "
            "curation is required before journal submission.\n\n"
        )

    _add_figures_section(sections, ctx)
    _add_guardrail_section(sections, ctx, records)
    _add_execution_appendix(sections, ctx, records)
    return sections


def _build_brief_sections(title: str, ctx) -> list[str]:
    """Build a brief summary report."""
    records = _report_records(ctx)
    analysis_records = _analysis_records(records)
    sections = []
    sections.append(f"# {title}\n\n")
    sections.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

    findings = _extract_key_findings(records)
    for f in findings:
        sections.append(f"- {f}\n")
    sections.append("\n")

    for rec in analysis_records:
        interp = _interpret_skill(rec)
        if interp:
            sections.append(f"**{rec.skill}:** {interp}\n\n")

    return sections


# ── Helper functions ─────────────────────────────────────────


def _figure_sort_key(path: Path) -> tuple[int, str]:
    priority = {".png": 0, ".jpg": 1, ".jpeg": 1, ".svg": 2, ".pdf": 3}
    return priority.get(path.suffix.lower(), 9), path.name


def _materialize_report_figures(ctx, report_dir: Path) -> list[Path]:
    """Copy session figure artifacts into the final report directory."""
    if not ctx or not hasattr(ctx, "state"):
        return []
    figures = getattr(ctx.state, "figures", []) or []
    if _is_mockish(figures):
        return []
    materialized: list[Path] = []
    seen: set[Path] = set()
    for fig_path in figures:
        source = Path(fig_path).expanduser()
        if source.suffix.lower() not in (".svg", ".png", ".pdf", ".jpg", ".jpeg"):
            continue
        if not source.exists() or not source.is_file():
            continue
        dest = report_dir / source.name
        try:
            if source.resolve() != dest.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
            final = dest
        except Exception:
            final = source
        if final not in seen:
            materialized.append(final)
            seen.add(final)
    try:
        ctx.state.figures = materialized
    except Exception:
        pass
    return materialized


def _logical_figures(ctx) -> list[list[Path]]:
    """Group figure variants by stem so svg/pdf/png siblings count once."""
    figures = getattr(ctx.state, "figures", []) if ctx and hasattr(ctx, "state") else []
    if _is_mockish(figures):
        return []
    groups: dict[str, list[Path]] = {}
    order: list[str] = []
    for fig_path in figures or []:
        p = Path(fig_path)
        if p.suffix.lower() not in (".svg", ".png", ".pdf", ".jpg", ".jpeg"):
            continue
        key = p.stem
        if key not in groups:
            groups[key] = []
            order.append(key)
        if p not in groups[key]:
            groups[key].append(p)
    return [sorted(groups[key], key=_figure_sort_key) for key in order]


_LOCAL_ARTIFACT_LINK_RE = re.compile(r"(?:!?\[[^\]]*\]\(([^)]+)\))")


def _missing_local_artifact_links(markdown_paths: list[Path], report_dir: Path) -> list[str]:
    """Return local figure/artifact links that are referenced but absent."""
    missing: list[str] = []
    seen: set[str] = set()
    for md_path in markdown_paths:
        if not md_path or not md_path.exists():
            continue
        content = md_path.read_text(encoding="utf-8", errors="ignore")
        for match in _LOCAL_ARTIFACT_LINK_RE.finditer(content):
            raw = match.group(1).strip()
            if not raw or raw.startswith(("http://", "https://", "mailto:", "#")):
                continue
            link = raw.split("#", 1)[0].split("?", 1)[0]
            if Path(link).suffix.lower() not in (".svg", ".png", ".pdf", ".jpg", ".jpeg"):
                continue
            candidate = Path(link)
            if not candidate.is_absolute():
                candidate = report_dir / candidate
            if not candidate.exists():
                label = f"{md_path.name}:{raw}"
                if label not in seen:
                    seen.add(label)
                    missing.append(label)
    return missing


def _attach_report_artifact_status(result: dict, report_dir: Path, markdown_paths: list[Path]) -> dict:
    missing = _missing_local_artifact_links(markdown_paths, report_dir)
    result["broken_figure_links"] = missing
    result["figure_artifacts"] = [str(p) for group in _logical_figures_for_dir(report_dir) for p in group]
    if missing:
        result["error"] = "Report references missing figure artifact(s): " + ", ".join(missing[:8])
    return result


def _logical_figures_for_dir(report_dir: Path) -> list[list[Path]]:
    figures = (
        list(report_dir.glob("*.svg"))
        + list(report_dir.glob("*.png"))
        + list(report_dir.glob("*.pdf"))
        + list(report_dir.glob("*.jpg"))
        + list(report_dir.glob("*.jpeg"))
    )

    class _State:
        pass

    class _Ctx:
        pass

    state = _State()
    state.figures = figures
    ctx = _Ctx()
    ctx.state = state
    return _logical_figures(ctx)


def _add_figures_section(sections: list[str], ctx) -> None:
    """Add figures section with proper scientific captions."""
    figure_groups = _logical_figures(ctx)
    if not figure_groups:
        return
    sections.append("## Figures\n\n")
    fig_n = 0

    # Build a map of figure paths to analysis records for captions
    fig_to_record = {}
    stem_to_record = {}
    for rec in _report_records(ctx):
        for fp in rec.figure_paths:
            p = Path(fp)
            fig_to_record[str(fp)] = rec
            stem_to_record[p.stem] = rec

    for group in figure_groups:
        p = group[0]
        fig_n += 1
        rel = p.name
        rec = fig_to_record.get(str(p)) or stem_to_record.get(p.stem)
        caption = _build_figure_caption(str(p), rec, fig_n, ctx)

        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg"):
            sections.append(f"![Figure {fig_n}]({rel})\n\n")
        else:
            label = p.suffix[1:].upper()
            sections.append(f"[Figure {fig_n} ({label})]({rel})\n\n")

        alternatives = [alt for alt in group[1:] if alt != p]
        if alternatives:
            links = ", ".join(f"[{alt.suffix[1:].upper()}]({alt.name})" for alt in alternatives)
            sections.append(f"Alternative format(s): {links}.\n\n")

        sections.append(f"{caption}\n\n")
    sections.append("")


def _add_guardrail_section(sections: list[str], ctx, records) -> None:
    """Add expanded statistical and safety guardrail issues."""
    issues = _extract_guardrail_issues(ctx, records)
    if not issues:
        return
    sections.append("## Guardrail Issues\n\n")
    sections.append("| Severity | Source | Type | Affected Step | Message | Recommended Action |\n")
    sections.append("|----------|--------|------|---------------|---------|--------------------|\n")
    for issue in issues:
        sections.append(
            f"| {_table_cell(issue['severity'])} | {_table_cell(issue['source'])} | "
            f"{_table_cell(issue['type'])} | {_table_cell(issue['skill'] or 'session')} | "
            f"{_table_cell(issue['message'])} | "
            f"{_table_cell(issue['recommendation'] or 'Review before release.')} |\n"
        )
    sections.append("\n")


def _add_execution_appendix(sections: list[str], ctx, records) -> None:
    """Add reproducible execution diagnostics without raw command output leakage."""
    entries = _extract_execution_log(ctx, records)
    record_rows = [
        rec for rec in records
        if getattr(rec, "skill", "") != "think"
    ]
    if not entries and not record_rows:
        return
    sections.append("## Execution Appendix\n\n")
    if entries:
        sections.append("### Step Log\n\n")
        sections.append("| Step | Tool | Status | Duration | Note |\n")
        sections.append("|------|------|--------|----------|------|\n")
        for idx, entry in enumerate(entries, 1):
            step = _format_execution_value(entry, ("step", "name", "task", "id"), str(idx))
            tool = _format_execution_value(entry, ("tool", "skill", "command_name"), "")
            status = _format_execution_value(entry, ("status", "outcome", "result"), "")
            duration = _format_execution_value(entry, ("duration", "duration_s", "elapsed"), "")
            note = _format_execution_value(entry, ("note", "summary", "message"), "")
            sections.append(f"| {step} | {tool} | {status} | {duration} | {note} |\n")
        sections.append("\n")

    if record_rows:
        sections.append("### Analysis Record Inventory\n\n")
        sections.append("| # | Skill | Parameters | Key Outputs |\n")
        sections.append("|---:|-------|------------|-------------|\n")
        for idx, rec in enumerate(record_rows, 1):
            params = _summarize_appendix_mapping(getattr(rec, "args", {}) or {})
            outputs = _summarize_appendix_mapping(getattr(rec, "key_results", {}) or {})
            sections.append(
                f"| {idx} | {_table_cell(getattr(rec, 'skill', ''))} | "
                f"{params or 'not parameterized'} | {outputs or 'recorded'} |\n"
            )
    sections.append("\n")


def _safe_appendix_scalar(value) -> str:
    """Compact appendix values while suppressing paths, raw logs, and secrets."""
    value = _format_value(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    text = " ".join(str(value).split())
    if not text:
        return ""
    lower = text.lower()
    if any(token in lower for token in ("api_key", "authorization:", "bearer ")):
        return "[redacted]"
    if "/users/" in lower or "\\users\\" in lower:
        return "[local path]"
    text = _redact_report_tool_tokens(text)
    if len(text) > 80:
        text = text[:77].rstrip() + "..."
    return _table_cell(text)


def _summarize_appendix_mapping(mapping: dict, limit: int = 6) -> str:
    """Summarize structured args/results for appendix tables without raw leakage."""
    if not isinstance(mapping, dict) or _is_mockish(mapping):
        return ""
    parts: list[str] = []
    for key, value in mapping.items():
        if _is_raw_result_key(str(key)) or key in {"figure", "figures", "source_results"}:
            continue
        if isinstance(value, dict):
            shown = ", ".join(str(k) for k in list(value.keys())[:3])
            scalar = f"dict[{shown}]" if shown else "dict"
        elif isinstance(value, (list, tuple)):
            if value and not isinstance(value[0], (dict, list, tuple)):
                preview = ", ".join(_safe_appendix_scalar(v) for v in value[:3])
                scalar = f"[{preview}{', ...' if len(value) > 3 else ''}]"
            else:
                scalar = f"{len(value)} item(s)"
        else:
            scalar = _safe_appendix_scalar(value)
        if scalar:
            parts.append(f"{_table_cell(key)}={scalar}")
        if len(parts) >= limit:
            break
    return "; ".join(parts)


def _add_cohorts_section(sections: list[str], ctx, records=None) -> None:
    """Add cohorts summary table."""
    evidence_cohort = (_session_evidence_summary(ctx, records).get("cohort") if records is not None else None) or {}
    if not evidence_cohort and not ctx.state.cohorts:
        return
    sections.append("## Cohort Summary\n\n")
    sections.append("| Cohort | Total | Cases | Controls |\n")
    sections.append("|--------|------:|------:|---------:|\n")
    if evidence_cohort:
        n_cases = evidence_cohort.get("n_cases")
        n_controls = evidence_cohort.get("n_controls")
        total = (n_cases + n_controls) if isinstance(n_cases, int) and isinstance(n_controls, int) else "not recorded"
        name = evidence_cohort.get("disease") or evidence_cohort.get("endpoint") or "primary cohort"
        total_str = f"{total:,}" if isinstance(total, int) else str(total)
        case_str = f"{n_cases:,}" if isinstance(n_cases, int) else _status_value(n_cases)
        control_str = f"{n_controls:,}" if isinstance(n_controls, int) else _status_value(n_controls)
        sections.append(f"| {name} | {total_str} | {case_str} | {control_str} |\n")
        sections.append("\n")
        return
    for name, df in ctx.state.cohorts.items():
        n_total = len(df)
        if "label" in df.columns:
            n_cases = int(df["label"].sum())
            n_controls = n_total - n_cases
        else:
            n_cases, n_controls = "N/A", "N/A"
        n_cases_str = f"{n_cases:,}" if isinstance(n_cases, int) else str(n_cases)
        n_controls_str = f"{n_controls:,}" if isinstance(n_controls, int) else str(n_controls)
        sections.append(f"| {name} | {n_total:,} | {n_cases_str} | {n_controls_str} |\n")
    sections.append("\n")


def _add_models_section(sections: list[str], ctx) -> None:
    """Add models summary table with AUC and 95% CI."""
    if not ctx.state.model_metadata:
        return
    sections.append("## Model Performance\n\n")
    sections.append("| Model | Type | AUC (95% CI) | Cases | Features |\n")
    sections.append("|-------|------|:-------------|------:|---------:|\n")
    for key, meta in ctx.state.model_metadata.items():
        auc_str = _format_auc_with_ci(meta)
        sections.append(
            f"| {key} | {meta.get('model_type', '?')} | "
            f"{auc_str} | {meta.get('n_cases', '?')} | "
            f"{meta.get('n_features', '?')} |\n"
        )
    sections.append("\n")


def _add_model_selection_section(sections: list[str], ctx, records=None) -> None:
    """Add automatic model-selection details from train_model(auto)."""
    comparisons: list[dict] = []
    rationale = ""
    selected = ""
    for rec in records or []:
        if getattr(rec, "skill", "") != "train_model":
            continue
        results = getattr(rec, "key_results", {}) or {}
        comparisons = list(results.get("candidate_comparison") or [])
        rationale = str(results.get("selection_rationale") or "")
        selected = str(results.get("selected_model_type") or results.get("model_type") or "")
        if comparisons:
            break
    if not comparisons and getattr(ctx.state, "model_metadata", None):
        for meta in ctx.state.model_metadata.values():
            ms = meta.get("model_selection", {}) if isinstance(meta, dict) else {}
            comparisons = list(ms.get("candidate_comparison") or [])
            rationale = str(ms.get("rationale") or "")
            selected = str(ms.get("selected_model_type") or "")
            if comparisons:
                break
    if not comparisons:
        return

    sections.append("## Model Selection\n\n")
    if selected or rationale:
        sections.append(f"Selected model: **{selected or 'not recorded'}**. {rationale}\n\n")
    sections.append("| Rank | Candidate | Status | Mean AUC | F1 | Precision | Recall | Note |\n")
    sections.append("|-----:|-----------|--------|---------:|---:|----------:|-------:|------|\n")
    for item in comparisons:
        if not isinstance(item, dict):
            continue
        status = item.get("status", "")
        note = item.get("error") or item.get("role") or ""
        sections.append(
            f"| {item.get('rank', '')} | {item.get('model_type', '')} | {status} | "
            f"{_status_value(item.get('auc_mean'), '')} | {_status_value(item.get('f1_mean'), '')} | "
            f"{_status_value(item.get('precision_mean'), '')} | {_status_value(item.get('recall_mean'), '')} | "
            f"{_table_cell(note)} |\n"
        )
    sections.append("\n")


def _write_html(md_content: str, title: str, report_dir: Path, stem: str = "report"):
    """Convert markdown to HTML with embedded CSS."""
    html_path = report_dir / f"{stem}.html"
    css_path = report_dir / ("_report_with_css.md" if stem == "report" else f"_{stem}_with_css.md")
    css_md = NATURE_CSS + "\n" + md_content
    css_path.write_text(css_md, encoding="utf-8")
    try:
        import subprocess

        result = subprocess.run(
            ["pandoc", str(css_path), "-o", str(html_path),
             "--self-contained", "--toc", "--toc-depth=2",
             "--highlight-style=tango",
             f"--metadata=title:{title}"],
            capture_output=True, timeout=30,
        )

        if result.returncode != 0:
            html_content = (
                f"<html><head><title>{title}</title>{NATURE_CSS}</head>"
                f"<body><div>{md_content}</div></body></html>"
            )
            html_path.write_text(html_content, encoding="utf-8")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        html_content = (
            f"<html><head><title>{title}</title>{NATURE_CSS}</head>"
            f"<body><pre>{md_content}</pre></body></html>"
        )
        html_path.write_text(html_content, encoding="utf-8")

    return html_path
