"""Report generator — compile analyses into structured, readable documents.

Supports two modes:
- 'report' (default): Technical report with executive summary, Key Findings, interpretive text
- 'paper': IMRaD paper draft with Nature-quality prose

Output formats: Markdown, HTML (self-contained with CSS), PDF (via pandoc)
"""

from pathlib import Path
from datetime import datetime

from biobank_agent.registry import skill
from biobank_agent.utils.report_templates import (
    REPORT_HEADER,
    KEY_FINDINGS_BOX,
    EXECUTIVE_SUMMARY,
    SECTION_HEADER,
    PAPER_ABSTRACT,
    PAPER_INTRODUCTION,
    PAPER_METHODS,
    PAPER_RESULTS,
    PAPER_DISCUSSION,
    NATURE_CSS,
)


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
            return (
                f"Prevalence analysis examined {n_diseases} disease codes. "
                f"Top finding: {top}. This establishes the epidemiological baseline "
                f"for downstream analyses."
            )

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

        elif skill_name == "train_model":
            auc = _format_value(results.get("mean_auc", results.get("auc", "?")))
            model_type = args.get("model_type", results.get("model_type", "?"))
            n_features = results.get("n_features", "?")
            if isinstance(auc, (int, float)):
                quality = (
                    "excellent" if auc > 0.9 else
                    "good" if auc > 0.8 else
                    "moderate" if auc > 0.7 else "limited"
                )
                return (
                    f"The {model_type} model achieved a mean AUC of {auc:.4f}, indicating "
                    f"{quality} discriminative ability using {n_features} features."
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
                    f"cases and controls (P={float(p_val):.2e}, Mann-Whitney U test)."
                )
            return f"Distribution analysis of {biomarker} completed."

        elif skill_name == "survival":
            icd10 = args.get("icd10_code", "?")
            p_val = results.get("log_rank_p", None)
            if p_val is not None:
                p_val = _format_value(p_val)
                return (
                    f"Survival analysis for {icd10}: log-rank test P={float(p_val):.2e}. "
                    f"See Kaplan-Meier curve for time-to-event distributions."
                )
            return f"Survival analysis for {icd10} completed."

        elif skill_name == "feature_importance":
            top_feat = results.get("top_feature")
            if not top_feat:
                top_feats = results.get("top_features", [])
                top_feat = top_feats[0] if isinstance(top_feats, list) and top_feats else "?"
            return (
                f"Feature importance analysis identified {top_feat} as the most "
                f"predictive biomarker. See SHAP beeswarm plot for feature contributions."
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
            pct = results.get("mean_missing_pct", results.get("overall_missing", "?"))
            return f"Missing data analysis: mean missingness = {pct}%. See pattern matrix."

    except Exception:
        pass

    # Generic fallback
    key_nums = {k: _format_value(v) for k, v in results.items()
                if isinstance(v, (int, float)) or (hasattr(v, "item") and callable(v.item))}
    if key_nums:
        nums_str = ", ".join(f"{k}={v}" for k, v in list(key_nums.items())[:3])
        return f"Analysis completed: {nums_str}."
    return f"Analysis step `{skill_name}` completed."


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

    return findings[:5] if findings else ["Analysis completed -- see details below"]


@skill(
    name="generate_report",
    description="Generate a structured analysis report from all session analyses. "
                "Supports 'report' format (technical with Key Findings) or 'paper' "
                "format (IMRaD structure). Produces Markdown, HTML, and optionally PDF output.",
    parameters={
        "title": {
            "type": "string",
            "description": "Report title",
            "default": "UK Biobank Analysis Report",
        },
        "format": {
            "type": "string",
            "description": "Output format: 'report' (technical), 'paper' (IMRaD draft), or 'brief' (summary only)",
            "default": "report",
        },
    },
    required=[],
)
def generate_report(
    title: str = "UK Biobank Analysis Report",
    format: str = "report",
    *,
    ctx=None,
) -> dict:
    """Generate a structured, readable analysis report."""
    report_dir = ctx.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    if format == "paper":
        sections = _build_paper_sections(title, ctx)
    elif format == "brief":
        sections = _build_brief_sections(title, ctx)
    else:
        sections = _build_report_sections(title, ctx)

    # Write markdown
    md_content = "\n".join(sections)
    md_path = report_dir / "report.md"
    md_path.write_text(md_content)

    # Write HTML with embedded CSS
    html_path = _write_html(md_content, title, report_dir)

    return {
        "report_dir": str(report_dir),
        "markdown": str(md_path),
        "html": str(html_path) if html_path else None,
        "format": format,
        "n_sections": len(ctx.state.records),
        "n_figures": len(ctx.state.figures),
    }


# ── Report builders ──────────────────────────────────────────


def _build_report_sections(title: str, ctx) -> list[str]:
    """Build technical report with executive summary and Key Findings."""
    sections = []

    sections.append(REPORT_HEADER.format(
        title=title,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        format_name="Technical Report",
    ))

    # Key Findings
    if ctx.state.records:
        findings = _extract_key_findings(ctx.state.records)
        findings_md = "\n".join(f"> - {f}" for f in findings)
        sections.append(KEY_FINDINGS_BOX.format(findings=findings_md))

    # Executive Summary
    if ctx.state.records:
        n_analyses = len([r for r in ctx.state.records if r.skill != "think"])
        n_figs = len(ctx.state.figures)
        n_cohorts = len(ctx.state.cohorts)
        n_models = len(ctx.state.models)
        parts = [f"This report summarizes {n_analyses} analyses performed on the UK Biobank dataset."]
        if n_cohorts:
            parts.append(f"{n_cohorts} disease cohort(s) were constructed.")
        if n_models:
            parts.append(f"{n_models} predictive model(s) were trained and evaluated.")
        if n_figs:
            parts.append(f"{n_figs} publication-quality figures were generated.")
        sections.append(EXECUTIVE_SUMMARY.format(summary=" ".join(parts)))

    # Analysis sections with interpretive text
    section_n = 0
    for rec in ctx.state.records:
        if rec.skill == "think":
            continue
        section_n += 1

        skill_title = rec.skill.replace("_", " ").title()
        sections.append(SECTION_HEADER.format(n=section_n, title=skill_title))

        if rec.args:
            args_str = ", ".join(f"`{k}={v}`" for k, v in rec.args.items())
            sections.append(f"**Parameters:** {args_str}\n\n")

        interp = _interpret_skill(rec)
        if interp:
            sections.append(f"{interp}\n\n")

        if rec.key_results and "error" not in rec.key_results:
            metrics = {k: _format_value(v) for k, v in rec.key_results.items()
                       if k not in ("figure", "figures", "_retried_with")}
            if metrics:
                sections.append("| Metric | Value |\n|--------|-------|\n")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        v_str = f"{v:.4f}" if abs(v) < 100 else f"{v:,.2f}"
                    elif isinstance(v, list) and len(v) > 5:
                        v_str = f"[{v[0]}, ..., {v[-1]}] ({len(v)} items)"
                    else:
                        v_str = str(v)
                    sections.append(f"| {k} | {v_str} |\n")
                sections.append("\n")

        if rec.key_results and "error" in rec.key_results:
            sections.append(f"> **Warning:** {rec.key_results['error']}\n\n")

        sections.append("")

    _add_figures_section(sections, ctx)
    _add_cohorts_section(sections, ctx)
    _add_models_section(sections, ctx)

    sections.append("## Methodology Notes\n\n")
    sections.append(
        "All analyses were performed on the UK Biobank cohort (N=502,370) using "
        "DuckDB for data access and Python scientific stack for computation. "
        "Statistical tests used two-sided P-values with significance threshold "
        "alpha=0.05. Multiple testing correction applied via FDR (Benjamini-Hochberg) "
        "where indicated. Figures follow Nature journal guidelines "
        "(Arial 7pt, 300 DPI, Okabe-Ito palette).\n\n"
    )

    return sections


def _build_paper_sections(title: str, ctx) -> list[str]:
    """Build IMRaD paper draft."""
    sections = []

    sections.append(f"# {title}\n\n")
    sections.append("*CHEN Pengan*\n\n")
    sections.append("*The Chinese University of Hong Kong*\n\n")
    sections.append("---\n\n")

    findings = _extract_key_findings(ctx.state.records)

    abstract = (
        "**Background:** We analysed the UK Biobank cohort (N=502,370) to identify "
        "disease-associated biomarkers and build predictive models. "
        "**Methods:** Case-control cohorts were constructed from ICD-10 coded hospital "
        "episode statistics. Gradient-boosted models were trained with 5-fold "
        "cross-validation. Feature importance was assessed via SHAP values. "
        "**Results:** " + ". ".join(findings[:3]) + ". "
        "**Conclusions:** These findings highlight potential biomarkers for further "
        "clinical investigation and risk stratification."
    )
    sections.append(PAPER_ABSTRACT.format(
        abstract=abstract,
        keywords="UK Biobank, biomarkers, machine learning, disease prediction, epidemiology",
    ))

    intro = (
        "The UK Biobank is a large-scale prospective cohort study comprising over "
        "500,000 participants aged 40-69 at recruitment, with extensive phenotypic, "
        "genetic, and health outcome data. This rich resource enables systematic "
        "identification of disease-associated biomarkers and construction of "
        "predictive models.\n\n"
        "In this analysis, we leverage the UK Biobank's hospital episode statistics "
        "(6.9 million ICD-10 coded diagnoses) alongside blood biochemistry, "
        "haematology, and anthropometric measurements to characterise disease "
        "cohorts and identify discriminative biomarker signatures."
    )
    sections.append(PAPER_INTRODUCTION.format(introduction=intro))

    cohort_info = ""
    if ctx.state.cohorts:
        for name, df in ctx.state.cohorts.items():
            n_cases = int(df["label"].sum()) if "label" in df.columns else "N/A"
            cohort_info += (
                f"For {name}, {n_cases:,} cases were identified from hospital episode "
                f"statistics and matched with {len(df) - n_cases:,} controls. "
            )

    model_info = ""
    if ctx.state.model_metadata:
        for key, meta in ctx.state.model_metadata.items():
            model_info += (
                f"A {meta.get('model_type', 'gradient-boosted')} classifier was trained "
                f"with {meta.get('n_features', 'all available')} features using "
                f"5-fold stratified cross-validation. "
            )

    sections.append(PAPER_METHODS.format(
        study_population=(
            f"This study utilised data from 502,370 UK Biobank participants. "
            f"{cohort_info}"
            "Diagnoses were extracted from hospital episode statistics (HES) using "
            "ICD-10 coding. Biomarker measurements were obtained from baseline assessment."
        ),
        statistical_analysis=(
            "Continuous variables were compared using Mann-Whitney U tests. "
            "Categorical variables were compared using chi-squared tests. "
            "Multiple testing correction was applied using the Benjamini-Hochberg "
            f"procedure. {model_info}"
            "Model performance was assessed using area under the receiver operating "
            "characteristic curve (AUC-ROC) with 95% confidence intervals."
        ),
    ))

    results_parts = []
    for rec in ctx.state.records:
        interp = _interpret_skill(rec)
        if interp:
            results_parts.append(interp)
    sections.append(PAPER_RESULTS.format(
        results="\n\n".join(results_parts) if results_parts else "Results pending.",
    ))

    sections.append(PAPER_DISCUSSION.format(
        discussion=(
            "Our analysis of the UK Biobank cohort revealed several notable findings. "
            + " ".join(findings[:3]) + ". "
            "These results are consistent with prior epidemiological evidence and suggest "
            "potential avenues for biomarker-based risk stratification."
        ),
        limitations=(
            "This study has several limitations. First, the UK Biobank represents a "
            "'healthy volunteer' cohort with known selection biases. Second, biomarker "
            "measurements were obtained at a single baseline time point. Third, the "
            "observational nature of the study precludes causal inference."
        ),
        conclusions=(
            "We identified disease-associated biomarker signatures using machine learning "
            "approaches applied to the UK Biobank. These findings warrant validation in "
            "independent cohorts and prospective studies."
        ),
    ))

    _add_figures_section(sections, ctx)
    return sections


def _build_brief_sections(title: str, ctx) -> list[str]:
    """Build a brief summary report."""
    sections = []
    sections.append(f"# {title}\n\n")
    sections.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

    findings = _extract_key_findings(ctx.state.records)
    for f in findings:
        sections.append(f"- {f}\n")
    sections.append("\n")

    for rec in ctx.state.records:
        if rec.skill == "think":
            continue
        interp = _interpret_skill(rec)
        if interp:
            sections.append(f"**{rec.skill}:** {interp}\n\n")

    return sections


# ── Helper functions ─────────────────────────────────────────


def _add_figures_section(sections: list[str], ctx) -> None:
    """Add figures section to report."""
    if not ctx.state.figures:
        return
    sections.append("## Figures\n\n")
    fig_n = 0
    for fig_path in ctx.state.figures:
        p = Path(fig_path)
        if p.suffix in (".svg", ".png", ".pdf"):
            fig_n += 1
            rel = p.name
            if p.suffix == ".svg":
                # SVG: embed inline in markdown for self-contained reports
                try:
                    svg_content = p.read_text(encoding="utf-8")
                    sections.append(
                        f"<!-- Figure {fig_n}: {p.stem} -->\n{svg_content}\n\n"
                    )
                except Exception:
                    # Fallback to image reference if read fails
                    sections.append(f"![Figure {fig_n}]({rel})\n\n")
            elif p.suffix == ".png":
                # PNG: embed as markdown image (backward compat)
                sections.append(f"![Figure {fig_n}]({rel})\n\n")
            else:
                # PDF: link instead of inline embed
                sections.append(f"[Figure {fig_n} (PDF)]({rel})\n\n")
    sections.append("")


def _add_cohorts_section(sections: list[str], ctx) -> None:
    """Add cohorts summary table."""
    if not ctx.state.cohorts:
        return
    sections.append("## Cohort Summary\n\n")
    sections.append("| Cohort | Total | Cases | Controls |\n")
    sections.append("|--------|------:|------:|---------:|\n")
    for name, df in ctx.state.cohorts.items():
        n_total = len(df)
        if "label" in df.columns:
            n_cases = int(df["label"].sum())
            n_controls = n_total - n_cases
        else:
            n_cases, n_controls = "N/A", "N/A"
        sections.append(f"| {name} | {n_total:,} | {n_cases:,} | {n_controls:,} |\n")
    sections.append("\n")


def _add_models_section(sections: list[str], ctx) -> None:
    """Add models summary table."""
    if not ctx.state.model_metadata:
        return
    sections.append("## Model Performance\n\n")
    sections.append("| Model | Type | AUC | Cases | Features |\n")
    sections.append("|-------|------|----:|------:|---------:|\n")
    for key, meta in ctx.state.model_metadata.items():
        auc_val = meta.get("auc")
        if auc_val is not None:
            auc_val = _format_value(auc_val)
            auc_str = f"{float(auc_val):.4f}"
        else:
            auc_str = "N/A"
        sections.append(
            f"| {key} | {meta.get('model_type', '?')} | "
            f"{auc_str} | {meta.get('n_cases', '?')} | "
            f"{meta.get('n_features', '?')} |\n"
        )
    sections.append("\n")


def _write_html(md_content: str, title: str, report_dir: Path):
    """Convert markdown to HTML with embedded CSS."""
    html_path = report_dir / "report.html"
    try:
        import subprocess
        css_md = NATURE_CSS + "\n" + md_content
        css_path = report_dir / "_report_with_css.md"
        css_path.write_text(css_md)

        result = subprocess.run(
            ["pandoc", str(css_path), "-o", str(html_path),
             "--self-contained", "--toc", "--toc-depth=2",
             "--highlight-style=tango",
             f"--metadata=title:{title}"],
            capture_output=True, timeout=30,
        )
        css_path.unlink(missing_ok=True)

        if result.returncode != 0:
            html_content = (
                f"<html><head><title>{title}</title>{NATURE_CSS}</head>"
                f"<body><div>{md_content}</div></body></html>"
            )
            html_path.write_text(html_content)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        html_content = (
            f"<html><head><title>{title}</title>{NATURE_CSS}</head>"
            f"<body><pre>{md_content}</pre></body></html>"
        )
        html_path.write_text(html_content)

    return html_path
