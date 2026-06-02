from pathlib import Path
from types import SimpleNamespace


def test_academic_report_polisher_rebuilds_juvenile_mechanism_narrative(tmp_path):
    from biobank_agent.skills.academic_report_polisher import polish_markdown_report

    report = tmp_path / "report.md"
    report.write_text(
        """# Juvenile Hair Whitening Multi-Omics Mechanism Analysis

**Date:** 2026-05-21 15:58
**Format:** Technical Report

---

## Executive Summary

This report summarizes 17 computational analysis step(s) performed on the VirtualCell/BWhair Multimodal Cohort dataset.

## Study Status

| Field | Status |
|-------|--------|
| Cohort evidence | not recorded cases; not recorded controls |
| Predictive model | not recorded; AUC=not recorded; 95% CI=not recorded |

## 1. Jh Variant Discovery

Recorded quantitative outputs include n_wgs_samples=28, n_candidate_variants=15.

| Metric | Value |
|--------|-------|
| n_wgs_samples | 28 |
| wgs_phenotype_counts | {'Senile_White': 12, 'Juvenile_White': 10, 'Vitiligo_White': 6} |
| n_candidate_variants | 15 |
| candidate_variants | [{'gene': 'TYR', 'evidence': 'curated_locus_fallback'}, {'gene': 'CXCL10'}] |

## 2. Tf Binding Disruption

| Metric | Value |
|--------|-------|
| n_tfbs_candidates | 28 |
| tf_binding_hits | [{'gene': 'TYR', 'evidence': 'motif_prior_fallback'}] |

## 3. Scatac Accessibility Differential

| Metric | Value |
|--------|-------|
| n_juvenile_donors | 10 |
| case_manifest_cells | 551978 |
| control_manifest_cells | 638143 |
| status | metadata_ready_matrix_deferred |

## 4. Multiomics Mechanism Prioritization

| Metric | Value |
|--------|-------|
| n_prioritized_hypotheses | 15 |
""",
        encoding="utf-8",
    )

    result = polish_markdown_report(report, strict=True)
    text = report.read_text(encoding="utf-8")
    main = text.split("## Execution Appendix", 1)[0]

    assert result["polished"] is True
    assert (tmp_path / "report_raw.md").exists()
    assert "## Abstract" in text
    assert "### Task 1: Genomic Candidate Variants" in text
    assert "### Task 2: Epigenomic Regulatory Context" in text
    assert "### Task 3: Accessibility and Expression Coupling" in text
    assert "### Task 4: Spatial Localization and Cell-Cell Interaction" in text
    assert "not causal" in text.lower() or "not sufficient to claim" in text.lower()
    assert "Recorded quantitative outputs" not in main
    assert "not recorded cases; not recorded controls" not in main
    assert "--------=-------" not in main
    assert "[{'gene':" not in main
    assert result["quality_checks"]["no_forbidden_main_body_patterns"] is True


def test_generate_report_calls_academic_polisher_by_default(tmp_path):
    from biobank_agent.skills.report import generate_report

    ctx = SimpleNamespace(
        report_dir=tmp_path / "reports",
        settings=SimpleNamespace(
            biobank_name="VirtualCell/BWhair Multimodal Cohort",
            biobank_description="donor-linked multimodal cohort",
            biobank_caveats="small donor count",
        ),
        state=SimpleNamespace(
            records=[],
            figures=[],
            cohorts={},
            models={},
            model_metadata={},
            provenances=[],
            executive_findings=[],
            execution_log=[],
            guardrail_issues=[],
        ),
    )
    result = generate_report("Juvenile Hair Whitening Multi-Omics Mechanism Analysis", format="report", ctx=ctx)

    assert result["polished"] is True
    assert result["polisher"]["quality_checks"]["has_abstract"] is True
    assert Path(result["polisher"]["raw_markdown"]).exists()
    assert Path(result["polisher"]["quality_json"]).exists()
