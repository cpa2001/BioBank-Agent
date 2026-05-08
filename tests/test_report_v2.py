"""Test report generation — report, paper, and brief formats."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace


def _make_record(skill_name, args=None, key_results=None, fig_paths=None):
    """Create a mock AnalysisRecord."""
    rec = SimpleNamespace(
        timestamp="2025-01-01 12:00",
        skill=skill_name,
        args=args or {},
        key_results=key_results or {},
        figure_paths=fig_paths or [],
        interpretation="",
    )
    return rec


def _make_ctx(tmp_path, records=None, figures=None, cohorts=None,
              models=None, model_metadata=None):
    """Build a mock context for report generation."""
    ctx = MagicMock()
    ctx.report_dir = tmp_path / "reports"
    ctx.report_dir.mkdir(parents=True, exist_ok=True)
    ctx.settings.biobank_name = "UK Biobank"
    ctx.settings.biobank_description = "a prospective population cohort"
    ctx.settings.biobank_caveats = "healthy volunteer bias"
    ctx.state.records = records or []
    ctx.state.figures = figures or []
    ctx.state.cohorts = cohorts or {}
    ctx.state.models = models or {}
    ctx.state.model_metadata = model_metadata or {}
    ctx.state.provenances = []
    return ctx


class TestReportImport:
    """Test generate_report skill registration."""

    def test_generate_report_imports(self):
        """Test that generate_report skill can be imported."""
        from biobank_agent.skills.report import generate_report
        assert callable(generate_report)
        assert generate_report._skill_name == "generate_report"

    def test_generate_report_schema(self):
        """Test generate_report has correct parameters."""
        from biobank_agent.skills.report import generate_report
        schema = generate_report._skill_schema
        func_def = schema["function"]
        assert "title" in func_def["parameters"]["properties"]
        assert "format" in func_def["parameters"]["properties"]


class TestReportFormat:
    """Test 'report' format — Key Findings + Executive Summary."""

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_report_format_produces_key_findings(self, mock_html, tmp_path):
        """Report format contains Key Findings section."""
        from biobank_agent.skills.report import generate_report

        records = [
            _make_record("train_model",
                         args={"model_type": "xgboost"},
                         key_results={"mean_auc": 0.87, "n_features": 25}),
        ]
        ctx = _make_ctx(tmp_path, records=records)
        result = generate_report(title="Test Report", format="report", ctx=ctx)

        md_path = Path(result["markdown"])
        md_text = md_path.read_text()

        assert "Key Findings" in md_text
        assert result["format"] == "report"

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_report_format_has_executive_summary(self, mock_html, tmp_path):
        """Report format contains Executive Summary section."""
        from biobank_agent.skills.report import generate_report

        records = [
            _make_record("prevalence",
                         key_results={"n_codes": 50, "top_code": "E11"}),
        ]
        ctx = _make_ctx(tmp_path, records=records)
        generate_report(title="Report", format="report", ctx=ctx)

        md_text = (ctx.report_dir / "report.md").read_text()
        assert "Executive Summary" in md_text

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_report_methodology_notes(self, mock_html, tmp_path):
        """Report format includes Methodology Notes section."""
        from biobank_agent.skills.report import generate_report

        ctx = _make_ctx(tmp_path, records=[_make_record("prevalence", key_results={"n_codes": 5})])
        generate_report(title="Report", format="report", ctx=ctx)

        md_text = (ctx.report_dir / "report.md").read_text()
        assert "Methodology Notes" in md_text

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_report_has_governance_and_no_placeholders(self, mock_html, tmp_path):
        """Technical report should avoid submission placeholders and include governance caveats."""
        from biobank_agent.skills.report import generate_report

        ctx = _make_ctx(tmp_path, records=[
            _make_record("train_model", key_results={"mean_auc": 0.82, "n_features": 20}),
        ])
        generate_report(title="Report", format="report", ctx=ctx)

        md_text = (ctx.report_dir / "report.md").read_text()
        assert "Reproducibility and Governance" in md_text
        assert "not causal effect estimates" in md_text
        assert "[CITATION_NEEDED]" not in md_text
        assert "References to be added" not in md_text


class TestPaperFormat:
    """Test 'paper' format — IMRaD sections."""

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_paper_format_produces_imrad_sections(self, mock_html, tmp_path):
        """Paper format contains Abstract, Introduction, Methods, Results, Discussion."""
        from biobank_agent.skills.report import generate_report

        records = [
            _make_record("cohort_summary",
                         args={"icd10_code": "E11"},
                         key_results={"n_cases": 5000, "n_controls": 10000}),
        ]
        ctx = _make_ctx(tmp_path, records=records)
        result = generate_report(title="Diabetes Paper", format="paper", ctx=ctx)

        md_text = Path(result["markdown"]).read_text()

        assert "## Abstract" in md_text
        assert "## Introduction" in md_text
        assert "## Methods" in md_text
        assert "## Results" in md_text
        assert "## Discussion" in md_text
        assert result["format"] == "paper"

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_paper_format_includes_limitations(self, mock_html, tmp_path):
        """Paper format includes Limitations and Conclusions sub-sections."""
        from biobank_agent.skills.report import generate_report

        ctx = _make_ctx(tmp_path, records=[_make_record("train_model", key_results={"mean_auc": 0.8})])
        generate_report(title="Paper", format="paper", ctx=ctx)

        md_text = (ctx.report_dir / "report.md").read_text()
        assert "Limitations" in md_text
        assert "Conclusions" in md_text

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_paper_format_has_data_availability_and_references(self, mock_html, tmp_path):
        """Paper draft should include governance/data availability and extract paper references."""
        from biobank_agent.skills.report import generate_report

        records = [
            _make_record("read_paper", key_results={
                "title": "Polygenic prediction in population biobanks",
                "authors": ["Smith A", "Jones B", "Lee C", "Patel D"],
                "doi": "10.1038/example",
            }),
            _make_record("cohort_summary", args={"icd10_code": "E11"}, key_results={"n_cases": 5000}),
        ]
        ctx = _make_ctx(tmp_path, records=records)
        generate_report(title="Paper", format="paper", ctx=ctx)

        md_text = (ctx.report_dir / "report.md").read_text()
        assert "Reproducibility, Governance and Data Availability" in md_text
        assert "Polygenic prediction in population biobanks" in md_text
        assert "10.1038/example" in md_text
        assert "External literature records were not attached" not in md_text


class TestBriefFormat:
    """Test 'brief' format — compact output."""

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_brief_format_is_compact(self, mock_html, tmp_path):
        """Brief format produces minimal output with key findings."""
        from biobank_agent.skills.report import generate_report

        records = [
            _make_record("train_model",
                         args={"model_type": "xgboost"},
                         key_results={"mean_auc": 0.85}),
        ]
        ctx = _make_ctx(tmp_path, records=records)
        result = generate_report(title="Brief", format="brief", ctx=ctx)

        md_text = Path(result["markdown"]).read_text()

        # Brief format should NOT have Executive Summary or Methodology Notes
        assert "Executive Summary" not in md_text
        assert "Methodology Notes" not in md_text
        assert result["format"] == "brief"


class TestReportWithEmptyRecords:
    """Test report with no analysis records."""

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_empty_records_produces_minimal_report(self, mock_html, tmp_path):
        """No records → report still created with minimal content."""
        from biobank_agent.skills.report import generate_report

        ctx = _make_ctx(tmp_path, records=[])
        result = generate_report(title="Empty Report", format="report", ctx=ctx)

        assert Path(result["markdown"]).exists()
        assert result["n_sections"] == 0
        assert result["n_figures"] == 0


class TestReportFigureEmbedding:
    """Test figure embedding across file types."""

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_svg_embedded_inline(self, mock_html, tmp_path):
        """SVG figures are embedded inline in markdown."""
        from biobank_agent.skills.report import generate_report

        svg_file = tmp_path / "reports" / "fig1.svg"
        svg_file.parent.mkdir(parents=True, exist_ok=True)
        svg_file.write_text('<svg xmlns="http://www.w3.org/2000/svg"><circle/></svg>')

        records = [_make_record("prevalence", key_results={"n_codes": 10})]
        ctx = _make_ctx(tmp_path, records=records, figures=[str(svg_file)])
        generate_report(title="SVG Report", format="report", ctx=ctx)

        md_text = (ctx.report_dir / "report.md").read_text()
        assert "<svg" in md_text

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_png_as_image_reference(self, mock_html, tmp_path):
        """PNG figures are embedded as markdown image references."""
        from biobank_agent.skills.report import generate_report

        png_file = tmp_path / "reports" / "fig1.png"
        png_file.parent.mkdir(parents=True, exist_ok=True)
        png_file.write_bytes(b"\x89PNG\r\n")

        records = [_make_record("prevalence", key_results={"n_codes": 10})]
        ctx = _make_ctx(tmp_path, records=records, figures=[str(png_file)])
        generate_report(title="PNG Report", format="report", ctx=ctx)

        md_text = (ctx.report_dir / "report.md").read_text()
        assert "![Figure" in md_text

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_pdf_as_link(self, mock_html, tmp_path):
        """PDF figures are embedded as links."""
        from biobank_agent.skills.report import generate_report

        pdf_file = tmp_path / "reports" / "fig1.pdf"
        pdf_file.parent.mkdir(parents=True, exist_ok=True)
        pdf_file.write_bytes(b"%PDF-1.4")

        records = [_make_record("prevalence", key_results={"n_codes": 10})]
        ctx = _make_ctx(tmp_path, records=records, figures=[str(pdf_file)])
        generate_report(title="PDF Report", format="report", ctx=ctx)

        md_text = (ctx.report_dir / "report.md").read_text()
        assert "(PDF)" in md_text


class TestReportInterpretiveText:
    """Test interpretive text generation for different skills."""

    def test_interpret_train_model(self):
        """train_model record → AUC quality text."""
        from biobank_agent.skills.report import _interpret_skill

        rec = _make_record("train_model",
                           args={"model_type": "xgboost"},
                           key_results={"mean_auc": 0.92, "n_features": 30})
        text = _interpret_skill(rec)

        assert "xgboost" in text.lower()
        assert "0.920" in text
        assert "excellent" in text

    def test_interpret_prevalence(self):
        """prevalence record → epidemiological baseline text."""
        from biobank_agent.skills.report import _interpret_skill

        rec = _make_record("prevalence",
                           key_results={"n_codes": 200, "top_disease": "E11"})
        text = _interpret_skill(rec)

        assert "200" in text
        assert "E11" in text
        assert "baseline" in text.lower()

    def test_interpret_think_returns_empty(self):
        """'think' skill records produce no interpretive text."""
        from biobank_agent.skills.report import _interpret_skill

        rec = _make_record("think", key_results={"query": "reasoning"})
        text = _interpret_skill(rec)

        assert text == ""

    def test_formatting_helpers_and_figure_captions(self, tmp_path):
        import numpy as np
        from biobank_agent.skills.report import _build_figure_caption, _format_auc_with_ci, _format_p_value, _format_value

        ctx = _make_ctx(tmp_path)

        assert _format_value(np.int64(5)) == 5
        assert _format_value(np.array([1, 2])) == [1, 2]
        assert _format_auc_with_ci({"auc_mean": 0.81234, "auc_95ci": "[0.80, 0.83]"}) == "AUC = 0.812 (95% CI: 0.80, 0.83)"
        assert _format_auc_with_ci({"auc": 0.7}) == "AUC = 0.700"
        assert _format_auc_with_ci({"auc": "bad"}) == "AUC = N/A"
        assert _format_p_value("NA") == "P = NA"
        assert _format_p_value(0.0001) == "P < 0.001"
        assert _format_p_value(0.004) == "P = 0.004"
        assert _format_p_value(0.03) == "P = 0.030"
        assert _format_p_value(0.5) == "P = 0.50 (n.s.)"

        caption_cases = [
            (_make_record("prevalence", key_results={"total_subjects": 12345}), "Prevalence of top diseases"),
            (_make_record("train_model", args={"model_type": "xgb"}, key_results={"mean_auc": 0.81, "n_cases": 10, "n_controls": 40}), "ROC curve"),
            (_make_record("survival", args={"icd10_code": "I21"}, key_results={"log_rank_p": 0.02}), "Kaplan-Meier"),
            (_make_record("survival", args={"icd10_code": "I21"}, key_results={}), "Kaplan-Meier"),
            (_make_record("feature_importance"), "SHAP beeswarm"),
            (_make_record("biomarker_dist", args={"field_name": "HbA1c"}, key_results={"p_value": 0.001}), "Mann-Whitney"),
            (_make_record("biomarker_dist", args={"field_name": "HbA1c"}, key_results={}), "Distribution of HbA1c"),
            (_make_record("correlation"), "correlation heatmap"),
            (_make_record("phewas"), "PheWAS Manhattan"),
            (_make_record("comorbidity"), "Comorbidity network"),
            (_make_record("custom_skill"), "Custom Skill visualization"),
        ]
        for rec, expected in caption_cases:
            assert expected in _build_figure_caption("fig.png", rec, 1, ctx)
        assert "Analysis visualization" in _build_figure_caption("fig.png", None, 1, ctx)

    def test_interpret_skill_branches_and_fallbacks(self):
        from biobank_agent.skills.report import _interpret_skill

        cases = [
            (_make_record("cohort_summary", args={"icd10_code": "E11"}, key_results={"n_cases": 1000, "n_controls": 4000}), "adequate"),
            (_make_record("cohort_summary", args={"icd10_code": "E11"}, key_results={"n_cases": 50, "n_controls": 200}), "moderate"),
            (_make_record("train_model", args={"model_type": "lr"}, key_results={"mean_auc": "NA"}), "Model training completed"),
            (_make_record("biomarker_dist", args={"field_id": "30750"}, key_results={}), "completed"),
            (_make_record("biomarker_dist", args={"field_name": "HbA1c"}, key_results={"p_value": 0.2}), "non-significant"),
            (_make_record("survival", args={"icd10_code": "I21"}, key_results={"log_rank_p": 0.02}), "P = 0.020"),
            (_make_record("survival", args={"icd10_code": "I21"}, key_results={}), "completed"),
            (_make_record("feature_importance", key_results={"top_feature": "HbA1c"}), "HbA1c"),
            (_make_record("feature_importance", key_results={"top_features": ["BMI", "LDL"]}), "BMI"),
            (_make_record("evaluate_model", key_results={"auc": 0.81234}), "AUC=0.8123"),
            (_make_record("calibration", key_results={}), "Model evaluation completed"),
            (_make_record("correlation"), "interdependencies"),
            (_make_record("phewas", key_results={"n_significant": 3}), "3 significant"),
            (_make_record("comorbidity"), "co-occurrence"),
            (_make_record("missing_data", key_results={"overall_missing": 12.5}), "12.5%"),
            (_make_record("phenotype_harmonize", args={"concept": "HbA1c"}, key_results={"mappings": [1, 2], "drift_risks": ["coding drift"]}), "coding drift"),
            (_make_record("cohort_card", args={"endpoint": "I21"}, key_results={"cohort_type": "survival", "status": "PASS"}), "Cohort card"),
            (_make_record("world_model_audit", key_results={"safety_status": "FAIL", "allowed_claim_type": "association"}), "World-model audit"),
            (_make_record("numeric_skill", key_results={"n": 10, "auc": 0.7}), "Analysis completed"),
            (_make_record("text_skill", key_results={"status": "ok"}), "Analysis step `text_skill` completed"),
        ]

        for rec, expected in cases:
            assert expected in _interpret_skill(rec)

        bad = _make_record("biomarker_dist", key_results={"p_value": "not-a-float"})
        assert "Analysis step `biomarker_dist` completed" in _interpret_skill(bad)

    def test_format_value_numpy_import_failure_fallback(self, monkeypatch):
        import builtins
        from biobank_agent.skills.report import _format_value

        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("no numpy")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)

        value = object()
        assert _format_value(value) is value


class TestReportDeduping:
    """Repeated adjacent tool calls should not bloat report sections."""

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_generate_report_dedupes_consecutive_same_skill_args(self, mock_html, tmp_path):
        from biobank_agent.skills.report import generate_report

        records = [
            _make_record("evaluate_model", args={"model_key": "I21_xgb"}, key_results={"auc": 0.78}),
            _make_record("evaluate_model", args={"model_key": "I21_xgb"}, key_results={"auc": 0.79}),
            _make_record("calibration", args={"model_key": "I21_xgb"}, key_results={"ece": 0.03}),
            _make_record("calibration", args={"model_key": "I21_xgb"}, key_results={"ece": 0.02}),
        ]
        ctx = _make_ctx(tmp_path, records=records)
        result = generate_report(title="Dedup Report", format="report", ctx=ctx)

        md_text = Path(result["markdown"]).read_text()
        # Only one section title per deduped skill should remain.
        assert md_text.count("## 1. Evaluate Model") == 1
        assert md_text.count("## 2. Calibration") == 1
        assert result["n_sections"] == 2

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_generate_report_hides_raw_evidence_gathering_sections(self, mock_html, tmp_path):
        from biobank_agent.skills.report import generate_report

        records = [
            _make_record("web_search", key_results={"results": [
                {"title": "bad", "url": "https://zhidao.baidu.com/question/1", "snippet": "noise"}
            ]}),
            _make_record("deep_research", key_results={"sources": [
                {"title": "Paper C", "doi": "10.1/c", "url": "https://paper-c"}
            ]}),
            _make_record("read_paper", key_results={"title": "Paper D", "doi": "10.1/d"}),
            _make_record("train_model", args={"model_type": "xgb"}, key_results={"mean_auc": 0.84}),
        ]
        ctx = _make_ctx(tmp_path, records=records)
        result = generate_report(title="Clean Report", format="report", ctx=ctx)

        md_text = Path(result["markdown"]).read_text()
        assert "## 1. Train Model" in md_text
        assert "Web Search" not in md_text
        assert "Deep Research" not in md_text
        assert "zhidao.baidu" not in md_text
        assert "Paper C" in md_text
        assert "Paper D" in md_text
        assert result["n_sections"] == 1


class TestReportExtractionAndRenderingHelpers:
    def test_key_findings_references_governance_and_dedupe(self, tmp_path):
        from biobank_agent.skills.report import (
            _dedupe_records,
            _extract_key_findings,
            _extract_references,
            _governance_text,
        )

        records = [
            _make_record("think"),
            _make_record("train_model", args={"model_type": "xgb"}, key_results={"mean_auc": 0.84}),
            _make_record("prevalence", key_results={"top_disease": "E11"}),
            _make_record("cohort_summary", args={"icd10_code": "E11"}, key_results={"n_cases": 1234}),
            _make_record("feature_importance", key_results={"top_feature": "BMI"}),
            _make_record("phewas", key_results={"n_significant": 7}),
            _make_record("survival", key_results={"log_rank_p": 1e-5}),
            _make_record("phenotype_harmonize", key_results={"label": "HbA1c", "harmonisation_status": "READY"}),
            _make_record("cohort_card", key_results={"endpoint": "I21", "cohort_type": "survival"}),
            _make_record("trajectory_tokenize", key_results={"n_tokens": 1000}),
            _make_record("world_model_audit", key_results={"safety_status": "PARTIAL", "allowed_claim_type": "association"}),
            _make_record("failed", key_results={"error": "bad"}),
        ]

        findings = _extract_key_findings(records)
        assert len(findings) == 5
        assert findings[0].startswith("xgb achieved")
        assert _extract_key_findings([_make_record("think")]) == ["Analysis completed -- see details below"]

        references = _extract_references([
            _make_record("read_paper", key_results={
                "title": "Paper A",
                "authors": ["A", "B", "C", "D"],
                "doi": "10.1/a",
            }),
            _make_record("fetch_paper", key_results={"paper_title": "Paper B", "source_url": "https://paper-b"}),
            _make_record("deep_research", key_results={"sources": [
                {"title": "Paper C", "doi": "10.1/c", "url": "https://paper-c"},
                {"name": "Paper D"},
                "Free text source",
                {"title": "Paper C", "doi": "10.1/c", "url": "https://paper-c"},
            ]}),
        ])
        assert any("A, B, C et al." in ref for ref in references)
        assert any("Paper B" in ref and "https://paper-b" in ref for ref in references)
        assert any("Free text source" == ref for ref in references)
        assert len(references) == len(set(references))

        branch_refs = _extract_references([
            _make_record("read_paper", key_results={"doi": "10.no/title", "authors": "Anon"}),
            _make_record("deep_research", key_results={"sources": [
                {"doi": "10.no/title"},
                123,
            ]}),
        ])
        assert branch_refs == []

        ctx = _make_ctx(tmp_path, records=records)
        ctx.state.provenances = [SimpleNamespace(provenance_id="p1")]
        text = _governance_text(ctx, records)
        assert "healthy volunteer bias" in text
        assert "1 provenance hash record" in text

        ctx.settings.biobank_name = 123
        ctx.settings.biobank_caveats = {"not": "text"}
        fallback_text = _governance_text(ctx, records)
        assert "configured Biobank data layer" in fallback_text
        assert "Cohort-level caveat" not in fallback_text

        duplicate_a = _make_record("train_model", args={"x": 1}, key_results={"auc": 0.7})
        duplicate_b = _make_record("train_model", args={"x": 1}, key_results={"auc": 0.8})
        different = _make_record("train_model", args={"x": 2}, key_results={"auc": 0.9})
        assert _dedupe_records([duplicate_a, duplicate_b, different]) == [duplicate_b, different]

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_default_title_and_think_records_are_skipped_in_report_and_brief(self, mock_html, tmp_path):
        from biobank_agent.skills.report import generate_report

        records = [
            _make_record("think", key_results={"status": "internal"}),
            _make_record("survival", args={"icd10_code": "I21"}, key_results={"log_rank_p": 0.03}),
        ]
        ctx = _make_ctx(tmp_path, records=records)

        result = generate_report(format="report", ctx=ctx)
        report_text = Path(result["markdown"]).read_text()
        assert "# UK Biobank Analysis Report" in report_text
        assert "## 1. Think" not in report_text
        assert "## 1. Survival" in report_text

        brief = generate_report(title="Brief", format="brief", ctx=ctx)
        brief_text = Path(brief["markdown"]).read_text()
        assert "**think:**" not in brief_text
        assert "**survival:**" in brief_text

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_empty_interpretation_records_are_omitted_from_report_and_brief(self, mock_html, tmp_path, monkeypatch):
        from biobank_agent.skills import report as report_mod

        ctx = _make_ctx(tmp_path, records=[_make_record("custom_step", key_results={"value": 1})])
        monkeypatch.setattr(report_mod, "_interpret_skill", lambda rec: "")

        report_mod.generate_report(title="No Interpretation", format="report", ctx=ctx)
        report_text = (ctx.report_dir / "report.md").read_text()
        assert "## 1. Custom Step" in report_text
        assert "**custom_step:**" not in report_text

        report_mod.generate_report(title="No Interpretation Brief", format="brief", ctx=ctx)
        brief_text = (ctx.report_dir / "report.md").read_text()
        assert "**custom_step:**" not in brief_text

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_report_sections_metrics_errors_models_cohorts_and_reference_fallback(self, mock_html, tmp_path):
        from biobank_agent.skills.report import generate_report

        records = [
            _make_record("custom_numeric", args={"a": 1}, key_results={
                "small_float": 0.123456,
                "big_float": 1234.567,
                "long_list": [1, 2, 3, 4, 5, 6],
                "_retried_with": {"x": 1},
                "figure": "skip",
            }),
            _make_record("figure_only", key_results={"figure": "skip", "figures": ["skip"], "_retried_with": {"x": 1}}),
            _make_record("failed_step", key_results={"error": "failed safely"}),
        ]
        ctx = _make_ctx(
            tmp_path,
            records=records,
            cohorts={
                "with_label": __import__("pandas").DataFrame({"label": [1, 0], "x": [1, 2]}),
                "without_label": __import__("pandas").DataFrame({"x": [1, 2, 3]}),
            },
            models={"m": object()},
            model_metadata={
                "m1": {"model_type": "xgb", "auc_mean": 0.812, "auc_95ci": "[0.80, 0.83]", "n_cases": 10, "n_features": 4},
            },
        )

        generate_report(title="Detailed", format="report", ctx=ctx)
        md_text = (ctx.report_dir / "report.md").read_text()

        assert "0.1235" in md_text
        assert "1,234.57" in md_text
        assert "[1, ..., 6] (6 items)" in md_text
        assert "failed safely" in md_text
        assert "## 2. Figure Only" in md_text
        assert "| figure |" not in md_text
        assert "| without_label | 3 | N/A | N/A |" in md_text
        assert "AUC = 0.812" in md_text
        assert "No external literature records were attached" in md_text

    def test_key_findings_skip_empty_branch_values(self):
        from biobank_agent.skills.report import _extract_key_findings

        records = [
            _make_record("train_model", key_results={"mean_auc": None}),
            _make_record("prevalence", key_results={}),
            _make_record("cohort_summary", args={"icd10_code": "E11"}, key_results={"n_cases": 0}),
            _make_record("feature_importance", key_results={}),
            _make_record("phewas", key_results={"n_significant": 0}),
            _make_record("survival", key_results={}),
            _make_record("phenotype_harmonize", key_results={}),
            _make_record("cohort_card", key_results={"endpoint": "I21"}),
            _make_record("trajectory_tokenize", key_results={"n_tokens": 0}),
            _make_record("world_model_audit", key_results={}),
        ]

        assert _extract_key_findings(records) == ["Analysis completed -- see details below"]

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_paper_sections_no_refs_no_cohorts_and_pending_results(self, mock_html, tmp_path):
        from biobank_agent.skills.report import generate_report

        ctx = _make_ctx(tmp_path, records=[_make_record("think")], cohorts={}, model_metadata={})
        generate_report(title="Paper", format="paper", ctx=ctx)
        md_text = (ctx.report_dir / "report.md").read_text()

        assert "Results pending" in md_text
        assert "External literature records were not attached" in md_text

    def test_add_figures_section_svg_fallback_unsupported_and_html_writers(self, tmp_path, monkeypatch):
        import subprocess
        from biobank_agent.skills import report as report_mod

        bad_svg = tmp_path / "reports" / "bad.svg"
        bad_svg.parent.mkdir(parents=True, exist_ok=True)
        bad_svg.write_text("<svg></svg>", encoding="utf-8")
        png = tmp_path / "reports" / "ok.png"
        png.write_bytes(b"png")
        txt = tmp_path / "reports" / "ignored.txt"
        txt.write_text("ignored", encoding="utf-8")

        original_read_text = Path.read_text

        def fake_read_text(self, *args, **kwargs):
            if self == bad_svg:
                raise OSError("cannot read")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", fake_read_text)
        ctx = _make_ctx(
            tmp_path,
            records=[_make_record("prevalence", fig_paths=[str(bad_svg)], key_results={"n_codes": 1})],
            figures=[str(bad_svg), str(png), str(txt)],
        )
        sections = []
        report_mod._add_figures_section(sections, ctx)
        joined = "\n".join(sections)
        assert "![Figure 1]" in joined
        assert "![Figure 2]" in joined
        assert "ignored.txt" not in joined

        class Completed:
            def __init__(self, returncode):
                self.returncode = returncode

        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Completed(0))
        html_success = report_mod._write_html("# Title", "Title", tmp_path)
        assert html_success.name == "report.html"

        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Completed(1))
        html_nonzero = report_mod._write_html("# Title", "Title", tmp_path)
        assert "<div># Title</div>" in html_nonzero.read_text()

        def timeout_run(*args, **kwargs):
            raise subprocess.TimeoutExpired("pandoc", 1)

        monkeypatch.setattr(subprocess, "run", timeout_run)
        html_timeout = report_mod._write_html("# Title", "Title", tmp_path)
        assert "<pre># Title</pre>" in html_timeout.read_text()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
