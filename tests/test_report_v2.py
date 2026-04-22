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
    ctx.state.records = records or []
    ctx.state.figures = figures or []
    ctx.state.cohorts = cohorts or {}
    ctx.state.models = models or {}
    ctx.state.model_metadata = model_metadata or {}
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
        assert "0.9200" in text
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
