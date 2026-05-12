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
              models=None, model_metadata=None, executive_findings=None,
              execution_log=None, guardrail_issues=None):
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
    ctx.state.executive_findings = executive_findings or []
    ctx.state.execution_log = execution_log or []
    ctx.state.guardrail_issues = guardrail_issues or []
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
        assert "output_dir" in func_def["parameters"]["properties"]

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_report_style_aliases_normalize_to_dual_styles(self, mock_html, tmp_path):
        """technical/report and nature/paper are accepted aliases."""
        from biobank_agent.skills.report import generate_report

        ctx = _make_ctx(tmp_path, records=[_make_record("prevalence", key_results={"n_codes": 5})])
        technical = generate_report(title="Technical", format="technical", ctx=ctx)
        assert technical["format"] == "report"
        assert "**Format:** Technical Report" in Path(technical["markdown"]).read_text()

        nature = generate_report(title="Nature", format="nature", ctx=ctx)
        assert nature["format"] == "paper"
        md_text = Path(nature["markdown"]).read_text()
        assert "## Abstract" in md_text

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_prompt_like_report_title_is_sanitized_for_human_reports(self, mock_html, tmp_path):
        """Long `/plan` prompts should not become report titles or leak reviewer/tool brands."""
        from biobank_agent.skills.report import generate_report

        ctx = _make_ctx(
            tmp_path,
            records=[_make_record("train_model", key_results={"auc": 0.91, "n_cases": 1000})],
        )
        long_prompt = (
            "I only have a broad research question: can UKB support metabolic health trajectories "
            "and Type 2 Diabetes risk prediction? Act as an autonomous biobank research agent. "
            "If Codex and Claude planning modes are available, merge their advice. You should "
            "run statistical_review, safety_check and world_model_audit."
        )

        result = generate_report(title=long_prompt, format="dual", ctx=ctx)
        technical = Path(result["paired_outputs"]["technical_markdown"]).read_text()
        nature = Path(result["paired_outputs"]["nature_markdown"]).read_text()

        assert technical.startswith("# UKB metabolic health trajectories and Type 2 Diabetes risk report")
        assert nature.startswith("# UKB metabolic health trajectories and Type 2 Diabetes risk report")
        assert "Codex" not in technical + nature
        assert "Claude" not in technical + nature
        assert "Act as an autonomous" not in technical + nature

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_nature_report_does_not_invent_model_methods_for_trajectory_only(self, mock_html, tmp_path):
        """Trajectory feasibility reports should not claim model training or SHAP when no model ran."""
        from biobank_agent.skills.report import generate_report

        ctx = _make_ctx(
            tmp_path,
            records=[
                _make_record("cohort_card", key_results={"endpoint": "E11", "cohort_type": "trajectory_prediction", "status": "PARTIAL"}),
                _make_record("trajectory_tokenize", key_results={
                    "status": "PARTIAL",
                    "n_tokens": 0,
                    "n_participants": 0,
                    "blocking_reasons": ["No longitudinal trajectory rows were supplied."],
                }),
                _make_record("world_model_audit", key_results={
                    "safety_status": "PARTIAL",
                    "allowed_claim_type": "association_conditioned_forecast",
                }),
            ],
        )

        result = generate_report(title="Trajectory", format="nature", ctx=ctx)
        md_text = Path(result["markdown"]).read_text()

        assert "trajectory feasibility" in md_text.lower()
        assert "No usable longitudinal rows" in md_text
        assert "Gradient-boosted models were trained" not in md_text
        assert "Feature importance was assessed via SHAP" not in md_text
        assert "candidate features for modelling" not in md_text
        assert "AUC-ROC" not in md_text

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_nature_report_describes_mixed_trajectory_and_model_branch(self, mock_html, tmp_path):
        """A successful model plus trajectory branch should not be summarized as baseline-only."""
        from biobank_agent.skills.report import generate_report

        ctx = _make_ctx(
            tmp_path,
            records=[
                _make_record("cohort_summary", key_results={"n_cases": 1000, "n_controls": 4000}),
                _make_record("trajectory_tokenize", key_results={
                    "status": "READY",
                    "n_tokens": 50000,
                    "n_participants": 4500,
                    "longitudinal_support": "multi_timepoint",
                    "trajectory_time_source": "synthetic_assessment_instance_dates",
                }),
                _make_record("train_model", key_results={
                    "model_type": "lgbm",
                    "auc": 0.91,
                    "auc_95ci": "[0.90, 0.92]",
                    "n_features": 120,
                }),
            ],
        )

        result = generate_report(title="Trajectory plus model", format="nature", ctx=ctx)
        md_text = Path(result["markdown"]).read_text()

        assert "HealthFormer-style feasibility layer" in md_text
        assert "baseline and repeated assessment instances" in md_text
        assert "single baseline time point" not in md_text
        assert "not externally validated forecasts" in md_text

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_report_has_replication_comparison_section_without_artifact_dump(self, mock_html, tmp_path):
        """Paper comparison should be rendered as a matrix, not raw artifact paths."""
        from biobank_agent.skills.report import generate_report

        ctx = _make_ctx(
            tmp_path,
            records=[
                _make_record("replicate_paper", key_results={
                    "status": "AWAITING_USER_APPROVAL",
                    "artifact": "/Users/example/report/paper_replication_spec.json",
                    "artifacts": {"spec_json": "/Users/example/report/paper_replication_spec.json"},
                    "comparison_checklist": ["do not overclaim"],
                    "proposed_spec": {"icd10_code": "E11"},
                }),
                _make_record("paper_replication_compare", key_results={
                    "paper_access": "text_available",
                    "local_endpoint": "UKB ICD-10 E11 diagnosis-centered cohort",
                    "overall_status": "partial_replication",
                    "acceptance_summary": {"verdict": "PASS_WITH_LIMITATIONS", "passed": 5, "partial": 1, "failed": 0},
                    "n_approximated_dimensions": 2,
                    "n_unavailable_dimensions": 0,
                    "comparison_rows": [
                        {
                            "dimension": "Endpoint",
                            "paper_target": "paper endpoint",
                            "local_result": "UKB ICD-10 E11 diagnosis-centered cohort",
                            "status": "approximation",
                            "evidence": "cohort_summary",
                        }
                    ],
                    "table_figure_diff": [
                        {
                            "target_type": "figure",
                            "paper_caption": "Figure 1: ROC curve",
                            "local_artifact": str(tmp_path / "roc_pr_E11_lgbm.svg"),
                            "status": "local_artifact_available",
                            "limitation": "Paper figure image unavailable.",
                        }
                    ],
                    "acceptance_gates": [
                        {
                            "gate": "model_auc",
                            "status": "PASS",
                            "observed": 0.82,
                            "threshold": "AUC >= 0.55",
                            "evidence": "train_model/evaluate_model",
                            "message": "Discrimination met the configured minimum.",
                        }
                    ],
                    "report_requirements": ["Separate paper target from local UKB result."],
                }),
            ],
        )

        result = generate_report(title="Replication", format="report", ctx=ctx)
        md_text = Path(result["markdown"]).read_text()

        assert "## Paper Replication Comparison" in md_text
        assert "Acceptance verdict: **PASS_WITH_LIMITATIONS**" in md_text
        assert "| model_auc | PASS | 0.82 | AUC >= 0.55 | train_model/evaluate_model |" in md_text
        assert "| Endpoint | paper endpoint | UKB ICD-10 E11 diagnosis-centered cohort | approximation | cohort_summary |" in md_text
        assert "Table/Figure diff targets" in md_text
        assert "roc_pr_E11_lgbm.svg" in md_text
        assert "/Users/example/report/paper_replication_spec.json" not in md_text

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_nature_report_renders_feature_importance_dict_as_name(self, mock_html, tmp_path):
        """Feature-importance records often store dict rows; prose should use the feature name."""
        from biobank_agent.skills.report import generate_report

        ctx = _make_ctx(
            tmp_path,
            records=[
                _make_record("cohort_summary", args={"icd10_code": "E11"}, key_results={"n_cases": 1000, "n_controls": 2000}),
                _make_record("train_model", args={"model_type": "auto"}, key_results={"model_type": "lgbm", "auc": 0.8, "auc_95ci": "[0.79, 0.81]"}),
                _make_record("feature_importance", key_results={"top_features": [{"feature": "HbA1c", "importance": 0.3}]}),
            ],
        )

        result = generate_report(title="Nature", format="nature", ctx=ctx)
        md_text = Path(result["markdown"]).read_text()

        assert "HbA1c was the leading model feature" in md_text
        assert "{'feature':" not in md_text

    def test_generate_report_dual_format_writes_paired_outputs_and_target_css(self, tmp_path):
        """Dual format produces technical and Nature-style markdown in a chosen directory."""
        from biobank_agent.skills.report import generate_report

        target_dir = tmp_path / "target_report"
        ctx = _make_ctx(
            tmp_path,
            records=[
                _make_record("train_model", args={"model_type": "auto"}, key_results={
                    "model_type": "auto-selected gradient boosted model",
                    "selected_model_type": "lgbm",
                    "mean_auc": 0.84,
                    "auc_95ci": "[0.82, 0.86]",
                    "n_cases": 1200,
                    "n_controls": 4800,
                    "n_features": 30,
                    "selection_rationale": "Selected lgbm because it had the highest mean CV AUC (0.8400).",
                    "candidate_comparison": [
                        {"rank": 1, "model_type": "lgbm", "status": "success", "auc_mean": 0.84},
                        {"rank": 2, "model_type": "sklearn_rf", "status": "success", "auc_mean": 0.78},
                    ],
                }),
                _make_record("safety_check", key_results={"overall": "PASS"}),
            ],
            executive_findings=["Auto-selected model separated the cohort with validated aggregate evidence"],
            execution_log=[{"step": "model", "tool": "train_model", "status": "success", "duration_s": "2.0", "note": "done"}],
        )

        result = generate_report(title="Dual Report", format="dual", output_dir=str(target_dir), ctx=ctx)

        assert result["format"] == "dual"
        assert Path(result["markdown"]).parent == target_dir
        for artifact in [
            "report.md",
            "report_technical.md",
            "report_nature.md",
            "_report_with_css.md",
            "_report_nature_with_css.md",
            "report.html",
            "report_nature.html",
        ]:
            path = target_dir / artifact
            assert path.exists()
            assert path.stat().st_size > 0
        technical = (target_dir / "report_technical.md").read_text()
        nature = (target_dir / "report_nature.md").read_text()
        assert "> **Executive Findings**" in technical
        assert "## Execution Appendix" in technical
        assert "## Model Selection" in technical
        assert "## Abstract" in nature
        assert "Generated by:" not in technical

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_report_renders_auto_model_selection_table(self, mock_html, tmp_path):
        """Automatic model selection should be visible in technical reports."""
        from biobank_agent.skills.report import generate_report

        ctx = _make_ctx(
            tmp_path,
            records=[
                _make_record("train_model", args={"model_type": "auto"}, key_results={
                    "selected_model_type": "lgbm",
                    "mean_auc": 0.84,
                    "auc_95ci": "[0.82, 0.86]",
                    "selection_rationale": "Selected lgbm because it had the highest mean CV AUC (0.8400).",
                    "candidate_comparison": [
                        {"rank": 1, "model_type": "lgbm", "status": "success", "auc_mean": 0.84, "f1_mean": 0.7, "precision_mean": 0.68, "recall_mean": 0.74},
                        {"rank": 2, "model_type": "sklearn_rf", "status": "success", "auc_mean": 0.79, "f1_mean": 0.65, "precision_mean": 0.62, "recall_mean": 0.7},
                        {"model_type": "catboost", "status": "failed", "error": "backend unavailable"},
                    ],
                }),
            ],
        )

        result = generate_report(title="Model Selection", format="report", ctx=ctx)
        md_text = Path(result["markdown"]).read_text()

        assert "## Model Selection" in md_text
        assert "Selected model: **lgbm**" in md_text
        assert "| 1 | lgbm | success | 0.840 |" in md_text
        assert "|  | catboost | failed |  |" in md_text
        assert "backend unavailable" in md_text


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

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_report_promotes_curated_executive_findings(self, mock_html, tmp_path):
        """State and record-level executive findings should lead the report."""
        from biobank_agent.skills.report import generate_report

        records = [
            _make_record("executive_findings", key_results={
                "findings": [{"summary": "Record-level finding is available"}],
            }),
            _make_record("train_model", args={"model_type": "xgb"}, key_results={"mean_auc": 0.81}),
        ]
        ctx = _make_ctx(
            tmp_path,
            records=records,
            executive_findings=["Curated finding takes priority"],
        )
        result = generate_report(title="Report", format="report", ctx=ctx)

        md_text = Path(result["markdown"]).read_text()
        assert "> **Executive Findings**" in md_text
        assert "Curated finding takes priority" in md_text
        assert "Record-level finding is available" in md_text
        assert md_text.index("> **Executive Findings**") < md_text.index("## Executive Summary")
        assert "## 1. Executive Findings" not in md_text


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

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_paper_format_has_no_author_or_institution_placeholders(self, mock_html, tmp_path):
        """Paper style should not inject author or institutional identity."""
        from biobank_agent.skills.report import generate_report

        ctx = _make_ctx(tmp_path, records=[_make_record("prevalence", key_results={"n_codes": 5})])
        generate_report(title="Paper", format="paper", ctx=ctx)

        md_text = (ctx.report_dir / "report.md").read_text()
        assert "CHEN Pengan" not in md_text
        assert "Chinese University of Hong Kong" not in md_text


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
    def test_svg_embedded_as_image_reference_not_inlined(self, mock_html, tmp_path):
        """SVG figures render as markdown image references without inline SVG dumps."""
        from biobank_agent.skills.report import generate_report

        svg_file = tmp_path / "reports" / "fig1.svg"
        svg_file.parent.mkdir(parents=True, exist_ok=True)
        svg_file.write_text('<svg xmlns="http://www.w3.org/2000/svg"><circle/></svg>')

        records = [_make_record("prevalence", key_results={"n_codes": 10})]
        ctx = _make_ctx(tmp_path, records=records, figures=[str(svg_file)])
        generate_report(title="SVG Report", format="report", ctx=ctx)

        md_text = (ctx.report_dir / "report.md").read_text()
        assert "<svg" not in md_text
        assert "![Figure 1](fig1.svg)" in md_text

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

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_figure_variants_deduped_by_stem(self, mock_html, tmp_path):
        """SVG/PDF/PNG variants with one stem count as one logical figure."""
        from biobank_agent.skills.report import generate_report

        report_dir = tmp_path / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        png = report_dir / "roc.png"
        svg = report_dir / "roc.svg"
        pdf = report_dir / "roc.pdf"
        png.write_bytes(b"\x89PNG\r\n")
        svg.write_text("<svg></svg>")
        pdf.write_bytes(b"%PDF-1.4")

        rec = _make_record("train_model", key_results={"mean_auc": 0.8}, fig_paths=[str(svg)])
        ctx = _make_ctx(tmp_path, records=[rec], figures=[str(svg), str(pdf), str(png)])
        result = generate_report(title="Variants", format="report", ctx=ctx)

        md_text = Path(result["markdown"]).read_text()
        assert result["n_figures"] == 1
        assert md_text.count("**Figure 1.**") == 1
        assert "![Figure 1](roc.png)" in md_text
        assert "[SVG](roc.svg)" in md_text
        assert "[PDF](roc.pdf)" in md_text


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
            (_make_record("feature_importance"), "Tree-based feature-importance"),
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
            (_make_record("missing_data", key_results={"overall_missing": 12.5}), "12.50%"),
            (_make_record("missing_data", key_results={"overall_missing_pct": 10.81}), "10.81%"),
            (_make_record("phenotype_harmonize", args={"concept": "HbA1c"}, key_results={"mappings": [1, 2], "drift_risks": ["coding drift"]}), "coding drift"),
            (_make_record("cohort_card", args={"endpoint": "I21"}, key_results={"cohort_type": "survival", "status": "PASS"}), "Cohort card"),
            (_make_record("world_model_audit", key_results={"safety_status": "FAIL", "allowed_claim_type": "association"}), "World-model audit"),
            (_make_record("numeric_skill", key_results={"n": 10, "auc": 0.7}), "Recorded quantitative outputs"),
        ]

        for rec, expected in cases:
            assert expected in _interpret_skill(rec)

        bad = _make_record("biomarker_dist", key_results={"p_value": "not-a-float"})
        assert _interpret_skill(bad) == ""
        assert _interpret_skill(_make_record("text_skill", key_results={"status": "ok"})) == ""

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

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_generate_report_dedupes_guardrails_and_hides_audit_execution_sections(self, mock_html, tmp_path):
        from biobank_agent.skills.report import generate_report

        report_dir = tmp_path / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        summary_svg = report_dir / "smart_summary_session.svg"
        summary_svg.write_text("<svg></svg>", encoding="utf-8")
        duplicated_issue = {
            "severity": "CRITICAL",
            "type": "diagnostic_biomarker_temporal_leakage_risk",
            "skill": "train_model",
            "message": "Endpoint-adjacent biomarkers require temporal review.",
            "recommendation": "Report as prevalent discrimination.",
        }
        records = [
            _make_record("train_model", args={"model_type": "auto"}, key_results={
                "mean_auc": 0.91,
                "analysis_design": "prevalent_or_ever_diagnosed_case_control",
                "prediction_target": "ever_diagnosed_E11_discrimination",
                "incident_risk_supported": False,
            }),
            _make_record("smart_plot", args={"plot_type": "summary"}, key_results={"plot_type": "summary"}, fig_paths=[str(summary_svg)]),
            _make_record("statistical_review", key_results={"issues": [duplicated_issue]}),
            _make_record("generate_report", key_results={"markdown": "old_report.md"}),
            _make_record("world_model_audit", key_results={"safety_status": "PARTIAL", "allowed_claim_type": "association_conditioned_forecast"}),
            _make_record("statistical_review", key_results={"issues": [duplicated_issue]}),
            _make_record("world_model_audit", key_results={"safety_status": "PARTIAL", "allowed_claim_type": "association_conditioned_forecast"}),
        ]
        ctx = _make_ctx(tmp_path, records=records, figures=[str(summary_svg)])
        result = generate_report(title="Dedup Guardrails", format="report", ctx=ctx)

        md_text = Path(result["markdown"]).read_text()
        assert "## 1. Train Model" in md_text
        assert "## 2. Smart Plot" in md_text
        assert "## 3. Generate Report" not in md_text
        assert "World Model Audit" not in md_text
        assert md_text.count("diagnostic_biomarker_temporal_leakage_risk") == 1
        assert "prevalent or ever diagnosed case control" in md_text
        assert "not incident-risk prediction" in md_text
        assert "Session-level diagnostic summary" in md_text


class TestReportExtractionAndRenderingHelpers:
    def test_key_findings_references_governance_and_dedupe(self, tmp_path):
        from biobank_agent.skills.report import (
            _dedupe_records,
            _extract_executive_findings,
            _extract_guardrail_issues,
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
        assert _extract_key_findings([_make_record("think")]) == []

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

        exec_ctx = _make_ctx(
            tmp_path,
            records=[
                _make_record("executive_findings", key_results={"findings": [{"message": "Record finding with validated biomarker evidence"}]}),
                _make_record("statistical_review", key_results={"issues": [
                    {
                        "severity": "WARNING",
                        "type": "small_sample",
                        "skill": "train_model",
                        "message": "Only 20 cases",
                        "recommendation": "Widen the cohort",
                    }
                ]}),
            ],
            executive_findings=["State finding with verified cohort signal"],
            guardrail_issues=[{"severity": "CRITICAL", "type": "privacy", "message": "Small cell"}],
        )
        assert _extract_executive_findings(exec_ctx, exec_ctx.state.records) == [
            "State finding with verified cohort signal",
            "Record finding with validated biomarker evidence",
        ]
        issue_rows = _extract_guardrail_issues(exec_ctx, exec_ctx.state.records)
        assert [row["severity"] for row in issue_rows] == ["CRITICAL", "WARNING"]
        assert issue_rows[0]["type"] == "privacy"

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
                "execution_log": [{"stdout": "raw leak"}],
            }),
            _make_record("figure_only", key_results={"figure": "skip", "figures": ["skip"], "_retried_with": {"x": 1}}),
            _make_record("failed_step", key_results={"error": "failed safely"}),
            _make_record("statistical_review", key_results={"issues": [
                {
                    "severity": "WARNING",
                    "type": "small_sample",
                    "skill": "train_model",
                    "message": "Only 20 cases",
                    "recommendation": "Widen the cohort",
                }
            ]}),
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
            execution_log=[
                {
                    "step": "fit model",
                    "tool": "train_model",
                    "status": "ok",
                    "duration_s": "2.1s",
                    "stdout": "raw output must not leak",
                    "summary": "completed fit",
                }
            ],
        )

        generate_report(title="Detailed", format="report", ctx=ctx)
        md_text = (ctx.report_dir / "report.md").read_text()

        assert "0.1235" in md_text
        assert "1,234.57" in md_text
        assert "[1, ..., 6] (6 items)" in md_text
        assert "failed safely" in md_text
        assert "## 2. Figure Only" in md_text
        assert "| figure |" not in md_text
        assert "| execution_log |" not in md_text
        assert "raw output must not leak" not in md_text
        assert "| without_label | 3 | N/A | N/A |" in md_text
        assert "AUC = 0.812" in md_text
        assert "## Guardrail Issues" in md_text
        assert "small_sample" in md_text
        assert "Only 20 cases" in md_text
        assert "## Execution Appendix" in md_text
        assert "fit model" in md_text
        assert "completed fit" in md_text
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

        assert _extract_key_findings(records) == []

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_paper_sections_no_refs_no_cohorts_and_pending_results(self, mock_html, tmp_path):
        from biobank_agent.skills.report import generate_report

        ctx = _make_ctx(tmp_path, records=[_make_record("think")], cohorts={}, model_metadata={})
        generate_report(title="Paper", format="paper", ctx=ctx)
        md_text = (ctx.report_dir / "report.md").read_text()

        assert "No analytical result records were available" in md_text
        assert "External literature records were not attached" in md_text

    def test_add_figures_section_svg_link_unsupported_and_html_writers(self, tmp_path, monkeypatch):
        import subprocess
        from biobank_agent.skills import report as report_mod

        svg = tmp_path / "reports" / "bad.svg"
        svg.parent.mkdir(parents=True, exist_ok=True)
        svg.write_text("<svg></svg>", encoding="utf-8")
        png = tmp_path / "reports" / "ok.png"
        png.write_bytes(b"png")
        txt = tmp_path / "reports" / "ignored.txt"
        txt.write_text("ignored", encoding="utf-8")
        ctx = _make_ctx(
            tmp_path,
            records=[_make_record("prevalence", fig_paths=[str(svg)], key_results={"n_codes": 1})],
            figures=[str(svg), str(png), str(txt)],
        )
        sections = []
        report_mod._add_figures_section(sections, ctx)
        joined = "\n".join(sections)
        assert "![Figure 1](bad.svg)" in joined
        assert "![Figure 2]" in joined
        assert "ignored.txt" not in joined

        class Completed:
            def __init__(self, returncode):
                self.returncode = returncode

        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Completed(0))
        html_success = report_mod._write_html("# Title", "Title", tmp_path)
        assert html_success.name == "report.html"
        assert (tmp_path / "_report_with_css.md").exists()

        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Completed(1))
        html_nonzero = report_mod._write_html("# Title", "Title", tmp_path)
        assert "<div># Title</div>" in html_nonzero.read_text()

        def timeout_run(*args, **kwargs):
            raise subprocess.TimeoutExpired("pandoc", 1)

        monkeypatch.setattr(subprocess, "run", timeout_run)
        html_timeout = report_mod._write_html("# Title", "Title", tmp_path)
        assert "<pre># Title</pre>" in html_timeout.read_text()

        html_named = report_mod._write_html("# Nature", "Nature", tmp_path, stem="report_nature")
        assert html_named.name == "report_nature.html"
        assert (tmp_path / "_report_nature_with_css.md").exists()

    @patch("biobank_agent.skills.report._write_html", return_value=None)
    def test_rendered_markdown_tables_are_contiguous_and_svg_embedded(self, mock_html, tmp_path):
        from biobank_agent.skills.report import generate_report

        svg = tmp_path / "reports" / "roc.svg"
        svg.parent.mkdir(parents=True, exist_ok=True)
        svg.write_text("<svg></svg>", encoding="utf-8")
        ctx = _make_ctx(
            tmp_path,
            records=[
                _make_record(
                    "train_model",
                    args={"model_type": "auto"},
                    key_results={"mean_auc": 0.82, "auc_95ci": "[0.80, 0.84]", "n_cases": 100},
                    fig_paths=[str(svg)],
                )
            ],
            figures=[str(svg)],
        )

        result = generate_report(title="Renderable", format="report", ctx=ctx)
        md_text = Path(result["markdown"]).read_text()

        assert "|\n\n|" not in md_text
        assert "![Figure 1](roc.svg)" in md_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
