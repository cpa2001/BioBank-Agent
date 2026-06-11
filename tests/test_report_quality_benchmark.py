"""Tests for the deterministic report-quality benchmark."""

from __future__ import annotations

from types import SimpleNamespace

from biobank_agent.eval.benchmarks import (
    AgentReportWorkflowBenchmark,
    LiveUKBReport20Benchmark,
    PUBLIC_REPORT_REFERENCES,
    Report20CaseBenchmark,
    ReportQualityBenchmark,
)
from biobank_agent.eval.harness import EvalHarness, TestCase as HarnessTestCase, TestResult as HarnessTestResult
from biobank_agent.eval.report_review import review_benchmark_artifacts, reviewer_specs


class FakeReportWorkflowAgent:
    def __init__(self, tmp_path, *, fail_stage=None):
        self.tmp_path = tmp_path
        self.fail_stage = fail_stage
        self.state = SimpleNamespace(records=[])
        self.calls = []

    def run(self, query):
        self.calls.append(query)
        lower = query.lower()
        if self.fail_stage and self.fail_stage in lower:
            raise RuntimeError(f"{self.fail_stage} failed")
        if "search/read" in lower:
            self.state.records.append(SimpleNamespace(skill="deep_research", key_results={}))
            self.state.records.append(SimpleNamespace(skill="read_paper", key_results={}))
            return "Proteomics literature recorded."
        if "search and summarize" in lower:
            self.state.records.append(SimpleNamespace(skill="deep_research", key_results={}))
            return "Biomarker prediction literature recorded."
        if "cohort" in lower or "proteomics risk" in lower:
            self.state.records.append(SimpleNamespace(skill="cohort_summary", key_results={}))
            self.state.records.append(SimpleNamespace(skill="train_model", key_results={}))
            return "Cohort and model evidence recorded."
        if "statistical_review" in lower or "safety_check" in lower:
            self.state.records.append(SimpleNamespace(skill="statistical_review", key_results={}))
            self.state.records.append(SimpleNamespace(skill="safety_check", key_results={}))
            return "Guardrails passed."
        if "technical report" in lower:
            return self._write_report("technical")
        if "imrad" in lower:
            return self._write_report("paper")
        return "ok"

    def _write_report(self, kind):
        report_dir = self.tmp_path / kind
        report_dir.mkdir()
        if kind == "paper":
            text = _paper_report_text()
        else:
            text = _technical_report_text()
        path = report_dir / "report.md"
        path.write_text(text, encoding="utf-8")
        self.state.records.append(SimpleNamespace(skill="generate_report", key_results={"markdown": str(path)}))
        return text[:500]


def _paper_report_text():
    return """# Biomarker prediction and proteomic risk signatures in UK Biobank

> **Executive Findings**
> - AUC = 0.842 with linked cohort, model, and governance evidence.
> - Small-cell suppression and non-causal interpretation are explicit.

## Abstract
AUC = 0.842 (95% CI: 0.831-0.853), P < 0.001, n = 62,250. Figure 1 summarizes discrimination.

## Methods
We used a case-control E11 workflow with stratified cross-validation.

## Results
Small-cell suppression was applied. Results are not causal.

## Discussion
Reproducibility, Governance and Data Availability are documented for aggregate reporting.

## Execution Appendix
Analysis record inventory: literature review, cohort construction, model training, statistical review, safety check, report synthesis.

## References
1. 10.1038/s41588-024-01898-1
2. 10.1038/s41586-023-06592-6
3. 10.1038/s41467-023-43575-7
"""


def _technical_report_text():
    return """# UK Biobank Proteomic Risk Technical Report

> **Executive Findings**
> - AUC = 0.842 with linked cohort, model, and governance evidence.
> - Small-cell suppression and non-causal interpretation are explicit.

## Key Findings
AUC = 0.842 (95% CI: 0.831-0.853), P < 0.001, n = 62,250. Figure 1 shows the ROC curve.

## Executive Summary
Cases and controls were analyzed with public-paper references.

## Methodology Notes
Cross-validation and statistical review were run before report synthesis.

## Reproducibility and Governance
Data availability follows the data access agreement. Small-cell suppression is required. This is not causal.

## Execution Appendix
Analysis record inventory: literature review, cohort construction, model training, statistical review, safety check, report synthesis.

## References
1. 10.1038/s41588-024-01898-1
2. 10.1038/s41586-023-06592-6
3. 10.1038/s41467-023-43575-7
"""


def test_report_quality_benchmark_generates_public_paper_grounded_reports():
    benchmark = ReportQualityBenchmark()
    result = EvalHarness().run(benchmark, SimpleNamespace(), mode="baseline")

    assert result.n_total == 2
    assert result.n_passed == 2
    assert result.gate_passed is True
    assert result.mean_score == 1.0
    assert result.observability["evidence_coverage"] == 1.0

    for case_result in result.results:
        checks = case_result.metadata["quality_checks"]
        assert all(checks.values())
        assert "generate_report" in case_result.actual_skills
        assert checks["has_public_references"] is True
        assert "Reproducibility" in case_result.actual_text
        assert case_result.metadata["public_references"] == [
            ref["doi"] for ref in PUBLIC_REPORT_REFERENCES
        ]


def test_report_20_case_benchmark_defines_ukb_oriented_synthetic_case_matrix():
    benchmark = Report20CaseBenchmark()

    assert benchmark.name == "report_20_case"
    assert len(benchmark.cases) == 20
    ids = {case.id for case in benchmark.cases}
    assert {"report_e11_hba1c", "report_i10_bmi_bp", "report_n18_creatinine"} <= ids
    assert all(case.metadata["icd10"] for case in benchmark.cases)
    assert all(case.metadata["field_id"] for case in benchmark.cases)
    assert all("ukb_oriented_synthetic" in case.tags for case in benchmark.cases)
    assert all(case.metadata["benchmark_kind"] == "ukb_oriented_synthetic" for case in benchmark.cases)


def test_live_ukb_report_20_benchmark_fails_closed_without_data_manager():
    benchmark = LiveUKBReport20Benchmark()
    benchmark.cases = benchmark.cases[:1]

    result = EvalHarness().run(benchmark, SimpleNamespace(), mode="baseline")

    assert result.n_total == 1
    assert result.n_passed == 0
    assert result.gate_passed is False
    assert result.results[0].metadata["benchmark_kind"] == "live_ukb"
    assert result.results[0].metadata["data_preflight"] == "FAIL"


def test_report_20_case_benchmark_runs_representative_cases():
    benchmark = Report20CaseBenchmark()
    benchmark.cases = benchmark.cases[:3]

    result = EvalHarness().run(benchmark, SimpleNamespace(), mode="baseline")

    assert result.n_total == 3
    assert result.n_passed == 3
    assert result.gate_passed is True
    for case_result in result.results:
        assert "UK Biobank" in case_result.actual_text
        assert any(
            key.endswith("has_public_references") and passed
            for key, passed in case_result.metadata["quality_checks"].items()
        )
        assert case_result.metadata["format"] == "dual"
        assert set(case_result.metadata["formats"]) == {"paper", "report"}
        assert all(case_result.metadata["quality_checks"].values())


def test_agent_report_workflow_benchmark_runs_staged_agent_and_reads_report(tmp_path):
    benchmark = AgentReportWorkflowBenchmark()
    agent = FakeReportWorkflowAgent(tmp_path)

    result = EvalHarness().run(benchmark, agent, mode="baseline")

    assert result.n_total == 2
    assert result.n_passed == 2
    assert result.mean_score > 0.9
    assert result.observability["evidence_coverage"] == 1.0
    for case_result in result.results:
        assert case_result.metadata["failure_stage"] is None
        assert [s["status"] for s in case_result.metadata["stage_results"]] == ["PASS"] * 4
        assert "generate_report" in case_result.actual_skills
        assert "10.1038/s41588-024-01898-1" in case_result.actual_text


def test_agent_report_workflow_benchmark_records_failure_stage(tmp_path):
    benchmark = AgentReportWorkflowBenchmark()
    agent = FakeReportWorkflowAgent(tmp_path, fail_stage="cohort")

    result = benchmark.run_case(benchmark.cases[0], agent)

    assert result.passed is False
    assert result.metadata["failure_stage"] == "cohort_model"
    assert any("Stage cohort_model failed" in error for error in result.errors)
    assert benchmark.score(result, benchmark.cases[0]) <= 0.2


def test_agent_report_workflow_benchmark_handles_missing_agent():
    benchmark = AgentReportWorkflowBenchmark()

    result = benchmark.run_case(benchmark.cases[0], SimpleNamespace())

    assert result.passed is False
    assert result.metadata["failure_stage"] == "agent_unavailable"


def test_agent_report_workflow_score_and_report_read_fallbacks(tmp_path):
    benchmark = AgentReportWorkflowBenchmark()
    empty = HarnessTestResult(case_id="empty", passed=False, metadata={})
    assert benchmark.score(empty, benchmark.cases[0]) == 0.0

    missing_path = tmp_path / "missing.md"
    agent = SimpleNamespace(
        state=SimpleNamespace(
            records=[
                SimpleNamespace(skill="cohort_summary", key_results={}),
                SimpleNamespace(skill="generate_report", key_results={"markdown": str(missing_path)}),
                SimpleNamespace(skill="generate_report", key_results={}),
            ]
        )
    )

    text, path = benchmark._latest_report_text(agent, "fallback report")

    assert text == "fallback report"
    assert path == missing_path


def test_report_quality_benchmark_flags_missing_public_references():
    benchmark = ReportQualityBenchmark()
    checks = benchmark._report_checks(
        """# Report

## Abstract
## Methods
## Results
AUC = 0.84 (95% CI: 0.83-0.85), P < 0.001, n = 10,000.

## Discussion
Figure 1. Reproducibility and Governance. Data availability. Not causal.
Small-cell suppression is required.

## References
1. Local-only note.
"""
    )

    assert checks["has_public_references"] is False
    assert checks["has_quantitative_statistics"] is True
    assert checks["has_noncausal_caveat"] is True


def test_report_quality_benchmark_flags_old_report_failure_modes():
    benchmark = ReportQualityBenchmark()
    checks = benchmark._report_checks(
        """# Bad Report

*CHEN Pengan*

*The Chinese University of Hong Kong*

## Results
Top finding: .
Analysis completed: returncode=0, elapsed_s=1.2.
PheWAS identified ? significant associations.

## References
External literature records were not attached.
"""
    )

    assert checks["has_executive_findings_upfront"] is False
    assert checks["has_execution_appendix"] is False
    assert checks["no_signature_or_generator_trace"] is False
    assert checks["no_raw_logs_or_secrets"] is False
    assert checks["no_placeholders"] is False
    assert checks["no_weak_main_body"] is False


def test_report_quality_benchmark_scores_empty_and_penalized_results():
    benchmark = ReportQualityBenchmark()

    assert benchmark.score(HarnessTestResult(case_id="empty", passed=False, metadata={}), benchmark.cases[0]) == 0.0

    result = HarnessTestResult(
        case_id="bad",
        passed=False,
        errors=["missing a", "missing b"],
        metadata={
            "quality_checks": {"a": True, "b": False},
            "quality_score": {"overall": 0.5},
        },
    )

    assert benchmark.score(result, benchmark.cases[0]) == 0.4


def test_report_quality_run_case_records_missing_content_and_failed_checks(monkeypatch):
    def fake_generate_report(title, format, ctx):
        md_path = ctx.report_dir / "bad.md"
        md_path.write_text("# Bad Report\n\nTODO\n", encoding="utf-8")
        return {"markdown": str(md_path)}

    monkeypatch.setattr("biobank_agent.skills.report.generate_report", fake_generate_report)
    benchmark = ReportQualityBenchmark()
    case = HarnessTestCase(
        id="bad_report",
        query="Generate a bad report",
        expected_contains=["Definitely Missing"],
        tags=["paper"],
    )

    result = benchmark.run_case(case, SimpleNamespace())

    assert result.passed is False
    assert "Missing expected report content: Definitely Missing" in result.errors
    assert any(error.startswith("Report quality check failed:") for error in result.errors)
    assert result.metadata["format"] == "paper"


class FakeReviewRegistry:
    def __init__(self):
        self.calls = []

    def execute(self, skill, args, ctx=None):
        self.calls.append((skill, args, ctx))
        return {
            "skill": skill,
            "task_kind": "review",
            "status": "success",
            "stdout": f"ALLOW: {skill} reviewed",
            "command_display": skill,
        }


def test_eval_harness_review_loop_runs_primary_and_optional_secondary(tmp_path):
    registry = FakeReviewRegistry()
    agent = SimpleNamespace(
        registry=registry,
        settings=SimpleNamespace(reports_dir=tmp_path),
        state=SimpleNamespace(records=[]),
    )
    benchmark = ReportQualityBenchmark()
    benchmark.cases = benchmark.cases[:1]

    result = EvalHarness().run(
        benchmark,
        agent,
        mode="baseline",
        review_loop=True,
        include_claude=True,
        review_timeout_s=12,
    )

    assert result.review_loop["status"] == "completed"
    assert result.review_loop["primary_reviewer"] == "report_review_primary"
    assert result.review_loop["old_report_overwrite_ready"] is True
    assert [v["verdict"] for v in result.review_loop["role_verdicts"]] == ["ALLOW"] * 3
    assert result.review_loop["artifact_checklist"]["executive_findings_upfront"]["passed"] is True
    assert [call[0] for call in registry.calls] == ["report_review_primary", "report_review_secondary"]
    assert registry.calls[0][1]["timeout_s"] == 12
    assert "report_quality" in registry.calls[0][1]["context"]
    assert result.review_loop["reviews"][0]["stdout"] == "ALLOW: report_review_primary reviewed"
    assert result.review_loop["reviews"][0]["verdict"] == "ALLOW"
    assert result.gate_passed is True


def test_eval_harness_review_loop_skips_without_registry():
    benchmark = ReportQualityBenchmark()
    benchmark.cases = benchmark.cases[:1]

    result = EvalHarness().run(benchmark, SimpleNamespace(), review_loop=True)

    assert result.review_loop["status"] == "skipped"
    assert result.review_loop["reviews"][0]["skill"] == "report_review_primary"
    assert result.review_loop["reviews"][0]["error"] == "agent registry unavailable"
    assert result.gate_passed is False


def test_report_review_helper_exposes_primary_and_optional_secondary_metadata():
    specs = reviewer_specs("report_review_primary", include_claude=True)

    assert specs[0]["reviewer"] == "report_review_primary"
    assert specs[0]["role"] == "primary_engineer_reviewer"
    assert specs[1]["reviewer"] == "report_review_secondary"
    assert specs[1]["optional"] is True


def test_report_review_helper_blocks_artifact_not_ready_for_old_report_overwrite():
    benchmark = ReportQualityBenchmark()
    result = SimpleNamespace(
        benchmark_name="report_quality",
        results=[
            HarnessTestResult(
                case_id="report_bad",
                passed=False,
                actual_skills=["generate_report"],
                actual_text=(
                    "# Report\n\n"
                    "## Authors\nA. Example, Example Institution\n\n"
                    "## Methods\nTODO\n\n"
                    "stdout: old report placeholder\n"
                ),
                metadata={"tags": ["report"], "format": "paper"},
            )
        ],
    )

    review = review_benchmark_artifacts(result, benchmark)

    assert review["status"] == "blocked"
    assert review["old_report_overwrite_ready"] is False
    assert review["artifact_checklist"]["executive_findings_upfront"]["passed"] is False
    assert review["artifact_checklist"]["no_author_or_institution_block"]["passed"] is False
    assert review["artifact_checklist"]["appendix_details"]["passed"] is False
    assert any(v["role"] == "engineer" and v["verdict"] == "BLOCK" for v in review["role_verdicts"])
