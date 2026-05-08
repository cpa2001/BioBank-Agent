"""Tests for the deterministic report-quality benchmark."""

from __future__ import annotations

from types import SimpleNamespace

from biobank_agent.eval.benchmarks import AgentReportWorkflowBenchmark, PUBLIC_REPORT_REFERENCES, ReportQualityBenchmark
from biobank_agent.eval.harness import EvalHarness, TestCase as HarnessTestCase, TestResult as HarnessTestResult


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

## Abstract
AUC = 0.842 (95% CI: 0.831-0.853), P < 0.001, n = 62,250. Figure 1 summarizes discrimination.

## Methods
We used a case-control E11 workflow with stratified cross-validation.

## Results
Small-cell suppression was applied. Results are not causal.

## Discussion
Reproducibility, Governance and Data Availability are documented for aggregate reporting.

## References
1. 10.1038/s41588-024-01898-1
2. 10.1038/s41586-023-06592-6
3. 10.1038/s41467-023-43575-7
"""


def _technical_report_text():
    return """# UK Biobank Proteomic Risk Technical Report

## Key Findings
AUC = 0.842 (95% CI: 0.831-0.853), P < 0.001, n = 62,250. Figure 1 shows the ROC curve.

## Executive Summary
Cases and controls were analyzed with public-paper references.

## Methodology Notes
Cross-validation and statistical review were run before report synthesis.

## Reproducibility and Governance
Data availability follows the data access agreement. Small-cell suppression is required. This is not causal.

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
        assert "10.1038/s41588-024-01898-1" in case_result.actual_text
        assert "Reproducibility" in case_result.actual_text
        assert case_result.metadata["public_references"] == [
            ref["doi"] for ref in PUBLIC_REPORT_REFERENCES
        ]


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
