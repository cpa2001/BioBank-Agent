"""Tests for research_eval_v1 benchmark and observability metrics."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from biobank_agent.eval.benchmarks import BiomedQABenchmark, ResearchEvalV1, SkillCallBenchmark, SkillSchemaBenchmark
from biobank_agent.eval.harness import (
    Benchmark,
    BenchmarkResult,
    EvalHarness,
    TestCase as HarnessTestCase,
    TestResult as HarnessCaseResult,
)


def test_skill_schema_init_falls_back_when_custom_discovery_fails(monkeypatch):
    from biobank_agent import config as config_mod
    from biobank_agent import registry as registry_mod

    calls = []

    monkeypatch.setattr(config_mod, "get_settings", lambda: (_ for _ in ()).throw(RuntimeError("settings failed")))
    monkeypatch.setattr(registry_mod, "autodiscover_skills", lambda: calls.append("auto"))
    monkeypatch.setattr(registry_mod, "discover_custom_skills", lambda path: calls.append(f"custom:{path}"))
    monkeypatch.setattr(registry_mod, "get_registry", lambda: SimpleNamespace(list_skills=lambda: []))

    suite = SkillSchemaBenchmark()

    assert suite.cases == []
    assert calls == ["auto"]


def test_research_eval_v1_has_120_cases():
    suite = ResearchEvalV1()
    assert len(suite.cases) == 120
    ids = {c.id for c in suite.cases}
    assert "complex_001" in ids
    assert "evidence_001" in ids
    assert "statistics_001" in ids
    assert "execution_001" in ids


def test_observability_metrics_computation():
    harness = EvalHarness()
    results = [
        HarnessCaseResult(
            case_id="c1",
            passed=True,
            elapsed_s=1.0,
            metadata={
                "claim_count": 2,
                "evidence_count": 2,
                "safety_status": "PASS",
                "debate_disagreement": True,
                "tags": ["complex"],
                "prompt_tokens_delta": 10,
                "completion_tokens_delta": 20,
            },
        ),
        HarnessCaseResult(
            case_id="c2",
            passed=False,
            elapsed_s=2.0,
            metadata={
                "claim_count": 2,
                "evidence_count": 0,
                "safety_status": "PARTIAL",
                "debate_disagreement": True,
                "tags": ["complex"],
                "prompt_tokens_delta": 5,
                "completion_tokens_delta": 5,
            },
        ),
    ]
    obs = harness._compute_observability(results)
    assert 0.0 <= obs["success_rate"] <= 1.0
    assert 0.0 <= obs["evidence_coverage"] <= 1.0
    assert obs["token_cost"] == 40.0
    assert obs["p95_latency_s"] >= 1.0


def test_reliability_gate_with_baseline_comparison():
    harness = EvalHarness()
    obs = {
        "evidence_coverage": 0.96,
        "stat_guardrail_violation": 0.0,
        "complex_task_success": 0.70,
        "wrong_consensus_rate": 0.30,
    }
    baseline = {
        "complex_task_success": 0.50,
        "wrong_consensus_rate": 0.60,
    }
    ok, failures = harness._check_reliability_gate(obs, baseline_observability=baseline)
    assert ok is True
    assert failures == []


def test_skill_schema_benchmark_does_not_call_agent_run():
    harness = EvalHarness()
    suite = SkillSchemaBenchmark()
    suite.cases = suite.cases[:3]
    agent = MagicMock()
    agent.run.side_effect = AssertionError("schema benchmark must not call LLM")

    result = harness.run(suite, agent, mode="baseline")

    assert result.n_total == 3
    assert result.n_passed == 3
    agent.run.assert_not_called()
    assert all(r.metadata.get("direct_schema_check") for r in result.results)


def test_skill_schema_score_failure_modes():
    suite = SkillSchemaBenchmark.__new__(SkillSchemaBenchmark)
    case = HarnessTestCase(id="schema_x", query="schema")

    def score_with(schemas, case_id="schema_x"):
        suite._registry = SimpleNamespace(tool_schemas=lambda: schemas)
        result = HarnessCaseResult(case_id=case_id, passed=True)
        score = suite.score(result, HarnessTestCase(id=case_id, query="schema"))
        return score, result

    score, result = score_with([{"function": {"name": "", "description": "desc", "parameters": {"type": "object"}}}], "schema_")
    assert score == 0.0
    assert result.errors == ["Missing function name"]

    score, result = score_with([{"function": {"name": "x", "parameters": {"type": "object"}}}])
    assert score == 0.5
    assert result.errors == ["Missing function description"]

    score, result = score_with([{"function": {"name": "x", "description": "desc", "parameters": {"type": "array"}}}])
    assert score == 0.3
    assert result.errors == ["Parameters type must be 'object'"]

    score, result = score_with([{"function": {"name": "other", "description": "desc", "parameters": {"type": "object"}}}])
    assert score == 0.0
    assert result.errors == ["Skill x not found in registry"]


def test_biomed_and_skill_call_benchmark_scoring_edges():
    qa = BiomedQABenchmark()
    assert len(qa.cases) == 10
    assert qa.score(HarnessCaseResult(case_id="empty", passed=False, actual_text=""), qa.cases[0]) == 0.0

    no_checks = HarnessTestCase(id="no_checks", query="q")
    assert qa.score(HarnessCaseResult(case_id="ok", passed=True, actual_text="text"), no_checks) == 1.0
    assert qa.score(HarnessCaseResult(case_id="bad", passed=False, actual_text="text"), no_checks) == 0.0
    assert qa.score(HarnessCaseResult(case_id="partial", passed=False, actual_text="text", errors=["missing"]), qa.cases[0]) == 0.0

    calls = SkillCallBenchmark()
    assert len(calls.cases) == 5
    assert calls.score(HarnessCaseResult(case_id="no_expected", passed=True), no_checks) == 1.0
    assert calls.score(HarnessCaseResult(case_id="all", passed=True, actual_skills=["prevalence"]), calls.cases[0]) == 1.0
    multi = HarnessTestCase(id="multi", query="q", expected_skills=["a", "b"])
    assert calls.score(HarnessCaseResult(case_id="partial", passed=False, actual_skills=["a"]), multi) == 0.5


def test_research_eval_score_edges():
    suite = ResearchEvalV1()
    case = suite.cases[0]

    no_claims = HarnessCaseResult(case_id=case.id, passed=True, metadata={"claim_count": 0, "evidence_count": 10})
    partial_evidence = HarnessCaseResult(case_id=case.id, passed=False, metadata={"claim_count": 4, "evidence_count": 2})

    assert suite.score(no_claims, case) == 0.8
    assert suite.score(partial_evidence, case) == 0.15


def test_benchmark_result_summary_empty_and_failure_modes():
    result = BenchmarkResult(
        benchmark_name="demo",
        mode="ab",
        observability={
            "success_rate": 0.0,
            "evidence_coverage": 0.5,
            "stat_guardrail_violation": 0.25,
            "wrong_consensus_rate": 0.75,
            "p95_latency_s": 1.2,
        },
        comparative={
            "complex_task_success_uplift": 0.2,
            "wrong_consensus_reduction_ratio": 0.5,
        },
        gate_passed=False,
        gate_failures=["evidence_coverage < 95%", "stat_guardrail_violation > 1%"],
    )

    summary = result.summary()

    assert result.n_total == 0
    assert result.pass_rate == 0.0
    assert result.mean_score == 0.0
    assert result.failures() == []
    assert "A/B delta" in summary
    assert "Reliability gate: FAIL" in summary

    passing = BenchmarkResult(
        benchmark_name="demo",
        results=[HarnessCaseResult(case_id="ok", passed=True, score=1.0)],
        observability={"success_rate": 1.0, "evidence_coverage": 1.0},
        gate_passed=True,
    )
    passing_summary = passing.summary()
    assert "A/B delta" not in passing_summary
    assert "Reliability gate: PASS" in passing_summary


def test_base_benchmark_defaults():
    benchmark = Benchmark()
    case = HarnessTestCase(id="base", query="noop")
    result = HarnessCaseResult(case_id="base", passed=False)

    assert benchmark.run_case(case, agent=MagicMock()) is None
    assert benchmark.score(result, case) == 0.0
    result.passed = True
    assert benchmark.score(result, case) == 1.0


def test_run_case_records_expected_errors_and_metadata():
    class FakeAgent:
        def __init__(self):
            self.state = SimpleNamespace(
                records=[
                    SimpleNamespace(skill="think"),
                    SimpleNamespace(skill="cohort_summary"),
                ],
                last_orchestration={
                    "claims": ["claim_a", "claim_b"],
                    "evidence_links": ["evidence_a"],
                    "debate_trace": {"disagreement": True},
                    "safety_status": "PARTIAL",
                },
                token_usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
            )

        def run(self, query):
            self.state.token_usage.prompt_tokens += 5
            self.state.token_usage.completion_tokens += 7
            return f"Response for {query} with endpoint"

    case = HarnessTestCase(
        id="case_success_with_errors",
        query="q",
        expected_skills=["cohort_summary", "missing_skill"],
        expected_contains=["endpoint", "absent text"],
        tags=["complex"],
    )

    result = EvalHarness()._run_case(case, FakeAgent())

    assert result.passed is False
    assert result.actual_skills == ["cohort_summary"]
    assert "Expected skill 'missing_skill' not called" in result.errors
    assert "Expected text 'absent text' not in response" in result.errors
    assert result.metadata["claim_count"] == 2
    assert result.metadata["evidence_count"] == 1
    assert result.metadata["safety_status"] == "PARTIAL"
    assert result.metadata["debate_disagreement"] is True
    assert result.metadata["prompt_tokens_delta"] == 5
    assert result.metadata["completion_tokens_delta"] == 7


def test_run_case_without_expected_text_can_pass():
    agent = MagicMock()
    agent.state = SimpleNamespace(
        records=[SimpleNamespace(skill="cohort_summary")],
        last_orchestration={"claims": [], "evidence_links": [], "safety_status": "PASS"},
        token_usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
    )
    agent.run.return_value = "any response"

    result = EvalHarness()._run_case(
        HarnessTestCase(id="no_text", query="q", expected_skills=["cohort_summary"]),
        agent,
    )

    assert result.passed is True
    assert result.errors == []


def test_run_case_exception_returns_failed_result():
    agent = MagicMock()
    agent.state = SimpleNamespace(token_usage=None)
    agent.run.side_effect = RuntimeError("boom")

    result = EvalHarness()._run_case(HarnessTestCase(id="case_error", query="q", tags=["error"]), agent)

    assert result.passed is False
    assert result.errors[0].startswith("Exception: RuntimeError: boom")
    assert result.metadata["safety_status"] == "FAIL"


def test_harness_run_uses_agent_path_when_benchmark_returns_none():
    class AgentPathBenchmark(Benchmark):
        name = "agent_path"

        def __init__(self):
            self.cases = [HarnessTestCase(id="agent_path_case", query="ok", expected_contains=["done"])]

    agent = MagicMock()
    agent.state = SimpleNamespace(
        records=[],
        last_orchestration={},
        token_usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
    )
    agent.run.return_value = "done"

    result = EvalHarness().run(AgentPathBenchmark(), agent)

    assert result.n_total == 1
    assert result.n_passed == 1
    agent.run.assert_called_once_with("ok")


def test_harness_enforce_gate_logs_failures(caplog):
    class GateFailBenchmark(Benchmark):
        name = "gate_fail"

        def __init__(self):
            self.cases = [HarnessTestCase(id="gate_fail_case", query="q", tags=["complex"])]

        def run_case(self, case, agent):
            return HarnessCaseResult(
                case_id=case.id,
                passed=False,
                metadata={
                    "tags": case.tags,
                    "claim_count": 1,
                    "evidence_count": 0,
                    "safety_status": "FAIL",
                    "debate_disagreement": True,
                },
            )

    with caplog.at_level(logging.WARNING):
        result = EvalHarness().run(GateFailBenchmark(), MagicMock(), enforce_gate=True)

    assert result.gate_passed is False
    assert "Reliability gate failed" in caplog.text


def test_observability_empty_and_single_result_paths():
    harness = EvalHarness()

    empty = harness._compute_observability([])
    one = harness._compute_observability([
        HarnessCaseResult(
            case_id="one",
            passed=True,
            elapsed_s=-3.0,
            metadata={
                "claim_count": 0,
                "evidence_count": 0,
                "safety_status": "PASS",
                "tags": [],
                "prompt_tokens_delta": 2,
                "completion_tokens_delta": 3,
            },
        )
    ])

    assert empty["success_rate"] == 0.0
    assert empty["p95_latency_s"] == 0.0
    assert one["evidence_coverage"] == 1.0
    assert one["complex_task_success"] == 1.0
    assert one["token_cost"] == 5.0
    assert one["p95_latency_s"] == 0.0


def test_compare_with_baseline_and_gate_failure_messages():
    harness = EvalHarness()
    obs = {
        "evidence_coverage": 0.5,
        "stat_guardrail_violation": 0.2,
        "complex_task_success": 0.1,
        "wrong_consensus_rate": 0.9,
    }
    baseline = {"complex_task_success": 0.5, "wrong_consensus_rate": 0.6}

    assert harness._compare_with_baseline(obs, None) == {}
    comparison = harness._compare_with_baseline(obs, baseline)
    ok_no_base, failures_no_base = harness._check_reliability_gate(obs)
    ok_with_base, failures_with_base = harness._check_reliability_gate(obs, baseline)

    assert comparison["complex_task_success_uplift"] == -0.4
    assert comparison["wrong_consensus_reduction"] == -0.30000000000000004
    assert ok_no_base is False
    assert "evidence_coverage < 95%" in failures_no_base
    assert "complex_task_success below minimum fallback threshold" in failures_no_base
    assert "wrong_consensus_rate reduction target not met" in failures_no_base
    assert ok_with_base is False
    assert "complex_task_success uplift < +15% vs baseline" in failures_with_base
    assert "wrong_consensus reduction < 40% vs baseline" in failures_with_base
