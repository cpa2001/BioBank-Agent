"""Tests for research_eval_v1 benchmark and observability metrics."""

from biobank_agent.eval.benchmarks import ResearchEvalV1
from biobank_agent.eval.harness import EvalHarness, TestResult as HarnessCaseResult


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
