"""Evaluation harness — run benchmarks against the agent systematically.

Usage::

    from biobank_agent.eval.harness import EvalHarness
    from biobank_agent.eval.benchmarks import SkillSchemaBenchmark

    harness = EvalHarness()
    result = harness.run(SkillSchemaBenchmark(), agent)
    print(result.summary())
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from statistics import quantiles
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single evaluation test case."""
    id: str
    query: str
    expected_skills: list[str] = field(default_factory=list)
    expected_contains: list[str] = field(default_factory=list)
    expected_result_keys: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of a single test case evaluation."""
    case_id: str
    passed: bool
    score: float = 0.0          # 0.0-1.0
    actual_skills: list[str] = field(default_factory=list)
    actual_text: str = ""
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregate result of a benchmark run."""
    benchmark_name: str
    results: list[TestResult] = field(default_factory=list)
    timestamp: str = ""
    total_elapsed_s: float = 0.0
    mode: str = "baseline"
    observability: dict[str, float] = field(default_factory=dict)
    baseline_observability: dict[str, float] = field(default_factory=dict)
    comparative: dict[str, float] = field(default_factory=dict)
    gate_passed: bool = True
    gate_failures: list[str] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_total if self.n_total else 0.0

    @property
    def mean_score(self) -> float:
        scores = [r.score for r in self.results]
        return sum(scores) / len(scores) if scores else 0.0

    def summary(self) -> str:
        obs = self.observability or {}
        obs_str = (
            f"success={obs.get('success_rate', 0):.1%}, "
            f"evidence={obs.get('evidence_coverage', 0):.1%}, "
            f"stat_violation={obs.get('stat_guardrail_violation', 0):.1%}, "
            f"wrong_consensus={obs.get('wrong_consensus_rate', 0):.1%}, "
            f"p95={obs.get('p95_latency_s', 0):.2f}s"
        )
        cmp = self.comparative or {}
        cmp_str = ""
        if cmp:
            cmp_str = (
                "\n  A/B delta: "
                f"complex_uplift={cmp.get('complex_task_success_uplift', 0):+.1%}, "
                f"wrong_consensus_reduction={cmp.get('wrong_consensus_reduction_ratio', 0):+.1%}"
            )
        gate_line = "PASS" if self.gate_passed else f"FAIL ({'; '.join(self.gate_failures[:3])})"
        return (
            f"Benchmark: {self.benchmark_name} [{self.mode}]\n"
            f"  Results: {self.n_passed}/{self.n_total} passed ({self.pass_rate:.0%})\n"
            f"  Mean score: {self.mean_score:.3f}\n"
            f"  Observability: {obs_str}\n"
            f"{cmp_str}"
            f"  Reliability gate: {gate_line}\n"
            f"  Time: {self.total_elapsed_s:.1f}s"
        )

    def failures(self) -> list[TestResult]:
        return [r for r in self.results if not r.passed]


class Benchmark:
    """Base class for evaluation benchmarks."""
    name: str = "base"
    cases: list[TestCase] = []

    def score(self, result: TestResult, case: TestCase) -> float:
        """Score a test result against its case. Override in subclasses."""
        return 1.0 if result.passed else 0.0


class EvalHarness:
    """Run benchmarks against the agent and collect results.

    Usage::

        harness = EvalHarness()
        result = harness.run(benchmark, agent)
        print(result.summary())
    """

    def run(
        self,
        benchmark: Benchmark,
        agent: Agent,
        mode: str = "baseline",
        enforce_gate: bool = False,
        baseline_observability: Optional[dict[str, float]] = None,
    ) -> BenchmarkResult:
        """Execute all test cases in a benchmark."""
        t0 = time.time()
        results = []

        for case in benchmark.cases:
            result = self._run_case(case, agent)
            result.score = benchmark.score(result, case)
            results.append(result)

        out = BenchmarkResult(
            benchmark_name=benchmark.name,
            results=results,
            timestamp=datetime.now().isoformat(),
            total_elapsed_s=time.time() - t0,
            mode=mode,
            baseline_observability=baseline_observability or {},
        )
        out.observability = self._compute_observability(results)
        out.comparative = self._compare_with_baseline(
            out.observability,
            out.baseline_observability,
        )
        out.gate_passed, out.gate_failures = self._check_reliability_gate(
            out.observability,
            baseline_observability=out.baseline_observability,
        )
        if enforce_gate and not out.gate_passed:
            logger.warning("Reliability gate failed: %s", "; ".join(out.gate_failures))
        return out

    def _run_case(self, case: TestCase, agent: Agent) -> TestResult:
        """Run a single test case."""
        t0 = time.time()
        errors = []
        tok_before = getattr(getattr(agent, "state", None), "token_usage", None)
        p_before = int(getattr(tok_before, "prompt_tokens", 0)) if tok_before else 0
        c_before = int(getattr(tok_before, "completion_tokens", 0)) if tok_before else 0

        try:
            response = agent.run(case.query)
            actual_skills = [r.skill for r in agent.state.records if r.skill != "think"]
            elapsed = time.time() - t0

            # Check expected skills
            if case.expected_skills:
                for skill in case.expected_skills:
                    if skill not in actual_skills:
                        errors.append(f"Expected skill '{skill}' not called")

            # Check expected text content
            if case.expected_contains:
                response_lower = response.lower()
                for text in case.expected_contains:
                    if text.lower() not in response_lower:
                        errors.append(f"Expected text '{text}' not in response")

            passed = len(errors) == 0
            orchestration = getattr(agent.state, "last_orchestration", {}) or {}
            claims = orchestration.get("claims", []) if isinstance(orchestration, dict) else []
            evidence = orchestration.get("evidence_links", []) if isinstance(orchestration, dict) else []
            debate = orchestration.get("debate_trace", {}) if isinstance(orchestration, dict) else {}
            safety_status = orchestration.get("safety_status", "PASS") if isinstance(orchestration, dict) else "PASS"
            tok_after = getattr(getattr(agent, "state", None), "token_usage", None)
            p_after = int(getattr(tok_after, "prompt_tokens", 0)) if tok_after else p_before
            c_after = int(getattr(tok_after, "completion_tokens", 0)) if tok_after else c_before

            return TestResult(
                case_id=case.id,
                passed=passed,
                actual_skills=actual_skills[-5:],
                actual_text=response[:500],
                errors=errors,
                elapsed_s=elapsed,
                metadata={
                    "tags": case.tags,
                    "claim_count": len(claims),
                    "evidence_count": len(evidence),
                    "safety_status": safety_status,
                    "debate_disagreement": bool(debate.get("disagreement")) if isinstance(debate, dict) else False,
                    "prompt_tokens_delta": max(0, p_after - p_before),
                    "completion_tokens_delta": max(0, c_after - c_before),
                },
            )

        except Exception as e:
            return TestResult(
                case_id=case.id,
                passed=False,
                errors=[f"Exception: {type(e).__name__}: {str(e)[:200]}"],
                elapsed_s=time.time() - t0,
                metadata={"tags": case.tags, "claim_count": 0, "evidence_count": 0, "safety_status": "FAIL"},
            )

    def _compute_observability(self, results: list[TestResult]) -> dict[str, float]:
        """Compute reliability dashboard metrics."""
        if not results:
            return {
                "success_rate": 0.0,
                "evidence_coverage": 0.0,
                "stat_guardrail_violation": 0.0,
                "wrong_consensus_rate": 0.0,
                "complex_task_success": 0.0,
                "token_cost": 0.0,
                "p95_latency_s": 0.0,
            }

        success_rate = sum(1 for r in results if r.passed) / len(results)
        claims_total = sum(int(r.metadata.get("claim_count", 0)) for r in results)
        evidence_total = sum(int(r.metadata.get("evidence_count", 0)) for r in results)
        evidence_coverage = 1.0 if claims_total == 0 else min(1.0, evidence_total / max(1, claims_total))

        stat_guardrail_violation = (
            sum(1 for r in results if str(r.metadata.get("safety_status", "PASS")).upper() != "PASS")
            / len(results)
        )

        disagreements = [r for r in results if bool(r.metadata.get("debate_disagreement", False))]
        unresolved = [r for r in disagreements if int(r.metadata.get("evidence_count", 0)) == 0]
        wrong_consensus_rate = (len(unresolved) / len(disagreements)) if disagreements else 0.0

        complex_results = [r for r in results if "complex" in (r.metadata.get("tags", []) or [])]
        complex_task_success = (
            sum(1 for r in complex_results if r.passed) / len(complex_results)
            if complex_results else success_rate
        )

        token_cost = float(sum(
            int(r.metadata.get("prompt_tokens_delta", 0)) + int(r.metadata.get("completion_tokens_delta", 0))
            for r in results
        ))
        latencies = [max(0.0, r.elapsed_s) for r in results]
        if len(latencies) >= 2:
            p95_latency_s = float(quantiles(latencies, n=20)[18])
        else:
            p95_latency_s = latencies[0] if latencies else 0.0

        return {
            "success_rate": success_rate,
            "evidence_coverage": evidence_coverage,
            "stat_guardrail_violation": stat_guardrail_violation,
            "wrong_consensus_rate": wrong_consensus_rate,
            "complex_task_success": complex_task_success,
            "token_cost": token_cost,
            "p95_latency_s": p95_latency_s,
        }

    def _compare_with_baseline(
        self,
        obs: dict[str, float],
        baseline_observability: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """Compute A/B deltas against baseline observability."""
        base = baseline_observability or {}
        if not base:
            return {}
        base_complex = float(base.get("complex_task_success", 0.0))
        base_wrong = float(base.get("wrong_consensus_rate", 0.0))
        cur_complex = float(obs.get("complex_task_success", 0.0))
        cur_wrong = float(obs.get("wrong_consensus_rate", 0.0))
        complex_uplift = cur_complex - base_complex
        wrong_consensus_reduction = base_wrong - cur_wrong
        reduction_ratio = wrong_consensus_reduction / max(base_wrong, 1e-6)
        return {
            "complex_task_success_uplift": complex_uplift,
            "wrong_consensus_reduction": wrong_consensus_reduction,
            "wrong_consensus_reduction_ratio": reduction_ratio,
        }

    def _check_reliability_gate(
        self,
        obs: dict[str, float],
        baseline_observability: Optional[dict[str, float]] = None,
    ) -> tuple[bool, list[str]]:
        """Check north-star thresholds."""
        failures = []
        if obs.get("evidence_coverage", 0.0) < 0.95:
            failures.append("evidence_coverage < 95%")
        if obs.get("stat_guardrail_violation", 1.0) > 0.01:
            failures.append("stat_guardrail_violation > 1%")
        base = baseline_observability or {}
        if base:
            complex_uplift = obs.get("complex_task_success", 0.0) - base.get("complex_task_success", 0.0)
            if complex_uplift < 0.15:
                failures.append("complex_task_success uplift < +15% vs baseline")
            base_wrong = max(base.get("wrong_consensus_rate", 0.0), 1e-6)
            reduction_ratio = (base_wrong - obs.get("wrong_consensus_rate", 0.0)) / base_wrong
            if reduction_ratio < 0.40:
                failures.append("wrong_consensus reduction < 40% vs baseline")
        else:
            if obs.get("complex_task_success", 0.0) < 0.15:
                failures.append("complex_task_success below minimum fallback threshold")
            if obs.get("wrong_consensus_rate", 1.0) > 0.60:
                failures.append("wrong_consensus_rate reduction target not met")
        return (len(failures) == 0, failures)
