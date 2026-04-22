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


@dataclass
class BenchmarkResult:
    """Aggregate result of a benchmark run."""
    benchmark_name: str
    results: list[TestResult] = field(default_factory=list)
    timestamp: str = ""
    total_elapsed_s: float = 0.0

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
        return (
            f"Benchmark: {self.benchmark_name}\n"
            f"  Results: {self.n_passed}/{self.n_total} passed ({self.pass_rate:.0%})\n"
            f"  Mean score: {self.mean_score:.3f}\n"
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

    def run(self, benchmark: Benchmark, agent: Agent) -> BenchmarkResult:
        """Execute all test cases in a benchmark."""
        t0 = time.time()
        results = []

        for case in benchmark.cases:
            result = self._run_case(case, agent)
            result.score = benchmark.score(result, case)
            results.append(result)

        return BenchmarkResult(
            benchmark_name=benchmark.name,
            results=results,
            timestamp=datetime.now().isoformat(),
            total_elapsed_s=time.time() - t0,
        )

    def _run_case(self, case: TestCase, agent: Agent) -> TestResult:
        """Run a single test case."""
        t0 = time.time()
        errors = []

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
            return TestResult(
                case_id=case.id,
                passed=passed,
                actual_skills=actual_skills[-5:],
                actual_text=response[:500],
                errors=errors,
                elapsed_s=elapsed,
            )

        except Exception as e:
            return TestResult(
                case_id=case.id,
                passed=False,
                errors=[f"Exception: {type(e).__name__}: {str(e)[:200]}"],
                elapsed_s=time.time() - t0,
            )
