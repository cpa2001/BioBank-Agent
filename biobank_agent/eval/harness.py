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
from pathlib import Path
from statistics import quantiles
from types import SimpleNamespace
from typing import Any, Optional, TYPE_CHECKING

from .report_review import (
    artifact_review_gate_failures,
    review_benchmark_artifacts,
    reviewer_specs as build_reviewer_specs,
)

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
    metadata: dict[str, Any] = field(default_factory=dict)


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
    review_loop: dict[str, Any] = field(default_factory=dict)

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
                f"wrong_consensus_reduction={cmp.get('wrong_consensus_reduction_ratio', 0):+.1%}\n"
            )
        review_str = ""
        if self.review_loop:
            reviews = self.review_loop.get("reviews", []) or []
            labels = [
                f"{r.get('reviewer', 'reviewer')}={r.get('status', 'unknown')}"
                for r in reviews[:3]
            ]
            review_str = (
                f"  Review loop: {self.review_loop.get('status', 'unknown')}"
                f" ({', '.join(labels) if labels else 'no reviews'})\n"
            )
        gate_line = "PASS" if self.gate_passed else f"FAIL ({'; '.join(self.gate_failures[:3])})"
        return (
            f"Benchmark: {self.benchmark_name} [{self.mode}]\n"
            f"  Results: {self.n_passed}/{self.n_total} passed ({self.pass_rate:.0%})\n"
            f"  Mean score: {self.mean_score:.3f}\n"
            f"  Observability: {obs_str}\n"
            f"{cmp_str}"
            f"{review_str}"
            f"  Reliability gate: {gate_line}\n"
            f"  Time: {self.total_elapsed_s:.1f}s"
        )

    def failures(self) -> list[TestResult]:
        return [r for r in self.results if not r.passed]


class Benchmark:
    """Base class for evaluation benchmarks."""
    name: str = "base"
    cases: list[TestCase] = []

    def run_case(self, case: TestCase, agent: Agent) -> TestResult | None:
        """Optionally run a case without going through ``agent.run``."""
        return None

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
        review_loop: bool = False,
        primary_reviewer: str = "report_review_primary",
        include_claude: bool = False,
        review_timeout_s: int = 600,
    ) -> BenchmarkResult:
        """Execute all test cases in a benchmark."""
        t0 = time.time()
        results = []

        for case in benchmark.cases:
            result = benchmark.run_case(case, agent)
            if result is None:
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
        if review_loop:
            out.review_loop = self._run_review_loop(
                result=out,
                benchmark=benchmark,
                agent=agent,
                primary_reviewer=primary_reviewer,
                include_claude=include_claude,
                review_timeout_s=review_timeout_s,
            )
            review_failures = self._review_gate_failures(out.review_loop)
            if review_failures:
                out.gate_passed = False
                out.gate_failures.extend(review_failures)
        if enforce_gate and not out.gate_passed:
            logger.warning("Reliability gate failed: %s", "; ".join(out.gate_failures))
        return out

    def _run_review_loop(
        self,
        result: BenchmarkResult,
        benchmark: Benchmark,
        agent: Agent,
        primary_reviewer: str = "report_review_primary",
        include_claude: bool = False,
        review_timeout_s: int = 600,
    ) -> dict[str, Any]:
        """Run optional external review diagnostics for a benchmark result."""
        reviewers = self._reviewer_specs(primary_reviewer, include_claude)
        artifact_review = review_benchmark_artifacts(
            result,
            benchmark,
            primary_reviewer=primary_reviewer,
            include_claude=include_claude,
        )
        review_context = self._review_context(result, benchmark, artifact_review=artifact_review)
        registry = getattr(agent, "registry", None)
        execute = getattr(registry, "execute", None)
        if not callable(execute):
            return {
                "enabled": True,
                "status": "skipped",
                "primary_reviewer": reviewers[0]["reviewer"],
                "include_claude": include_claude,
                "reviewer_specs": reviewers,
                "artifact_review": artifact_review,
                "artifact_checklist": artifact_review.get("artifact_checklist", {}),
                "role_verdicts": artifact_review.get("role_verdicts", []),
                "old_report_overwrite_ready": artifact_review.get("old_report_overwrite_ready", False),
                "reviews": [
                    {
                        **spec,
                        "status": "skipped",
                        "verdict": "MISSING",
                        "error": "agent registry unavailable",
                    }
                    for spec in reviewers
                ],
            }

        ctx = self._review_ctx(agent)
        reviews = []
        for spec in reviewers:
            args = {
                "focus": (
                    f"{result.benchmark_name} evaluation review. "
                    f"Primary reviewer concept: {reviewers[0]['reviewer']}. "
                    "Assess failures, report quality, statistical caveats, and whether this eval should ship."
                ),
                "context": review_context,
                "timeout_s": int(review_timeout_s),
            }
            try:
                payload = execute(spec["skill"], args, ctx=ctx)
                reviews.append(self._normalize_review_payload(spec, payload))
            except Exception as e:
                reviews.append({
                    **spec,
                    "status": "error",
                    "verdict": "ERROR",
                    "error": f"{type(e).__name__}: {str(e)[:300]}",
                })

        statuses = {str(r.get("status", "")).lower() for r in reviews}
        verdicts = {str(r.get("verdict", "")).upper() for r in reviews}
        if artifact_review.get("status") == "blocked":
            status = "blocked"
        elif "BLOCK" in verdicts:
            status = "blocked"
        elif "error" in statuses:
            status = "error"
        elif statuses and statuses <= {"success"}:
            status = "completed"
        elif statuses and statuses <= {"skipped"}:
            status = "skipped"
        elif "success" in statuses:
            status = "partial"
        else:
            status = "unknown"

        return {
            "enabled": True,
            "status": status,
            "primary_reviewer": reviewers[0]["reviewer"],
            "include_claude": include_claude,
            "reviewer_specs": reviewers,
            "artifact_review": artifact_review,
            "artifact_checklist": artifact_review.get("artifact_checklist", {}),
            "role_verdicts": artifact_review.get("role_verdicts", []),
            "old_report_overwrite_ready": artifact_review.get("old_report_overwrite_ready", False),
            "reviews": reviews,
        }

    @staticmethod
    def _reviewer_specs(primary_reviewer: str, include_claude: bool) -> list[dict[str, Any]]:
        return build_reviewer_specs(primary_reviewer, include_claude)

    @staticmethod
    def _review_context(
        result: BenchmarkResult,
        benchmark: Benchmark,
        artifact_review: Optional[dict[str, Any]] = None,
    ) -> str:
        failures = []
        for case_result in result.failures()[:10]:
            failures.append({
                "case_id": case_result.case_id,
                "score": case_result.score,
                "errors": case_result.errors[:6],
                "metadata": case_result.metadata,
            })
        payload = {
            "suite": result.benchmark_name,
            "mode": result.mode,
            "benchmark_case_count": len(getattr(benchmark, "cases", []) or []),
            "n_total": result.n_total,
            "n_passed": result.n_passed,
            "mean_score": result.mean_score,
            "observability": result.observability,
            "gate_passed": result.gate_passed,
            "gate_failures": result.gate_failures,
            "artifact_review": artifact_review or {},
            "failures": failures,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str)[:12000]

    @staticmethod
    def _review_ctx(agent: Agent):
        settings = getattr(agent, "settings", None)
        report_root = getattr(settings, "reports_dir", Path("./reports"))
        report_dir = Path(report_root) / "eval" / "review_loop"
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        build_ctx = getattr(agent, "_build_ctx", None)
        if callable(build_ctx):
            try:
                return build_ctx(report_dir)
            except Exception:
                pass
        return SimpleNamespace(
            settings=settings,
            state=getattr(agent, "state", None),
            memory=getattr(agent, "memory", None),
            report_dir=report_dir,
        )

    @staticmethod
    def _normalize_review_payload(spec: dict[str, Any], payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            stdout = str(payload)[:8000]
            return {
                **spec,
                "status": "success",
                "stdout": stdout,
                "verdict": EvalHarness._review_verdict(stdout),
            }
        status = str(payload.get("status") or "success").lower()
        stdout = str(payload.get("stdout") or "")[:8000]
        stderr = str(payload.get("stderr") or "")[:4000]
        error = str(payload.get("error") or "")[:1000]
        return {
            **spec,
            "status": status,
            "stdout": stdout,
            "stderr": stderr,
            "error": error,
            "verdict": EvalHarness._review_verdict("\n".join([stdout, stderr, error])),
            "command_display": str(payload.get("command_display") or ""),
        }

    @staticmethod
    def _review_verdict(text: str) -> str:
        """Parse ALLOW/BLOCK review output without treating free text as approval."""
        for raw in str(text or "").splitlines()[:80]:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("BLOCK:") or upper == "BLOCK":
                return "BLOCK"
            if upper.startswith("ALLOW:") or upper == "ALLOW":
                return "ALLOW"
        return "UNKNOWN"

    @staticmethod
    def _review_gate_failures(review_loop: dict[str, Any]) -> list[str]:
        """Translate review-loop diagnostics into reliability gate failures."""
        if not review_loop:
            return []
        failures: list[str] = []
        reviews = review_loop.get("reviews", []) or []
        if not reviews:
            return ["review loop produced no reviewer output"]
        for review in reviews:
            reviewer = review.get("reviewer") or review.get("skill") or "reviewer"
            status = str(review.get("status", "")).lower()
            verdict = str(review.get("verdict", "")).upper()
            if verdict == "BLOCK":
                failures.append(f"{reviewer} returned BLOCK")
            if status in {"error", "skipped"}:
                reason = review.get("error") or status
                failures.append(f"{reviewer} review {status}: {reason}")
        loop_status = str(review_loop.get("status", "")).lower()
        if loop_status in {"blocked", "error", "skipped"} and not failures:
            failures.append(f"review loop status={loop_status}")
        failures.extend(artifact_review_gate_failures(review_loop.get("artifact_review", {})))
        return failures

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
