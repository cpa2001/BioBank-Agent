"""Online evaluation over a run tree: pure scorers that grade a session's spans.

An ``Evaluator`` is a pure function ``RunNode -> list[EvalResult]`` — it reads the tree
(run_tree.py) and emits scored, pass/fail judgements with a note. No model calls, no IO, so a
session can be graded deterministically as it runs (or after the fact). Built-ins cover the
cheap, always-useful signals: tool success rate, latency budget, error-free. Add domain
evaluators (methodology-gate respected, evidence linked, ...) by composing more of the same.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from biobank_agent.runtime.run_tree import ERROR, OK, PHASE, TOOL, RunNode, summarize_run_tree


@dataclass
class EvalResult:
    name: str
    score: float          # 0.0 – 1.0
    passed: bool
    note: str = ""
    node_id: str | None = None


Evaluator = Callable[[RunNode], list[EvalResult]]


@dataclass
class RunEvalReport:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def score(self) -> float:
        return round(sum(r.score for r in self.results) / len(self.results), 4) if self.results else 1.0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "results": [vars(r) for r in self.results],
        }


def tool_success_rate(min_rate: float = 0.8) -> Evaluator:
    """Fraction of tool spans that ended ``ok`` must meet ``min_rate`` (vacuously true if none)."""
    def _eval(root: RunNode) -> list[EvalResult]:
        s = summarize_run_tree(root)
        rate = s["tool_success_rate"]
        if rate is None:
            return [EvalResult("tool_success_rate", 1.0, True, "no tool calls")]
        return [EvalResult("tool_success_rate", rate, rate >= min_rate,
                           f"{rate:.0%} of {s['tool_calls']} tool call(s) succeeded")]
    return _eval


def error_free() -> Evaluator:
    """No span anywhere in the tree ended in ``error``."""
    def _eval(root: RunNode) -> list[EvalResult]:
        errs = [n for n in root.walk() if n.status == ERROR]
        return [EvalResult("error_free", 0.0 if errs else 1.0, not errs,
                           "no errors" if not errs else f"{len(errs)} error span(s): " +
                           ", ".join(sorted({n.name for n in errs}))[:200])]
    return _eval


def latency_budget(max_seconds: float) -> Evaluator:
    """Every tool span's wall-clock latency must be within ``max_seconds``."""
    def _eval(root: RunNode) -> list[EvalResult]:
        over = [n for n in root.walk()
                if n.kind == TOOL and n.duration is not None and n.duration > max_seconds]
        worst = max((n.duration for n in root.walk() if n.kind == TOOL and n.duration is not None), default=0.0)
        return [EvalResult("latency_budget", 0.0 if over else 1.0, not over,
                           f"slowest tool {worst:.2f}s (budget {max_seconds:.0f}s); "
                           f"{len(over)} over budget")]
    return _eval


def no_repeated_tool_failures(max_repeats: int = 2) -> Evaluator:
    """Flag a stuck retry loop: a single tool name that ends in ``error`` more than
    ``max_repeats`` times (default >2, i.e. a 3rd identical failure)."""
    def _eval(root: RunNode) -> list[EvalResult]:
        fails = Counter(n.name for n in root.walk() if n.kind == TOOL and n.status == ERROR)
        offenders = sorted(name for name, count in fails.items() if count > max_repeats)
        passed = not offenders
        note = "no stuck tool loops" if passed else (
            f"repeated failures (>{max_repeats}x): " + ", ".join(offenders))
        return [EvalResult("no_repeated_tool_failures", 1.0 if passed else 0.0, passed, note)]
    return _eval


def phase_errors_free() -> Evaluator:
    """No ``phase`` span anywhere in the tree ended in ``error``."""
    def _eval(root: RunNode) -> list[EvalResult]:
        bad = [n for n in root.walk() if n.kind == PHASE and n.status == ERROR]
        passed = not bad
        note = "no phase errors" if passed else (
            f"{len(bad)} failed phase(s): " + ", ".join(sorted({n.name for n in bad}))[:200])
        return [EvalResult("phase_errors_free", 1.0 if passed else 0.0, passed, note)]
    return _eval


def evaluate_run_tree(root: RunNode, evaluators: list[Evaluator]) -> RunEvalReport:
    """Apply each evaluator and collect results (order-preserving). Pure."""
    report = RunEvalReport()
    for ev in evaluators:
        report.results.extend(ev(root))
    return report


def default_evaluators() -> list[Evaluator]:
    return [tool_success_rate(), error_free()]


__all__ = ["EvalResult", "Evaluator", "RunEvalReport", "evaluate_run_tree",
           "tool_success_rate", "error_free", "latency_budget", "default_evaluators",
           "no_repeated_tool_failures", "phase_errors_free"]
