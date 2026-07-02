"""Lightweight workflow DAG runner: steps + dependencies, fan-out / fan-in.

Executes steps honouring their dependency edges — each topological "level" (all steps whose
dependencies are already satisfied) runs concurrently via the council thread-pool, then the run advances.
A step's work is an injected callable receiving its upstream results, so the runner is substrate-agnostic
and offline-testable: a step may call a skill, an external agent (see ``external_orchestration``), or a
biobank subagent. Failures are isolated — a failed/timed-out step marks its dependents ``skipped`` and the
run returns a per-step status map rather than raising. A dependency cycle is a programming error and is
rejected up front.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from .council import CouncilContext, map_parallel


@dataclass
class WorkflowStep:
    """One node in the workflow DAG.

    ``run(upstream)`` receives a dict ``{dependency_id: result_value}`` of its resolved dependencies and
    returns this step's result. ``dependencies`` are the incoming edges (ids of other steps).
    """

    id: str
    run: Callable[[dict], Any]
    dependencies: tuple[str, ...] = ()
    label: str = ""


@dataclass
class WorkflowStepResult:
    id: str
    status: str  # "ok" | "failed" | "skipped"
    value: Any = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class WorkflowResult:
    steps: dict[str, WorkflowStepResult] = field(default_factory=dict)
    order: list[list[str]] = field(default_factory=list)  # ids per execution batch (topological levels)

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(r.status == "ok" for r in self.steps.values())

    def result(self, step_id: str) -> Any:
        r = self.steps.get(step_id)
        return r.value if r is not None else None


def _index_and_validate(steps: Sequence[WorkflowStep]) -> dict[str, WorkflowStep]:
    by_id: dict[str, WorkflowStep] = {}
    for step in steps:
        if step.id in by_id:
            raise ValueError(f"duplicate workflow step id: {step.id!r}")
        by_id[step.id] = step
    for step in steps:
        for dep in step.dependencies:
            if dep not in by_id:
                raise ValueError(f"step {step.id!r} depends on unknown step {dep!r}")
    # Kahn cycle check: if not every node can be emitted, there is a cycle. Count UNIQUE dependency
    # edges — a duplicate dep (e.g. ("a", "a")) is acyclic and must not be miscounted as a cycle, since
    # the decrement below fires once per predecessor node.
    indegree = {sid: 0 for sid in by_id}
    for step in steps:
        for _dep in set(step.dependencies):
            indegree[step.id] += 1
    ready = [sid for sid, d in indegree.items() if d == 0]
    emitted = 0
    while ready:
        sid = ready.pop()
        emitted += 1
        for other in by_id.values():
            if sid in other.dependencies:
                indegree[other.id] -= 1
                if indegree[other.id] == 0:
                    ready.append(other.id)
    if emitted != len(by_id):
        raise ValueError("workflow has a dependency cycle")
    return by_id


def run_workflow(
    steps: Sequence[WorkflowStep],
    *,
    max_workers: int = 4,
    timeout_s: float = 240.0,
    emit: Optional[Callable[..., None]] = None,
) -> WorkflowResult:
    """Run a workflow DAG, executing each ready level concurrently. Never raises on step failure.

    ``timeout_s`` bounds how long the runner WAITS for a level, after which unfinished steps are recorded
    as timed out and the run advances. It cannot force-kill a worker thread (a Python limitation), so a
    step's ``run`` callable must be self-bounding — e.g. an external-agent step relies on the subprocess
    timeout; a long compute step should enforce its own deadline — or its thread may linger past the run.
    """
    by_id = _index_and_validate(steps)
    result = WorkflowResult()
    remaining = dict(by_id)
    ctx = CouncilContext(provider_router=None, max_workers=max(1, int(max_workers)), timeout_s=float(timeout_s), emit=emit)

    while remaining:
        ready: list[WorkflowStep] = []
        blocked: list[WorkflowStep] = []
        for sid, step in remaining.items():
            statuses = [result.steps.get(dep) for dep in step.dependencies]
            if any(s is None for s in statuses):
                continue  # a dependency hasn't run yet — wait for a later level
            if all(s.status == "ok" for s in statuses):
                ready.append(step)
            else:
                blocked.append(step)  # an upstream dep failed/skipped — this step cannot run

        for step in blocked:
            result.steps[step.id] = WorkflowStepResult(step.id, "skipped", error="dependency did not succeed")
            remaining.pop(step.id, None)

        if not ready:
            if blocked:
                continue  # made progress by skipping; re-evaluate
            break  # nothing ready and nothing blocked left (all remaining are done) — done

        result.order.append([s.id for s in ready])

        def _run(step: WorkflowStep) -> Any:
            upstream = {dep: result.steps[dep].value for dep in step.dependencies}
            return step.run(upstream)

        mapped = map_parallel(ctx, ready, _run, stage="workflow", label_fn=lambda i, s: s.label or s.id)
        # map_parallel returns results sorted by index over `ready`, so they align with `ready`.
        for item, step in zip(mapped, ready):
            if item.ok:
                result.steps[step.id] = WorkflowStepResult(step.id, "ok", value=item.value)
            else:
                result.steps[step.id] = WorkflowStepResult(step.id, "failed", error=item.error or "step failed")
            remaining.pop(step.id, None)

    return result


__all__ = [
    "WorkflowStep",
    "WorkflowStepResult",
    "WorkflowResult",
    "run_workflow",
]
