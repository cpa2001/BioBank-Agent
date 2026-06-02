"""Completion gates for the LLM-driven runtime.

The runtime executes approved plans via the ``run_turn`` model/tool loop rather
than a rigid skill+args DAG, so the legacy ``PlanExecutor`` is NOT ported
wholesale. Instead this module supplies the *gates* the legacy executor enforced,
adapted to the runtime:

* report-contract check (did a plan that promises a report actually produce one?)
* goal-acceptance (an LLM judge decides whether the objective was met, given the
  evidence — not just an assistant "done")
* completion warnings (unfinished steps, missing verification, missing report)
* LLM-proposed, *bounded* repair when the goal is not yet accepted (recorded via
  the runtime's existing ``record_repair_attempt`` / ``verification_repair_loop``)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from biobank_agent.runtime.council import CouncilContext, CouncilJob, extract_json, run_parallel
from biobank_agent.runtime.types import ProviderRole

logger = logging.getLogger(__name__)

_TERMINAL = {"done", "completed", "skipped"}


@dataclass
class CompletionAssessment:
    accepted: bool = False
    reasons: str = ""
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    needs_report: bool = False
    report_present: bool = False
    goal_checked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": self.reasons,
            "missing": list(self.missing),
            "warnings": list(self.warnings),
            "needs_report": self.needs_report,
            "report_present": self.report_present,
            "goal_checked": self.goal_checked,
        }


class CompletionGate:
    """Assess whether an approved plan's objective has actually been achieved."""

    def __init__(self, provider_router: Any, config: Any | None = None, *, timeout_s: float = 120.0, clock: Callable[[], float] | None = None) -> None:
        self.provider_router = provider_router
        self.config = config
        self.timeout_s = float(timeout_s)
        self._clock = clock or (lambda: 0.0)

    # ------------------------------------------------------------------ public
    def assess(
        self,
        objective: str,
        plan: Any,
        *,
        evidence_summary: str = "",
        report_present: bool = False,
        emit: Callable[..., None] | None = None,
        session_id: str = "completion",
        turn_id: str = "completion",
    ) -> CompletionAssessment:
        ctx = CouncilContext(
            provider_router=self.provider_router,
            session_id=session_id,
            turn_id=turn_id,
            emit=emit,
            timeout_s=self.timeout_s,
            clock=self._clock,
        )
        warnings: list[str] = []
        needs_report = self._plan_requires_report(plan)
        if needs_report and not report_present:
            warnings.append("Plan declares a reporting step but no report artifact was detected.")
        unfinished = [s.id for s in getattr(plan, "steps", []) if getattr(s, "status", "") not in _TERMINAL]
        if unfinished:
            warnings.append(f"{len(unfinished)} plan step(s) not marked complete: {', '.join(unfinished[:6])}")
        if not str(evidence_summary or "").strip():
            warnings.append("No execution evidence (tool calls / verification) recorded yet.")

        ctx.emit_event("Report", status="success" if (not needs_report or report_present) else "error",
                       message=("report present" if report_present else ("no report required" if not needs_report else "report missing")))

        verdict = self._goal_acceptance(ctx, objective, plan, evidence_summary)
        # Strict identity: only an explicit boolean true accepts (a string like
        # "false" is truthy and must NOT slip through as acceptance).
        goal_accepted = verdict.get("accepted") is True
        accepted = goal_accepted and not (needs_report and not report_present)
        ctx.emit_event("Review hooks", status="success" if accepted else "error",
                       message="goal accepted" if accepted else "goal not yet accepted")
        return CompletionAssessment(
            accepted=accepted,
            reasons=str(verdict.get("reasons") or ""),
            missing=[str(x) for x in (verdict.get("missing") or [])],
            warnings=warnings,
            needs_report=needs_report,
            report_present=report_present,
            goal_checked=verdict.get("checked", False),
        )

    def propose_repair(
        self,
        objective: str,
        plan: Any,
        assessment: CompletionAssessment,
        *,
        emit: Callable[..., None] | None = None,
        session_id: str = "completion",
        turn_id: str = "completion",
    ) -> str:
        ctx = CouncilContext(provider_router=self.provider_router, session_id=session_id, turn_id=turn_id, emit=emit, timeout_s=self.timeout_s, clock=self._clock)
        gaps = "; ".join(assessment.missing) or assessment.reasons or "the objective is not yet fully satisfied"
        prompt = (
            f"Objective: {objective}\n"
            f"Outstanding gaps: {gaps}\n"
            f"Warnings: {'; '.join(assessment.warnings) or 'none'}\n\n"
            "Propose the smallest set of concrete next actions (tools to run, checks "
            "to perform) that would close these gaps. Be specific and ordered. Plain text."
        )
        results = run_parallel(
            ctx,
            [CouncilJob(role=ProviderRole.PLANNER, messages=[
                {"role": "system", "content": "You are a meticulous repair planner."},
                {"role": "user", "content": prompt},
            ], label="repair-proposal", stage="Repair")],
        )
        if results and results[0].ok:
            return results[0].text.strip()
        return ""

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _plan_requires_report(plan: Any) -> bool:
        for step in getattr(plan, "steps", []):
            text = f"{getattr(step, 'id', '')} {getattr(step, 'title', '')}".lower()
            if "report" in text or any("report" in str(t).lower() for t in getattr(step, "tool_scope", [])):
                return True
        if any("report" in str(t).lower() for t in getattr(plan, "proposed_tool_scope", [])):
            return True
        return any("report" in str(v).lower() for v in getattr(plan, "verification_plan", []))

    def _goal_acceptance(self, ctx: CouncilContext, objective: str, plan: Any, evidence_summary: str) -> dict[str, Any]:
        steps = "; ".join(f"{getattr(s, 'id', '')}:{getattr(s, 'title', '')}[{getattr(s, 'status', '')}]" for s in getattr(plan, "steps", []))
        prompt = (
            f"Objective: {objective}\n\n"
            f"Plan steps (id:title[status]): {steps}\n\n"
            f"Evidence of work performed:\n{evidence_summary[:4000] or '(none recorded)'}\n\n"
            "Decide whether the objective has actually been achieved based ONLY on the "
            "evidence. Do not assume work that is not shown. Return ONLY JSON: "
            '{"accepted": true|false, "missing": ["..."], "reasons": "short justification"}'
        )
        results = run_parallel(
            ctx,
            [CouncilJob(role=ProviderRole.CRITIC, messages=[
                {"role": "system", "content": "You are a strict completion auditor. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ], label="goal-acceptance", stage="Review hooks")],
        )
        if results and results[0].ok:
            try:
                data = extract_json(results[0].text)
                if isinstance(data, dict):
                    return {
                        "accepted": data.get("accepted") is True,
                        "missing": [str(x) for x in (data.get("missing") or [])],
                        "reasons": str(data.get("reasons") or ""),
                        "checked": True,
                    }
            except Exception:
                logger.debug("goal-acceptance parse failed", exc_info=True)
        # If the judge is unavailable, do NOT auto-accept; report unknown.
        return {"accepted": False, "missing": ["goal-acceptance check unavailable"], "reasons": "completion judge unavailable", "checked": False}
