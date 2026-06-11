"""Reflexion engine — 4 action-class repair planner.

The legacy ``biobank_agent/reflexion.py`` only proposes parameter
corrections. v3 expands the action space to four mutually exclusive
classes so a single failure can drive any of:

    1. RetryWithCorrectedArgs   - patch the arguments and retry
    2. SwapSkill                - call a different tool (e.g.
                                  ``train_model`` → ``train_model_lite``)
    3. InsertPrerequisiteStep   - inject a missing setup step (e.g.
                                  ``build_cohort`` before ``train_model``)
    4. SkipStep                 - mark optional/diagnostic step skipped

PlanExecutor's existing repair loop accepts these in its
``PlanRepairAction`` adapter (``plan_executor.py:107``); this module
is the *generator* that proposes them, not the consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# These imports are intentional aliases so callers can do
# ``from biobank_agent.core.evolution.reflexion import ReflexionAction``
# without touching legacy reflexion modules.


class ActionClass(str, Enum):
    RETRY_WITH_CORRECTED_ARGS = "retry_with_corrected_args"
    SWAP_SKILL = "swap_skill"
    INSERT_PREREQUISITE_STEP = "insert_prerequisite_step"
    SKIP_STEP = "skip_step"


@dataclass
class ReflexionAction:
    cls: ActionClass
    rationale: str = ""
    confidence: float = 0.5
    corrected_args: dict[str, Any] = field(default_factory=dict)
    swap_to_skill: Optional[str] = None
    insert_skill: Optional[str] = None
    insert_args: dict[str, Any] = field(default_factory=dict)
    target_step_id: Optional[str] = None

    def to_plan_repair_payload(self) -> dict[str, Any]:
        """Translate to ``plan_executor.PlanRepairAction`` shape."""
        if self.cls == ActionClass.RETRY_WITH_CORRECTED_ARGS:
            return {
                "action": "retry_args",
                "reason": self.rationale,
                "args": dict(self.corrected_args),
            }
        if self.cls == ActionClass.SWAP_SKILL:
            return {
                "action": "create_custom_skill" if self.swap_to_skill is None else "retry_args",
                "reason": self.rationale,
                "replacement_step": {
                    "skill": self.swap_to_skill,
                    "args": dict(self.corrected_args),
                },
            }
        if self.cls == ActionClass.INSERT_PREREQUISITE_STEP:
            return {
                "action": "insert_prerequisite_steps",
                "reason": self.rationale,
                "steps": [{
                    "skill": self.insert_skill or "think",
                    "args": dict(self.insert_args),
                    "description": f"Repair: {self.rationale[:80]}",
                }],
            }
        if self.cls == ActionClass.SKIP_STEP:
            return {
                "action": "pause_with_trace",
                "reason": f"skip: {self.rationale}",
            }
        return {"action": "noop"}


# ── Pure-heuristic generator (LLM optional) ────────────────


_KEYWORD_TO_PREREQUISITE = {
    "cohort": "build_cohort",
    "no biomarkers": "field_search",
    "missing field": "field_search",
    "schema": "field_search",
    "label": "build_cohort",
    "train_model": "build_cohort",
}

_LITE_FALLBACK = {
    "train_model": "train_model_lite",
    "phewas": "phewas_lite",
}


def propose_action(
    *,
    skill: str,
    args: dict[str, Any],
    error: str,
    step_id: Optional[str] = None,
    history: Optional[list[dict[str, Any]]] = None,
) -> ReflexionAction:
    """Heuristic ReflexionAction proposer.

    Strategy (ordered):
    1. If error references a missing prerequisite (cohort / fields),
       propose ``InsertPrerequisiteStep``.
    2. If error mentions "n_cases" too small or "OOM", swap to a
       ``_lite`` variant of the skill if known.
    3. If error mentions an arg name verbatim, propose
       ``RetryWithCorrectedArgs`` with that arg dropped or halved.
    4. Otherwise, ``SkipStep`` if the step is non-required.
    """
    err = (error or "").lower()
    history = history or []
    repeated_failures = sum(
        1 for h in history if h.get("skill") == skill and h.get("error")
    )

    # Tier 1 — InsertPrerequisiteStep
    for needle, prereq in _KEYWORD_TO_PREREQUISITE.items():
        if needle in err and skill != prereq:
            return ReflexionAction(
                cls=ActionClass.INSERT_PREREQUISITE_STEP,
                rationale=f"error mentions {needle!r}; inserting {prereq} before retry",
                confidence=0.7,
                insert_skill=prereq,
                insert_args={},
                target_step_id=step_id,
            )

    # Tier 2 — SwapSkill (resource issues / repeated failures)
    if any(token in err for token in ("oom", "memory", "n_cases too small", "minimum")) or repeated_failures >= 2:
        swap = _LITE_FALLBACK.get(skill)
        if swap:
            return ReflexionAction(
                cls=ActionClass.SWAP_SKILL,
                rationale=f"swap to lighter variant {swap} due to repeated/limit failure",
                confidence=0.6,
                swap_to_skill=swap,
                corrected_args=dict(args or {}),
                target_step_id=step_id,
            )

    # Tier 3 — RetryWithCorrectedArgs
    corrected = dict(args or {})
    halved = False
    for key in ("n_folds", "top_n", "max_iter"):
        if key in corrected and isinstance(corrected[key], int):
            new = max(2, corrected[key] // 2)
            if new < corrected[key]:
                corrected[key] = new
                halved = True
    if halved:
        return ReflexionAction(
            cls=ActionClass.RETRY_WITH_CORRECTED_ARGS,
            rationale="halve heuristic-bounded numeric arguments and retry",
            confidence=0.55,
            corrected_args=corrected,
            target_step_id=step_id,
        )

    # Tier 4 — SkipStep
    return ReflexionAction(
        cls=ActionClass.SKIP_STEP,
        rationale="no usable repair found; recommend skipping",
        confidence=0.3,
        target_step_id=step_id,
    )


__all__ = ["ActionClass", "ReflexionAction", "propose_action"]
