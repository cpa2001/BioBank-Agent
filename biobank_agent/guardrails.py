"""Delegation guardrails — anti-pattern enforcement for multi-agent safety.

Ported from heathcliff233/my_codex skills/multi-agent-delegation/SKILL.md:
  - No re-delegation: leaf agents never spawn sub-agents
  - No history inheritance: subagent messages are self-contained
  - No write-scope overlap: parallel workers never touch the same files
  - No vague specs: worker specs must reference concrete file paths
  - Trust but verify: orchestrator re-checks worker output

These guardrails run BEFORE subagent dispatch, failing fast on violations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GuardrailViolation:
    """A violated guardrail with context."""
    rule: str
    description: str
    severity: str = "BLOCK"  # BLOCK (prevent dispatch) or WARN (log and continue)


class DelegationGuardrails:
    """Enforce safe multi-agent delegation patterns.

    Usage::

        g = DelegationGuardrails()
        violations = g.check_all(role, messages, workers)
        if any(v.severity == "BLOCK" for v in violations):
            raise GuardrailError(violations)
    """

    def check_all(
        self,
        role: str,
        messages: list[dict] | None = None,
        spec: str = "",
        workers: list[dict] | None = None,
    ) -> list[GuardrailViolation]:
        """Run all applicable guardrails and return violations."""
        violations = []
        if messages:
            violations.extend(self.check_no_history_inheritance(messages))
        if spec:
            violations.extend(self.check_no_vague_spec(spec))
        if workers:
            violations.extend(self.check_no_write_overlap(workers))
        return violations

    def check_no_history_inheritance(
        self,
        messages: list[dict],
    ) -> list[GuardrailViolation]:
        """Subagent messages must not contain parent conversation history.

        From SKILL.md: "Subagents start with NO parent conversation history.
        Every child message must be self-contained."
        """
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        if user_turns > 1:
            return [GuardrailViolation(
                rule="no_history_inheritance",
                description=(
                    f"Subagent message list contains {user_turns} user turns. "
                    "Expected at most 1 (self-contained). "
                    "Parent should bake all context into a single message."
                ),
                severity="WARN",
            )]
        return []

    def check_no_vague_spec(self, spec: str) -> list[GuardrailViolation]:
        """Worker specs must reference concrete file paths.

        From SKILL.md: "Never delegate understanding — if don't know exact
        files/patterns/changes, keep exploring + synthesize first."
        """
        vague_signals = [
            "figure out", "look into", "research how",
            "implement something", "find a way", "explore options",
            "try to", "maybe", "probably",
        ]
        violations = []
        spec_lower = spec.lower()
        for signal in vague_signals:
            if signal in spec_lower:
                violations.append(GuardrailViolation(
                    rule="no_vague_spec",
                    description=(
                        f"Worker spec contains vague language: '{signal}'. "
                        "Spec must be decision-complete with exact file paths "
                        "and concrete changes."
                    ),
                    severity="WARN",
                ))
                break  # One warning is enough

        # Check for concrete file references
        has_file_ref = bool(re.search(r"[\w/]+\.\w{1,4}", spec))  # e.g., path/to/file.py
        if not has_file_ref and len(spec) > 50:
            violations.append(GuardrailViolation(
                rule="no_vague_spec",
                description=(
                    "Worker spec does not reference any file paths. "
                    "Add exact paths (e.g., biobank_agent/skills/foo.py) "
                    "for decision-complete specification."
                ),
                severity="WARN",
            ))

        return violations

    def check_no_write_overlap(
        self,
        workers: list[dict],
    ) -> list[GuardrailViolation]:
        """Parallel workers must not have overlapping write scopes.

        From SKILL.md: "Do NOT assign overlapping write ownership.
        Give each worker concrete spec with owned files, off-limits files."
        """
        violations = []
        for i, w1 in enumerate(workers):
            owned1 = set(w1.get("owned_files", []))
            for j, w2 in enumerate(workers[i + 1:], i + 1):
                owned2 = set(w2.get("owned_files", []))
                overlap = owned1 & owned2
                if overlap:
                    violations.append(GuardrailViolation(
                        rule="no_write_overlap",
                        description=(
                            f"Workers {i} and {j} have overlapping write scope: "
                            f"{sorted(overlap)}. Each file must be owned by "
                            "exactly one worker."
                        ),
                        severity="BLOCK",
                    ))
        return violations

    def check_decision_complete(self, plan: str) -> list[GuardrailViolation]:
        """Check if a plan spec is decision-complete.

        From AGENTS.md: "A plan is decision-complete only when another engineer
        could execute it in order without making new design decisions."

        Required elements:
        - Exact file names (path references)
        - Ordered implementation steps (numbered)
        - First verification command
        """
        violations = []

        # Check for file references
        file_refs = re.findall(r"[\w/]+\.\w{1,4}", plan)
        if not file_refs:
            violations.append(GuardrailViolation(
                rule="decision_complete",
                description="Plan has no file path references. Must specify exact files to modify.",
                severity="WARN",
            ))

        # Check for numbered steps
        has_numbered = bool(re.search(r"^\s*\d+[\.\)]\s", plan, re.MULTILINE))
        if not has_numbered:
            violations.append(GuardrailViolation(
                rule="decision_complete",
                description="Plan has no numbered implementation steps. Use ordered steps.",
                severity="WARN",
            ))

        # Check for verification command
        has_verify = any(kw in plan.lower() for kw in
                        ("verify", "test", "pytest", "check", "assert", "validate"))
        if not has_verify:
            violations.append(GuardrailViolation(
                rule="decision_complete",
                description="Plan has no verification step. Add a first check command.",
                severity="WARN",
            ))

        return violations


# ── Temporal Safety Rules ─────────────────────────────────────────────
# Lightweight temporal precedence constraints (inspired by LTL safety specs).
# NOT full linear temporal logic — just "skill A must execute before skill B"
# in any valid plan. Checked during LongHorizonPlan construction.


@dataclass
class TemporalRule:
    """A temporal precedence rule: `before` must execute before `after`.

    These encode domain knowledge about safe analysis ordering:
      - Always check prevalence before training a disease model
      - Always run safety_check before generating final reports
      - Always define cohort before running survival analysis
    """
    name: str
    before: str       # skill that must run first
    after: str        # skill that must run after
    severity: str = "WARN"  # "BLOCK" or "WARN"
    reason: str = ""


# Default temporal rules for biobank research safety
TEMPORAL_RULES: list[TemporalRule] = [
    TemporalRule(
        name="prevalence_before_training",
        before="prevalence",
        after="train_model",
        severity="WARN",
        reason="Must verify disease prevalence (n_cases ≥ 100) before training predictive models",
    ),
    TemporalRule(
        name="safety_before_report",
        before="safety_check",
        after="report",
        severity="WARN",
        reason="Safety review required before generating publication-ready reports",
    ),
    TemporalRule(
        name="cohort_before_survival",
        before="cohort_summary",
        after="survival",
        severity="WARN",
        reason="Must define and validate cohort before running survival analysis",
    ),
    TemporalRule(
        name="cohort_before_training",
        before="cohort_summary",
        after="train_model",
        severity="WARN",
        reason="Must define cohort (verify sample sizes) before training models",
    ),
    TemporalRule(
        name="prevalence_before_gwas",
        before="prevalence",
        after="gwas_proxy",
        severity="WARN",
        reason="Must verify case count before running GWAS (need n_cases ≥ 500 for power)",
    ),
    TemporalRule(
        name="missing_data_before_training",
        before="missing_data",
        after="train_model",
        severity="WARN",
        reason="Should assess missing data patterns before model training",
    ),
]


class TemporalSafetyChecker:
    """Check temporal precedence rules against a plan's skill ordering.

    Usage::

        checker = TemporalSafetyChecker()
        violations = checker.check_plan(plan_steps)
        for v in violations:
            if v.severity == "BLOCK":
                raise TemporalViolation(v)
    """

    def __init__(self, rules: list[TemporalRule] | None = None) -> None:
        self.rules = rules if rules is not None else TEMPORAL_RULES

    def check_plan(self, plan_steps: list[dict]) -> list[GuardrailViolation]:
        """Check a plan's step ordering against temporal rules.

        Parameters
        ----------
        plan_steps : list of dicts with at least {"skill": str, "id": str}
            The ordered list of steps in a plan (topological order).

        Returns
        -------
        List of violations (empty if plan is safe).
        """
        violations: list[GuardrailViolation] = []

        # Build skill → position mapping (first occurrence)
        skill_positions: dict[str, int] = {}
        for i, step in enumerate(plan_steps):
            skill_name = step.get("skill", "")
            if skill_name and skill_name not in skill_positions:
                skill_positions[skill_name] = i

        # Check each rule
        for rule in self.rules:
            if rule.after in skill_positions:
                # The "after" skill is in the plan — check if "before" precedes it
                after_pos = skill_positions[rule.after]
                if rule.before not in skill_positions:
                    # "before" skill is missing entirely
                    violations.append(GuardrailViolation(
                        rule=f"temporal:{rule.name}",
                        description=(
                            f"Temporal rule violated: '{rule.before}' must execute before "
                            f"'{rule.after}', but '{rule.before}' is not in the plan. "
                            f"Reason: {rule.reason}"
                        ),
                        severity=rule.severity,
                    ))
                elif skill_positions[rule.before] > after_pos:
                    # "before" skill comes AFTER "after" skill — wrong order
                    violations.append(GuardrailViolation(
                        rule=f"temporal:{rule.name}",
                        description=(
                            f"Temporal rule violated: '{rule.before}' (step {skill_positions[rule.before]}) "
                            f"must execute BEFORE '{rule.after}' (step {after_pos}), "
                            f"but is scheduled after. Reason: {rule.reason}"
                        ),
                        severity=rule.severity,
                    ))

        return violations

    def suggest_reorder(self, plan_steps: list[dict]) -> list[dict]:
        """Suggest a corrected ordering that satisfies all temporal rules.

        Returns the reordered steps (best-effort; may not resolve all conflicts).
        """
        # Build dependency edges from temporal rules
        deps: dict[str, set[str]] = {}  # skill → set of skills that must come before it
        for rule in self.rules:
            if rule.after not in deps:
                deps[rule.after] = set()
            deps[rule.after].add(rule.before)

        # Simple topological adjustment: move "before" skills earlier
        result = list(plan_steps)
        for rule in self.rules:
            before_idx = None
            after_idx = None
            for i, step in enumerate(result):
                if step.get("skill") == rule.before and before_idx is None:
                    before_idx = i
                if step.get("skill") == rule.after and after_idx is None:
                    after_idx = i

            if before_idx is not None and after_idx is not None and before_idx > after_idx:
                # Move "before" to just before "after"
                step = result.pop(before_idx)
                result.insert(after_idx, step)

        return result
