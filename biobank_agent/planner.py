"""Plan mode engine — structured planning workflow with stage gates.

Follows the INTAKE → ALIGNMENT → EXECUTION → DONE state machine.
Each plan is stored as a markdown file in the plans/ directory.

Enhanced with long-horizon planning (dependency graph) and 4-phase
pipeline discipline from heathcliff233/my_codex.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMClient

logger = logging.getLogger(__name__)


# ── Long-horizon plan structures ──────────────────────────────


@dataclass
class PlanStep:
    """A single step in a long-horizon plan with dependencies."""
    id: str
    skill: str
    args: dict = field(default_factory=dict)
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    can_parallelize: bool = False
    status: str = "pending"   # pending | running | done | failed | skipped
    result: dict = field(default_factory=dict)
    error: str = ""

    @property
    def is_done(self) -> bool:
        return self.status in ("done", "skipped")

    @property
    def is_blocked(self) -> bool:
        return self.status == "pending" and bool(self.depends_on)


@dataclass
class LongHorizonPlan:
    """A dependency-aware multi-step analysis plan.

    Steps form a DAG where each step lists its dependencies.
    The plan executor runs steps in topological order, parallelizing
    independent branches where possible.
    """
    goal: str
    steps: list[PlanStep] = field(default_factory=list)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def done_steps(self) -> int:
        return sum(1 for s in self.steps if s.is_done)

    @property
    def failed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == "failed")

    def progress(self) -> float:
        """0.0 to 1.0 completion ratio."""
        return self.done_steps / self.total_steps if self.total_steps else 0.0

    def next_runnable(self) -> list[PlanStep]:
        """Return all steps whose dependencies are satisfied."""
        done_ids = {s.id for s in self.steps if s.is_done}
        return [
            s for s in self.steps
            if s.status == "pending"
            and set(s.depends_on).issubset(done_ids)
        ]

    def mark_done(self, step_id: str, result: dict) -> None:
        """Mark a step as completed with its result."""
        for s in self.steps:
            if s.id == step_id:
                s.status = "done"
                s.result = result
                return

    def mark_failed(self, step_id: str, error: str) -> None:
        """Mark a step as failed and skip dependents."""
        for s in self.steps:
            if s.id == step_id:
                s.status = "failed"
                s.error = error
                break
        # Skip all transitive dependents
        failed_ids = {step_id}
        changed = True
        while changed:
            changed = False
            for s in self.steps:
                if s.status == "pending" and set(s.depends_on) & failed_ids:
                    s.status = "skipped"
                    s.error = f"Skipped: dependency {step_id} failed"
                    failed_ids.add(s.id)
                    changed = True

    def summary(self) -> str:
        """One-line progress summary."""
        return (
            f"Plan: {self.done_steps}/{self.total_steps} done, "
            f"{self.failed_steps} failed, "
            f"{len(self.next_runnable())} ready"
        )

    def to_markdown(self) -> str:
        """Render plan as numbered markdown checklist."""
        lines = [f"# Plan: {self.goal}\n"]
        for i, s in enumerate(self.steps, 1):
            check = "x" if s.is_done else ("!" if s.status == "failed" else " ")
            deps = f" (after: {', '.join(s.depends_on)})" if s.depends_on else ""
            lines.append(f"{i}. [{check}] `{s.skill}({s.args})`{deps}")
            if s.description:
                lines.append(f"   {s.description}")
            if s.error:
                lines.append(f"   **Error:** {s.error}")
        lines.append(f"\n**Progress:** {self.progress():.0%}")
        return "\n".join(lines)


class LongHorizonPlanner:
    """Decompose complex goals into dependency-aware step graphs.

    Uses LLM to analyze a goal and produce a LongHorizonPlan with
    skill-level steps and dependency ordering.

    Usage::

        planner = LongHorizonPlanner(llm)
        plan = planner.decompose("Discover T2DM biomarkers", available_skills)
        while plan.next_runnable():
            for step in plan.next_runnable():
                result = registry.execute(step.skill, step.args, ctx)
                plan.mark_done(step.id, result)
    """

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm

    def decompose(
        self,
        goal: str,
        available_skills: list[str],
        context: str = "",
    ) -> LongHorizonPlan:
        """Use LLM to decompose a goal into a step graph.

        Falls back to a sensible default plan if LLM fails.
        """
        if not self.llm:
            return self._default_plan(goal)

        prompt = f"""Decompose this biobank research goal into concrete analysis steps.

**Goal:** {goal}
{f"**Context:** {context[:500]}" if context else ""}

**Available tools:** {', '.join(available_skills[:30])}

Output a JSON array where each step has:
- "id": unique step ID (e.g., "s1", "s2")
- "skill": tool name from the list above
- "args": dict of arguments
- "description": what this step does
- "depends_on": array of step IDs that must complete first
- "can_parallelize": true if this can run alongside other ready steps

Example:
[
  {{"id": "s1", "skill": "prevalence", "args": {{"top_n": 10}}, "description": "Check disease prevalence", "depends_on": [], "can_parallelize": false}},
  {{"id": "s2", "skill": "train_model", "args": {{"icd10_code": "E11"}}, "description": "Train predictor", "depends_on": ["s1"], "can_parallelize": false}}
]

Respond with ONLY the JSON array."""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "You are a biobank research planner. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
            )

            text = response.text.strip()
            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    clean = part.strip().removeprefix("json").strip()
                    if clean.startswith("["):
                        text = clean
                        break

            data = json.loads(text)
            if not isinstance(data, list):
                return self._default_plan(goal)

            steps = []
            for item in data:
                steps.append(PlanStep(
                    id=item.get("id", f"s{len(steps)+1}"),
                    skill=item.get("skill", "think"),
                    args=item.get("args", {}),
                    description=item.get("description", ""),
                    depends_on=item.get("depends_on", []),
                    can_parallelize=item.get("can_parallelize", False),
                ))

            return LongHorizonPlan(goal=goal, steps=steps)

        except Exception as e:
            logger.warning("LLM plan decomposition failed: %s. Using default.", e)
            return self._default_plan(goal)

    def _default_plan(self, goal: str) -> LongHorizonPlan:
        """Sensible default plan for common biobank analyses."""
        return LongHorizonPlan(
            goal=goal,
            steps=[
                PlanStep(id="s1", skill="think", args={"reasoning": f"Planning: {goal}"}, description="Analyze the goal"),
                PlanStep(id="s2", skill="prevalence", args={"top_n": 10}, description="Check disease prevalence", depends_on=["s1"]),
            ],
        )

PLAN_TEMPLATE = """\
# Plan: {title}

## Metadata
- Plan ID: {plan_id}
- Status: INTAKE
- Created: {created}
- Updated: {created}

## Goal
{goal}

## Scope
- In scope: {scope}
- Out of scope: TBD
- Constraints: TBD

## Open Questions
{questions}

## User Decisions
(none yet)

## Execution Checklist
{checklist}

## Progress Log
- {created} Plan created

## Definition of Done
- [ ] All checklist items completed
- [ ] Results verified
"""


class PlanMode:
    """File-first planning workflow with strict stage gates.

    States: INACTIVE → INTAKE → ALIGNMENT → EXECUTION → DONE
    """

    VALID_STATES = ("INACTIVE", "INTAKE", "ALIGNMENT", "EXECUTION", "BLOCKED", "DONE")

    def __init__(self, plans_dir: Path) -> None:
        self.plans_dir = plans_dir
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.current_plan: Optional[Path] = None
        self.status: str = "INACTIVE"

    @property
    def is_active(self) -> bool:
        return self.status not in ("INACTIVE", "DONE")

    def enter(self, task_description: str) -> str:
        """Start plan mode: create plan file, set status to INTAKE."""
        if self.is_active:
            return f"Plan mode already active (status: {self.status}). Use /plan-exit to leave first."

        now = datetime.now()
        slug = task_description[:40].lower().replace(" ", "-").replace("/", "-")
        slug = "".join(c for c in slug if c.isalnum() or c == "-")
        plan_id = now.strftime(f"%Y-%m-%d-%H%M-{slug}")
        filename = f"{plan_id}.md"
        self.current_plan = self.plans_dir / filename

        # Analyze task to create initial structure
        goal = task_description
        scope = task_description
        questions = "1. [ ] Clarify requirements\n"
        checklist = "- [ ] Step 1: Analyze requirements\n- [ ] Step 2: Execute\n- [ ] Step 3: Verify\n"

        content = PLAN_TEMPLATE.format(
            title=task_description,
            plan_id=plan_id,
            created=now.strftime("%Y-%m-%d %H:%M"),
            goal=goal,
            scope=scope,
            questions=questions,
            checklist=checklist,
        )
        self.current_plan.write_text(content)
        self.status = "INTAKE"

        logger.info("Plan mode entered: %s", self.current_plan)
        return (
            f"Plan mode activated. Plan file: `{self.current_plan}`\n"
            f"Status: **INTAKE** — gathering context and identifying unknowns.\n"
            f"Use natural language to refine the plan. Type `/plan-approve` when ready to execute."
        )

    def get_plan_content(self) -> str:
        """Read current plan file content."""
        if not self.current_plan or not self.current_plan.exists():
            return "(no active plan)"
        return self.current_plan.read_text()

    def update_plan(self, new_content: str) -> str:
        """Update the plan file with new content."""
        if not self.current_plan:
            return "No active plan."

        # Update the Updated timestamp
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = new_content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("- Updated:"):
                lines[i] = f"- Updated: {now}"
                break

        self.current_plan.write_text("\n".join(lines))
        return f"Plan updated at {now}."

    def set_status(self, new_status: str) -> str:
        """Transition plan status with gate validation."""
        if new_status not in self.VALID_STATES:
            return f"Invalid status: {new_status}. Valid: {', '.join(self.VALID_STATES)}"

        # Validate transitions
        if new_status == "EXECUTION" and self.status != "ALIGNMENT":
            return "Cannot execute — plan must be in ALIGNMENT status first. Use /plan-approve."

        old = self.status
        self.status = new_status

        # Update status in plan file
        if self.current_plan and self.current_plan.exists():
            content = self.current_plan.read_text()
            for old_status in self.VALID_STATES:
                content = content.replace(f"- Status: {old_status}", f"- Status: {new_status}", 1)
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            content = content.replace(
                "## Progress Log",
                f"## Progress Log\n- {now} Status: {old} → {new_status}",
                1,
            )
            self.current_plan.write_text(content)

        logger.info("Plan status: %s → %s", old, new_status)
        return f"Plan status changed: {old} → **{new_status}**"

    def _has_open_questions(self) -> bool:
        """Check whether unresolved open questions remain in the plan."""
        if not self.current_plan or not self.current_plan.exists():
            return False
        content = self.current_plan.read_text()
        marker = "## Open Questions"
        if marker not in content:
            return False
        after_marker = content.split(marker, 1)[1]
        # Extract only up to the next ## heading
        next_section = after_marker.split("##", 1)[0]
        return "[ ]" in next_section

    def approve(self) -> str:
        """Move from ALIGNMENT to EXECUTION after user approval."""
        if self.status == "INTAKE":
            self.set_status("ALIGNMENT")

        if self.status != "ALIGNMENT":
            return f"Cannot approve — status is {self.status}, expected ALIGNMENT."

        # Check for unresolved questions
        if self._has_open_questions():
            return "Cannot approve — there are unresolved open questions. Address them first."

        return self.set_status("EXECUTION")

    def complete(self) -> str:
        """Mark plan as DONE."""
        if self.status != "EXECUTION":
            return f"Cannot complete — status is {self.status}, expected EXECUTION."
        result = self.set_status("DONE")
        return result + "\nPlan completed successfully."

    def exit(self) -> str:
        """Exit plan mode (can be called from any state)."""
        if not self.is_active:
            return "Not in plan mode."

        old_status = self.status
        plan_path = self.current_plan
        self.status = "INACTIVE"
        self.current_plan = None
        return f"Exited plan mode (was: {old_status}). Plan saved at: {plan_path}"

    def list_plans(self) -> list[dict]:
        """List all plan files in the plans directory."""
        plans = []
        for f in sorted(self.plans_dir.glob("*.md"), reverse=True):
            content = f.read_text()
            status = "UNKNOWN"
            for line in content.split("\n"):
                if line.startswith("- Status:"):
                    status = line.split(":", 1)[1].strip()
                    break
            plans.append({"file": f.name, "status": status, "path": str(f)})
        return plans
