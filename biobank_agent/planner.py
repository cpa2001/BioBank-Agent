"""Plan mode engine — structured planning workflow with stage gates.

Follows the INTAKE → ALIGNMENT → EXECUTION → DONE state machine.
Each plan is stored as a markdown file in the plans/ directory.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

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

    def approve(self) -> str:
        """Move from ALIGNMENT to EXECUTION after user approval."""
        if self.status == "INTAKE":
            self.set_status("ALIGNMENT")

        if self.status != "ALIGNMENT":
            return f"Cannot approve — status is {self.status}, expected ALIGNMENT."

        # Check for unresolved questions
        if self.current_plan and self.current_plan.exists():
            content = self.current_plan.read_text()
            if "[ ]" in content.split("## Open Questions")[1].split("##")[0] if "## Open Questions" in content else "":
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
