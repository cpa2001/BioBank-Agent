"""Test PlanMode state machine and plan file management."""

import pytest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime


class TestPlanModeImport:
    """Test PlanMode class import and construction."""

    def test_planmode_imports(self):
        """PlanMode can be imported."""
        from biobank_agent.planner import PlanMode
        assert PlanMode is not None

    def test_initial_state_is_inactive(self, tmp_path):
        """Freshly constructed PlanMode starts INACTIVE."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path / "plans")

        assert pm.status == "INACTIVE"
        assert not pm.is_active
        assert pm.current_plan is None

    def test_creates_plans_directory(self, tmp_path):
        """Constructor creates the plans directory if missing."""
        from biobank_agent.planner import PlanMode

        plans_dir = tmp_path / "new_plans_dir"
        assert not plans_dir.exists()

        PlanMode(plans_dir=plans_dir)

        assert plans_dir.exists()


class TestPlanModeEnter:
    """Test entering plan mode."""

    def test_enter_creates_plan_file(self, tmp_path):
        """enter() creates a markdown plan file and sets status to INTAKE."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        msg = pm.enter("Analyze diabetes biomarkers")

        assert pm.status == "INTAKE"
        assert pm.is_active
        assert pm.current_plan is not None
        assert pm.current_plan.exists()
        assert pm.current_plan.suffix == ".md"
        assert "INTAKE" in msg

    def test_plan_file_contains_skeleton(self, tmp_path):
        """Plan file has required sections: Goal, Scope, Checklist, etc."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.enter("Build E11 cohort")

        content = pm.current_plan.read_text()

        assert "# Plan:" in content
        assert "## Goal" in content
        assert "## Scope" in content
        assert "## Open Questions" in content
        assert "## Execution Checklist" in content
        assert "## Progress Log" in content
        assert "## Definition of Done" in content
        assert "- Status: INTAKE" in content

    def test_enter_when_already_active_returns_error(self, tmp_path):
        """Entering plan mode twice -> error message."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.enter("First task")
        msg = pm.enter("Second task")

        assert "already active" in msg.lower()
        assert pm.status == "INTAKE"  # unchanged

    def test_plan_id_derived_from_description(self, tmp_path):
        """Plan filename is slug-ified from task description."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.enter("Analyze diabetes biomarkers")

        assert "analyze-diabetes" in pm.current_plan.name.lower()


class TestPlanModeTransitions:
    """Test status transitions and gate validation."""

    def test_intake_to_alignment(self, tmp_path):
        """INTAKE -> ALIGNMENT is valid."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.enter("task")
        msg = pm.set_status("ALIGNMENT")

        assert pm.status == "ALIGNMENT"
        assert "INTAKE" in msg
        assert "ALIGNMENT" in msg

    def test_alignment_to_execution(self, tmp_path):
        """ALIGNMENT -> EXECUTION is valid."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.enter("task")
        pm.set_status("ALIGNMENT")
        msg = pm.set_status("EXECUTION")

        assert pm.status == "EXECUTION"

    def test_execution_to_done(self, tmp_path):
        """EXECUTION -> DONE via complete() is valid."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.enter("task")
        pm.set_status("ALIGNMENT")
        pm.set_status("EXECUTION")
        msg = pm.complete()

        assert pm.status == "DONE"
        assert "completed successfully" in msg.lower()

    def test_cannot_jump_to_execution_from_intake(self, tmp_path):
        """INTAKE -> EXECUTION directly is rejected."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.enter("task")
        msg = pm.set_status("EXECUTION")

        assert "Cannot execute" in msg
        assert pm.status == "INTAKE"  # unchanged

    def test_invalid_status_rejected(self, tmp_path):
        """Bogus status name -> error message."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.enter("task")
        msg = pm.set_status("FLYING")

        assert "Invalid status" in msg
        assert pm.status == "INTAKE"

    def test_complete_from_non_execution_rejected(self, tmp_path):
        """complete() from ALIGNMENT -> rejection."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.enter("task")
        pm.set_status("ALIGNMENT")
        msg = pm.complete()

        assert "Cannot complete" in msg
        assert pm.status == "ALIGNMENT"

    def test_status_change_updates_plan_file(self, tmp_path):
        """Status transition writes new status into the plan file."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.enter("task")
        pm.set_status("ALIGNMENT")

        content = pm.current_plan.read_text()

        assert "- Status: ALIGNMENT" in content
        assert "INTAKE" in content  # in the progress log


class TestPlanModeApprove:
    """Test approve() shortcut logic."""

    def test_approve_from_intake_moves_through_alignment(self, tmp_path):
        """approve() from INTAKE first transitions to ALIGNMENT, then to EXECUTION."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.enter("task")

        # Remove open questions to allow approval
        content = pm.current_plan.read_text()
        content = content.replace("1. [ ] Clarify requirements",
                                  "1. [x] Clarify requirements")
        pm.current_plan.write_text(content)

        msg = pm.approve()

        assert pm.status == "EXECUTION"

    def test_approve_from_done_rejected(self, tmp_path):
        """approve() when status is DONE -> rejection."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.enter("task")
        pm.set_status("ALIGNMENT")
        pm.set_status("EXECUTION")
        pm.set_status("DONE")

        msg = pm.approve()

        assert "Cannot approve" in msg


class TestPlanModeExit:
    """Test exiting plan mode."""

    def test_exit_from_active_state(self, tmp_path):
        """exit() from any active state -> INACTIVE."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.enter("task")
        pm.set_status("ALIGNMENT")

        msg = pm.exit()

        assert pm.status == "INACTIVE"
        assert not pm.is_active
        assert pm.current_plan is None
        assert "ALIGNMENT" in msg

    def test_exit_when_not_active(self, tmp_path):
        """exit() when not in plan mode -> informational message."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        msg = pm.exit()

        assert "Not in plan mode" in msg


class TestPlanModeListPlans:
    """Test listing saved plans."""

    def test_list_plans_empty_directory(self, tmp_path):
        """No plan files -> empty list."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        plans = pm.list_plans()

        assert plans == []

    def test_list_plans_returns_correct_data(self, tmp_path):
        """Multiple plans -> list with file, status, path."""
        from biobank_agent.planner import PlanMode

        # Create two plan files manually
        (tmp_path / "plan-a.md").write_text(
            "# Plan A\n- Status: DONE\n"
        )
        (tmp_path / "plan-b.md").write_text(
            "# Plan B\n- Status: INTAKE\n"
        )

        pm = PlanMode(plans_dir=tmp_path)
        plans = pm.list_plans()

        assert len(plans) == 2
        names = {p["file"] for p in plans}
        assert "plan-a.md" in names
        assert "plan-b.md" in names

        statuses = {p["file"]: p["status"] for p in plans}
        assert statuses["plan-a.md"] == "DONE"
        assert statuses["plan-b.md"] == "INTAKE"

    def test_list_plans_has_path_field(self, tmp_path):
        """Each plan entry includes its full path."""
        from biobank_agent.planner import PlanMode

        (tmp_path / "test-plan.md").write_text("# Test\n- Status: EXECUTION\n")

        pm = PlanMode(plans_dir=tmp_path)
        plans = pm.list_plans()

        assert len(plans) == 1
        assert str(tmp_path) in plans[0]["path"]


class TestPlanModeUpdatePlan:
    """Test plan file update mechanism."""

    def test_update_plan_writes_content(self, tmp_path):
        """update_plan() writes new content to plan file."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.enter("task")

        new_content = pm.get_plan_content().replace(
            "## Goal\ntask",
            "## Goal\nRevised goal: build E11 model",
        )
        msg = pm.update_plan(new_content)

        assert "updated" in msg.lower()
        assert "Revised goal" in pm.get_plan_content()

    def test_get_plan_content_no_active_plan(self, tmp_path):
        """get_plan_content() with no active plan -> fallback string."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        content = pm.get_plan_content()

        assert "no active plan" in content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
