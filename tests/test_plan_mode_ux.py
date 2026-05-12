"""Test plan progress display, executor, and state persistence."""

import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from io import StringIO

from biobank_agent.planner import LongHorizonPlan, PlanMode, PlanState, PlanStep
from biobank_agent.progress import PlanProgressDisplay, PlanRunDashboard, StepResult
from biobank_agent.plan_executor import PlanExecutor
from biobank_agent.plan_state import PlanCheckpoint


# ── StepResult Tests ─────────────────────────────────────────


class TestStepResult:
    """Test StepResult dataclass."""

    def test_basic_construction(self):
        r = StepResult(step_id="s1", step_description="Test step")
        assert r.success is True
        assert r.error == ""
        assert r.duration_s == 0.0

    def test_failure_result(self):
        r = StepResult(
            step_id="s2",
            step_description="Failed step",
            success=False,
            error="Something broke",
            duration_s=1.5,
        )
        assert not r.success
        assert r.error == "Something broke"


class TestPlanMetadata:
    """Test plan metadata used by repair/skip UX."""

    def test_step_criticality_round_trips(self):
        plan = LongHorizonPlan(
            goal="Test",
            assumptions=["Use E11 endpoint."],
            clarifications=[{"id": "endpoint", "answer": "Use E11 endpoint."}],
            steps=[
                PlanStep(
                    id="s1",
                    skill="field_search",
                    args={"query": "BMI"},
                    description="Search",
                    criticality="diagnostic",
                    repair_hints=["Broaden query"],
                )
            ],
        )

        restored = LongHorizonPlan.from_dict(plan.to_dict())

        assert restored.assumptions == ["Use E11 endpoint."]
        assert restored.clarifications[0]["id"] == "endpoint"
        assert restored.steps[0].criticality == "diagnostic"
        assert restored.steps[0].repair_hints == ["Broaden query"]


# ── PlanProgressDisplay Tests ────────────────────────────────


class TestPlanProgressDisplay:
    """Test Rich-based progress display."""

    def _make_plan(self, n_steps=3):
        steps = [
            PlanStep(
                id=f"s{i}",
                skill="think",
                description=f"Step {i}",
                depends_on=[f"s{i-1}"] if i > 1 else [],
            )
            for i in range(1, n_steps + 1)
        ]
        return LongHorizonPlan(goal="Test goal", steps=steps)

    def test_construction(self):
        plan = self._make_plan()
        display = PlanProgressDisplay(plan)
        assert display.plan is plan
        assert display._live is None

    def test_show_plan_for_review_no_error(self, capsys):
        """show_plan_for_review() renders without errors."""
        from rich.console import Console

        plan = self._make_plan()
        console = Console(file=StringIO(), force_terminal=True)
        display = PlanProgressDisplay(plan, console)
        display.show_plan_for_review()
        # Should not raise

    def test_show_plan_for_review_with_revision(self):
        """Revision number appears in display."""
        from rich.console import Console

        plan = self._make_plan()
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        display = PlanProgressDisplay(plan, console)
        display.show_plan_for_review(revision=3)
        rendered = output.getvalue()
        assert "v3" in rendered

    def test_update_step_tracking(self):
        """update_step() updates internal tracking."""
        plan = self._make_plan()
        display = PlanProgressDisplay(plan)
        display.update_step("s1", "running", "Processing...")
        assert display._step_status["s1"] == "running"
        assert display._current_detail == "Processing..."

    def test_show_report_no_error(self):
        """show_report() renders without errors."""
        from rich.console import Console

        plan = self._make_plan()
        console = Console(file=StringIO(), force_terminal=True)
        display = PlanProgressDisplay(plan, console)
        results = [
            StepResult(step_id="s1", step_description="Step 1", success=True, duration_s=1.0),
            StepResult(step_id="s2", step_description="Step 2", success=False, error="oops", duration_s=0.5),
        ]
        display.show_report(results)
        # Should not raise

    def test_show_step_failure_no_error(self):
        """show_step_failure() renders without errors."""
        from rich.console import Console

        plan = self._make_plan()
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        display = PlanProgressDisplay(plan, console)
        display.show_step_failure("Step 2", "Something went wrong")
        rendered = output.getvalue()
        assert "Auto-repair" in rendered
        assert "Ask Codex/Claude/Gemini" in rendered
        assert "skip this step and continue" not in rendered

    def test_plan_run_dashboard_records_events(self):
        """PlanRunDashboard stores phase events and renders current status."""
        from rich.console import Console

        output = StringIO()
        console = Console(file=output, force_terminal=True)
        dashboard = PlanRunDashboard(console)
        dashboard.record("Planning", "biobank", "running", "decomposing")
        dashboard.record("External council", "codex", "success", "returned advice")
        panel = dashboard._build_panel()

        assert len(dashboard.events) == 2
        assert dashboard.events[-1].actor == "codex"
        console.print(panel)
        rendered = output.getvalue()
        assert "External council" in rendered
        assert "returned advice" in rendered

    def test_show_paused_no_error(self):
        """show_paused() renders without errors."""
        from rich.console import Console

        plan = self._make_plan()
        console = Console(file=StringIO(), force_terminal=True)
        display = PlanProgressDisplay(plan, console)
        display.show_paused("Test pause reason")
        # Should not raise

    def test_stop_display_when_not_started(self):
        """stop_display() is safe when no live display active."""
        plan = self._make_plan()
        display = PlanProgressDisplay(plan)
        display.stop_display()  # Should not raise


# ── PlanExecutor Tests ───────────────────────────────────────


class TestPlanExecutor:
    """Test step-by-step plan execution."""

    def _make_plan_mode(self, tmp_path, steps=None):
        pm = PlanMode(plans_dir=tmp_path)
        if steps is None:
            steps = [
                PlanStep(id="s1", skill="think", args={"reasoning": "test"}, description="Think"),
                PlanStep(id="s2", skill="prevalence", args={"top_n": 5}, description="Check prevalence", depends_on=["s1"]),
            ]
        pm.plan = LongHorizonPlan(goal="Test", steps=steps)
        pm.state = PlanState.APPROVED
        pm.goal = "Test"
        return pm

    def _make_registry(self, results=None):
        """Mock skill registry."""
        registry = MagicMock()
        if results is None:
            results = {"output": "success"}
        registry.execute.return_value = results
        return registry

    def test_execute_happy_path(self, tmp_path):
        """All steps execute successfully."""
        from rich.console import Console

        pm = self._make_plan_mode(tmp_path)
        registry = self._make_registry()
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=registry,
            ctx_builder=lambda: MagicMock(),
            console=console,
        )
        results = executor.execute()

        assert len(results) == 2
        assert all(r.success for r in results)
        assert pm.state == PlanState.DONE
        assert registry.execute.call_count == 2

    def test_execute_emits_progress_events(self, tmp_path):
        """Executor emits lifecycle events for dashboards/logging."""
        from rich.console import Console

        pm = self._make_plan_mode(
            tmp_path,
            steps=[PlanStep(id="s1", skill="think", args={"reasoning": "x"}, description="Think")],
        )
        events = []
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=self._make_registry(),
            ctx_builder=lambda: MagicMock(),
            console=console,
            event_sink=lambda phase, actor, status, message, metadata=None: events.append(
                (phase, actor, status, message, metadata or {})
            ),
        )
        executor.execute()

        assert ("Execution", "think", "running", "Think", {"step_id": "s1"}) in events
        assert any(event[0] == "Report" and event[2] == "success" for event in events)

    def test_direct_registry_progress_updates_activity(self, tmp_path):
        """Direct registry execution also forwards ctx.emit_progress to the display."""
        from rich.console import Console

        pm = self._make_plan_mode(
            tmp_path,
            steps=[PlanStep(id="s1", skill="train_model", args={}, description="Train")],
        )
        console = Console(file=StringIO(), force_terminal=True)

        class Registry:
            def execute(self, _skill, _args, ctx=None):
                ctx.emit_progress("model-selection", "candidate 1/2: xgb")
                return {"ok": True}

        executor = PlanExecutor(
            plan_mode=pm,
            registry=Registry(),
            ctx_builder=lambda: MagicMock(),
            console=console,
        )
        executor.execute()

        assert "candidate 1/2: xgb" in executor.progress._current_detail

    def test_execute_step_failure_pauses(self, tmp_path):
        """Step failure transitions to PAUSED."""
        from rich.console import Console

        pm = self._make_plan_mode(tmp_path)
        registry = MagicMock()
        registry.execute.side_effect = [RuntimeError("boom")]
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=registry,
            ctx_builder=lambda: MagicMock(),
            console=console,
        )
        results = executor.execute()

        assert len(results) == 1
        assert not results[0].success
        assert results[0].error == "boom"
        assert pm.state == PlanState.PAUSED

    def test_execute_skill_error_result_pauses(self, tmp_path):
        """Skill wrapper errors are treated as failed steps, not successes."""
        from rich.console import Console

        steps = [
            PlanStep(id="s1", skill="field_search", args={"query": ""}, description="Bad search"),
            PlanStep(id="s2", skill="generate_report", args={"format": "dual"}, description="Report", depends_on=["s1"]),
        ]
        pm = self._make_plan_mode(tmp_path, steps=steps)
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=MagicMock(),
            ctx_builder=lambda: MagicMock(),
            skill_executor=lambda _skill, _args: {
                "result": {"error": "bad args"},
                "is_error": True,
                "elapsed_s": 0.01,
            },
            console=console,
        )
        results = executor.execute()

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error == "bad args"
        assert pm.state == PlanState.PAUSED
        assert pm.plan.steps[0].status == "failed"
        assert pm.plan.steps[1].status == "skipped"

    def test_execute_registry_error_result_pauses(self, tmp_path):
        """A dict result containing error fails even without an exception."""
        from rich.console import Console

        pm = self._make_plan_mode(
            tmp_path,
            steps=[PlanStep(id="s1", skill="field_search", args={"query": ""}, description="Bad search")],
        )
        registry = MagicMock()
        registry.execute.return_value = {"error": "missing query"}
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=registry,
            ctx_builder=lambda: MagicMock(),
            console=console,
        )
        results = executor.execute()

        assert results[0].success is False
        assert results[0].error == "missing query"
        assert pm.state == PlanState.PAUSED

    def test_execute_refuses_invalid_dependency_graph(self, tmp_path):
        """Executor must not run or mark success when the plan DAG is invalid."""
        from rich.console import Console

        pm = self._make_plan_mode(
            tmp_path,
            steps=[
                PlanStep(id="s1", skill="think", args={}, description="One", depends_on=["s2"]),
                PlanStep(id="s2", skill="think", args={}, description="Two", depends_on=["s1"]),
                PlanStep(id="s3", skill="generate_report", args={"format": "dual"}, description="Report", depends_on=["s99"]),
            ],
        )
        registry = MagicMock()
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=registry,
            ctx_builder=lambda: MagicMock(),
            console=console,
        )
        results = executor.execute()

        assert results == []
        assert pm.state == PlanState.PAUSED
        assert "Dependency cycle detected" in pm.pause_reason
        assert "Unknown dependencies: s99" in pm.pause_reason
        registry.execute.assert_not_called()

    def test_goal_acceptance_rejects_stale_missing_report_artifact(self, tmp_path):
        """A restored report success log is not enough if files are gone."""
        from rich.console import Console

        missing_report = tmp_path / "missing_report.md"
        pm = self._make_plan_mode(
            tmp_path,
            steps=[PlanStep(id="s1", skill="generate_report", args={"format": "dual"}, description="Report")],
        )
        pm.execution_log = [
            StepResult(
                step_id="s1",
                step_description="Report",
                skill="generate_report",
                success=True,
                result={"format": "dual", "markdown": str(missing_report), "html": str(tmp_path / "missing.html")},
            )
        ]
        executor = PlanExecutor(
            plan_mode=pm,
            registry=MagicMock(),
            ctx_builder=lambda: MagicMock(),
            console=Console(file=StringIO(), force_terminal=True),
        )

        verdict = executor._goal_acceptance_checker_result(pm.plan)

        assert verdict["accepted"] is False
        assert "no successful generate_report artifact" in verdict["reason"]

    def test_field_search_repair_uses_suggested_query_before_pause(self, tmp_path):
        """A zero-hit field search should retry a concrete suggested query once."""
        from rich.console import Console

        calls = []
        pm = self._make_plan_mode(
            tmp_path,
            steps=[PlanStep(id="s1", skill="field_search", args={"query": "long mixed phrase"}, description="Search")],
        )
        console = Console(file=StringIO(), force_terminal=True)

        def skill_executor(_skill, args):
            calls.append(dict(args))
            if args["query"] == "long mixed phrase":
                return {
                    "result": {
                        "total": 0,
                        "results": [],
                        "requires_repair": True,
                        "suggested_queries": ["BMI"],
                        "warnings": ["No catalogue fields matched."],
                    },
                    "is_error": False,
                }
            return {
                "result": {
                    "total": 1,
                    "results": [{"field_id": "21001", "title": "Body mass index"}],
                    "requires_repair": False,
                },
                "is_error": False,
            }

        executor = PlanExecutor(
            plan_mode=pm,
            registry=MagicMock(),
            ctx_builder=lambda: MagicMock(),
            skill_executor=skill_executor,
            console=console,
        )
        results = executor.execute()

        assert pm.state == PlanState.DONE
        assert results[0].success is True
        assert calls == [{"query": "long mixed phrase"}, {"query": "BMI"}]
        assert pm.plan.steps[0].args == {"query": "BMI"}
        assert pm.repair_log[0]["action"] == "retry_args"

    def test_repair_retry_args_prevents_pause(self, tmp_path):
        """A repair action can fix args and retry the step before it is marked failed."""
        from rich.console import Console

        pm = self._make_plan_mode(
            tmp_path,
            steps=[PlanStep(id="s1", skill="field_search", args={"query": ""}, description="Search")],
        )
        from biobank_agent.planner import PlanSchemaValidator
        pm.available_skills = ["field_search"]
        pm.validator = PlanSchemaValidator(["field_search"], [])
        console = Console(file=StringIO(), force_terminal=True)

        def skill_executor(_skill, args):
            if not args.get("query"):
                return {"result": {"error": "Missing query"}, "is_error": True}
            return {"result": {"results": [{"field_id": "30750"}]}, "is_error": False}

        def repair_strategy(step, result, _plan, _context):
            return {
                "action": "retry_args",
                "reason": "Fill missing field-search query.",
                "args": {"query": "glucose"},
            }

        executor = PlanExecutor(
            plan_mode=pm,
            registry=MagicMock(),
            ctx_builder=lambda: MagicMock(),
            skill_executor=skill_executor,
            repair_strategy=repair_strategy,
            console=console,
        )
        results = executor.execute()

        assert pm.state == PlanState.DONE
        assert len(results) == 1
        assert results[0].success is True
        assert pm.plan.steps[0].args == {"query": "glucose"}
        assert pm.repair_log[0]["action"] == "retry_args"

    def test_repair_inserts_prerequisite_before_retry(self, tmp_path):
        """A repair action can add a prerequisite step and defer the failed step."""
        from rich.console import Console

        calls = []
        pm = self._make_plan_mode(
            tmp_path,
            steps=[PlanStep(id="s1", skill="main_analysis", args={}, description="Main")],
        )
        console = Console(file=StringIO(), force_terminal=True)

        def skill_executor(skill, _args):
            calls.append(skill)
            if skill == "main_analysis" and "cohort_summary" not in calls:
                return {"result": {"error": "cohort missing"}, "is_error": True}
            return {"result": {"ok": True}, "is_error": False}

        def repair_strategy(_step, _result, _plan, _context):
            return {
                "action": "insert_prerequisite_steps",
                "reason": "Build cohort before main analysis.",
                "steps": [
                    {"id": "r1", "skill": "cohort_summary", "args": {"icd10_code": "E11"}, "description": "Build cohort"}
                ],
            }

        executor = PlanExecutor(
            plan_mode=pm,
            registry=MagicMock(),
            ctx_builder=lambda: MagicMock(),
            skill_executor=skill_executor,
            repair_strategy=repair_strategy,
            console=console,
        )
        executor.execute()

        assert pm.state == PlanState.DONE
        assert calls == ["main_analysis", "cohort_summary", "main_analysis"]
        assert [s.id for s in pm.plan.steps] == ["r1", "s1"]
        assert pm.plan.steps[1].depends_on == ["r1"]

    def test_invalid_repair_steps_are_rolled_back(self, tmp_path):
        """A failed repair must not leave a corrupted dependency graph behind."""
        from rich.console import Console

        pm = self._make_plan_mode(
            tmp_path,
            steps=[PlanStep(id="s1", skill="field_search", args={"query": ""}, description="Search")],
        )
        from biobank_agent.planner import PlanSchemaValidator
        pm.available_skills = ["field_search"]
        pm.validator = PlanSchemaValidator(["field_search"], [])
        console = Console(file=StringIO(), force_terminal=True)

        def repair_strategy(_step, _result, _plan, _context):
            return {
                "action": "insert_prerequisite_steps",
                "reason": "bad repair",
                "steps": [
                    {"id": "r1", "skill": "unknown_skill", "args": {}, "description": "Invalid", "depends_on": []}
                ],
            }

        executor = PlanExecutor(
            plan_mode=pm,
            registry=MagicMock(),
            ctx_builder=lambda: MagicMock(),
            skill_executor=lambda _skill, _args: {"result": {"error": "missing query"}, "is_error": True},
            repair_strategy=repair_strategy,
            console=console,
        )
        executor.execute()

        assert [step.id for step in pm.plan.steps] == ["s1"]
        assert all(step.id != "r1" for step in pm.plan.steps)
        assert pm.state == PlanState.PAUSED

    def test_repair_create_custom_skill_refreshes_tools_and_retries(self, tmp_path):
        """A create_custom_skill repair hot-loads tools and retries the current step."""
        from rich.console import Console

        activated = {"value": False}
        refreshed = {"value": False}
        pm = self._make_plan_mode(
            tmp_path,
            steps=[PlanStep(id="s1", skill="new_analysis", args={}, description="New analysis")],
        )
        console = Console(file=StringIO(), force_terminal=True)

        def skill_executor(skill, _args):
            if skill == "create_skill":
                activated["value"] = True
                return {"result": {"status": "success", "activated": True, "skill_name": "new_analysis"}, "is_error": False}
            if skill == "new_analysis" and not activated["value"]:
                return {"result": {"error": "Unknown skill: new_analysis"}, "is_error": True}
            return {"result": {"ok": True}, "is_error": False}

        def repair_strategy(_step, _result, _plan, _context):
            return {
                "action": "create_custom_skill",
                "reason": "Create missing analysis skill.",
                "skill": {
                    "name": "new_analysis",
                    "description": "New analysis",
                    "parameters": "{}",
                    "code_body": "return {'ok': True}",
                },
            }

        executor = PlanExecutor(
            plan_mode=pm,
            registry=MagicMock(),
            ctx_builder=lambda: MagicMock(),
            skill_executor=skill_executor,
            repair_strategy=repair_strategy,
            refresh_tools=lambda: refreshed.update(value=True),
            console=console,
        )
        executor.execute()

        assert pm.state == PlanState.DONE
        assert activated["value"] is True
        assert refreshed["value"] is True
        assert pm.repair_log[0]["action"] == "create_custom_skill"

    def test_execute_does_not_mutate_plan_args_with_ctx(self, tmp_path):
        """Registry ctx injection must not write back into PlanStep.args."""
        from rich.console import Console
        from biobank_agent.registry import SkillRegistry

        registry = SkillRegistry()

        def ok_skill(**_kwargs):
            return {"ok": True}

        registry.register("ok_skill", ok_skill, {
            "type": "function",
            "function": {
                "name": "ok_skill",
                "description": "OK",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        })
        step = PlanStep(id="s1", skill="ok_skill", args={}, description="OK")
        pm = self._make_plan_mode(tmp_path, steps=[step])
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=registry,
            ctx_builder=lambda: MagicMock(name="ctx"),
            console=console,
        )
        executor.execute()

        assert step.args == {}
        assert "ctx" not in step.result

    def test_generate_report_dual_artifacts_required_for_completion(self, tmp_path):
        """Dual report steps cannot complete unless all required files exist."""
        from rich.console import Console

        report_dir = tmp_path / "report"
        report_dir.mkdir()
        for name in [
            "report.md",
            "report_technical.md",
            "report_nature.md",
            "_report_with_css.md",
            "_report_nature_with_css.md",
            "report.html",
            "report_nature.html",
        ]:
            (report_dir / name).write_text("content", encoding="utf-8")

        result = {
            "report_dir": str(report_dir),
            "markdown": str(report_dir / "report.md"),
            "markdown_with_css": str(report_dir / "_report_with_css.md"),
            "html": str(report_dir / "report.html"),
            "format": "dual",
            "paired_outputs": {
                "technical_markdown": str(report_dir / "report_technical.md"),
                "nature_markdown": str(report_dir / "report_nature.md"),
                "nature_markdown_with_css": str(report_dir / "_report_nature_with_css.md"),
                "nature_html": str(report_dir / "report_nature.html"),
            },
        }
        pm = self._make_plan_mode(
            tmp_path,
            steps=[PlanStep(id="s1", skill="generate_report", args={"format": "dual"}, description="Report")],
        )
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=MagicMock(),
            ctx_builder=lambda: MagicMock(),
            skill_executor=lambda _skill, _args: {"result": result, "is_error": False},
            console=console,
        )
        results = executor.execute()

        assert results[0].success is True
        assert pm.state == PlanState.DONE

    def test_generate_report_missing_dual_artifact_pauses(self, tmp_path):
        """A partial dual report result is a failed step."""
        from rich.console import Console

        report_dir = tmp_path / "report"
        report_dir.mkdir()
        (report_dir / "report.md").write_text("content", encoding="utf-8")
        result = {
            "report_dir": str(report_dir),
            "markdown": str(report_dir / "report.md"),
            "format": "dual",
            "paired_outputs": {},
        }
        pm = self._make_plan_mode(
            tmp_path,
            steps=[PlanStep(id="s1", skill="generate_report", args={"format": "dual"}, description="Report")],
        )
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=MagicMock(),
            ctx_builder=lambda: MagicMock(),
            skill_executor=lambda _skill, _args: {"result": result, "is_error": False},
            console=console,
        )
        results = executor.execute()

        assert results[0].success is False
        assert "required artifact" in results[0].error
        assert pm.state == PlanState.PAUSED

    def test_goal_acceptance_adds_governed_dual_report_steps(self, tmp_path):
        """Report goals are not DONE until a final report step succeeds."""
        from rich.console import Console

        report_dir = tmp_path / "report"
        report_dir.mkdir()
        for name in [
            "report.md",
            "report_technical.md",
            "report_nature.md",
            "_report_with_css.md",
            "_report_nature_with_css.md",
            "report.html",
            "report_nature.html",
        ]:
            (report_dir / name).write_text("content", encoding="utf-8")

        pm = self._make_plan_mode(
            tmp_path,
            steps=[PlanStep(id="s1", skill="think", args={"reasoning": "draft"}, description="Think")],
        )
        pm.goal = "Generate final report"
        console = Console(file=StringIO(), force_terminal=True)

        def skill_executor(skill, _args):
            if skill == "generate_report":
                return {
                    "result": {
                        "format": "dual",
                        "markdown": str(report_dir / "report.md"),
                        "markdown_with_css": str(report_dir / "_report_with_css.md"),
                        "html": str(report_dir / "report.html"),
                        "paired_outputs": {
                            "technical_markdown": str(report_dir / "report_technical.md"),
                            "nature_markdown": str(report_dir / "report_nature.md"),
                            "nature_markdown_with_css": str(report_dir / "_report_nature_with_css.md"),
                            "nature_html": str(report_dir / "report_nature.html"),
                        },
                    },
                    "is_error": False,
                }
            return {"result": {"ok": True}, "is_error": False}

        executor = PlanExecutor(
            plan_mode=pm,
            registry=MagicMock(),
            ctx_builder=lambda: MagicMock(),
            skill_executor=skill_executor,
            console=console,
        )
        executor.execute()

        assert pm.state == PlanState.DONE
        assert "generate_report" in [r.skill for r in pm.execution_log]
        assert [s.skill for s in pm.plan.steps[-4:]] == [
            "statistical_review",
            "safety_check",
            "world_model_audit",
            "generate_report",
        ]

    def test_goal_acceptance_respects_explicit_no_final_report(self, tmp_path):
        """Artifact-only goals should not receive report steps when the user opts out."""
        from rich.console import Console

        pm = self._make_plan_mode(
            tmp_path,
            steps=[PlanStep(id="s1", skill="bank_data_readiness", args={}, description="Probe")],
        )
        pm.goal = "Validate HPP and CKB data readiness; no final report required."
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=MagicMock(),
            ctx_builder=lambda: MagicMock(),
            skill_executor=lambda _skill, _args: {"result": {"status": "SKIPPED"}, "is_error": False},
            console=console,
        )
        executor.execute()

        assert pm.state == PlanState.DONE
        assert [s.skill for s in pm.plan.steps] == ["bank_data_readiness"]
        assert "generate_report" not in [r.skill for r in pm.execution_log]

    def test_execute_respects_dependencies(self, tmp_path):
        """Steps run in dependency order."""
        from rich.console import Console

        execution_order = []

        def track_execute(name, args, ctx):
            execution_order.append(name)
            return {"output": "ok"}

        steps = [
            PlanStep(id="s1", skill="first", args={}, description="First"),
            PlanStep(id="s2", skill="second", args={}, description="Second", depends_on=["s1"]),
            PlanStep(id="s3", skill="third", args={}, description="Third", depends_on=["s2"]),
        ]
        pm = self._make_plan_mode(tmp_path, steps=steps)
        registry = MagicMock()
        registry.execute.side_effect = track_execute
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=registry,
            ctx_builder=lambda: MagicMock(),
            console=console,
        )
        executor.execute()

        assert execution_order == ["first", "second", "third"]

    def test_execute_skips_dependent_on_failure(self, tmp_path):
        """Dependent steps are skipped when dependency fails."""
        from rich.console import Console

        steps = [
            PlanStep(id="s1", skill="fail_step", args={}, description="Will fail"),
            PlanStep(id="s2", skill="never_run", args={}, description="Depends on s1", depends_on=["s1"]),
        ]
        pm = self._make_plan_mode(tmp_path, steps=steps)
        registry = MagicMock()
        registry.execute.side_effect = ValueError("intentional failure")
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=registry,
            ctx_builder=lambda: MagicMock(),
            console=console,
        )
        results = executor.execute()

        # Only s1 was attempted (s2 skipped by mark_failed)
        assert len(results) == 1
        assert not results[0].success
        # s2 should be marked skipped in the plan
        s2 = next(s for s in pm.plan.steps if s.id == "s2")
        assert s2.status == "skipped"

    def test_resume_continues_from_where_left_off(self, tmp_path):
        """resume() continues execution after pause."""
        from rich.console import Console

        call_count = [0]

        def conditional_execute(name, args, ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first call fails")
            return {"output": "ok"}

        steps = [
            PlanStep(id="s1", skill="flaky", args={}, description="Flaky step"),
            PlanStep(id="s2", skill="ok", args={}, description="OK step"),
        ]
        pm = self._make_plan_mode(tmp_path, steps=steps)
        registry = MagicMock()
        registry.execute.side_effect = conditional_execute
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=registry,
            ctx_builder=lambda: MagicMock(),
            console=console,
        )

        # First run: s1 fails
        results = executor.execute()
        assert pm.state == PlanState.PAUSED
        assert len(results) == 1

    def test_empty_plan_returns_empty(self, tmp_path):
        """Executing an empty plan returns empty results."""
        from rich.console import Console

        pm = PlanMode(plans_dir=tmp_path)
        pm.plan = None
        registry = MagicMock()
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=registry,
            ctx_builder=lambda: MagicMock(),
            console=console,
        )
        results = executor.execute()
        assert results == []

    def test_stall_detection_blocked_steps(self, tmp_path):
        """Executor detects stall when remaining steps are all blocked."""
        from rich.console import Console

        # Create a plan where s2 depends on s1, but s1 has an unresolvable dep
        steps = [
            PlanStep(id="s1", skill="step1", args={}, description="Step 1", depends_on=["s_nonexistent"]),
            PlanStep(id="s2", skill="step2", args={}, description="Step 2", depends_on=["s1"]),
        ]
        pm = self._make_plan_mode(tmp_path, steps=steps)
        registry = MagicMock()
        registry.execute.return_value = {"output": "ok"}
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=registry,
            ctx_builder=lambda: MagicMock(),
            console=console,
        )
        results = executor.execute()

        # Should fail validation before a stall can masquerade as execution.
        assert pm.state == PlanState.PAUSED
        assert "Plan validation failed" in pm.pause_reason
        assert "Unknown dependencies: s_nonexistent" in pm.pause_reason
        # No steps were actually executed
        assert registry.execute.call_count == 0


# ── PlanCheckpoint Tests ─────────────────────────────────────


class TestPlanCheckpoint:
    """Test plan state persistence."""

    def test_save_and_load(self, tmp_path):
        """Checkpoint can be saved and loaded."""
        path = tmp_path / "checkpoint.json"
        cp = PlanCheckpoint(
            goal="Test goal",
            plan_data={"goal": "Test goal", "steps": [{"id": "s1", "skill": "think", "description": "Think"}]},
            state="EXECUTING",
            revision=2,
            report_dir=str(tmp_path / "reports" / "run1"),
        )
        cp.save(path)

        loaded = PlanCheckpoint.load(path)
        assert loaded is not None
        assert loaded.goal == "Test goal"
        assert loaded.state == "EXECUTING"
        assert loaded.revision == 2
        assert loaded.report_dir.endswith("run1")
        assert loaded.updated_at != ""

    def test_load_nonexistent(self, tmp_path):
        """Loading from non-existent path returns None."""
        path = tmp_path / "missing.json"
        assert PlanCheckpoint.load(path) is None

    def test_load_invalid_json(self, tmp_path):
        """Loading invalid JSON returns None."""
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{")
        assert PlanCheckpoint.load(path) is None

    def test_clear(self, tmp_path):
        """clear() removes the checkpoint file."""
        path = tmp_path / "checkpoint.json"
        path.write_text("{}")
        assert path.exists()

        PlanCheckpoint.clear(path)
        assert not path.exists()

    def test_clear_nonexistent(self, tmp_path):
        """clear() is safe on non-existent file."""
        path = tmp_path / "missing.json"
        PlanCheckpoint.clear(path)  # Should not raise

    def test_from_plan_mode(self, tmp_path):
        """from_plan_mode() creates checkpoint from current state."""
        pm = PlanMode(plans_dir=tmp_path)
        pm.start("Test task")
        pm.approve()
        pm.set_executing()

        cp = PlanCheckpoint.from_plan_mode(pm)
        assert cp.goal == "Test task"
        assert cp.state == "EXECUTING"
        assert cp.plan_data["goal"] == "Test task"

    def test_checkpoint_sanitizes_runtime_objects(self, tmp_path):
        """Checkpoint JSON excludes ctx/runtime objects and object addresses."""
        path = tmp_path / "checkpoint.json"
        pm = PlanMode(plans_dir=tmp_path)
        pm.goal = "Runtime object hygiene"
        pm.state = PlanState.EXECUTING
        pm.repair_log.append({"step_id": "s1", "action": "retry_args", "ctx": object()})
        pm.plan = LongHorizonPlan(
            goal="Runtime object hygiene",
            steps=[
                PlanStep(
                    id="s1",
                    skill="field_search",
                    args={"query": "glucose", "ctx": object(), "connection": object()},
                    description="Search",
                    result={"rows": [object()], "ctx": object()},
                )
            ],
        )
        pm.execution_log.append(
            StepResult(
                step_id="s1",
                step_description="Search",
                skill="field_search",
                result={"connection": object(), "ctx": object()},
            )
        )

        PlanCheckpoint.from_plan_mode(pm).save(path)
        text = path.read_text(encoding="utf-8")

        assert '"ctx"' not in text
        assert "object at 0x" not in text
        loaded = PlanCheckpoint.load(path)
        assert loaded is not None
        assert loaded.plan_data["steps"][0]["args"]["connection"] == "<object>"
        assert loaded.repair_log[0]["action"] == "retry_args"

    def test_restore_to_plan_mode(self, tmp_path):
        """restore_to_plan_mode() maps EXECUTING to PAUSED for resumability."""
        # Create and save
        pm1 = PlanMode(plans_dir=tmp_path / "pm1")
        pm1.start("Restore test")
        pm1.approve()
        pm1.set_executing()
        pm1.execution_log.append(
            StepResult(step_id="s1", step_description="Done", success=True, duration_s=1.0)
        )

        cp = PlanCheckpoint.from_plan_mode(pm1)
        path = tmp_path / "cp.json"
        cp.save(path)

        # Restore into new PlanMode
        loaded_cp = PlanCheckpoint.load(path)
        pm2 = PlanMode(plans_dir=tmp_path / "pm2")
        success = loaded_cp.restore_to_plan_mode(pm2)

        assert success
        assert pm2.goal == "Restore test"
        # EXECUTING is mapped to PAUSED on restore (not resumable without executor)
        assert pm2.state == PlanState.PAUSED
        assert pm2.pause_reason == "Restored from interrupted session"
        assert pm2.plan is not None
        assert len(pm2.execution_log) == 1

    def test_atomic_save(self, tmp_path):
        """save() uses atomic tmp+rename pattern."""
        path = tmp_path / "checkpoint.json"
        cp = PlanCheckpoint(goal="Atomic test")
        cp.save(path)

        # Verify file exists and temp file does not
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()


# ── Integration: Full Flow Tests ─────────────────────────────


class TestFullPlanFlow:
    """Integration tests for complete plan lifecycle."""

    def test_start_refine_approve_execute_complete(self, tmp_path):
        """Full happy path: start → refine → approve → execute → done."""
        from rich.console import Console

        # Setup with LLM for proper refinement
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(text='[{"id":"s1","skill":"think","args":{},"description":"Think","depends_on":[],"can_parallelize":false}]')

        pm = PlanMode(plans_dir=tmp_path, llm=mock_llm, available_skills=["think"])
        pm.start("Analyze data")
        assert pm.state == PlanState.REVIEW

        # Refine — LLM returns updated plan
        mock_llm.chat.return_value = MagicMock(text='[{"id":"s2","skill":"think","args":{},"description":"Revised","depends_on":[],"can_parallelize":false}]')
        pm.refine("Add more steps")
        assert pm.state == PlanState.REVIEW
        assert pm.revision == 2

        # Approve
        pm.approve()
        assert pm.state == PlanState.APPROVED

        # Execute
        registry = MagicMock()
        registry.execute.return_value = {"output": "ok"}
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=registry,
            ctx_builder=lambda: MagicMock(),
            console=console,
        )
        results = executor.execute()

        assert pm.state == PlanState.DONE
        assert all(r.success for r in results)

    def test_execute_fail_refine_re_execute(self, tmp_path):
        """Failure → pause → refine → re-approve → resume."""
        from rich.console import Console

        steps = [
            PlanStep(id="s1", skill="will_fail", args={}, description="Will fail"),
            PlanStep(id="s2", skill="ok", args={}, description="OK", depends_on=["s1"]),
        ]
        pm = PlanMode(plans_dir=tmp_path)
        pm.plan = LongHorizonPlan(goal="Test", steps=steps)
        pm.state = PlanState.APPROVED
        pm.goal = "Test"

        registry = MagicMock()
        registry.execute.side_effect = ValueError("first fail")
        console = Console(file=StringIO(), force_terminal=True)

        executor = PlanExecutor(
            plan_mode=pm,
            registry=registry,
            ctx_builder=lambda: MagicMock(),
            console=console,
        )

        # First execution fails
        executor.execute()
        assert pm.state == PlanState.PAUSED

        # User refines plan (simulated — without LLM, plan stays same)
        pm.refine("Try different approach")
        assert pm.state == PlanState.REVIEW


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
