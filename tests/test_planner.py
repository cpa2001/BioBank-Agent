"""Test PlanMode state machine — new human-in-the-loop lifecycle.

Tests the complete Plan → Review → Approve → Execute → Report cycle
plus iterative refinement, pause/resume, and edge cases.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestPlanModeImport:
    """Test PlanMode class import and construction."""

    def test_planmode_imports(self):
        """PlanMode can be imported."""
        from biobank_agent.planner import PlanMode
        assert PlanMode is not None

    def test_initial_state_is_inactive(self, tmp_path):
        """Freshly constructed PlanMode starts INACTIVE."""
        from biobank_agent.planner import PlanMode, PlanState

        pm = PlanMode(plans_dir=tmp_path / "plans")

        assert pm.status == "INACTIVE"
        assert pm.state == PlanState.INACTIVE
        assert not pm.is_active
        assert pm.current_plan_file is None
        assert pm.plan is None

    def test_creates_plans_directory(self, tmp_path):
        """Constructor creates the plans directory if missing."""
        from biobank_agent.planner import PlanMode

        plans_dir = tmp_path / "new_plans_dir"
        assert not plans_dir.exists()

        PlanMode(plans_dir=plans_dir)

        assert plans_dir.exists()


class TestPlanModeStart:
    """Test starting plan mode (single-command entry)."""

    def test_start_creates_plan_and_enters_review(self, tmp_path):
        """start() decomposes goal and sets state to REVIEW."""
        from biobank_agent.planner import PlanMode, PlanState

        pm = PlanMode(plans_dir=tmp_path)
        msg = pm.start("Analyze diabetes biomarkers")

        assert pm.state == PlanState.REVIEW
        assert pm.is_active
        assert pm.plan is not None
        assert pm.plan.total_steps > 0
        assert pm.goal == "Analyze diabetes biomarkers"
        assert pm.revision == 1
        assert "steps" in msg.lower()

    def test_start_creates_plan_file(self, tmp_path):
        """start() persists a plan markdown file."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.start("Build E11 cohort")

        assert pm.current_plan_file is not None
        assert pm.current_plan_file.exists()
        assert pm.current_plan_file.suffix == ".md"

    def test_start_when_already_active_returns_error(self, tmp_path):
        """Starting plan mode twice -> error message."""
        from biobank_agent.planner import PlanMode, PlanState

        pm = PlanMode(plans_dir=tmp_path)
        pm.start("First task")
        msg = pm.start("Second task")

        assert "already active" in msg.lower()
        assert pm.state == PlanState.REVIEW  # unchanged

    def test_start_with_llm_decomposes_properly(self, tmp_path):
        """start() uses LLM when available to decompose goal."""
        from biobank_agent.planner import PlanMode

        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(text='[{"id":"s1","skill":"prevalence","args":{"top_n":5},"description":"Check prevalence","depends_on":[],"can_parallelize":false}]')

        pm = PlanMode(plans_dir=tmp_path, llm=mock_llm, available_skills=["prevalence", "think"])
        pm.start("Check T2D prevalence")

        assert pm.plan.total_steps == 1
        assert pm.plan.steps[0].skill == "prevalence"

    def test_start_passes_compiled_study_spec_to_decomposer(self, tmp_path):
        """Plan mode should use StudySpec as a real planning gate."""
        from types import SimpleNamespace
        from biobank_agent.planner import LongHorizonPlan, PlanMode, PlanStep

        captured = {}

        class FakeSpec:
            design = SimpleNamespace(value="case_control")
            tool_budget = 3

            def spec_hash(self):
                return "spec123"

            def constrain_skills(self, skills):
                return ["think"]

        class FakeCompiler:
            def compile(self, goal):
                captured["compiled_goal"] = goal
                return FakeSpec()

        class FakePlanner:
            def decompose(self, goal, available_skills, context="", spec=None, tool_schemas=None):
                captured["available_skills"] = list(available_skills)
                captured["spec"] = spec
                return LongHorizonPlan(
                    goal=goal,
                    steps=[PlanStep(id="s1", skill="think", args={"reasoning": goal})],
                )

        pm = PlanMode(
            plans_dir=tmp_path,
            available_skills=["think", "train_model"],
            study_spec_compiler=FakeCompiler(),
        )
        pm.planner = FakePlanner()

        pm.start("Train E11 model")

        assert captured["compiled_goal"] == "Train E11 model"
        assert captured["spec"] is pm.current_study_spec
        assert any("StudySpec gate: hash=spec123" in a for a in pm.plan.assumptions)

    def test_plan_file_slug_from_goal(self, tmp_path):
        """Plan filename is slug-ified from goal."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.start("Analyze diabetes/biomarkers")

        assert "analyze-diabetes" in pm.current_plan_file.name.lower()

    def test_default_plan_for_paper_replication_is_actionable(self):
        """LLM fallback for paper goals should be a real replication workflow."""
        from biobank_agent.planner import LongHorizonPlanner

        goal = (
            "Paper reproduction task: I am giving you this paper PDF: "
            "/tmp/s41588-024-01898-1.pdf. Replicate it and generate report."
        )
        plan = LongHorizonPlanner(llm=None).decompose(goal, available_skills=[])
        skills = [step.skill for step in plan.steps]

        assert skills[:7] == ["replicate_paper", "read_paper", "deep_research", "field_search", "cohort_summary", "missing_data", "train_model"]
        assert "missing_data" in skills
        assert "evaluate_model" in skills
        assert "calibration" in skills
        assert "feature_importance" in skills
        assert "paper_replication_compare" in skills
        assert skills[-4:] == ["statistical_review", "safety_check", "world_model_audit", "generate_report"]
        assert plan.steps[0].args["source"] == "/tmp/s41588-024-01898-1.pdf"
        assert plan.steps[0].args["source_type"] == "path"
        assert plan.steps[1].args["paper_path_or_doi"] == "/tmp/s41588-024-01898-1.pdf"
        assert plan.steps[-1].args["format"] == "dual"

    def test_paper_replication_template_does_not_block_on_llm(self):
        """High-value paper replication tasks use deterministic templates before LLM planning."""
        from biobank_agent.planner import LongHorizonPlanner

        mock_llm = MagicMock()
        goal = "Paper reproduction task: read /tmp/s41588-024-01898-1.pdf and generate report"
        plan = LongHorizonPlanner(llm=mock_llm).decompose(goal, available_skills=[])

        assert plan.steps[0].skill == "replicate_paper"
        assert plan.steps[1].skill == "read_paper"
        mock_llm.chat.assert_not_called()

    def test_paper_replication_template_extracts_doi_without_pdf_path(self):
        """Paper prompts that only provide a DOI should pass the DOI to read_paper."""
        from biobank_agent.planner import LongHorizonPlanner

        goal = "Please reproduce the paper DOI 10.1038/s41588-024-01898-1 and compare results."
        plan = LongHorizonPlanner(llm=None).decompose(goal, available_skills=[])

        assert plan.steps[0].skill == "replicate_paper"
        assert "10.1038/s41588-024-01898-1" in plan.steps[0].args["source"]
        assert plan.steps[0].args["source_type"] == "text"
        assert plan.steps[1].skill == "read_paper"
        assert plan.steps[1].args["paper_path_or_doi"] == "10.1038/s41588-024-01898-1"

    def test_default_plan_for_literature_discovery_is_actionable(self):
        """LLM fallback for open-ended discovery should include literature, model, guardrails, report."""
        from biobank_agent.planner import LongHorizonPlanner

        goal = "Natural language discovery task: find related work and predict cardiometabolic biomarkers"
        plan = LongHorizonPlanner(llm=None).decompose(goal, available_skills=[])
        skills = [step.skill for step in plan.steps]

        assert skills[:5] == ["deep_research", "field_search", "cohort_summary", "missing_data", "train_model"]
        assert "evaluate_model" in skills
        assert "calibration" in skills
        assert "feature_importance" in skills
        assert skills[-4:] == ["statistical_review", "safety_check", "world_model_audit", "generate_report"]
        assert plan.steps[4].args["model_type"] == "auto"
        assert plan.steps[-1].args["format"] == "dual"

    def test_without_supplied_paper_does_not_route_to_read_paper(self):
        """Mentioning papers as comparison context should not imply a supplied-paper workflow."""
        from biobank_agent.planner import LongHorizonPlanner

        goal = (
            "Natural language discovery task: Without using a supplied paper, "
            "find related papers, analyze Type 2 Diabetes biomarkers, and generate a dual report."
        )
        plan = LongHorizonPlanner(llm=None).decompose(goal, available_skills=[])
        skills = [step.skill for step in plan.steps]

        assert skills[0] == "deep_research"
        assert "read_paper" not in skills
        assert skills[-1] == "generate_report"

    def test_incomplete_publication_task_uses_conservative_discovery_template(self):
        """Broad incomplete prompts should not explode into huge external-agent plans."""
        from biobank_agent.planner import LongHorizonPlanner

        goal = (
            "This request is intentionally incomplete: study diabetes/obesity progression in UKB "
            "and make it publication quality. You choose the endpoint and report format."
        )
        plan = LongHorizonPlanner(llm=MagicMock()).decompose(goal, available_skills=[])
        skills = [step.skill for step in plan.steps]

        assert len(plan.steps) <= 14
        assert skills[:5] == ["deep_research", "field_search", "cohort_summary", "missing_data", "train_model"]
        assert plan.steps[-1].skill == "generate_report"
        assert plan.steps[-1].args["format"] == "dual"

    def test_invalid_causal_privacy_request_gets_safe_dual_plan(self):
        """Unsafe causal/privacy prompts should push back and use aggregate reporting."""
        from biobank_agent.planner import LongHorizonPlanner

        goal = (
            "I want a causal proof that HbA1c causes Type 2 Diabetes and patient-level IDs "
            "in a technical plus Nature-style report."
        )
        plan = LongHorizonPlanner(llm=MagicMock()).decompose(goal, available_skills=[])
        skills = [step.skill for step in plan.steps]

        assert skills[:3] == ["safety_check", "critical_thinking", "think"]
        assert "train_model" in skills
        assert "world_model_audit" in skills
        assert plan.steps[-1].skill == "generate_report"
        assert plan.steps[-1].args["format"] == "dual"

    def test_report_contract_forces_dual_when_goal_asks_nature_and_technical(self):
        """LLM plans should not downgrade explicit paired report requests."""
        from biobank_agent.planner import LongHorizonPlanner

        mock_llm = MagicMock()
        mock_llm.chat.return_value.text = (
            '[{"id":"s1","skill":"generate_report","args":{"title":"x","format":"report"},'
            '"description":"Report","depends_on":[],"can_parallelize":false}]'
        )

        goal = "Generate a technical plus Nature-style report."
        plan = LongHorizonPlanner(llm=mock_llm).decompose(goal, available_skills=["generate_report"])

        assert plan.steps[0].args["format"] == "dual"

    def test_report_contract_inserts_pre_report_guardrails_when_available(self):
        """Report steps should wait for statistical, safety, and world-model review."""
        from biobank_agent.planner import LongHorizonPlanner

        mock_llm = MagicMock()
        mock_llm.chat.return_value.text = (
            '[{"id":"s1","skill":"field_search","args":{"query":"E11"},"description":"Search","depends_on":[],"can_parallelize":false},'
            '{"id":"s2","skill":"generate_report","args":{"title":"x","format":"dual"},"description":"Report","depends_on":["s1"],"can_parallelize":false}]'
        )

        plan = LongHorizonPlanner(llm=mock_llm).decompose(
            "Generate a technical plus Nature-style report.",
            available_skills=["field_search", "statistical_review", "safety_check", "world_model_audit", "generate_report"],
        )
        skills = [step.skill for step in plan.steps]
        report = next(step for step in plan.steps if step.skill == "generate_report")
        stat = next(step for step in plan.steps if step.skill == "statistical_review")
        safety = next(step for step in plan.steps if step.skill == "safety_check")
        world = next(step for step in plan.steps if step.skill == "world_model_audit")

        assert skills == ["field_search", "statistical_review", "safety_check", "world_model_audit", "generate_report"]
        assert report.args["format"] == "dual"
        assert set(report.depends_on) >= {stat.id, safety.id, world.id}
        assert stat.depends_on == ["s1"]
        assert safety.depends_on == [stat.id]
        assert world.depends_on == [safety.id]

    def test_explicit_trajectory_task_uses_feasibility_template(self):
        """Longitudinal trajectory requests should not be routed as generic biomarker discovery."""
        from biobank_agent.planner import LongHorizonPlanner

        goal = (
            "Build a longitudinal HealthFormer-style trajectory forecast over time for UKB "
            "diabetes progression and generate a technical plus Nature-style report."
        )
        plan = LongHorizonPlanner(llm=MagicMock()).decompose(goal, available_skills=[])
        skills = [step.skill for step in plan.steps]
        by_skill = {step.skill: step for step in plan.steps}

        assert skills[:9] == [
            "project_doc",
            "ukb_data_inventory",
            "deep_research",
            "field_search",
            "ukb_field_resolve",
            "bank_data_probe",
            "cohort_summary",
            "cohort_card",
            "trajectory_tokenize",
        ]
        assert skills[9:14] == ["missing_data", "train_model", "evaluate_model", "calibration", "feature_importance"]
        assert "smart_plot" in skills
        assert skills[-4:] == ["statistical_review", "safety_check", "world_model_audit", "generate_report"]
        assert by_skill["ukb_field_resolve"].depends_on == ["s2"]
        assert "6153" in by_skill["ukb_field_resolve"].args["field_ids"]
        assert by_skill["bank_data_probe"].args["probe_fields"] == "hba1c,bmi,glucose,systolic_bp,diastolic_bp"
        assert by_skill["cohort_summary"].depends_on == ["s2a", "s2b"]
        assert by_skill["trajectory_tokenize"].args["target_modality"] == "bmi"
        assert by_skill["train_model"].args["model_type"] == "auto"
        assert set(by_skill["statistical_review"].depends_on) >= {"s5", "s8", "s9", "s10"}
        assert by_skill["world_model_audit"].depends_on == [by_skill["safety_check"].id]
        assert plan.steps[-1].args["format"] == "dual"
        from biobank_agent.planner import PlanSchemaValidator
        assert PlanSchemaValidator().validate(plan) == []

    def test_bank_readiness_task_uses_readiness_probe_template(self):
        """Credential/data-path readiness requests should use the aggregate readiness skill."""
        from biobank_agent.planner import LongHorizonPlanner, PlanSchemaValidator

        goal = "Validate HPP, CKB and UKB-RAP data readiness and credentialed aggregate data paths before modelling."
        plan = LongHorizonPlanner(llm=MagicMock()).decompose(
            goal,
            available_skills=["bank_data_readiness"],
        )

        assert [step.skill for step in plan.steps] == ["bank_data_readiness"]
        assert plan.steps[0].args["banks"] == "hpp,ckb,ukb_rap"
        assert plan.steps[0].args["icd10_code"] == "E11"
        assert "systolic_bp" in plan.steps[0].args["probe_fields"]
        assert PlanSchemaValidator(["bank_data_readiness"]).validate(plan) == []

    def test_ukb_only_future_ports_grand_challenge_still_uses_trajectory_template(self):
        """Mentioning HPP/CKB/RAP as future ports must not collapse the demo into readiness-only."""
        from biobank_agent.planner import LongHorizonPlanner, PlanSchemaValidator

        goal = (
            "I only have a broad research question: can UKB support a compelling study of metabolic health "
            "trajectories, Type 2 Diabetes risk prediction, and potentially actionable cardiometabolic biomarkers? "
            "Act as an autonomous biobank research agent. First inspect project documentation, available skills, "
            "and the current UKB data inventory. Treat HPP/CKB/RAP as future ports only; this benchmark is UKB-only. "
            "If Codex and Claude planning modes are available, ask them for independent plans and merge their advice "
            "with your own plan. Then design and execute the strongest feasible UKB-only workflow. You should discover "
            "relevant fields from both the existing Milton parquet subset and the full UKB raw CSV inventory; search "
            "repeated BMI, HbA1c, glucose, blood pressure, lipid and diagnosis fields; attempt a HealthFormer-style "
            "trajectory feasibility branch; train the best feasible tabular prediction model with model_type=\"auto\"; "
            "run statistical_review, safety_check and world_model_audit before conclusions; and produce final technical "
            "plus Nature-style reports with exact output paths."
        )
        plan = LongHorizonPlanner(llm=MagicMock()).decompose(goal, available_skills=[])
        skills = [step.skill for step in plan.steps]

        assert skills[0] == "project_doc"
        assert skills[1] == "ukb_data_inventory"
        assert "bank_data_readiness" not in skills
        for skill in ("deep_research", "field_search", "ukb_field_resolve", "cohort_summary", "cohort_card", "trajectory_tokenize", "train_model", "smart_plot"):
            assert skill in skills
        assert plan.steps[-1].skill == "generate_report"
        assert plan.steps[-1].args["format"] == "dual"
        assert PlanSchemaValidator().validate(plan) == []

    def test_short_broad_grand_challenge_prompt_expands_to_showcase_workflow(self):
        """The human-style short prompt should be enough for the agent to infer the full workflow."""
        from biobank_agent.planner import LongHorizonPlanner, PlanSchemaValidator

        goal = (
            "I only have a broad research question: can UKB support a compelling study of metabolic health "
            "trajectories, Type 2 Diabetes risk prediction, and potentially actionable cardiometabolic biomarkers?"
        )
        plan = LongHorizonPlanner(llm=MagicMock()).decompose(goal, available_skills=[])
        skills = [step.skill for step in plan.steps]
        by_skill = {step.skill: step for step in plan.steps}

        assert skills[:2] == ["project_doc", "ukb_data_inventory"]
        for skill in (
            "deep_research",
            "field_search",
            "ukb_field_resolve",
            "bank_data_probe",
            "cohort_summary",
            "cohort_card",
            "trajectory_tokenize",
            "missing_data",
            "train_model",
            "evaluate_model",
            "calibration",
            "feature_importance",
            "smart_plot",
            "statistical_review",
            "safety_check",
            "world_model_audit",
            "generate_report",
        ):
            assert skill in skills
        assert by_skill["train_model"].args["model_type"] == "auto"
        assert by_skill["generate_report"].args["format"] == "dual"
        assert PlanSchemaValidator().validate(plan) == []

    def test_biobank_t2d_wording_expands_to_showcase_workflow(self, tmp_path):
        """The setup detector should not require the user to literally type UKB or Type 2 Diabetes."""
        from biobank_agent.planner import LongHorizonPlanner, PlanMode, PlanSchemaValidator

        goal = (
            "Can biobank data support a compelling study of metabolic health trajectories, "
            "T2D risk prediction, and actionable cardiometabolic biomarkers?"
        )
        planner = LongHorizonPlanner(llm=MagicMock())
        plan = planner.decompose(goal, available_skills=[])
        skills = [step.skill for step in plan.steps]

        assert planner._is_metabolic_showcase_goal(goal) is True
        assert PlanMode(tmp_path).identify_clarifications(goal) == []
        assert skills[:2] == ["project_doc", "ukb_data_inventory"]
        assert "trajectory_tokenize" in skills
        assert "train_model" in skills
        assert plan.steps[-1].skill == "generate_report"
        assert PlanSchemaValidator().validate(plan) == []

    def test_trajectory_only_setup_can_disable_tabular_fallback_branch(self):
        """If the user explicitly chooses trajectory-only setup, the deterministic plan should honor it."""
        from biobank_agent.planner import LongHorizonPlanner, PlanSchemaValidator

        goal = (
            "I only have a broad research question: can UKB support a compelling study of metabolic health "
            "trajectories, Type 2 Diabetes risk prediction, and potentially actionable cardiometabolic biomarkers?\n\n"
            "Autonomous research setup:\n"
            "- research_setup_trajectory_policy: Do not train a tabular fallback model unless trajectory data are sufficient."
        )
        plan = LongHorizonPlanner(llm=MagicMock()).decompose(goal, available_skills=[])
        skills = [step.skill for step in plan.steps]

        assert "trajectory_tokenize" in skills
        assert "train_model" not in skills
        assert "evaluate_model" not in skills
        assert "calibration" not in skills
        assert "feature_importance" not in skills
        assert plan.steps[-1].skill == "generate_report"
        assert PlanSchemaValidator().validate(plan) == []

    def test_validator_blocks_missing_deps_cycles_and_normalizes_criticality(self):
        """Invalid DAGs should be rejected before /plan-approve can execute."""
        from biobank_agent.planner import LongHorizonPlan, PlanSchemaValidator, PlanStep

        plan = LongHorizonPlan(
            goal="bad graph",
            steps=[
                PlanStep(id="s1", skill="think", args={}, description="one", depends_on=["s2"], criticality="high"),
                PlanStep(id="s2", skill="think", args={}, description="two", depends_on=["s1", "s99"]),
            ],
        )

        issues = PlanSchemaValidator().validate(plan)
        messages = "\n".join(issue.message for issue in issues)

        assert "Unknown dependencies: s99." in messages
        assert "Dependency cycle detected:" in messages
        assert plan.steps[0].criticality == "required"

    def test_external_planning_merge_adds_missing_model_branch(self, tmp_path):
        """External planner advice should be merged as schema-safe executable steps."""
        from biobank_agent.planner import LongHorizonPlan, PlanMode, PlanStep

        pm = PlanMode(
            plans_dir=tmp_path,
            available_skills=[
                "field_search",
                "cohort_summary",
                "missing_data",
                "train_model",
                "evaluate_model",
                "calibration",
                "feature_importance",
                "statistical_review",
                "safety_check",
                "generate_report",
            ],
        )
        pm.goal = "Build the best possible prediction model for Type 2 Diabetes"
        pm.plan = LongHorizonPlan(
            goal=pm.goal,
            steps=[
                PlanStep(id="s1", skill="field_search", args={"query": "diabetes"}, description="Find fields"),
                PlanStep(id="s2", skill="cohort_summary", args={"icd10_code": "E11"}, description="Cases", depends_on=["s1"]),
                PlanStep(id="s3", skill="statistical_review", args={"scope": "session"}, description="Review", depends_on=["s2"]),
                PlanStep(id="s4", skill="safety_check", args={"scope": "session"}, description="Safety", depends_on=["s3"]),
                PlanStep(id="s5", skill="generate_report", args={"title": "x", "format": "dual"}, description="Report", depends_on=["s4"]),
            ],
        )
        pm.attach_planning_council([
            {
                "agent": "codex",
                "status": "success",
                "summary": "Add model training, AUC, calibration, and feature importance checks.",
                "command_display": "codex exec -",
                "prompt_hash": "abc123",
            }
        ])

        message = pm.merge_external_plans()
        skills = [step.skill for step in pm.plan.steps]
        train_step = next(step for step in pm.plan.steps if step.skill == "train_model")
        report_step = next(step for step in pm.plan.steps if step.skill == "generate_report")

        assert "merged" in message.lower()
        assert ["missing_data", "train_model", "evaluate_model", "calibration", "feature_importance"] == skills[5:10]
        assert train_step.args == {"icd10_code": "E11", "model_type": "auto", "n_folds": 5}
        assert {"s6", "s7", "s8", "s9", "s10"}.issubset(set(report_step.depends_on))
        appendix = pm._planning_council_markdown()
        assert "`codex exec -`" in appendix
        assert "Prompt hash: `abc123`" in appendix


class TestPlanModeRefine:
    """Test iterative plan refinement."""

    def test_refine_in_review_state(self, tmp_path):
        """refine() works in REVIEW state — without LLM, reports unchanged."""
        from biobank_agent.planner import PlanMode, PlanState

        pm = PlanMode(plans_dir=tmp_path)
        pm.start("Task")

        assert pm.revision == 1
        msg = pm.refine("Add a validation step")

        # Without LLM, plan cannot be refined
        assert pm.state == PlanState.REVIEW
        assert "unchanged" in msg.lower() or "updated" in msg.lower()

    def test_refine_in_review_state_with_llm(self, tmp_path):
        """refine() with LLM increments revision and updates plan."""
        from biobank_agent.planner import PlanMode, PlanState
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        # decompose returns initial plan
        mock_llm.chat.return_value = MagicMock(text='[{"id":"s1","skill":"think","args":{},"description":"Think","depends_on":[],"can_parallelize":false}]')

        pm = PlanMode(plans_dir=tmp_path, llm=mock_llm, available_skills=["think"])
        pm.start("Task")
        assert pm.revision == 1

        # Now refine — LLM returns a different step
        mock_llm.chat.return_value = MagicMock(text='[{"id":"s2","skill":"prevalence","args":{},"description":"Check","depends_on":[],"can_parallelize":false}]')
        msg = pm.refine("Add prevalence check")

        assert pm.state == PlanState.REVIEW
        assert pm.revision == 2
        assert "v2" in msg

    def test_refine_in_paused_state(self, tmp_path):
        """refine() works in PAUSED state — without LLM, stays in REVIEW."""
        from biobank_agent.planner import PlanMode, PlanState

        pm = PlanMode(plans_dir=tmp_path)
        pm.start("Task")
        pm.approve()
        pm.set_executing()
        pm.pause("test")

        msg = pm.refine("Replace step with different approach")
        # Without LLM, refine is a no-op but state transitions to REVIEW
        assert pm.state == PlanState.REVIEW

    def test_refine_from_inactive_rejected(self, tmp_path):
        """refine() from INACTIVE state -> rejection."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        msg = pm.refine("Some feedback")

        assert "cannot refine" in msg.lower()

    def test_refine_with_llm_preserves_completed(self, tmp_path):
        """refine() preserves completed steps when using LLM."""
        from biobank_agent.planner import PlanMode, PlanStep, LongHorizonPlan

        mock_llm = MagicMock()
        # First call: decompose
        mock_llm.chat.return_value = MagicMock(text='[{"id":"s1","skill":"think","args":{},"description":"Analyze","depends_on":[],"can_parallelize":false},{"id":"s2","skill":"prevalence","args":{},"description":"Check prevalence","depends_on":["s1"],"can_parallelize":false}]')

        pm = PlanMode(plans_dir=tmp_path, llm=mock_llm, available_skills=["think", "prevalence"])
        pm.start("Task")

        # Mark s1 as done
        pm.plan.mark_done("s1", {"output": "done"})

        # Refine with LLM returning new step
        mock_llm.chat.return_value = MagicMock(text='[{"id":"s3","skill":"correlation","args":{},"description":"Check correlations","depends_on":[],"can_parallelize":false}]')
        pm.refine("Replace prevalence with correlation")

        # s1 should still be present and done
        assert any(s.id == "s1" and s.status == "done" for s in pm.plan.steps)


class TestPlanModeApprove:
    """Test approval flow."""

    def test_approve_from_review(self, tmp_path):
        """approve() from REVIEW transitions to APPROVED."""
        from biobank_agent.planner import PlanMode, PlanState

        pm = PlanMode(plans_dir=tmp_path)
        pm.start("Task")
        msg = pm.approve()

        assert pm.state == PlanState.APPROVED
        assert "approved" in msg.lower()

    def test_approve_from_non_review_rejected(self, tmp_path):
        """approve() from non-REVIEW state -> rejection."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        msg = pm.approve()

        assert "cannot approve" in msg.lower()

    def test_approve_empty_plan_rejected(self, tmp_path):
        """approve() with no steps -> rejection."""
        from biobank_agent.planner import PlanMode, PlanState, LongHorizonPlan

        pm = PlanMode(plans_dir=tmp_path)
        pm.state = PlanState.REVIEW
        pm.plan = LongHorizonPlan(goal="empty", steps=[])
        msg = pm.approve()

        assert "no steps" in msg.lower()

    def test_approve_blocks_schema_invalid_plan(self, tmp_path):
        """Schema-invalid generated args stay in REVIEW and cannot execute."""
        from biobank_agent.planner import PlanMode, PlanState

        schema = {
            "type": "function",
            "function": {
                "name": "generate_report",
                "description": "Generate report",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "default": "Report"},
                        "format": {"type": "string", "default": "report"},
                        "output_dir": {"type": "string", "default": ""},
                    },
                    "required": [],
                },
            },
        }
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            text='[{"id":"s1","skill":"generate_report","args":{"format":"dual","sections":["bad"]},'
                 '"description":"Write report","depends_on":[],"can_parallelize":false}]'
        )

        pm = PlanMode(
            plans_dir=tmp_path,
            llm=mock_llm,
            available_skills=["generate_report"],
            tool_schemas=[schema],
        )
        msg = pm.start("Write final report")

        assert pm.state == PlanState.REVIEW
        assert "schema validation failed" in msg.lower()
        assert "Unsupported args: sections" in pm.validation_summary()
        approve_msg = pm.approve()
        assert "cannot approve" in approve_msg.lower()
        assert pm.state == PlanState.REVIEW

    def test_plan_edit_revalidates_required_args(self, tmp_path):
        """Refined plans are checked against real required args before approval."""
        from biobank_agent.planner import PlanMode

        schema = {
            "type": "function",
            "function": {
                "name": "field_search",
                "description": "Search field catalogue",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["query"],
                },
            },
        }
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            text='[{"id":"s1","skill":"field_search","args":{"query":"glucose"},'
                 '"description":"Search","depends_on":[],"can_parallelize":false}]'
        )
        pm = PlanMode(
            plans_dir=tmp_path,
            llm=mock_llm,
            available_skills=["field_search"],
            tool_schemas=[schema],
        )
        pm.start("Find fields")
        assert pm.validation_issues == []

        mock_llm.chat.return_value = MagicMock(
            text='[{"id":"s2","skill":"field_search","args":{"limit":5},'
                 '"description":"Broken search","depends_on":[],"can_parallelize":false}]'
        )
        msg = pm.refine("remove query")

        assert "schema validation failed" in msg.lower()
        assert "Missing required args: query" in pm.validation_summary()
        assert "Cannot approve" in pm.approve()

    def test_refresh_tools_updates_validator_after_dynamic_skill_load(self, tmp_path):
        """PlanMode can see schemas added after custom skill activation."""
        from biobank_agent.planner import LongHorizonPlan, PlanMode, PlanStep

        class FakeRegistry:
            def list_skills(self):
                return [{"name": "dynamic_skill"}]

            def tool_schemas(self):
                return [{
                    "type": "function",
                    "function": {
                        "name": "dynamic_skill",
                        "description": "Dynamic",
                        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                    },
                }]

        pm = PlanMode(plans_dir=tmp_path, available_skills=["old"], tool_schemas=[])
        pm.plan = LongHorizonPlan(
            goal="dynamic",
            steps=[PlanStep(id="s1", skill="dynamic_skill", args={"x": "ok"})],
        )

        pm.refresh_tools(FakeRegistry())

        assert pm.available_skills == ["dynamic_skill"]
        assert pm.validation_issues == []


class TestPlanModePauseResume:
    """Test pause/resume during execution."""

    def test_pause_from_executing(self, tmp_path):
        """pause() from EXECUTING -> PAUSED."""
        from biobank_agent.planner import PlanMode, PlanState

        pm = PlanMode(plans_dir=tmp_path)
        pm.start("Task")
        pm.approve()
        pm.set_executing()

        msg = pm.pause("Test pause")

        assert pm.state == PlanState.PAUSED
        assert pm.pause_reason == "Test pause"

    def test_pause_from_non_executing_rejected(self, tmp_path):
        """pause() from non-EXECUTING state -> rejection."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.start("Task")
        msg = pm.pause("test")

        assert "cannot pause" in msg.lower()

    def test_resume_from_paused(self, tmp_path):
        """resume() from PAUSED -> EXECUTING."""
        from biobank_agent.planner import PlanMode, PlanState

        pm = PlanMode(plans_dir=tmp_path)
        pm.start("Task")
        pm.approve()
        pm.set_executing()
        pm.pause("test")

        msg = pm.resume()

        assert pm.state == PlanState.EXECUTING
        assert "resuming" in msg.lower()

    def test_resume_from_non_paused_rejected(self, tmp_path):
        """resume() from non-PAUSED state -> rejection."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        msg = pm.resume()

        assert "cannot resume" in msg.lower()


class TestPlanModeComplete:
    """Test plan completion."""

    def test_complete_sets_done(self, tmp_path):
        """complete() transitions to DONE."""
        from biobank_agent.planner import PlanMode, PlanState

        pm = PlanMode(plans_dir=tmp_path)
        pm.start("Task")
        pm.approve()
        pm.set_executing()
        msg = pm.complete()

        assert pm.state == PlanState.DONE
        assert "complete" in msg.lower()


class TestPlanModeExit:
    """Test exiting plan mode."""

    def test_exit_from_active_state(self, tmp_path):
        """exit() from any active state -> INACTIVE."""
        from biobank_agent.planner import PlanMode, PlanState

        pm = PlanMode(plans_dir=tmp_path)
        pm.start("Task")

        msg = pm.exit()

        assert pm.state == PlanState.INACTIVE
        assert not pm.is_active
        assert pm.plan is None
        assert "REVIEW" in msg

    def test_exit_when_not_active(self, tmp_path):
        """exit() when not in plan mode -> informational message."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        msg = pm.exit()

        assert "Not in plan mode" in msg

    def test_exit_from_executing_resets(self, tmp_path):
        """exit() during EXECUTING properly resets."""
        from biobank_agent.planner import PlanMode, PlanState

        pm = PlanMode(plans_dir=tmp_path)
        pm.start("Task")
        pm.approve()
        pm.set_executing()

        msg = pm.exit()
        assert pm.state == PlanState.INACTIVE
        assert "EXECUTING" in msg


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

        (tmp_path / "plan-a.md").write_text("# Plan A\n- Status: DONE\n")
        (tmp_path / "plan-b.md").write_text("# Plan B\n- Status: REVIEW\n")

        pm = PlanMode(plans_dir=tmp_path)
        plans = pm.list_plans()

        assert len(plans) == 2
        statuses = {p["file"]: p["status"] for p in plans}
        assert statuses["plan-a.md"] == "DONE"
        assert statuses["plan-b.md"] == "REVIEW"

    def test_list_plans_has_path_field(self, tmp_path):
        """Each plan entry includes its full path."""
        from biobank_agent.planner import PlanMode

        (tmp_path / "test-plan.md").write_text("# Test\n- Status: EXECUTING\n")

        pm = PlanMode(plans_dir=tmp_path)
        plans = pm.list_plans()

        assert len(plans) == 1
        assert str(tmp_path) in plans[0]["path"]


class TestPlanModeGetContent:
    """Test plan content retrieval."""

    def test_get_plan_content_returns_markdown(self, tmp_path):
        """get_plan_content() returns markdown of current plan."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        pm.start("Task")

        content = pm.get_plan_content()
        assert "Plan:" in content
        assert "think" in content or "prevalence" in content

    def test_get_plan_content_no_active_plan(self, tmp_path):
        """get_plan_content() with no active plan -> fallback string."""
        from biobank_agent.planner import PlanMode

        pm = PlanMode(plans_dir=tmp_path)
        content = pm.get_plan_content()

        assert "no active plan" in content.lower()


class TestPlanStateTransitions:
    """Test all state transition rules."""

    def test_full_happy_path(self, tmp_path):
        """INACTIVE → REVIEW → APPROVED → EXECUTING → DONE."""
        from biobank_agent.planner import PlanMode, PlanState

        pm = PlanMode(plans_dir=tmp_path)
        assert pm.state == PlanState.INACTIVE

        pm.start("Goal")
        assert pm.state == PlanState.REVIEW

        pm.approve()
        assert pm.state == PlanState.APPROVED

        pm.set_executing()
        assert pm.state == PlanState.EXECUTING

        pm.complete()
        assert pm.state == PlanState.DONE

    def test_refine_loop(self, tmp_path):
        """REVIEW → REFINING → REVIEW (multiple times) with LLM."""
        from biobank_agent.planner import PlanMode, PlanState
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        # decompose
        mock_llm.chat.return_value = MagicMock(text='[{"id":"s1","skill":"think","args":{},"description":"Think","depends_on":[],"can_parallelize":false}]')

        pm = PlanMode(plans_dir=tmp_path, llm=mock_llm, available_skills=["think"])
        pm.start("Goal")

        for i in range(3):
            # Each refine returns a new step list
            mock_llm.chat.return_value = MagicMock(text=f'[{{"id":"s{i+2}","skill":"think","args":{{}},"description":"Step {i+2}","depends_on":[],"can_parallelize":false}}]')
            pm.refine(f"Change {i}")
            assert pm.state == PlanState.REVIEW
            assert pm.revision == i + 2

    def test_pause_and_refine_loop(self, tmp_path):
        """EXECUTING → PAUSED → REVIEW (via refine) → APPROVED → EXECUTING."""
        from biobank_agent.planner import PlanMode, PlanState

        pm = PlanMode(plans_dir=tmp_path)
        pm.start("Goal")
        pm.approve()
        pm.set_executing()
        pm.pause("issue")
        assert pm.state == PlanState.PAUSED

        pm.refine("Fix the issue")
        assert pm.state == PlanState.REVIEW

        pm.approve()
        assert pm.state == PlanState.APPROVED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
