"""CLI command tests — verify all slash commands in _handle_command."""

import json
import os
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from rich.console import Console


def test_collect_pasted_plan_lines_merges_available_stdin(monkeypatch):
    """Multi-line pasted /plan goals should be one command, not refinements."""
    from biobank_agent.cli import _collect_pasted_command_lines

    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"I am giving you this paper PDF: /tmp/paper.pdf\nAct as an autonomous replication agent.\n")
    os.close(write_fd)
    with os.fdopen(read_fd, "r", encoding="utf-8") as stream:
        monkeypatch.setattr(sys, "stdin", stream)
        merged = _collect_pasted_command_lines("/plan Paper reproduction task:")

    assert merged == (
        "/plan Paper reproduction task:\n"
        "I am giving you this paper PDF: /tmp/paper.pdf\n"
        "Act as an autonomous replication agent."
    )


def test_collect_pasted_plan_lines_does_not_swallow_next_command(monkeypatch):
    """Queued slash commands after /plan should remain executable."""
    from biobank_agent import cli
    from biobank_agent.planner import PlanState

    cli._PENDING_STDIN_LINES.clear()
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"/plan-exit\nquit\n")
    os.close(write_fd)
    with os.fdopen(read_fd, "r", encoding="utf-8") as stream:
        monkeypatch.setattr(sys, "stdin", stream)
        merged = cli._collect_pasted_command_lines("/plan Paper reproduction task:")
        pending = cli._read_query(None, SimpleNamespace(is_active=False, state=PlanState.INACTIVE))

    assert merged == "/plan Paper reproduction task:"
    assert pending == "/plan-exit"
    assert cli._PENDING_STDIN_LINES == []


def test_plan_paste_header_heuristic_is_narrow():
    """Only likely pasted plan headers get the longer drain window."""
    from biobank_agent.cli import _looks_like_plan_paste_start

    assert _looks_like_plan_paste_start("/plan Paper reproduction task:")
    assert _looks_like_plan_paste_start("/plan 论文复现:")
    assert not _looks_like_plan_paste_start("/skills")
    assert not _looks_like_plan_paste_start("/plan Train a model for E11 and generate the final report")


def test_absorb_pasted_plan_continuation_updates_review_plan():
    """A split pasted /plan block should update the plan goal, not trigger refine."""
    from biobank_agent.cli import _absorb_pasted_plan_continuation
    from biobank_agent.planner import LongHorizonPlan, PlanState, PlanStep

    saved = {"called": False}
    planner = SimpleNamespace(
        state=PlanState.REVIEW,
        goal="Paper reproduction task:",
        plan=LongHorizonPlan(
            goal="Paper reproduction task:",
            steps=[
                PlanStep(
                    id="s1",
                    skill="read_paper",
                    args={"paper_path_or_doi": "Paper reproduction task:"},
                )
            ],
        ),
        revision=1,
        validate_current_plan=lambda: [],
        _save_plan_file=lambda: saved.update(called=True),
        _awaiting_pasted_plan_continuation=True,
    )

    absorbed = _absorb_pasted_plan_continuation(
        planner,
        "I am giving you this paper PDF: /tmp/s41588-024-01898-1.pdf",
    )

    assert absorbed is True
    assert "/tmp/s41588-024-01898-1.pdf" in planner.goal
    assert planner.plan.goal == planner.goal
    assert planner.plan.steps[0].args["paper_path_or_doi"] == "/tmp/s41588-024-01898-1.pdf"
    assert planner.revision == 2
    assert planner._needs_replan_from_pasted_goal is True
    assert saved["called"] is True


def test_refresh_plan_from_pasted_goal_replans_before_approval():
    """Approval should use the complete pasted prompt, not the first header line."""
    from biobank_agent.cli import _refresh_plan_from_pasted_goal
    from biobank_agent.planner import LongHorizonPlan, PlanState, PlanStep

    captured = {}

    class FakePlanner:
        def decompose(self, goal, available_skills, tool_schemas):
            captured["goal"] = goal
            return LongHorizonPlan(
                goal=goal,
                steps=[
                    PlanStep(
                        id="s1",
                        skill="read_paper",
                        args={"paper_path_or_doi": "/tmp/paper.pdf"},
                    )
                ],
            )

    saved = {"called": False}
    planner = SimpleNamespace(
        state=PlanState.REVIEW,
        goal="Paper reproduction task:\nI am giving you this paper PDF: /tmp/paper.pdf",
        plan=LongHorizonPlan(goal="Paper reproduction task:", steps=[]),
        revision=2,
        planner=FakePlanner(),
        available_skills=["read_paper"],
        tool_schemas={},
        validate_current_plan=lambda: [],
        _save_plan_file=lambda: saved.update(called=True),
        _needs_replan_from_pasted_goal=True,
        _awaiting_pasted_plan_continuation=True,
    )

    message = _refresh_plan_from_pasted_goal(planner)

    assert "pasted multi-line goal" in message
    assert captured["goal"].endswith("/tmp/paper.pdf")
    assert planner.plan.goal == captured["goal"]
    assert planner.revision == 3
    assert planner._needs_replan_from_pasted_goal is False
    assert planner._awaiting_pasted_plan_continuation is False
    assert saved["called"] is True


def test_stream_agent_response_renders_message_and_tool_events(mock_agent, cli_capture_console, monkeypatch):
    """Normal CLI turns should consume AsyncAgent events instead of waiting silently."""
    from biobank_agent.cli import _stream_agent_response
    from biobank_agent.core.events import AgentEvent, AgentEventType
    from biobank_agent.core.runtime import AsyncAgent

    mock_agent.settings.async_runtime_enabled = True

    monkeypatch.setattr(AsyncAgent, "is_safe_for_async", staticmethod(lambda agent: (True, "")))
    monkeypatch.setattr(AsyncAgent, "__init__", lambda self, legacy: None)

    async def fake_stream_events(self, query):
        yield AgentEvent.make(AgentEventType.MESSAGE_DELTA, text="partial answer")
        yield AgentEvent.make(
            AgentEventType.TOOL_STARTED,
            tool_call_id="c1",
            skill="prevalence",
            args={"icd10_code": "E11"},
        )
        yield AgentEvent.make(
            AgentEventType.TOOL_RESULT,
            tool_call_id="c1",
            skill="prevalence",
            summary={"n_cases": 100},
        )
        yield AgentEvent.make(AgentEventType.TURN_FINISHED, final_text="partial answer")

    monkeypatch.setattr(AsyncAgent, "stream_events", fake_stream_events)

    text, streamed = _stream_agent_response(mock_agent, "test")

    assert streamed is True
    assert text == "partial answer"
    output = cli_capture_console.getvalue()
    assert "partial answer" in output
    assert "prevalence(icd10_code)" in output
    assert "n_cases" in output


@pytest.fixture
def mock_agent(tmp_path):
    """Create a mock Agent with enough state for CLI commands."""
    agent = MagicMock()
    agent.settings = MagicMock()
    agent.settings.llm_model = "test-model"
    agent.settings.multi_model_enabled = True
    agent.settings.llm_base_url = "http://relay.local"
    agent.settings.data_dir = "/fake/data"
    agent.settings.biobank_name = "Test Biobank"
    agent.settings.plans_dir = tmp_path / "plans"
    agent.settings.plans_dir.mkdir(parents=True, exist_ok=True)
    agent.settings.reports_dir = tmp_path / "reports"
    agent.settings.reports_dir.mkdir(parents=True, exist_ok=True)
    agent.settings.plan_clarification_enabled = True
    agent.settings.plan_external_council_enabled = False
    agent.settings.plan_external_council_timeout_s = 12
    agent.settings.plan_review_hook_mode = "never"
    agent.settings.plan_review_hook_agents = "codex,claude"
    agent.registry = MagicMock()
    agent.registry.__len__ = MagicMock(return_value=43)
    agent.registry.list_skills.return_value = [
        {"name": "prevalence", "description": "Calculate prevalence"},
        {"name": "train_model", "description": "Train a model"},
    ]
    agent.state = MagicMock()
    agent.state.records = []
    agent.state.figures = []
    agent.state.cohorts = {}
    agent.state.models = {}
    agent.state.model_metadata = {}
    agent.state.token_usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    agent.state.last_orchestration = {}
    agent.state.custom_data = {}
    agent.state.context_summary.return_value = "context summary"
    agent.messages = []
    agent.memory = MagicMock()
    agent.memory.summary.return_value = "1 saved pipeline"
    agent.memory.list_pipelines.return_value = ["test_pipe"]
    agent.memory.get_pipeline.return_value = [{"skill": "prevalence", "args": {}}]
    agent.memory.most_common_errors.return_value = []
    agent.memory.explain_claim.return_value = "Claim `c1` evidence chain:\n1. supports result:r1"
    agent.orchestrator = MagicMock()
    agent.orchestrator.model_pool = [
        SimpleNamespace(model_id="test-model", role="generalist", priority=10),
        SimpleNamespace(model_id="gpt-5.4", role="generalist", priority=8),
    ]
    agent.available_models = ["test-model", "gpt-5.4", "gemini-3.1-pro-preview"]
    agent.refresh_available_models = MagicMock(return_value=agent.available_models)
    return agent


@pytest.fixture
def mock_planner():
    """Create a mock PlanMode compatible with new state machine."""
    from biobank_agent.planner import PlanState

    planner = MagicMock()
    planner.is_active = False
    planner.status = "INACTIVE"
    planner.state = PlanState.INACTIVE
    planner.current_plan_file = None
    planner.plan = None
    planner.goal = ""
    planner.revision = 0
    planner.list_plans.return_value = []
    planner.start.return_value = "Plan generated with 2 steps."
    planner.identify_clarifications.return_value = []
    planner.validation_issues = []
    planner.validation_summary.return_value = "No validation errors."
    return planner


@pytest.fixture
def cli_capture_console():
    """Capture rich console output for CLI rendering assertions."""
    from biobank_agent import cli

    stream = StringIO()
    test_console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=160,
    )
    with patch.object(cli, "console", test_console):
        yield stream


class TestSlashCommands:
    """Test all slash commands dispatch correctly."""

    def test_help(self, mock_agent, mock_planner, cli_capture_console):
        from biobank_agent.cli import _handle_command

        _handle_command("/help", mock_agent, mock_planner, {})
        output = cli_capture_console.getvalue()
        assert "Command Palette" in output
        assert "Session" in output
        assert "Configuration" in output
        assert "/export [format]" in output
        assert "/models-available" in output
        assert "/external-agents" in output
        assert "[bold]" not in output
        assert "[/]" not in output

    def test_skills(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/skills", mock_agent, mock_planner, {})
        mock_agent.registry.list_skills.assert_called_once()

    def test_history_empty(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/history", mock_agent, mock_planner, {})

    def test_figures_empty(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/figures", mock_agent, mock_planner, {})

    def test_cohorts_empty(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/cohorts", mock_agent, mock_planner, {})

    def test_models_empty(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/models", mock_agent, mock_planner, {})

    def test_model_show_current(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/model", mock_agent, mock_planner, {})

    def test_model_switch(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/model gpt-4o", mock_agent, mock_planner, {})
        assert mock_agent.settings.llm_model == "gpt-4o"
        assert mock_agent.llm.model == "gpt-4o"

    def test_compact_short_history(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        mock_agent.messages = [{"role": "user", "content": "hi"}]
        _handle_command("/compact", mock_agent, mock_planner, {})

    def test_clear(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        mock_agent.messages = [{"role": "user", "content": "hi"}]
        _handle_command("/clear", mock_agent, mock_planner, {})
        assert mock_agent.messages == []

    def test_pipelines(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/pipelines", mock_agent, mock_planner, {})
        mock_agent.memory.list_pipelines.assert_called()

    def test_errors(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/errors", mock_agent, mock_planner, {})
        mock_agent.memory.most_common_errors.assert_called()

    def test_memory(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/memory", mock_agent, mock_planner, {})
        mock_agent.memory.summary.assert_called()

    def test_plan_no_arg(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/plan", mock_agent, mock_planner, {})

    def test_plan_with_arg(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        from biobank_agent.planner import PlanState, LongHorizonPlan, PlanStep

        mock_planner.state = PlanState.INACTIVE
        mock_planner.is_active = False
        # After start(), plan should exist for display
        mock_planner.plan = LongHorizonPlan(
            goal="test task",
            steps=[PlanStep(id="s1", skill="think", description="Test step")],
        )
        mock_planner.revision = 1
        mock_planner.start.return_value = "Plan generated with 1 steps."
        _handle_command("/plan test task", mock_agent, mock_planner, {})
        mock_planner.start.assert_called_with("test task")

    def test_plan_short_wgs_prompt_adds_framework_clarifications(self, mock_agent, mock_planner, monkeypatch):
        from biobank_agent import cli
        from biobank_agent.cli import _handle_command
        from biobank_agent.planner import PlanState, LongHorizonPlan, PlanStep

        monkeypatch.setattr(cli, "_stdin_is_interactive", lambda: False)
        mock_planner.state = PlanState.INACTIVE
        mock_planner.is_active = False
        mock_planner.planner._is_wgs_vitiligo_goal.return_value = True
        mock_planner.identify_clarifications.return_value = [
            {
                "id": "wgs_data_source",
                "header": "Data",
                "question": "Which WGS data source should anchor this plan?",
                "options": [
                    {
                        "label": "Auto Discover",
                        "value": "Auto-discover WGS data and record fallback.",
                        "description": "Recommended.",
                    }
                ],
            }
        ]
        mock_planner.plan = LongHorizonPlan(
            goal="wgs",
            steps=[PlanStep(id="s1", skill="wgs_environment_check", description="Check")],
        )

        _handle_command("/plan 分析这批白癜风WGS数据", mock_agent, mock_planner, {})

        started_goal = mock_planner.start.call_args.args[0]
        assert "wgs_data_source" in started_goal
        assert "wgs_agent_policy" in started_goal
        assert "QC, annotation" in started_goal
        mock_planner.record_clarification_answers.assert_called()

    def test_plan_title_renames_active_plan(self, mock_agent, cli_capture_console, tmp_path):
        from biobank_agent.cli import _handle_command
        from biobank_agent.planner import LongHorizonPlan, PlanMode, PlanState, PlanStep

        planner = PlanMode(plans_dir=tmp_path / "plans")
        planner.goal = "分析这批白癜风WGS数据"
        planner.plan = LongHorizonPlan(
            goal=planner.goal,
            title="VirtualCell WGS Vitiligo Case-Control Analysis",
            steps=[
                PlanStep(id="s1", skill="think", description="Think"),
                PlanStep(
                    id="s2",
                    skill="generate_report",
                    args={"title": "VirtualCell WGS Vitiligo Case-Control Analysis", "format": "dual"},
                    description="Report",
                    depends_on=["s1"],
                ),
            ],
        )
        planner.state = PlanState.REVIEW
        planner.revision = 1

        _handle_command("/plan-title Short WGS Smoke", mock_agent, planner, {})

        assert planner.plan.title == "Short WGS Smoke"
        assert planner.plan.steps[1].args["title"] == "Short WGS Smoke"
        assert planner.goal == "分析这批白癜风WGS数据"
        assert "Plan title updated" in cli_capture_console.getvalue()

    def test_plan_skip_refuses_required_and_allows_optional(self, mock_agent, cli_capture_console, tmp_path):
        from biobank_agent.cli import _handle_command
        from biobank_agent.planner import LongHorizonPlan, PlanMode, PlanState, PlanStep

        planner = PlanMode(plans_dir=tmp_path / "plans")
        planner.state = PlanState.PAUSED
        planner.goal = "skip test"
        planner.plan = LongHorizonPlan(
            goal="skip test",
            steps=[
                PlanStep(id="s1", skill="cohort_summary", description="Required", criticality="required"),
                PlanStep(id="s2", skill="field_search", description="Diagnostic", criticality="diagnostic"),
            ],
        )

        _handle_command("/plan-skip s1", mock_agent, planner, {})
        _handle_command("/plan-skip s2", mock_agent, planner, {})

        assert planner.plan.steps[0].status == "pending"
        assert planner.plan.steps[1].status == "skipped"
        output = cli_capture_console.getvalue()
        assert "Refusing to skip required step" in output
        assert "Skipped s2" in output

    def test_collect_external_planning_council_records_codex_and_unavailable_claude(self, mock_agent, cli_capture_console):
        from biobank_agent.cli import _collect_external_planning_council

        mock_agent.settings.plan_external_council_enabled = True
        mock_agent.settings.plan_external_council_timeout_s = 12
        mock_agent.registry.execute.side_effect = [
            {
                "agents": {
                    "codex": {"available": True, "version": "codex test"},
                    "claude": {
                        "available": False,
                        "error": "auth required",
                        "remediation": "Run claude auth login",
                    },
                }
            },
            {
                "agent": "codex",
                "status": "success",
                "available": True,
                "elapsed_s": 1.2,
                "stdout": "Add model training, calibration, report quality gates.",
            },
        ]

        records = _collect_external_planning_council(mock_agent, "design a prediction study")

        assert records == [
            {
                "agent": "codex",
                "status": "success",
                "available": True,
                "elapsed_s": 1.2,
                "summary": "Add model training, calibration, report quality gates.",
                "stdout": "Add model training, calibration, report quality gates.",
                "stderr": "",
                "error": "",
                "command_display": "",
                "prompt_hash": "",
            },
            {
                "agent": "claude",
                "status": "unavailable",
                "available": False,
                "error": "auth required",
                "summary": "Run claude auth login",
                "remediation": "Run claude auth login",
                "command_display": "",
            },
            {
                "agent": "gemini",
                "status": "unavailable",
                "available": False,
                "error": "External planner unavailable",
                "summary": "",
                "remediation": "",
                "command_display": "",
            },
        ]
        assert mock_agent.registry.execute.call_args_list[0].args[:2] == (
            "external_agent_status",
            {"agent": "all"},
        )
        assert mock_agent.registry.execute.call_args_list[1].args[:2] == (
            "codex_plan",
            {"task": "design a prediction study", "timeout_s": 12},
        )
        assert "Planning council: 1/3" in cli_capture_console.getvalue()

    def test_collect_external_planning_council_emits_events(self, mock_agent):
        from biobank_agent.cli import _collect_external_planning_council

        mock_agent.settings.plan_external_council_enabled = True
        mock_agent.registry.execute.side_effect = [
            {"agents": {"codex": {"available": False, "error": "missing"}, "claude": {"available": False}}},
        ]
        events = []

        records = _collect_external_planning_council(
            mock_agent,
            "task",
            event_sink=lambda phase, actor, status, message, metadata=None: events.append(
                (phase, actor, status, message)
            ),
        )

        assert len(records) == 3
        assert ("External council", "biobank", "running", "checking codex/claude/gemini availability") in events
        assert any(event[1] == "codex" and event[2] == "skipped" for event in events)

    def test_extract_external_planner_blocking_questions(self):
        from biobank_agent.cli import _extract_external_planner_questions

        records = [
            {
                "agent": "gemini",
                "status": "success",
                "stdout": "\n".join([
                    "BLOCKING QUESTIONS:",
                    "1. Should the diabetes endpoint be incident-only or ever-diagnosed?",
                    "2. May the planner use raw UKB CSV columns if parquet coverage is incomplete?",
                    "",
                    "Plan:",
                    "- continue",
                ]),
            },
            {"agent": "codex", "status": "success", "stdout": "BLOCKING QUESTIONS: none\n- plan"},
        ]

        questions = _extract_external_planner_questions(records)

        assert questions == [
            {
                "agent": "gemini",
                "question": "Should the diabetes endpoint be incident-only or ever-diagnosed?",
            },
            {
                "agent": "gemini",
                "question": "May the planner use raw UKB CSV columns if parquet coverage is incomplete?",
            },
        ]

    def test_external_planner_questions_are_auto_answered_from_explicit_goal(self):
        from biobank_agent.cli import _collect_external_planner_clarifications

        text, answers = _collect_external_planner_clarifications(
            [
                {"agent": "codex", "question": "Should HPP or CKB be included in this bank workflow?"},
                {"agent": "gemini", "question": "May raw CSV fields be materialized if parquet is incomplete?"},
            ],
            "This benchmark is UKB-only and should inspect the full UKB raw CSV inventory.",
        )

        assert "UKB-only" in text
        assert "raw CSV" in text
        assert [item["label"] for item in answers] == ["Auto from task context", "Auto from task context"]

    def test_research_setup_defaults_for_short_grand_challenge(self, mock_agent, monkeypatch):
        from biobank_agent import cli

        monkeypatch.setattr(cli, "_stdin_is_interactive", lambda: False)
        monkeypatch.setattr(cli, "_quick_external_agent_status", lambda: {
            "codex": {"available": True},
            "claude": {"available": False},
            "gemini": {"available": True},
        })
        goal = (
            "I only have a broad research question: can UKB support a compelling study of metabolic health "
            "trajectories, Type 2 Diabetes risk prediction, and potentially actionable cardiometabolic biomarkers?"
        )

        clarified, answers, meta = cli._collect_research_setup(mock_agent, goal)

        assert "Autonomous research setup" in clarified
        assert "Use UKB as the active execution dataset" in clarified
        assert "Use external planning council with: codex, gemini" in clarified
        assert "available skills, and the current UKB data inventory" in clarified
        assert meta["external_agents"] == "codex,gemini"
        assert [item["id"] for item in answers] == [
            "research_setup_dataset",
            "research_setup_external_council",
            "research_setup_trajectory_policy",
            "research_setup_model_policy",
        ]

    def test_research_setup_accepts_biobank_t2d_wording(self, mock_agent, monkeypatch):
        from biobank_agent import cli

        monkeypatch.setattr(cli, "_stdin_is_interactive", lambda: False)
        monkeypatch.setattr(cli, "_quick_external_agent_status", lambda: {
            "codex": {"available": True},
            "claude": {"available": True},
            "gemini": {"available": True},
        })
        goal = (
            "Can biobank data support a compelling study of metabolic health trajectories, "
            "T2D risk prediction, and actionable cardiometabolic biomarkers?"
        )

        clarified, answers, meta = cli._collect_research_setup(mock_agent, goal)

        assert cli._is_broad_metabolic_showcase_goal(goal) is True
        assert "Autonomous research setup" in clarified
        assert "Use external planning council with: codex, claude, gemini" in clarified
        assert meta["external_council_requested"] is True
        assert len(answers) == 4

    def test_research_setup_interactive_uses_three_separate_questions(self, mock_agent, monkeypatch):
        from biobank_agent import cli

        monkeypatch.setattr(cli, "_stdin_is_interactive", lambda: True)
        monkeypatch.setattr(cli, "_quick_external_agent_status", lambda: {
            "codex": {"available": True},
            "claude": {"available": True},
            "gemini": {"available": True},
        })
        entered_prompts = []
        choices = iter(["", "1,3", "3"])

        def fake_input(prompt):
            entered_prompts.append(prompt)
            return next(choices)

        monkeypatch.setattr(cli.console, "input", fake_input)
        goal = (
            "I only have a broad research question: can UKB support a compelling study of metabolic health "
            "trajectories, Type 2 Diabetes risk prediction, and potentially actionable cardiometabolic biomarkers?"
        )

        clarified, answers, meta = cli._collect_research_setup(mock_agent, goal)

        assert len(entered_prompts) == 3
        assert all("Choice" in prompt for prompt in entered_prompts)
        assert "Use external planning council with: codex, gemini" in clarified
        assert "Prefer an interpretable model baseline" in clarified
        assert meta["external_agents"] == "codex,gemini"
        assert meta["model_policy"] == "interpretable"
        assert [item["question"] for item in answers] == [
            "Which data sources are active for this benchmark?",
            "Should external planning agents be used?",
            "How should incomplete trajectory support be handled?",
            "How should model choice be handled?",
        ]

    def test_plan_short_grand_challenge_uses_research_setup_before_start(self, mock_agent, mock_planner, monkeypatch):
        from biobank_agent import cli
        from biobank_agent.cli import _handle_command
        from biobank_agent.planner import LongHorizonPlan, PlanState, PlanStep

        monkeypatch.setattr(cli, "_stdin_is_interactive", lambda: False)
        monkeypatch.setattr(cli, "_quick_external_agent_status", lambda: {
            "codex": {"available": True},
            "claude": {"available": False},
            "gemini": {"available": True},
        })
        mock_agent.settings.plan_external_council_enabled = False
        mock_agent.settings.plan_research_setup_enabled = True
        mock_planner.state = PlanState.INACTIVE
        mock_planner.is_active = False
        mock_planner.plan = LongHorizonPlan(
            goal="showcase",
            steps=[PlanStep(id="s1", skill="project_doc", description="Inspect docs")],
        )
        mock_planner.start.return_value = "Plan generated with 1 steps."
        goal = (
            "I only have a broad research question: can UKB support a compelling study of metabolic health "
            "trajectories, Type 2 Diabetes risk prediction, and potentially actionable cardiometabolic biomarkers?"
        )

        _handle_command(f"/plan {goal}", mock_agent, mock_planner, {})

        started_goal = mock_planner.start.call_args.args[0]
        assert started_goal.startswith(goal)
        assert "Autonomous research setup" in started_goal
        assert "planning council" in started_goal
        mock_planner.record_clarification_answers.assert_called()
        assert mock_agent.settings.plan_external_council_agents == "codex,gemini"

    def test_unknown_command(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/nonexistent", mock_agent, mock_planner, {})

    def test_cost(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        token_usage = {"prompt_tokens": 100, "completion_tokens": 50}
        _handle_command("/cost", mock_agent, mock_planner, token_usage)

    def test_models_pool(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/models-pool", mock_agent, mock_planner, {})

    def test_models_available(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/models-available", mock_agent, mock_planner, {})
        mock_agent.refresh_available_models.assert_called_once()

    def test_strategy_show(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/strategy", mock_agent, mock_planner, {})

    def test_strategy_set(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/strategy single", mock_agent, mock_planner, {})
        assert mock_agent.settings.multi_model_enabled is False

    def test_routing_status(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        mock_agent.state.last_orchestration = {
            "safety_status": "PASS",
            "claims": [{"claim_id": "c1"}],
            "evidence_links": [{"claim_id": "c1"}],
            "debate_trace": {"strategy": "debate", "disagreement": False},
        }
        _handle_command("/routing-status", mock_agent, mock_planner, {})

    def test_evidence(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        _handle_command("/evidence c1", mock_agent, mock_planner, {})
        mock_agent.memory.explain_claim.assert_called_once_with("c1", limit=10)

    def test_external_agents_status(self, mock_agent, mock_planner, cli_capture_console):
        from biobank_agent.cli import _handle_command
        mock_agent.registry.execute.return_value = {
            "agents": {
                "codex": {"available": True, "version": "codex-cli test"},
                "claude": {
                    "available": False,
                    "version": "Claude Code test",
                    "error": "auth required",
                    "remediation": "Run `claude auth login`",
                },
            }
        }
        _handle_command("/external-agents", mock_agent, mock_planner, {})
        mock_agent.registry.execute.assert_called_with(
            "external_agent_status",
            {"agent": "all"},
            ctx=mock_agent._build_ctx.return_value,
        )
        assert "claude auth login" in cli_capture_console.getvalue()

    def test_codex_plan_command(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        mock_agent.registry.execute.return_value = {
            "agent": "codex",
            "task_kind": "plan",
            "status": "success",
            "stdout": "plan",
        }
        _handle_command("/codex-plan design workflow", mock_agent, mock_planner, {})
        mock_agent.registry.execute.assert_called_with(
            "codex_plan",
            {"task": "design workflow"},
            ctx=mock_agent._build_ctx.return_value,
        )

    def test_claude_check_command(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command
        mock_agent.registry.execute.return_value = {
            "agent": "claude",
            "task_kind": "review",
            "status": "success",
            "stdout": "review",
        }
        _handle_command("/claude-check report quality", mock_agent, mock_planner, {})
        mock_agent.registry.execute.assert_called_with(
            "claude_check_execution",
            {"focus": "report quality"},
            ctx=mock_agent._build_ctx.return_value,
        )

    def test_gemini_plan_and_check_commands(self, mock_agent, mock_planner):
        from biobank_agent.cli import _handle_command

        mock_agent.registry.execute.return_value = {
            "agent": "gemini",
            "task_kind": "plan",
            "status": "success",
            "stdout": "plan",
        }
        _handle_command("/gemini-plan design workflow", mock_agent, mock_planner, {})
        mock_agent.registry.execute.assert_called_with(
            "gemini_plan",
            {"task": "design workflow"},
            ctx=mock_agent._build_ctx.return_value,
        )

        mock_agent.registry.execute.return_value = {
            "agent": "gemini",
            "task_kind": "review",
            "status": "success",
            "stdout": "review",
        }
        _handle_command("/gemini-check report quality", mock_agent, mock_planner, {})
        mock_agent.registry.execute.assert_called_with(
            "gemini_check_execution",
            {"focus": "report quality"},
            ctx=mock_agent._build_ctx.return_value,
        )

    def test_review_repair_loop_writes_artifacts_and_reruns_safe_steps(self, tmp_path):
        from biobank_agent.cli import _run_review_repair_loop

        calls = []

        def fake_execute(skill, args, report_dir, allow_retry=True):
            calls.append((skill, args, Path(report_dir), allow_retry))
            return {"is_error": False, "result": {"status": "ok"}}

        agent = SimpleNamespace(
            settings=SimpleNamespace(plan_review_repair_mode="auto_safe", plan_review_repair_max_loops=2),
            state=SimpleNamespace(custom_data={}),
            _execute_skill_and_record=fake_execute,
        )
        review_records = [
            {
                "agent": "codex",
                "status": "success",
                "stdout": "High severity: prevalent model is not incident risk prediction. SHAP caption is wrong.",
            }
        ]

        _run_review_repair_loop(agent, review_records, focus="E11 risk workflow", report_dir=tmp_path)

        assert (tmp_path / "external_review_summary.json").exists()
        assert (tmp_path / "review_repair_plan.md").exists()
        assert [call[0] for call in calls] == [
            "statistical_review",
            "safety_check",
            "world_model_audit",
            "generate_report",
        ]
        assert agent.state.custom_data["external_review_summary"]["status"] == "needs_repair"

    def test_remaining_dispatch_branches(self, mock_agent, mock_planner, cli_capture_console):
        from biobank_agent.cli import _handle_command

        mock_planner.approve.return_value = "approved"
        mock_planner.exit.return_value = "exited"
        mock_planner.list_plans.return_value = [
            {"status": "DONE", "file": "done.md"},
            {"status": "EXECUTION", "file": "run.md"},
            {"status": "BLOCKED", "file": "blocked.md"},
            {"status": "DRAFT", "file": "draft.md"},
        ]

        _handle_command("/status", mock_agent, mock_planner, {"prompt_tokens": 1, "completion_tokens": 2})
        _handle_command("/plan-approve", mock_agent, mock_planner, {})
        _handle_command("/plan-exit", mock_agent, mock_planner, {})
        _handle_command("/plans", mock_agent, mock_planner, {})
        _handle_command("/evidence", mock_agent, mock_planner, {})
        _handle_command("/codex-plan", mock_agent, mock_planner, {})
        _handle_command("/claude-plan", mock_agent, mock_planner, {})
        _handle_command("/codex-check", mock_agent, mock_planner, {})
        _handle_command("/claude-plan draft review", mock_agent, mock_planner, {})
        _handle_command("/record", mock_agent, mock_planner, {})
        _handle_command("/strategy auto", mock_agent, mock_planner, {})
        _handle_command("/strategy ensemble", mock_agent, mock_planner, {})
        _handle_command("/strategy nonsense", mock_agent, mock_planner, {})
        _handle_command("/debate", mock_agent, mock_planner, {})

        output = cli_capture_console.getvalue()
        assert "Usage: /evidence" in output
        assert "Usage: /codex-plan" in output
        assert "Usage: /claude-plan" in output
        assert "Strategy set to auto" in output
        assert "Unknown strategy" in output


def populated_cli_agent(tmp_path):
    from biobank_agent.state import AnalysisRecord

    agent = SimpleNamespace()
    agent.settings = SimpleNamespace(
        llm_model="claude-sonnet",
        multi_model_enabled=False,
        llm_base_url="http://relay.local",
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
    )
    agent.settings.reports_dir.mkdir(exist_ok=True)
    agent.registry = MagicMock()
    agent.registry.__len__.return_value = 3
    agent.registry.list_skills.return_value = [
        {"name": "prevalence", "description": "x" * 120},
    ]
    agent.registry.tool_schemas.return_value = [{"type": "function"}]
    agent.state = SimpleNamespace(
        records=[
            AnalysisRecord(
                timestamp="2026-01-01T00:00:00",
                skill="cohort_summary",
                args={"icd10_code": "E11"},
                key_results={"n_cases": 10, "n_controls": 40, "extra": "value"},
                figure_paths=["fig.png"],
            ),
            AnalysisRecord(
                timestamp="2026-01-01T00:01:00",
                skill="think",
                args={},
                key_results={"thought": "skip"},
                figure_paths=[],
            ),
        ],
        figures=[str(tmp_path / "figure.png")],
        cohorts={
            "labeled": pd.DataFrame({"label": [1, 0, 0], "x": [1, 2, 3]}),
            "unlabeled": pd.DataFrame({"x": [1, 2]}),
        },
        model_metadata={
            "model_auc": {"model_type": "xgb", "auc": 0.8123, "n_cases": 10, "n_features": 5},
            "model_no_auc": {"model_type": "lr"},
        },
        models={},
        embeddings={},
        custom_data={},
        feature_matrix=None,
        labels=None,
        last_orchestration={},
        token_usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
        context_summary=lambda: "context summary",
    )
    agent.messages = [{"role": "user", "content": str(i)} for i in range(25)]
    agent.memory = SimpleNamespace(
        summary=lambda: "",
        save_pipeline=MagicMock(),
        list_pipelines=lambda: [],
        get_pipeline=lambda name: None,
        most_common_errors=lambda n: [
            {"error_type": "ValueError", "skill": "cohort", "count": 3, "last_seen": ""},
            {"error_type": "RuntimeError", "skill": "model", "count": 1, "last_seen": "2026-01-01T00:00:00"},
        ],
        explain_claim=MagicMock(return_value="no linked evidence for c1"),
    )
    agent.orchestrator = SimpleNamespace(
        model_pool=[
            SimpleNamespace(model_id="claude-sonnet", role="generalist", priority=10),
            SimpleNamespace(model_id="gpt-5.4", role="critic", priority=8),
        ],
        debate=MagicMock(return_value=SimpleNamespace(text="debate result", safety_status="PARTIAL")),
    )
    agent.llm = SimpleNamespace(model="claude-sonnet", list_models=MagicMock(return_value=["claude-sonnet", "other"]))
    agent.available_models = []
    agent._system_message = MagicMock(return_value={"role": "system", "content": "sys"})
    return agent


class TestCliHelperBranches:
    def test_prompt_session_prompt_message_read_query_and_dashboard(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli

        settings = SimpleNamespace(
            memory_dir=tmp_path / "memory",
            llm_model="model-a",
            multi_model_enabled=True,
            biobank_name="UKB",
            data_dir=tmp_path / "data",
        )
        session = cli._build_prompt_session(settings)
        assert (settings.memory_dir / "cli_history.txt").exists() is False
        assert session is not None

        plan_prompt = cli._prompt_message(True, "REVIEW")
        normal_prompt = cli._prompt_message(False)
        assert "plan" in str(plan_prompt)
        assert "biobank" in str(normal_prompt)

        monkeypatch.setattr(cli.console, "input", lambda prompt: "typed query")
        assert cli._read_query(None, SimpleNamespace(is_active=False, status="INACTIVE")) == "typed query"
        fake_session = SimpleNamespace(prompt=lambda message: f"prompted:{message!s}")
        assert cli._read_query(fake_session, SimpleNamespace(is_active=True, status="REVIEW")).startswith("prompted:")

        cli._render_startup_dashboard(
            settings,
            n_skills=5,
            n_subjects=None,
            n_fields=12,
            model_pool=["a", "b", "c", "d"],
            available_model_count=9,
        )
        cli._render_startup_dashboard(
            settings,
            n_skills=5,
            n_subjects=10,
            n_fields=12,
            model_pool=["a"],
            available_model_count=0,
        )
        assert "BioBank Agent" in cli_capture_console.getvalue()

    def test_rebuild_parquet_cmd_uses_categories_and_renders_results(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli

        settings = SimpleNamespace(
            raw_csv_dir=tmp_path / "raw",
            category_parquet_dir=tmp_path / "category",
            biomarker_parquet=tmp_path / "existing",
        )
        calls = {}
        all_calls = []

        def fake_batch_rebuild(**kwargs):
            calls.update(kwargs)
            all_calls.append(kwargs)
            kwargs["callback"]("blood", "processing blood")
            return {
                "total_new_fields": 2,
                "total_new_columns": 4,
                "total_existing_fields": 10,
                "output_dir": str(tmp_path / "out"),
                "categories": {
                    "blood": {"n_fields": 2, "n_cols": 4, "elapsed_s": 0.1},
                    "urine": {"skipped": True, "reason": "missing csv"},
                },
            }

        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        monkeypatch.setattr("biobank_agent.data.parquet_builder.batch_rebuild", fake_batch_rebuild)
        monkeypatch.setattr(cli.sys, "argv", ["biobank", "rebuild-parquet", "--categories=blood,urine"])

        cli.rebuild_parquet_cmd()
        monkeypatch.setattr(cli.sys, "argv", ["biobank", "rebuild-parquet", "--dry-run"])
        cli.rebuild_parquet_cmd()

        assert all_calls[0]["categories"] == ["blood", "urine"]
        assert calls["categories"] is None
        output = cli_capture_console.getvalue()
        assert "Rebuild Complete" in output
        assert "urine: skipped" in output

    def test_build_ukb_full_parquet_cmd_parses_dry_run_and_fields(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli

        settings = SimpleNamespace(
            raw_dir=tmp_path / "raw",
            full_ukb_feature_store=tmp_path / "full_store",
        )
        calls = {}

        def fake_build_full_ukb_feature_store(**kwargs):
            calls.update(kwargs)
            kwargs["callback"]("main_672073", "writing chunk 1/1")
            return {
                "status": "READY",
                "sources": [
                    {"source_key": "main_672073", "status": "READY", "n_chunks": 1},
                ],
                "n_selected_feature_columns": 2,
                "n_chunks": 1,
                "manifest_path": str(tmp_path / "full_store" / "manifest.json"),
            }

        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        monkeypatch.setattr(
            "biobank_agent.data.parquet_builder.build_full_ukb_feature_store",
            fake_build_full_ukb_feature_store,
        )
        monkeypatch.setattr(
            cli.sys,
            "argv",
            [
                "biobank",
                "build-ukb-full-parquet",
                "--dry-run",
                "--field-ids=6153,2443",
                "--sources=main_672073",
                "--chunk-cols=10",
            ],
        )

        cli.build_ukb_full_parquet_cmd()

        assert calls["dry_run"] is True
        assert calls["field_ids"] == ["6153", "2443"]
        assert calls["sources"] == ["main_672073"]
        assert calls["chunk_cols"] == 10
        assert "Full UKB Feature Store" in cli_capture_console.getvalue()

    def test_export_history_figures_cohorts_models_pipelines_errors_and_memory(self, tmp_path, cli_capture_console):
        from biobank_agent import cli

        agent = populated_cli_agent(tmp_path)
        cli._export(agent, "json")
        cli._export(agent, "md")
        cli._export(agent, "bad")
        cli._show_history(agent)
        cli._show_figures(agent)
        cli._show_cohorts(agent)
        cli._show_models(agent)
        cli._record_pipeline(agent, "paper_flow")

        agent.state.records = [SimpleNamespace(skill="think", args={})]
        cli._record_pipeline(agent, "empty_flow")
        cli._show_pipelines(agent)
        cli._show_errors(agent)
        cli._show_memory(agent)

        assert agent.memory.save_pipeline.call_args[0][0] == "paper_flow"
        exported = list((tmp_path / "reports" / "exports").glob("session_*"))
        assert any(path.suffix == ".json" for path in exported)
        assert any(path.suffix == ".md" for path in exported)
        output = cli_capture_console.getvalue()
        assert "Unknown format" in output
        assert "No saved pipelines" in output
        assert "Long-term memory is empty" in output

    def test_status_routing_evidence_external_cost_models_strategy_and_debate(self, tmp_path, cli_capture_console):
        from biobank_agent import cli

        agent = populated_cli_agent(tmp_path)
        planner = SimpleNamespace(is_active=True, status="EXECUTING", state="EXECUTING", goal="Test goal", plan=SimpleNamespace(done_steps=2, total_steps=5), revision=1, current_plan_file=SimpleNamespace(name="plan.md"))
        cli._show_status(agent, planner, {"prompt_tokens": 1234, "completion_tokens": 5678})
        cli._show_status(agent, SimpleNamespace(is_active=True, status="REVIEW", state="REVIEW", goal="", plan=None, revision=0, current_plan_file=None), {"prompt_tokens": 0, "completion_tokens": 0})

        agent.state.last_orchestration = {
            "safety_status": "PARTIAL",
            "claims": [{"claim_id": "c1"}, "not-dict"],
            "evidence_links": [{"claim_id": "c1"}],
            "debate_trace": {"strategy": "debate", "disagreement": True, "participants": ["a", "b"]},
            "initial_strategy": "single",
            "final_strategy": "debate",
            "turn_orchestrations": [{}, {}],
        }
        cli._show_routing_status(agent)
        agent.state.last_orchestration = {
            "claims": [{"claim_id": ""}, "not-dict"],
            "evidence_links": [],
            "debate_trace": {"strategy": "single"},
            "safety_status": "PASS",
        }
        cli._show_routing_status(agent)
        agent.state.last_orchestration = {
            "claims": [],
            "evidence_links": [],
            "debate_trace": {"strategy": "single"},
            "safety_status": "PASS",
        }
        cli._show_routing_status(agent)
        cli._show_evidence(agent, " c1 ")

        no_build_ctx = SimpleNamespace(**agent.__dict__)
        no_build_ctx.settings.reports_dir = tmp_path / "missing-parent" / "reports"
        no_build_ctx.registry = SimpleNamespace(
            execute=MagicMock(side_effect=[RuntimeError("external down"), "plain result", {"status": "error", "stderr": "stderr", "command_display": "cmd"}])
        )
        cli._run_external_agent_skill(no_build_ctx, "x", {})
        cli._run_external_agent_skill(no_build_ctx, "x", {})
        cli._run_external_agent_skill(no_build_ctx, "x", {})

        cli._show_cost({"prompt_tokens": 1000, "completion_tokens": 2000}, agent)
        agent.settings.llm_model = "gpt-4.1"
        cli._show_cost({"prompt_tokens": 1000, "completion_tokens": 2000}, agent)

        cli._show_model_pool(agent)
        agent.settings.multi_model_enabled = True
        agent.refresh_available_models = MagicMock(side_effect=RuntimeError("relay down"))
        cli._show_available_llm_models(agent)
        delattr(agent, "refresh_available_models")
        agent.llm._model_cache = ["cached"]
        agent.llm._model_cache_ts = 1.0
        cli._show_available_llm_models(agent)
        agent.llm.list_models = MagicMock(return_value=[])
        cli._show_available_llm_models(agent)

        agent.orchestrator.model_pool = [SimpleNamespace(model_id="solo", role="generalist", priority=1)]
        cli._force_debate(agent, "query", {})
        agent.orchestrator.model_pool = [
            SimpleNamespace(model_id="m1", role="generalist", priority=1),
            SimpleNamespace(model_id="m2", role="critic", priority=2),
        ]
        cli._force_debate(agent, "query", {})
        agent.orchestrator.debate.side_effect = RuntimeError("debate failed")
        cli._force_debate(agent, "query", {})

        output = cli_capture_console.getvalue()
        assert "Session Status" in output
        assert "no linked evidence" in output
        assert "External agent command failed" in output
        assert "Relay Models" in output
        assert "No models returned" in output
        assert "Debate requires" in output
        assert "safety_status=PARTIAL" in output
        assert "Debate failed" in output

    def test_evolve_writes_review_only_proposals(self, tmp_path, cli_capture_console):
        from biobank_agent import cli
        from biobank_agent.tool_learner import ToolLearner

        agent = populated_cli_agent(tmp_path)
        agent.tool_learner = ToolLearner()
        for _ in range(3):
            agent.tool_learner.record(
                "generate_report",
                {"format": "dual"},
                {"error": "report prerequisites missing: no statistical_review"},
                elapsed_s=0.1,
            )

        cli._show_evolution_status(agent, "--write-proposals")

        files = list((tmp_path / "reports" / "generated_skills").glob("evolve_proposals_*.md"))
        assert files
        text = files[0].read_text(encoding="utf-8")
        assert "review-only patch plans" in text
        assert "generate_report" in text
        assert "```diff" in text
        assert "Evolution proposals written" in cli_capture_console.getvalue()

    def test_evolve_write_history_creates_scheduled_pattern_artifacts(self, tmp_path, cli_capture_console):
        from biobank_agent import cli
        from biobank_agent.tool_learner import ToolLearner

        agent = populated_cli_agent(tmp_path)
        agent.tool_learner = ToolLearner()
        for _ in range(3):
            agent.tool_learner.record(
                "generate_report",
                {"format": "dual"},
                {"error": "report prerequisites missing: no statistical_review"},
                elapsed_s=0.1,
            )

        cli._show_evolution_status(agent, "--write-history")

        output_dir = tmp_path / "reports" / "eval" / "evolution"
        latest = json.loads((output_dir / "latest.json").read_text(encoding="utf-8"))
        assert latest["status"] == "NEEDS_REVIEW"
        assert latest["n_patterns"] == 1
        assert latest["artifacts"]["history_jsonl"].endswith("evolution_pattern_history.jsonl")
        assert (output_dir / "evolution_pattern_history.jsonl").exists()
        assert "Evolution pattern history written" in cli_capture_console.getvalue()

        cli._show_evolution_status(agent, "--scheduled-run")
        history = (output_dir / "evolution_pattern_history.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(history) >= 2

    def test_evolve_apply_low_routes_generated_proposals_through_auto_merger(
        self, tmp_path, monkeypatch, cli_capture_console
    ):
        import subprocess

        from biobank_agent import cli
        from biobank_agent.tool_learner import ToolLearner

        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        (tmp_path / "seed.txt").write_text("seed\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=tmp_path, check=True)
        monkeypatch.chdir(tmp_path)

        agent = populated_cli_agent(tmp_path)
        agent.tool_learner = ToolLearner()
        for _ in range(3):
            agent.tool_learner.record(
                "generate_report",
                {"format": "dual"},
                {"error": "report prerequisites missing: no statistical_review"},
                elapsed_s=0.1,
            )

        cli._show_evolution_status(agent, "--apply-low")

        proposal = tmp_path / "reports" / "generated_skills" / "generate_report_repair_plan.md"
        assert proposal.exists()
        assert "Evolution proposal: generate_report" in proposal.read_text(encoding="utf-8")
        assert "LOW proposal generate_report: merged" in cli_capture_console.getvalue()

    def test_evolve_apply_medium_requires_and_uses_confirmation(
        self, tmp_path, monkeypatch, cli_capture_console
    ):
        import subprocess

        from biobank_agent import cli

        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        (tmp_path / "seed.txt").write_text("seed\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=tmp_path, check=True)
        monkeypatch.chdir(tmp_path)

        medium_diff = (
            "diff --git a/reports/generated_skills/medium_repair_plan.md "
            "b/reports/generated_skills/medium_repair_plan.md\n"
            "new file mode 100644\n"
            "index 0000000..1111111\n"
            "--- /dev/null\n"
            "+++ b/reports/generated_skills/medium_repair_plan.md\n"
            "@@ -0,0 +1,3 @@\n"
            "+# Medium proposal\n"
            "+if x > 0:\n"
            "+    fix = True\n"
        )
        proposal = {
            "skill": "generate_report",
            "failure_count": 3,
            "error_signature": "branch failure",
            "suggested_action": "adjust branch after confirmation",
            "risk": "medium",
            "risk_reason": "touches control flow",
            "target_path": "reports/generated_skills/medium_repair_plan.md",
            "candidate_patch": medium_diff,
        }

        class Learner:
            def mine_failure_patterns(self, min_count=3):
                return []

            def auto_propose_skill_improvement(self, min_count=3):
                return [proposal]

        agent = populated_cli_agent(tmp_path)
        agent.tool_learner = Learner()

        cli._show_evolution_status(agent, "--apply-medium")
        assert "MEDIUM proposals require an explicit confirmation callback" in cli_capture_console.getvalue()
        assert "MEDIUM proposal generate_report: user_rejected" in cli_capture_console.getvalue()
        assert not (tmp_path / "reports" / "generated_skills" / "medium_repair_plan.md").exists()

        async def confirm(_assessment, _patch):
            return True

        agent.state.custom_data["evolve_confirm_fn"] = confirm
        cli._show_evolution_status(agent, "--apply-medium")

        applied = tmp_path / "reports" / "generated_skills" / "medium_repair_plan.md"
        assert applied.exists()
        assert "if x > 0" in applied.read_text(encoding="utf-8")
        audit = [
            json.loads(line)
            for line in (tmp_path / "reports" / "generated_skills" / "evolution_decisions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert audit[-1]["risk"] == "medium"
        assert audit[-1]["confirmed"] is True
        assert "MEDIUM proposal generate_report: merged" in cli_capture_console.getvalue()


class TestSlashAutocomplete:
    """Slash command autocomplete should be discoverable and precise."""

    def test_only_completes_slash_commands(self):
        from biobank_agent.cli import SlashCommandCompleter, _command_hints

        completer = SlashCommandCompleter(_command_hints())
        doc = Document("what are top biomarkers?", cursor_position=22)
        completions = list(completer.get_completions(doc, CompleteEvent(completion_requested=True)))
        assert completions == []

    def test_completes_matching_prefix(self):
        from biobank_agent.cli import SlashCommandCompleter, _command_hints

        completer = SlashCommandCompleter(_command_hints())
        doc = Document("/mo", cursor_position=3)
        completions = list(completer.get_completions(doc, CompleteEvent(completion_requested=True)))
        texts = [c.text for c in completions]
        assert "/model " in texts
        assert "/models" in texts
        assert "/models-pool" in texts
        assert "/models-available" in texts

    def test_inserts_trailing_space_for_arg_commands(self):
        from biobank_agent.cli import SlashCommandCompleter, _command_hints

        completer = SlashCommandCompleter(_command_hints())
        doc = Document("/pla", cursor_position=4)
        completions = list(completer.get_completions(doc, CompleteEvent(completion_requested=True)))
        text_map = {c.display_text: c.text for c in completions}
        assert text_map["/plan <task>"] == "/plan "

    def test_does_not_complete_after_command_space(self):
        from biobank_agent.cli import SlashCommandCompleter, _command_hints

        completer = SlashCommandCompleter(_command_hints())
        doc = Document("/plan ", cursor_position=6)
        completions = list(completer.get_completions(doc, CompleteEvent(completion_requested=True)))
        assert completions == []

    def test_completes_command_token_when_arguments_are_present(self):
        from biobank_agent.cli import SlashCommandCompleter, _command_hints

        completer = SlashCommandCompleter(_command_hints())
        doc = Document("/pla draft cohort", cursor_position=17)
        completions = list(completer.get_completions(doc, CompleteEvent(completion_requested=True)))

        assert any(c.display_text == "/plan <task>" for c in completions)


class TestEvalArgParsing:
    """Eval CLI should accept common argparse-style flag forms."""

    def test_parse_eval_args_supports_equals_form(self):
        from biobank_agent.cli import _parse_eval_args

        parsed = _parse_eval_args([
            "--suite=skill_schemas",
            "--mode=mas_v2",
            "--ab",
            "--enforce-gate",
            "--baseline-report=reports/eval/baseline.json",
        ])

        assert parsed.suite == "skill_schemas"
        assert parsed.mode == "mas_v2"
        assert parsed.enforce_gate is True
        assert parsed.ab_compare is True
        assert parsed.policy == "all"
        assert parsed.baseline_report == "reports/eval/baseline.json"
        assert parsed.review_loop is False

    def test_parse_eval_args_supports_space_form(self):
        from biobank_agent.cli import _parse_eval_args

        parsed = _parse_eval_args([
            "--suite",
            "skill_schemas",
            "--mode",
            "baseline",
            "--policy",
            "always",
            "--baseline-report",
            "reports/eval/baseline.json",
        ])

        assert parsed.suite == "skill_schemas"
        assert parsed.mode == "baseline"
        assert parsed.policy == "always"
        assert parsed.enforce_gate is False
        assert parsed.ab_compare is False
        assert parsed.baseline_report == "reports/eval/baseline.json"

    def test_parse_eval_args_accepts_report_quality_suites(self):
        from biobank_agent.cli import _parse_eval_args
        from biobank_agent.eval.benchmarks import (
            AgentReportWorkflowBenchmark,
            LiveUKBReport20Benchmark,
            Report20CaseBenchmark,
            ReportQualityBenchmark,
        )

        parsed = _parse_eval_args(["--suite", "report_quality"])
        workflow = _parse_eval_args(["--suite", "agent_report_workflow"])
        report_20 = _parse_eval_args(["--suite", "report_20_case"])
        live_20 = _parse_eval_args(["--suite", "live_ukb_report_20"])

        assert parsed.suite == "report_quality"
        assert workflow.suite == "agent_report_workflow"
        assert report_20.suite == "report_20_case"
        assert live_20.suite == "live_ukb_report_20"
        assert ReportQualityBenchmark.name == "report_quality"
        assert AgentReportWorkflowBenchmark.name == "agent_report_workflow"
        assert Report20CaseBenchmark.name == "report_20_case"
        assert LiveUKBReport20Benchmark.name == "live_ukb_report_20"

    def test_parse_eval_args_accepts_review_loop_flags(self):
        from biobank_agent.cli import _parse_eval_args

        parsed = _parse_eval_args([
            "--suite=report_20_case",
            "--review-loop",
            "--reviewer",
            "codex-gpt-5.5-xhigh",
            "--include-claude",
            "--review-timeout=45",
        ])

        assert parsed.suite == "report_20_case"
        assert parsed.review_loop is True
        assert parsed.primary_reviewer == "codex-gpt-5.5-xhigh"
        assert parsed.include_claude is True
        assert parsed.review_timeout_s == 45

    def test_parse_eval_args_ignores_unknown_flags(self):
        from biobank_agent.cli import _parse_eval_args

        parsed = _parse_eval_args(["--unknown", "--review-timeout", "not-an-int"])
        assert parsed.suite == "research_eval_v1"
        assert parsed.mode == "baseline"
        assert parsed.enforce_gate is False
        assert parsed.ab_compare is False
        assert parsed.baseline_report == ""
        assert parsed.review_loop is False
        assert parsed.primary_reviewer == "codex-gpt-5.5-xhigh"
        assert parsed.review_timeout_s == 600

    def test_parse_eval_args_accepts_behavioral_policy(self):
        from biobank_agent.cli import _parse_eval_args

        parsed = _parse_eval_args(["--suite=behavioral", "--policy=usually"])

        assert parsed.suite == "behavioral"
        assert parsed.policy == "usually"

    def test_parse_eval_args_accepts_scheduled_evolution_history(self):
        from biobank_agent.cli import _parse_eval_args

        parsed = _parse_eval_args([
            "--suite",
            "scheduled",
            "--evolution-history",
            "reports/eval/evolution_history.jsonl",
            "--fail-on-evolution-patterns",
        ])

        assert parsed.suite == "scheduled"
        assert parsed.evolution_history == "reports/eval/evolution_history.jsonl"
        assert parsed.fail_on_evolution_patterns is True

    def test_parse_eval_args_accepts_live_artifact_run_dir(self):
        from biobank_agent.cli import _parse_eval_args

        parsed = _parse_eval_args([
            "--suite=live_artifacts",
            "--run-dir",
            "reports/biobank_live_tests/manual/20260511_100909",
        ])

        assert parsed.suite == "live_artifacts"
        assert parsed.run_dir == "reports/biobank_live_tests/manual/20260511_100909"

    def test_parse_eval_args_accepts_v3_completion_evidence_paths(self):
        from biobank_agent.cli import _parse_eval_args

        parsed = _parse_eval_args([
            "--suite=v3_completion",
            "--run-dir=reports/biobank_live_tests/manual/20260511_100909",
            "--hpp-ckb-rap-readiness",
            "reports/bank_readiness.json",
            "--mcp-compat-evidence",
            "reports/mcp_compat.json",
            "--remote-ci-evidence",
            "reports/ci.json",
            "--high-pr-evidence",
            "https://example.test/pull/1",
            "--external-evidence-dir",
            "reports/eval/external_evidence",
            "--collect-external",
        ])

        assert parsed.suite == "v3_completion"
        assert parsed.run_dir.endswith("20260511_100909")
        assert parsed.hpp_ckb_rap_readiness == "reports/bank_readiness.json"
        assert parsed.mcp_compat_evidence == "reports/mcp_compat.json"
        assert parsed.remote_ci_evidence == "reports/ci.json"
        assert parsed.high_pr_evidence == "https://example.test/pull/1"
        assert parsed.external_evidence_dir == "reports/eval/external_evidence"
        assert parsed.collect_external is True

    def test_parse_eval_args_accepts_external_evidence_options(self):
        from biobank_agent.cli import _parse_eval_args

        parsed = _parse_eval_args([
            "--suite",
            "external_evidence",
            "--collect",
            "bank-readiness",
            "--banks=hpp,ckb,ukb_rap",
            "--icd10-code",
            "E11",
            "--probe-fields",
            "hba1c,bmi",
            "--collect",
            "mcp",
            "--mcp-config",
            "reports/mcp_servers.json",
            "--mcp-call-args=reports/mcp_args.json",
            "--mcp-min-servers",
            "3",
            "--workflow",
            "biobank-scheduled-eval.yml",
            "--pr-url",
            "https://github.com/example/repo/pull/1",
            "--branch",
            "auto-improve/demo-00001",
        ])

        assert parsed.suite == "external_evidence"
        assert parsed.collect == "mcp"
        assert parsed.banks == "hpp,ckb,ukb_rap"
        assert parsed.icd10_code == "E11"
        assert parsed.probe_fields == "hba1c,bmi"
        assert parsed.mcp_config == "reports/mcp_servers.json"
        assert parsed.mcp_call_args == "reports/mcp_args.json"
        assert parsed.mcp_min_servers == 3
        assert parsed.workflow == "biobank-scheduled-eval.yml"
        assert parsed.pr_url == "https://github.com/example/repo/pull/1"
        assert parsed.branch == "auto-improve/demo-00001"


class FakeCliBenchmark:
    name = "fake"


class FakeBenchmarkResult:
    def __init__(self, gate_passed=True, comparative=None, baseline=None, review_loop=None):
        self.observability = {
            "success_rate": 1.0,
            "evidence_coverage": 0.95,
            "stat_guardrail_violation": 0.0,
            "wrong_consensus_rate": 0.0,
            "complex_task_success": 0.8,
            "token_cost": 42.0,
            "p95_latency_s": 1.2,
        }
        self.comparative = comparative or {}
        self.baseline_observability = baseline or {}
        self.review_loop = review_loop or {}
        self.gate_passed = gate_passed
        self.gate_failures = [] if gate_passed else ["gate failed"]
        self.n_total = 1
        self.n_passed = 1 if gate_passed else 0
        self.timestamp = "2026-01-01T00:00:00"
        self.results = [
            SimpleNamespace(
                case_id="fake_case",
                passed=gate_passed,
                score=1.0 if gate_passed else 0.0,
                actual_skills=["generate_report"],
                actual_text="fake output",
                errors=[] if gate_passed else ["failed"],
                elapsed_s=1.2,
                metadata={"stage_results": [{"stage": "report", "status": "PASS"}]},
            )
        ]

    def summary(self):
        return "fake summary"


class FakeEvalHarness:
    def __init__(self, gate_passed=True):
        self.gate_passed = gate_passed
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        comparative = (
            {
                "complex_task_success_uplift": 0.1,
                "wrong_consensus_reduction_ratio": 0.2,
            }
            if kwargs.get("baseline_observability")
            else {}
        )
        return FakeBenchmarkResult(
            gate_passed=self.gate_passed,
            comparative=comparative,
            baseline=kwargs.get("baseline_observability") or {},
            review_loop=(
                {
                    "enabled": True,
                    "status": "completed",
                    "primary_reviewer": kwargs.get("primary_reviewer"),
                    "include_claude": kwargs.get("include_claude"),
                    "reviews": [
                        {"reviewer": kwargs.get("primary_reviewer"), "status": "success"},
                    ],
                }
                if kwargs.get("review_loop")
                else {}
            ),
        )


class FakeCliAgent:
    def __init__(self, settings):
        self.settings = settings
        self.orchestrator = SimpleNamespace(
            model_pool=[
                SimpleNamespace(model_id="model-a"),
                SimpleNamespace(model_id="missing-model"),
            ]
        )
        self.available_models = ["model-a"]


def cli_eval_settings(tmp_path):
    settings = SimpleNamespace(
        multi_model_enabled=True,
        reports_dir=tmp_path / "reports",
        llm_base_url="http://relay.local",
        llm_model="model-a",
        ensure_dirs=lambda: (tmp_path / "reports").mkdir(exist_ok=True),
    )
    return settings


class TestEvalCommand:
    def test_eval_cmd_behavioral_suite_writes_history_without_agent(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli

        monkeypatch.setattr(cli, "get_settings", lambda: cli_eval_settings(tmp_path))
        monkeypatch.setattr(cli.sys, "argv", [
            "biobank",
            "eval",
            "--suite",
            "behavioral",
            "--policy",
            "always",
        ])
        agent_calls = []
        monkeypatch.setattr(cli, "Agent", lambda settings: agent_calls.append(settings))

        cli.eval_cmd()

        assert agent_calls == []
        latest = tmp_path / "reports" / "eval" / "behavioral" / "latest.json"
        assert latest.exists()
        payload = json.loads(latest.read_text(encoding="utf-8"))
        assert payload["status"] == "PASS"
        assert payload["policies"] == ["always"]
        assert "Behavioral Evaluation" in cli_capture_console.getvalue()

    def test_eval_cmd_behavioral_rejects_unknown_policy(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli

        monkeypatch.setattr(cli, "get_settings", lambda: cli_eval_settings(tmp_path))
        monkeypatch.setattr(cli.sys, "argv", [
            "biobank",
            "eval",
            "--suite",
            "behavioral",
            "--policy",
            "bad",
        ])

        with pytest.raises(SystemExit) as exc:
            cli.eval_cmd()

        assert exc.value.code == 2
        assert "Unknown behavioral policy" in cli_capture_console.getvalue()

    def test_eval_cmd_scheduled_suite_writes_manifest(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli

        history = tmp_path / "failures.jsonl"
        history.write_text(
            "\n".join(
                json.dumps({"skill": "generate_report", "success": False, "error": "same failure"})
                for _ in range(3)
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(cli, "get_settings", lambda: cli_eval_settings(tmp_path))
        monkeypatch.setattr(cli.sys, "argv", [
            "biobank",
            "eval",
            "--suite",
            "scheduled",
            "--policy",
            "always",
            "--evolution-history",
            str(history),
        ])

        cli.eval_cmd()

        latest = tmp_path / "reports" / "eval" / "scheduled" / "latest.json"
        assert latest.exists()
        payload = json.loads(latest.read_text(encoding="utf-8"))
        assert payload["status"] == "NEEDS_REVIEW"
        assert payload["behavioral_status"] == "PASS"
        assert payload["evolution_status"] == "NEEDS_REVIEW"
        assert "Scheduled Quality Gates" in cli_capture_console.getvalue()

    def test_eval_cmd_live_artifacts_suite_writes_strict_audit(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli

        run_dir = tmp_path / "live"
        run_dir.mkdir()
        run_dir.joinpath("LIVE_TEST_AUDIT.md").write_text("- Status: **PASS**\n", encoding="utf-8")
        worker = run_dir / "worker-02"
        report_dir = worker / "workspace" / "reports" / "20260511_100913"
        plan_dir = worker / "workspace" / "plans"
        plan_dir.mkdir(parents=True)
        report_dir.mkdir(parents=True)
        plan_dir.joinpath("plan.md").write_text(
            "10.1038/s41588-024-01898-1 replicate_paper read_paper paper_replication_compare "
            "statistical_review safety_check world_model_audit generate_report format='dual'",
            encoding="utf-8",
        )
        for name in ("report.md", "report_technical.md", "report_nature.md", "_report_with_css.md", "_report_nature_with_css.md", "report.html", "report_nature.html"):
            report_dir.joinpath(name).write_text("# Report\n", encoding="utf-8")
        report_dir.joinpath("paper_replication_comparison.md").write_text(
            "Acceptance verdict: PASS_WITH_LIMITATIONS\n"
            "| Gate | Status | Observed |\n|---|---:|---|\n"
            "| paper_access | PASS | full |\n| cohort_count | PASS | ok |\n"
            "| model_auc | PASS | ok |\n| calibration_ece | PASS | ok |\n"
            "| feature_importance | PASS | ok |\n| figure_artifacts | PASS | ok |\n",
            encoding="utf-8",
        )
        worker.joinpath("artifact_index.json").write_text(json.dumps({"audit": {"report_dirs": [str(report_dir)]}}), encoding="utf-8")

        trajectory = run_dir / "worker-06"
        trajectory_report = trajectory / "workspace" / "reports" / "20260511_101416"
        trajectory_plan = trajectory / "workspace" / "plans"
        trajectory_plan.mkdir(parents=True)
        trajectory_report.mkdir(parents=True)
        trajectory_plan.joinpath("plan.md").write_text(
            "HealthFormer trajectory trajectory_tokenize statistical_review safety_check world_model_audit generate_report format='dual'",
            encoding="utf-8",
        )
        technical = (
            "Trajectory layer: 3,536,009 tokens\n"
            "| available_tokens | 3536009 |\n"
            "| training_distribution_coverage | 0 |\n"
            "| allowed_claim_type | association_conditioned_forecast |\n"
            "| trajectory_time_source | synthetic_assessment_instance_dates |\n"
            "| world_model_audit | PARTIAL |\n"
        )
        for name in ("report.md", "report_technical.md", "report_nature.md", "_report_with_css.md", "_report_nature_with_css.md", "report.html", "report_nature.html"):
            trajectory_report.joinpath(name).write_text(technical if name == "report_technical.md" else "# Report\n", encoding="utf-8")
        trajectory.joinpath("artifact_index.json").write_text(json.dumps({"audit": {"report_dirs": [str(trajectory_report)]}}), encoding="utf-8")

        monkeypatch.setattr(cli, "get_settings", lambda: cli_eval_settings(tmp_path))
        monkeypatch.setattr(cli.sys, "argv", [
            "biobank",
            "eval",
            "--suite",
            "live_artifacts",
            "--run-dir",
            str(run_dir),
            "--enforce-gate",
        ])

        cli.eval_cmd()

        assert list(run_dir.glob("strict_live_artifact_audit_*.json"))
        assert "Strict Live Artifact Audit" in cli_capture_console.getvalue()

    def test_eval_cmd_v3_completion_suite_reports_external_blockers(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli
        from biobank_agent.eval.v3_completion import CompletionCriterion, V3CompletionAudit

        monkeypatch.setattr(cli, "get_settings", lambda: cli_eval_settings(tmp_path))
        monkeypatch.setattr(cli.sys, "argv", [
            "biobank",
            "eval",
            "--suite",
            "v3_completion",
            "--run-dir",
            "reports/biobank_live_tests/manual/20260511_100909",
        ])

        def fake_audit(**kwargs):
            assert kwargs["live_run_dir"] == "reports/biobank_live_tests/manual/20260511_100909"
            return V3CompletionAudit(
                status="BLOCKED_EXTERNAL",
                generated_at="2026-05-11T00:00:00",
                objective="objective",
                criteria=[
                    CompletionCriterion(id="local", requirement="local", status="PASS"),
                    CompletionCriterion(id="remote", requirement="remote", status="BLOCKED_EXTERNAL"),
                ],
                artifacts={"run_md": str(tmp_path / "reports" / "eval" / "v3_completion" / "audit.md")},
            )

        monkeypatch.setattr("biobank_agent.eval.v3_completion.run_v3_completion_audit", fake_audit)

        cli.eval_cmd()

        output = cli_capture_console.getvalue()
        assert "v3 Completion Audit" in output
        assert "BLOCKED_EXTERNAL" in output
        assert "Blocked external: 1" in output

    def test_eval_cmd_v3_completion_can_collect_external_before_audit(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli
        from biobank_agent.eval.v3_completion import CompletionCriterion, V3CompletionAudit

        monkeypatch.setattr(cli, "get_settings", lambda: cli_eval_settings(tmp_path))
        monkeypatch.setattr(cli.sys, "argv", [
            "biobank",
            "eval",
            "--suite",
            "v3_completion",
            "--run-dir",
            "reports/biobank_live_tests/manual/20260511_100909",
            "--collect-external",
            "--banks",
            "hpp,ckb,ukb_rap",
            "--pr-url",
            "https://github.com/example/repo/pull/1",
            "--branch",
            "auto-improve/demo-00001",
        ])
        calls = []

        def fake_collect(**kwargs):
            calls.append(kwargs)
            return {"bank_data_readiness": str(tmp_path / "bank_readiness.json")}

        def fake_audit(**kwargs):
            assert kwargs["external_evidence_dir"] == str(tmp_path / "reports" / "eval" / "external_evidence")
            return V3CompletionAudit(
                status="BLOCKED_EXTERNAL",
                generated_at="2026-05-11T00:00:00",
                objective="objective",
                criteria=[
                    CompletionCriterion(id="local", requirement="local", status="PASS"),
                    CompletionCriterion(id="remote", requirement="remote", status="BLOCKED_EXTERNAL"),
                ],
                artifacts={"run_md": str(tmp_path / "reports" / "eval" / "v3_completion" / "audit.md")},
            )

        monkeypatch.setattr("biobank_agent.eval.v3_completion.collect_external_evidence_for_completion", fake_collect)
        monkeypatch.setattr("biobank_agent.eval.v3_completion.run_v3_completion_audit", fake_audit)

        cli.eval_cmd()

        assert calls
        assert calls[0]["banks"] == "hpp,ckb,ukb_rap"
        assert calls[0]["pr_url"] == "https://github.com/example/repo/pull/1"
        output = cli_capture_console.getvalue()
        assert "collected bank_data_readiness" in output
        assert "v3 Completion Audit" in output

    def test_eval_cmd_external_evidence_suite_collects_requested_artifacts(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli
        from biobank_agent.eval.external_evidence import EvidenceArtifact

        monkeypatch.setattr(cli, "get_settings", lambda: cli_eval_settings(tmp_path))
        monkeypatch.setattr(cli.sys, "argv", [
            "biobank",
            "eval",
            "--suite",
            "external_evidence",
            "--collect",
            "high-pr",
            "--pr-url",
            "https://github.com/example/repo/pull/1",
            "--branch",
            "auto-improve/demo-00001",
        ])

        def fake_high_pr(**kwargs):
            assert kwargs["pr_url"] == "https://github.com/example/repo/pull/1"
            assert kwargs["branch"] == "auto-improve/demo-00001"
            return EvidenceArtifact("high_risk_pr_evidence", "PR_OPENED", str(tmp_path / "high_pr.json"))

        monkeypatch.setattr("biobank_agent.eval.external_evidence.collect_high_pr_evidence", fake_high_pr)

        cli.eval_cmd()

        output = cli_capture_console.getvalue()
        assert "External Evidence" in output
        assert "high_risk_pr_evidence" in output
        assert "PR_OPENED" in output

    def test_eval_cmd_unknown_mode_and_suite(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli

        monkeypatch.setattr(cli, "get_settings", lambda: cli_eval_settings(tmp_path))
        monkeypatch.setattr(cli.sys, "argv", ["biobank", "eval", "--mode", "bad"])
        with pytest.raises(SystemExit) as bad_mode:
            cli.eval_cmd()
        assert bad_mode.value.code == 2

        monkeypatch.setattr(cli.sys, "argv", ["biobank", "eval", "--suite", "unknown"])
        with pytest.raises(SystemExit) as bad_suite:
            cli.eval_cmd()
        assert bad_suite.value.code == 2

        output = cli_capture_console.getvalue()
        assert "Unknown mode" in output
        assert "Unknown suite" in output

    def test_eval_cmd_ab_compare_baseline_report_and_gate_failure(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli
        import biobank_agent.eval.benchmarks as benchmarks_mod
        import biobank_agent.eval.harness as harness_mod

        for name in ("AgentReportWorkflowBenchmark", "LiveUKBReport20Benchmark", "ResearchEvalV1", "Report20CaseBenchmark", "ReportQualityBenchmark", "SkillSchemaBenchmark", "BiomedQABenchmark", "SkillCallBenchmark"):
            monkeypatch.setattr(benchmarks_mod, name, FakeCliBenchmark)
        harness = FakeEvalHarness()
        monkeypatch.setattr(harness_mod, "EvalHarness", lambda: harness)
        monkeypatch.setattr(cli, "Agent", FakeCliAgent)
        monkeypatch.setattr(cli, "get_settings", lambda: cli_eval_settings(tmp_path))

        monkeypatch.setattr(cli.sys, "argv", ["biobank", "eval", "--suite", "skill_schemas", "--mode", "mas_v2", "--ab"])
        cli.eval_cmd()
        assert len(harness.calls) == 2
        assert harness.calls[0]["mode"] == "baseline"
        assert harness.calls[1]["mode"] == "mas_v2"
        saved = sorted((tmp_path / "reports" / "eval").glob("skill_schemas_mas_v2_*.json"))[-1]
        payload = json.loads(saved.read_text(encoding="utf-8"))
        assert payload["results"][0]["case_id"] == "fake_case"
        assert payload["results"][0]["metadata"]["stage_results"][0]["stage"] == "report"

        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps({"observability": {"complex_task_success": 0.4}}), encoding="utf-8")
        harness.calls.clear()
        monkeypatch.setattr(cli.sys, "argv", [
            "biobank",
            "eval",
            "--suite",
            "skill_schemas",
            "--mode",
            "baseline",
            "--baseline-report",
            str(baseline_path),
        ])
        cli.eval_cmd()
        assert harness.calls[-1]["baseline_observability"] == {"complex_task_success": 0.4}

        bad_baseline = tmp_path / "bad.json"
        bad_baseline.write_text("{bad", encoding="utf-8")
        monkeypatch.setattr(cli.sys, "argv", [
            "biobank",
            "eval",
            "--suite",
            "skill_schemas",
            "--mode",
            "baseline",
            "--baseline-report",
            str(bad_baseline),
        ])
        cli.eval_cmd()
        assert "Failed to parse baseline report" in cli_capture_console.getvalue()

        failing_harness = FakeEvalHarness(gate_passed=False)
        monkeypatch.setattr(harness_mod, "EvalHarness", lambda: failing_harness)
        monkeypatch.setattr(cli.sys, "argv", [
            "biobank",
            "eval",
            "--suite",
            "skill_schemas",
            "--mode",
            "baseline",
            "--enforce-gate",
        ])
        with pytest.raises(SystemExit) as gate_exit:
            cli.eval_cmd()
        assert gate_exit.value.code == 3

    def test_eval_cmd_report_20_case_review_loop_flags(self, tmp_path, monkeypatch):
        from biobank_agent import cli
        import biobank_agent.eval.benchmarks as benchmarks_mod
        import biobank_agent.eval.harness as harness_mod

        for name in ("AgentReportWorkflowBenchmark", "LiveUKBReport20Benchmark", "ResearchEvalV1", "Report20CaseBenchmark", "ReportQualityBenchmark", "SkillSchemaBenchmark", "BiomedQABenchmark", "SkillCallBenchmark"):
            monkeypatch.setattr(benchmarks_mod, name, FakeCliBenchmark)
        harness = FakeEvalHarness()
        monkeypatch.setattr(harness_mod, "EvalHarness", lambda: harness)
        monkeypatch.setattr(cli, "Agent", FakeCliAgent)
        monkeypatch.setattr(cli, "get_settings", lambda: cli_eval_settings(tmp_path))
        monkeypatch.setattr(cli.sys, "argv", [
            "biobank",
            "eval",
            "--suite",
            "report_20_case",
            "--review-loop",
            "--reviewer",
            "codex-gpt-5.5-xhigh",
            "--include-claude",
            "--review-timeout",
            "42",
        ])

        cli.eval_cmd()

        call = harness.calls[-1]
        assert call["benchmark"].name == "fake"
        assert call["review_loop"] is True
        assert call["primary_reviewer"] == "codex-gpt-5.5-xhigh"
        assert call["include_claude"] is True
        assert call["review_timeout_s"] == 42
        saved = sorted((tmp_path / "reports" / "eval").glob("report_20_case_baseline_*.json"))[-1]
        payload = json.loads(saved.read_text(encoding="utf-8"))
        assert payload["suite"] == "report_20_case"
        assert payload["review_loop"]["primary_reviewer"] == "codex-gpt-5.5-xhigh"

    def test_eval_cmd_missing_baseline_file_and_unavailable_model_list(self, tmp_path, monkeypatch):
        from biobank_agent import cli
        import biobank_agent.eval.benchmarks as benchmarks_mod
        import biobank_agent.eval.harness as harness_mod

        for name in ("AgentReportWorkflowBenchmark", "LiveUKBReport20Benchmark", "ResearchEvalV1", "Report20CaseBenchmark", "ReportQualityBenchmark", "SkillSchemaBenchmark", "BiomedQABenchmark", "SkillCallBenchmark"):
            monkeypatch.setattr(benchmarks_mod, name, FakeCliBenchmark)
        harness = FakeEvalHarness()
        monkeypatch.setattr(harness_mod, "EvalHarness", lambda: harness)

        class NoModelListAgent(FakeCliAgent):
            def __init__(self, settings):
                super().__init__(settings)
                self.available_models = []

        monkeypatch.setattr(cli, "Agent", NoModelListAgent)
        monkeypatch.setattr(cli, "get_settings", lambda: cli_eval_settings(tmp_path))
        monkeypatch.setattr(cli.sys, "argv", [
            "biobank",
            "eval",
            "--suite",
            "skill_schemas",
            "--mode",
            "baseline",
            "--baseline-report",
            str(tmp_path / "does-not-exist.json"),
        ])

        cli.eval_cmd()

        assert harness.calls[-1]["baseline_observability"] == {}


class FakeMainSettings:
    def __init__(self, tmp_path):
        self.llm_model = "model-a"
        self.multi_model_enabled = True
        self.plans_dir = tmp_path / "plans"
        self.memory_dir = tmp_path / "memory"
        self.data_dir = tmp_path / "data"
        self.field_txt = tmp_path / "field.txt"
        self.biobank_name = "UKB"
        self.reports_dir = tmp_path / "reports"
        self.reports_dir.mkdir(exist_ok=True)

    def ensure_dirs(self):
        self.plans_dir.mkdir(exist_ok=True)
        self.memory_dir.mkdir(exist_ok=True)


class FakeMainAgent:
    def __init__(self, settings, *, count_subjects=10, fields=None, run_side_effect=None):
        self.settings = settings
        self.dm = SimpleNamespace(count_subjects=MagicMock(side_effect=count_subjects if isinstance(count_subjects, Exception) else None))
        if not isinstance(count_subjects, Exception):
            self.dm.count_subjects.return_value = count_subjects
        self.catalog = SimpleNamespace(fields=fields if fields is not None else {"30740": {}})
        self.orchestrator = SimpleNamespace(model_pool=[SimpleNamespace(model_id="model-a")])
        self.registry = MagicMock()
        self.registry.__len__.return_value = 3
        self.registry.list_skills.return_value = [{"name": "think", "description": "Think"}]
        self.available_models = ["model-a"]
        self.llm = MagicMock()
        self.state = SimpleNamespace(
            figures=[],
            token_usage=SimpleNamespace(prompt_tokens=11, completion_tokens=22),
            interrupted=False,
        )
        self.messages = []
        self.run = MagicMock(side_effect=run_side_effect) if run_side_effect is not None else MagicMock(return_value="agent response")


class FakeMainPlanner:
    def __init__(self, plans_dir, *, active=False, **kwargs):
        from biobank_agent.planner import PlanState
        self.plans_dir = plans_dir
        self.is_active = active
        self.status = "EXECUTING" if active else "INACTIVE"
        self.state = PlanState.EXECUTING if active else PlanState.INACTIVE
        self.current_plan_file = None
        self.plan = None
        self.goal = ""
        self.revision = 0

    def get_plan_content(self):
        return "Plan body"


class TestMainCommand:
    def test_main_dispatches_subcommands(self, monkeypatch):
        from biobank_agent import cli

        calls = []
        monkeypatch.setattr(cli.sys, "argv", ["biobank", "rebuild-parquet"])
        monkeypatch.setattr(cli, "rebuild_parquet_cmd", lambda: calls.append("rebuild"))
        cli.main()

        monkeypatch.setattr(cli.sys, "argv", ["biobank", "eval"])
        monkeypatch.setattr(cli, "eval_cmd", lambda: calls.append("eval"))
        cli.main()

        assert calls == ["rebuild", "eval"]

    def test_main_opens_interactive_shell_by_default(self, tmp_path, monkeypatch):
        from biobank_agent import cli

        settings = FakeMainSettings(tmp_path)
        captured = {}

        def fake_interactive(settings_arg, *, initial_task="", console=None, workspace=""):
            captured["settings"] = settings_arg
            captured["initial_task"] = initial_task
            captured["console"] = console

        monkeypatch.setattr(cli.sys, "argv", ["biobank"])
        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        monkeypatch.setattr(cli, "run_interactive_shell", fake_interactive)
        monkeypatch.setattr(cli, "Agent", lambda settings_arg: FakeMainAgent(settings_arg))

        cli.main()

        assert captured["settings"] is settings
        assert captured["initial_task"] == ""
        assert captured["console"] is cli.console

    def test_main_model_space_form_does_not_contaminate_task(self, tmp_path, monkeypatch):
        # Regression: `--model foo` must consume `foo` as the model value, not also
        # leak it into the seeded interactive task.
        from biobank_agent import cli

        settings = FakeMainSettings(tmp_path)
        captured = {}

        def fake_interactive(settings_arg, *, initial_task="", console=None, workspace=""):
            captured["initial_task"] = initial_task

        monkeypatch.setattr(cli.sys, "argv", ["biobank", "--model", "model-b", "Analyze", "cohort"])
        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        monkeypatch.setattr(cli, "run_interactive_shell", fake_interactive)
        monkeypatch.setattr(cli, "Agent", lambda settings_arg: FakeMainAgent(settings_arg))

        cli.main()

        assert settings.llm_model == "model-b"
        assert captured["initial_task"] == "Analyze cohort"

    def test_main_compatibility_task_seeds_interactive_shell(self, tmp_path, monkeypatch):
        from biobank_agent import cli

        settings = FakeMainSettings(tmp_path)
        captured = {}

        def fake_interactive(settings_arg, *, initial_task="", console=None, workspace=""):
            captured["settings"] = settings_arg
            captured["initial_task"] = initial_task

        monkeypatch.setattr(cli.sys, "argv", ["biobank", "Analyze the cohort"])
        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        monkeypatch.setattr(cli, "run_interactive_shell", fake_interactive)
        monkeypatch.setattr(cli, "Agent", lambda settings_arg: FakeMainAgent(settings_arg))

        cli.main()

        assert captured["settings"] is settings
        assert captured["initial_task"] == "Analyze the cohort"

    def test_main_workspace_flag_is_parsed(self, tmp_path, monkeypatch):
        # `--workspace <dir>` (and `--cwd`) must be consumed as the workspace value
        # and forwarded to the shell, not leaked into the seeded task (issue #4).
        from biobank_agent import cli

        settings = FakeMainSettings(tmp_path)
        captured = {}

        def fake_interactive(settings_arg, *, initial_task="", console=None, workspace=""):
            captured["initial_task"] = initial_task
            captured["workspace"] = workspace

        monkeypatch.setattr(cli.sys, "argv", ["biobank", "--workspace", "/proj/X", "Analyze", "cohort"])
        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        monkeypatch.setattr(cli, "run_interactive_shell", fake_interactive)
        monkeypatch.setattr(cli, "Agent", lambda settings_arg: FakeMainAgent(settings_arg))

        cli.main()

        assert captured["workspace"] == "/proj/X"
        assert captured["initial_task"] == "Analyze cohort"

    def test_main_workspace_equals_form_is_parsed(self, tmp_path, monkeypatch):
        from biobank_agent import cli

        settings = FakeMainSettings(tmp_path)
        captured = {}

        def fake_interactive(settings_arg, *, initial_task="", console=None, workspace=""):
            captured["workspace"] = workspace

        monkeypatch.setattr(cli.sys, "argv", ["biobank", "--cwd=/data/proj"])
        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        monkeypatch.setattr(cli, "run_interactive_shell", fake_interactive)
        monkeypatch.setattr(cli, "Agent", lambda settings_arg: FakeMainAgent(settings_arg))

        cli.main()

        assert captured["workspace"] == "/data/proj"


class TestCliRemainingBranches:
    def test_dispatch_and_helper_edge_branches(self, tmp_path, mock_agent, mock_planner, cli_capture_console):
        from biobank_agent import cli

        agent = populated_cli_agent(tmp_path)
        cli._compact(agent)
        assert len(agent.messages) == 20

        cli._record_pipeline(SimpleNamespace(state=SimpleNamespace(records=[]), memory=agent.memory), "none")

        cli._handle_command("/plans", mock_agent, mock_planner, {})
        cli._handle_command("/export", agent, mock_planner, {})
        cli._handle_command("/record paper_flow", agent, mock_planner, {})
        cli._handle_command("/debate compare models", agent, mock_planner, {})
        mock_agent.switch_model = "not callable"
        cli._handle_command("/model no-switch", mock_agent, mock_planner, {})

        empty_trace_agent = populated_cli_agent(tmp_path)
        empty_trace_agent.state.last_orchestration = {}
        cli._show_routing_status(empty_trace_agent)

        fallback_models = SimpleNamespace(
            settings=SimpleNamespace(llm_base_url="http://relay.local", llm_model="fallback"),
            orchestrator=SimpleNamespace(model_pool=[]),
            available_models=["fallback"],
        )
        cli._show_available_llm_models(fallback_models)

        external_agent = SimpleNamespace(
            settings=SimpleNamespace(reports_dir=tmp_path / "reports-file"),
            state=SimpleNamespace(),
            memory=None,
            _build_ctx=MagicMock(side_effect=RuntimeError("ctx failed")),
            registry=SimpleNamespace(execute=MagicMock(return_value={"status": "success", "stdout": "ok"})),
        )
        external_agent.settings.reports_dir.write_text("file", encoding="utf-8")
        cli._run_external_agent_skill(external_agent, "codex_plan", {"task": "x"})

        pass_debate_agent = populated_cli_agent(tmp_path)
        pass_debate_agent.orchestrator.debate.return_value = SimpleNamespace(text="safe debate", safety_status="PASS")
        cli._force_debate(pass_debate_agent, "query", {})

        output = cli_capture_console.getvalue()
        assert "No saved plans" in output
        assert "No routing trace yet" in output
        assert "Relay Models" in output
        assert "safe debate" in output
