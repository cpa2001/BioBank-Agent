"""CLI command tests — verify all slash commands in _handle_command."""

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from rich.console import Console


@pytest.fixture
def mock_agent():
    """Create a mock Agent with enough state for CLI commands."""
    agent = MagicMock()
    agent.settings = MagicMock()
    agent.settings.llm_model = "test-model"
    agent.settings.multi_model_enabled = True
    agent.settings.llm_base_url = "http://relay.local"
    agent.settings.data_dir = "/fake/data"
    agent.settings.biobank_name = "Test Biobank"
    agent.settings.reports_dir = MagicMock()
    agent.settings.reports_dir.__truediv__ = MagicMock(return_value=MagicMock())
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
    """Create a mock PlanMode."""
    planner = MagicMock()
    planner.is_active = False
    planner.status = "INACTIVE"
    planner.current_plan = None
    planner.list_plans.return_value = []
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
        mock_planner.enter.return_value = "Plan mode activated."
        _handle_command("/plan test task", mock_agent, mock_planner, {})
        mock_planner.enter.assert_called_with("test task")

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

        plan_prompt = cli._prompt_message(True)
        normal_prompt = cli._prompt_message(False)
        assert "plan" in str(plan_prompt)
        assert "biobank" in str(normal_prompt)

        monkeypatch.setattr(cli.console, "input", lambda prompt: "typed query")
        assert cli._read_query(None, SimpleNamespace(is_active=False)) == "typed query"
        fake_session = SimpleNamespace(prompt=lambda message: f"prompted:{message!s}")
        assert cli._read_query(fake_session, SimpleNamespace(is_active=True)).startswith("prompted:")

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
        assert "Biobank Agent" in cli_capture_console.getvalue()

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
        planner = SimpleNamespace(is_active=True, status="EXECUTION", current_plan=SimpleNamespace(name="plan.md"))
        cli._show_status(agent, planner, {"prompt_tokens": 1234, "completion_tokens": 5678})
        cli._show_status(agent, SimpleNamespace(is_active=True, status="INTAKE", current_plan=None), {"prompt_tokens": 0, "completion_tokens": 0})

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

        assert parsed == (
            "skill_schemas",
            "mas_v2",
            True,
            True,
            "reports/eval/baseline.json",
        )

    def test_parse_eval_args_supports_space_form(self):
        from biobank_agent.cli import _parse_eval_args

        parsed = _parse_eval_args([
            "--suite",
            "skill_schemas",
            "--mode",
            "baseline",
            "--baseline-report",
            "reports/eval/baseline.json",
        ])

        assert parsed == (
            "skill_schemas",
            "baseline",
            False,
            False,
            "reports/eval/baseline.json",
        )

    def test_parse_eval_args_accepts_report_quality_suites(self):
        from biobank_agent.cli import _parse_eval_args
        from biobank_agent.eval.benchmarks import AgentReportWorkflowBenchmark, ReportQualityBenchmark

        parsed = _parse_eval_args(["--suite", "report_quality"])
        workflow = _parse_eval_args(["--suite", "agent_report_workflow"])

        assert parsed[0] == "report_quality"
        assert workflow[0] == "agent_report_workflow"
        assert ReportQualityBenchmark.name == "report_quality"
        assert AgentReportWorkflowBenchmark.name == "agent_report_workflow"

    def test_parse_eval_args_ignores_unknown_flags(self):
        from biobank_agent.cli import _parse_eval_args

        assert _parse_eval_args(["--unknown"]) == (
            "research_eval_v1",
            "baseline",
            False,
            False,
            "",
        )


class FakeCliBenchmark:
    name = "fake"


class FakeBenchmarkResult:
    def __init__(self, gate_passed=True, comparative=None, baseline=None):
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

        for name in ("AgentReportWorkflowBenchmark", "ResearchEvalV1", "ReportQualityBenchmark", "SkillSchemaBenchmark", "BiomedQABenchmark", "SkillCallBenchmark"):
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

    def test_eval_cmd_missing_baseline_file_and_unavailable_model_list(self, tmp_path, monkeypatch):
        from biobank_agent import cli
        import biobank_agent.eval.benchmarks as benchmarks_mod
        import biobank_agent.eval.harness as harness_mod

        for name in ("AgentReportWorkflowBenchmark", "ResearchEvalV1", "ReportQualityBenchmark", "SkillSchemaBenchmark", "BiomedQABenchmark", "SkillCallBenchmark"):
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
        self.available_models = ["model-a"]
        self.state = SimpleNamespace(
            figures=[],
            token_usage=SimpleNamespace(prompt_tokens=11, completion_tokens=22),
            interrupted=False,
        )
        self.messages = []
        self.run = MagicMock(side_effect=run_side_effect) if run_side_effect is not None else MagicMock(return_value="agent response")


class FakeMainPlanner:
    def __init__(self, plans_dir, *, active=False):
        self.plans_dir = plans_dir
        self.is_active = active
        self.status = "EXECUTION"
        self.current_plan = None

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

    def test_main_repl_slash_and_exit_paths(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli

        settings = FakeMainSettings(tmp_path)
        agents = []

        def fake_agent_factory(settings_arg):
            agent = FakeMainAgent(settings_arg, count_subjects=RuntimeError("no data"), fields={})
            agents.append(agent)
            return agent

        monkeypatch.setattr(cli.sys, "argv", ["biobank", "--model=override-model"])
        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        monkeypatch.setattr(cli, "Agent", fake_agent_factory)
        monkeypatch.setattr(cli, "PlanMode", lambda plans_dir: FakeMainPlanner(plans_dir))
        monkeypatch.setattr(cli, "_read_query", MagicMock(side_effect=["", "/help", "quit"]))
        monkeypatch.setenv("UKB_PARQUET_DIR", "/old")
        monkeypatch.delenv("DATA_DIR", raising=False)

        cli.main()

        assert settings.llm_model == "override-model"
        output = cli_capture_console.getvalue()
        assert "deprecated" in output
        assert "Biomarker data not found" in output
        assert "Field catalogue empty" in output
        assert "Goodbye" in output

    def test_main_repl_runs_queries_figures_interrupts_and_errors(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli

        settings = FakeMainSettings(tmp_path)
        agent = FakeMainAgent(
            settings,
            count_subjects=123,
            fields={"30740": {}},
            run_side_effect=[KeyboardInterrupt(), RuntimeError("boom"), "final response"],
        )

        def fake_agent_factory(settings_arg):
            return agent

        def fake_read_query(prompt_session, planner):
            if not hasattr(fake_read_query, "queries"):
                fake_read_query.queries = iter(["interrupt", "error", "normal", "quit"])
            return next(fake_read_query.queries)

        def fake_status(message):
            class DummyStatus:
                def __enter__(self):
                    if "Thinking" in message and agent.run.call_count == 2:
                        agent.state.figures.append(str(tmp_path / "new.png"))
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            return DummyStatus()

        monkeypatch.setattr(cli.sys, "argv", ["biobank", "--noop"])
        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        monkeypatch.setattr(cli, "Agent", fake_agent_factory)
        monkeypatch.setattr(cli, "PlanMode", lambda plans_dir: FakeMainPlanner(plans_dir, active=True))
        monkeypatch.setattr(cli, "_read_query", fake_read_query)
        monkeypatch.setattr(cli.console, "status", fake_status)

        cli.main()

        assert agent.run.call_count == 3
        assert "[PLAN MODE" in agent.run.call_args_list[0].args[0]
        output = cli_capture_console.getvalue()
        assert "Interrupted" in output
        assert "Error: boom" in output
        assert "final response" in output

    def test_main_repl_runs_plain_query_outside_plan_mode(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli

        settings = FakeMainSettings(tmp_path)
        agent = FakeMainAgent(settings, run_side_effect=["plain response"])

        monkeypatch.setattr(cli.sys, "argv", ["biobank"])
        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        monkeypatch.setattr(cli, "Agent", lambda settings_arg: agent)
        monkeypatch.setattr(cli, "PlanMode", lambda plans_dir: FakeMainPlanner(plans_dir, active=False))
        monkeypatch.setattr(cli, "_read_query", MagicMock(side_effect=["plain question", "quit"]))

        cli.main()

        agent.run.assert_called_once_with("plain question")
        assert "plain response" in cli_capture_console.getvalue()

    def test_main_repl_eof_exit(self, tmp_path, monkeypatch, cli_capture_console):
        from biobank_agent import cli

        settings = FakeMainSettings(tmp_path)
        monkeypatch.setattr(cli.sys, "argv", ["biobank"])
        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        monkeypatch.setattr(cli, "Agent", lambda settings_arg: FakeMainAgent(settings_arg))
        monkeypatch.setattr(cli, "PlanMode", lambda plans_dir: FakeMainPlanner(plans_dir))
        monkeypatch.setattr(cli, "_read_query", MagicMock(side_effect=EOFError()))

        cli.main()

        assert "Goodbye" in cli_capture_console.getvalue()


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
