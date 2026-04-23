"""CLI command tests — verify all slash commands in _handle_command."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
