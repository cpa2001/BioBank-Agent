from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from biobank_agent.cli.interactive import InteractiveShell
from biobank_agent.core.tools.approval import ApprovalPolicy, builtin_profile
from biobank_agent.runtime import (
    AgentEvent,
    AgentEventType,
    HarnessTask,
    RuntimeHarnessRunner,
    ProviderResponse,
    ToolCall,
    audit_session,
    reject_persistent_apply,
    learn_from_session,
    replay_trajectory,
    write_audit_report,
    write_learning_report,
)
from biobank_agent.runtime.harness import write_harness_report


class _Settings:
    def __init__(self, root: Path) -> None:
        self.llm_base_url = "http://fake.local/v1"
        self.llm_api_key = "fake"
        self.llm_model = "fake-model"
        self.preferred_multi_models = "fake-model,fake-model,fake-model"
        self.max_tool_rounds = 2
        self.reports_dir = root / "reports"
        self.memory_dir = root / "memory"
        self.tool_call_content_mode = "null"
        self.auto_compact_event_threshold = 120
        self.auto_compact_turn_threshold = 20

    def ensure_dirs(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)


class _FakeProvider:
    def __init__(self) -> None:
        from biobank_agent.runtime import ProviderResponse

        self.responses = [ProviderResponse(text="done", provider="fake", model="fake-model")]

    def complete(self, request):
        return self.responses.pop(0) if self.responses else type("Resp", (), {"text": "done", "tool_calls": [], "usage": {}, "provider": "fake", "model": "fake-model"})()


class _FakeLegacyAgent:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.state = SimpleNamespace(
            custom_data={},
            current_report_dir=None,
            figures=[],
            records=[],
            cohorts={},
            models={},
            model_metadata={},
        )
        self.memory = SimpleNamespace()
        self.orchestrator = SimpleNamespace(dispatch_subagent=lambda *args, **kwargs: SimpleNamespace(text="ok", model="planner-model"))

    def _build_ctx(self, report_dir):
        return SimpleNamespace(settings=self.settings, state=self.state, report_dir=report_dir, memory=self.memory)


class _FakeTool:
    @property
    def name(self):
        return "demo_tool"

    def spec(self):
        from biobank_agent.core.tools.protocol import ToolSpec

        return ToolSpec(name="demo_tool", description="demo", parameters={})

    def required_capabilities(self):
        from biobank_agent.core.tools.protocol import Capability

        return frozenset({Capability.READ_DATA})

    @property
    def is_mutating(self):
        return False

    async def handle(self, ctx):
        return {"ok": True}


def _shell(tmp_path: Path):
    output = StringIO()
    shell = InteractiveShell(settings=_Settings(tmp_path), console=Console(file=output, force_terminal=False, width=120))
    shell.legacy_agent_factory = _FakeLegacyAgent
    shell.initialize()
    shell.runtime.provider_router.providers = {"fake-model": _FakeProvider()}
    shell.runtime.tool_registry.register(_FakeTool())
    shell.legacy_agent.orchestrator = SimpleNamespace(dispatch_subagent=lambda *args, **kwargs: SimpleNamespace(text="subagent ok", model="planner-model"))
    return shell, output


def test_audit_report_serializes_runtime_trajectory(tmp_path):
    shell, _ = _shell(tmp_path)
    shell.handle_line("/goal Track audit")
    shell.handle_line("/plan build audit")
    shell.handle_line("plain user turn")

    report = audit_session(shell.runtime, shell.session)
    artifacts = write_audit_report(report, tmp_path / "audit")

    assert report.session_id == shell.session.session_id
    assert report.task_summary["turns"] >= 1
    assert report.event_type_counts["command_started"] >= 2
    assert "session" in report.to_dict()["task_summary"]["session_file"]
    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["markdown"]).exists()
    assert "Timeline" in Path(artifacts["markdown"]).read_text(encoding="utf-8")


def test_replay_trajectory_validates_event_order_and_file_exists(tmp_path):
    shell, _ = _shell(tmp_path)
    shell.handle_line("/goal Replay check")
    shell.handle_line("user text")

    trajectory = shell.runtime.session_store.rollout_file(shell.session.session_id)
    report = replay_trajectory(trajectory)

    assert report.status == "ok"
    assert report.session_id == shell.session.session_id
    assert report.event_count > 0
    assert report.model_calls == 0


def test_replay_command_accepts_direct_trajectory_path(tmp_path):
    shell, output = _shell(tmp_path)
    shell.handle_line("/goal Replay direct file")
    trajectory = shell.runtime.session_store.rollout_file(shell.session.session_id)

    shell.handle_line(f"/replay {trajectory}")

    assert "Replay" in output.getvalue()
    assert any(event["type"] == "plan_phase" and event["payload"].get("phase") == "Replay" for event in shell.session.events)


def test_harness_runner_executes_through_runtime_and_writes_report(tmp_path):
    shell, _ = _shell(tmp_path)
    task = HarnessTask.from_dict(
        {
            "task_id": "runtime-harness",
            "slash_commands": ["/status"],
            "user_inputs": ["hello runtime"],
            "fake_provider_script": [{"text": "runtime answer"}],
            "fake_tool_responses": {"demo_tool": {"result": {"ok": True}, "capabilities": ["read_data"]}},
            "expected": {
                "event_types": ["session_started", "user_turn_started", "user_turn_completed", "session_saved"],
                "min_turns": 1,
                "min_events": 1,
                "provider_roles": ["primary_executor"],
                "require_replay_ok": True,
                "require_session_saved": True,
            },
        }
    )
    runner = RuntimeHarnessRunner(shell)
    report = runner.run(task, output_dir=tmp_path / "harness")
    artifacts = write_harness_report(report, tmp_path / "harness_write")

    assert report.task_id == "runtime-harness"
    assert report.status in {"passed", "failed"}
    assert Path(report.trajectory_path).exists()
    assert Path(artifacts["json"]).exists()


def test_learning_report_links_trajectory_feedback_and_writes_artifacts(tmp_path):
    shell, _ = _shell(tmp_path)
    shell.handle_line("/goal Learn from session")
    shell.handle_line("plain text")

    report = learn_from_session(shell.runtime, shell.session)
    artifacts = write_learning_report(report, tmp_path / "learning")

    assert report.session_id == shell.session.session_id
    assert report.proposals
    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["markdown"]).exists()


def test_report_helpers_redact_secret_like_values():
    payload = {
        "api_key": "sk-test-123456789012",
        "nested": {"token": "rk-abcdef1234567890"},
        "note": "value sk-test-123456789012 remains hidden",
    }
    redacted = json.loads(json.dumps(payload))
    from biobank_agent.runtime.audit import redact_secrets

    redacted = redact_secrets(redacted)
    assert redacted["api_key"] == "<REDACTED>"
    assert redacted["nested"]["token"] == "<REDACTED>"
    assert "<REDACTED>" in redacted["note"]


def test_session_contains_action_graph_and_trajectory_records_after_tool_call(tmp_path):
    shell, _ = _shell(tmp_path)
    shell.runtime.provider_router.providers["fake-model"].responses = [
        ProviderResponse(
            text="",
            provider="fake",
            model="fake-model",
            tool_calls=[ToolCall(id="call_demo", name="demo_tool", args={})],
        ),
        ProviderResponse(text="tool complete", provider="fake", model="fake-model"),
    ]
    shell.handle_line("use a tool")

    assert shell.session.action_graph_refs
    assert shell.legacy_agent.state.trajectory.records
    replay = replay_trajectory(shell.runtime.session_store.rollout_file(shell.session.session_id))
    assert replay.status == "ok"


def test_runtime_audit_reports_replay_and_compaction_status(tmp_path):
    shell, _ = _shell(tmp_path)
    shell.settings.auto_compact_event_threshold = 1
    shell.handle_line("trigger compaction")
    report = audit_session(shell.runtime, shell.session)

    assert report.compactions
    assert report.replay_status["status"] == "ok"


def test_approval_policy_and_memory_like_mutating_tools_escalate():
    from biobank_agent.core.tools.protocol import ToolSpec
    from biobank_agent.core.tools.approval import Decision

    class MutatingTool:
        name = "mutator"

        def spec(self):
            return ToolSpec(name="mutator", description="mutates", parameters={})

        def required_capabilities(self):
            from biobank_agent.core.tools.protocol import Capability

            return frozenset({Capability.MUTATE_MEMORY})

        @property
        def is_mutating(self):
            return True

    policy = ApprovalPolicy(builtin_profile("read_only"))
    outcome = policy.decide(MutatingTool(), {})
    assert outcome.decision in {Decision.DENY, Decision.ASK_USER}


def test_reject_persistent_evolution_apply_is_review_only():
    out = reject_persistent_apply("no persistent apply from shell")
    assert out["status"] == "rejected"
    assert out["approval_required"] is True


def test_harness_and_learning_outputs_do_not_rely_on_fixed_fixture_tokens():
    production_files = list(Path("biobank_agent").rglob("*.py"))
    forbidden = ("runtime-harness", "hello runtime", "Track audit", "Learn from session")
    hits = []
    for path in production_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                hits.append((str(path), token))
    assert hits == []
