"""Tests for v3 CLI package modules and SDK entrypoints."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from biobank_agent.cli.commands.registry import build_core_registry
from biobank_agent.cli.live import StreamingRenderer
from biobank_agent.cli.tui import textual_available
from biobank_agent.core.events import AgentEvent, AgentEventType
from biobank_agent.core.tools.protocol import Capability, ToolContext, ToolSpec
from biobank_agent.registry import SkillRegistry


def test_cli_package_exposes_importable_command_modules():
    import biobank_agent.cli_commands.registry as compat_registry
    import biobank_agent.cli.commands.registry as native_registry

    registry = build_core_registry()

    assert native_registry.build_core_registry.__module__ == "biobank_agent.cli.commands.registry"
    assert compat_registry.build_core_registry is native_registry.build_core_registry
    assert "/status" in registry
    assert registry["/status"].usage == "/status"
    for command in (
        "/goal",
        "/resume",
        "/new",
        "/fork",
        "/diff",
        "/permissions",
        "/mcp",
        "/agent",
        "/subagents",
        "/review",
        "/doctor",
        "/audit",
        "/harness",
        "/learn",
        "/verify",
        "/quit",
        "/plan",
        "/plan-approve",
        "/plan-edit",
        "/plan-reject",
        "/plan-resume",
        "/plan-diagnose",
        "/plan-retry",
        "/plan-use",
        "/plan-option",
        "/tools",
        "/mcp-list",
        "/mcp-start",
        "/mcp-health",
        "/mcp-call",
        "/mcp-stop",
        "/replay",
        "/evolve",
        "/record",
        "/pipelines",
        "/errors",
        "/memory",
        "/debate",
    ):
        assert command in registry
    from biobank_agent.cli.__main__ import main

    assert callable(main)


def test_command_registry_discovers_split_builtin_modules():
    from biobank_agent.cli.commands import mcp, memory, plan, reproducibility, research, runtime, session
    from biobank_agent.cli.commands.registry import BUILTIN_COMMAND_MODULES, iter_registered_commands

    assert BUILTIN_COMMAND_MODULES == ("session", "runtime", "plan", "research", "mcp", "reproducibility", "memory")
    modules = [session, runtime, plan, research, mcp, reproducibility, memory]
    expected = {command.name for module in modules for command in module.commands()}
    discovered = {command.name for command in iter_registered_commands()}

    assert expected == discovered
    assert len(discovered) == len(iter_registered_commands())


def test_command_registry_discovers_env_plugin_modules(monkeypatch):
    from biobank_agent.cli.commands.base import CommandContext, RegisteredCommand
    from biobank_agent.cli.commands.registry import EXTRA_COMMAND_MODULES_ENV, build_core_registry

    module = types.ModuleType("biobank_test_plugin_commands")

    def plugin_commands():
        def handle(ctx: CommandContext, arg: str) -> None:
            ctx.action("plugin_demo")(arg)

        return [RegisteredCommand("/plugin-demo", "/plugin-demo <arg>", "Plugin demo command", handle)]

    module.commands = plugin_commands
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv(EXTRA_COMMAND_MODULES_ENV, module.__name__)

    registry = build_core_registry()

    assert "/plugin-demo" in registry
    assert registry["/plugin-demo"].usage == "/plugin-demo <arg>"


def test_streaming_renderer_tracks_tool_events():
    renderer = StreamingRenderer()
    event = AgentEvent.make(
        AgentEventType.TOOL_RESULT,
        skill="train_model",
        summary={"auc": 0.8},
    )

    renderer.handle(event)

    assert renderer.recent[-1][0] == "train_model"
    assert renderer.recent[-1][1] == "done"


def test_textual_availability_probe_returns_bool():
    assert isinstance(textual_available(), bool)


@pytest.mark.asyncio
async def test_confirm_modal_defaults_and_builds_auto_merger_callback():
    from biobank_agent.cli.tui.confirm_modal import (
        ConfirmationModal,
        ConfirmationRequest,
        auto_merger_confirm_fn,
        confirm,
        confirm_with_app,
        _patch_review_body,
    )

    request = ConfirmationRequest(title="Approve", body="Body")
    assert await confirm(request, default=False) is False
    assert ConfirmationModal(request).request is request

    class FakeApp:
        def __init__(self):
            self.screens = []

        async def push_screen_wait(self, screen):
            self.screens.append(screen)
            return True

    app = FakeApp()
    approved = await confirm_with_app(app, request, default=False)
    assert approved is True
    assert app.screens[-1].request.title == "Approve"

    assessment = type("Assessment", (), {"risk": type("Risk", (), {"value": "medium"})(), "reason": "logic change"})()
    patch = type(
        "Patch",
        (),
        {
            "target_skill": "demo",
            "target_path": "custom_skills/demo.py",
            "unified_diff": "--- a/custom_skills/demo.py\n+++ b/custom_skills/demo.py\n@@\n-if x:\n+if y:\n",
            "metadata": {"eval_summary": "ALWAYS_PASSES eval gate passed"},
        },
    )()
    body = _patch_review_body(assessment, patch)
    assert "Diff preview" in body
    assert "ALWAYS_PASSES eval gate passed" in body
    assert "+if y:" in body
    callback = auto_merger_confirm_fn(app)
    assert await callback(assessment, patch) is True
    assert app.screens[-1].request.confirm_label == "Approve patch"
    assert "Diff preview" in app.screens[-1].request.body


def test_tui_panels_consume_agent_events_without_textual_runtime():
    from biobank_agent.cli.tui.disclosure_panel import DisclosurePanel
    from biobank_agent.cli.tui.main_screen import BiobankTuiApp
    from biobank_agent.cli.tui.progress_panel import ProgressPanel
    from biobank_agent.cli.tui.run_detail import RunDetailPanel
    from biobank_agent.cli.tui.tool_status import ToolStatusPanel
    from biobank_agent.cli.tui.worker_review import WorkerReviewPanel

    progress = ProgressPanel()
    progress.apply_event(AgentEvent.make(AgentEventType.TURN_STARTED, turn_id="turn_1"))
    progress.apply_event(AgentEvent.make(
        AgentEventType.PLAN_PHASE,
        phase="External council",
        actor="codex",
        status="running",
        message="planning",
    ))
    progress.apply_event(AgentEvent.make(AgentEventType.TOOL_STARTED, skill="deep_research"))
    progress.apply_event(AgentEvent.make(
        AgentEventType.TOOL_PROGRESS,
        skill="deep_research",
        message="query 2/5",
    ))
    progress.apply_event(AgentEvent.make(AgentEventType.TOOL_RESULT, skill="deep_research"))
    progress_text = progress.render_text()
    assert "External council" in progress.render_text()
    assert "planning" in progress_text
    assert "Run dashboard" in progress_text
    assert "tools: 1 done, 0 failed, 0 active" in progress_text
    assert "Current: Tool completed: deep_research" in progress_text
    assert "Execution | deep_research | running | query 2/5" in progress_text

    tools = ToolStatusPanel()
    tools.apply_event(AgentEvent.make(AgentEventType.TOOL_STARTED, skill="train_model"))
    tools.apply_event(AgentEvent.make(
        AgentEventType.TOOL_PROGRESS,
        skill="train_model",
        message="candidate 1/5",
    ))
    tools.apply_event(AgentEvent.make(AgentEventType.TOOL_RESULT, skill="train_model"))
    text = tools.render_text()
    assert "candidate 1/5" in text
    assert "train_model: done" in text

    disclosure = DisclosurePanel()
    disclosure.apply_event(AgentEvent.make(
        AgentEventType.SCHEMA_VIOLATION,
        skill="generate_report",
        reason="invalid arg sections",
    ))
    assert "schema gate" in disclosure.render_text()

    detail = RunDetailPanel()
    detail.apply_event(AgentEvent.make(
        AgentEventType.PLAN_PHASE,
        phase="External council",
        actor="codex",
        status="running",
        message="planning trajectory branch",
    ))
    detail.apply_event(AgentEvent.make(
        AgentEventType.PLAN_PHASE,
        phase="Repair",
        actor="biobank",
        status="failed",
        message="plan stalled with blocked steps",
        metadata={
            "pause_reason": "Stall: s14 waiting for s12",
            "choices": [
                {"key": "A", "title": "Auto-repair blocked dependencies", "detail": "Update the graph and resume."},
                {"key": "N", "title": "Abort", "detail": "Stop execution."},
            ],
        },
    ))
    detail.apply_event(AgentEvent.make(
        AgentEventType.PLAN_PHASE,
        phase="Report",
        actor="biobank",
        status="success",
        message="all required plan steps completed",
        metadata={
            "report_dir": "/tmp/biobank-report",
            "report_artifacts": [
                "/tmp/biobank-report/report_technical.md",
                "/tmp/biobank-report/report_nature.md",
            ],
        },
    ))
    detail_text = detail.render_text()
    assert "codex: running: planning trajectory branch" in detail_text
    assert "Pause: Stall: s14 waiting for s12" in detail_text
    assert "A: Auto-repair blocked dependencies" in detail_text
    assert "dir: /tmp/biobank-report" in detail_text
    assert "report_technical.md" in detail_text

    worker_review = WorkerReviewPanel()
    worker_review.apply_event(AgentEvent.make(
        AgentEventType.PLAN_PHASE,
        phase="External council",
        actor="codex",
        status="running",
        message="planning trajectory branch",
        metadata={"worker_id": "codex-01", "case": "codex plan"},
    ))
    worker_review.apply_event(AgentEvent.make(
        AgentEventType.PLAN_PHASE,
        phase="Repair",
        actor="biobank",
        status="paused",
        message="plan stalled",
        metadata={
            "worker_id": "worker-06",
            "case": "trajectory feasibility",
            "choices": [{"key": "A", "title": "Auto-repair blocked dependencies"}],
            "transcript": "/tmp/worker-06/transcript.clean.log",
        },
    ))
    worker_review.select_next()
    worker_review_text = worker_review.render_text()
    assert "codex plan: running" in worker_review_text
    assert "trajectory feasibility: paused" in worker_review_text
    assert "A: Auto-repair blocked dependencies" in worker_review_text
    assert "transcript.clean.log" in worker_review_text

    app = BiobankTuiApp(settings=object())
    app.handle_agent_event(AgentEvent.make(AgentEventType.TURN_STARTED))
    assert "started" in app.progress_panel.render_text()
    app.handle_agent_event(AgentEvent.make(
        AgentEventType.PLAN_PHASE,
        phase="Report",
        actor="biobank",
        status="success",
        message="complete",
        metadata={"report_dir": "/tmp/final-report"},
    ))
    assert "/tmp/final-report" in app.run_detail_panel.render_text()
    assert "biobank: success" in app.worker_review_panel.render_text()


def test_tui_app_handles_text_with_injected_handler_without_textual_runtime():
    from biobank_agent.cli.tui.main_screen import BiobankTuiApp

    calls = []

    async def handler(text: str) -> str:
        calls.append(text)
        return "handled"

    app = BiobankTuiApp(settings=object(), command_handler=handler)
    output = asyncio.run(app.handle_text_async("/status"))

    assert calls == ["/status"]
    assert output == "handled"
    assert "biobank > /status" in "\n".join(app.log_lines)
    assert "handled" in "\n".join(app.log_lines)


def test_tui_review_shortcuts_dispatch_plan_commands_without_typing_slashes():
    from biobank_agent.cli.tui.main_screen import BiobankTuiApp

    calls = []

    async def handler(text: str) -> str:
        calls.append(text)
        return f"handled {text}"

    async def run() -> BiobankTuiApp:
        app = BiobankTuiApp(settings=object(), command_handler=handler)
        await app.action_approve_plan()
        await app.action_resume_plan()
        await app.action_repair_a()
        await app.action_repair_b()
        await app.action_repair_n()
        app.worker_review_panel.apply_event(AgentEvent.make(
            AgentEventType.PLAN_PHASE,
            phase="Execution",
            actor="worker",
            status="running",
            message="one",
            metadata={"worker_id": "w1", "case": "case one"},
        ))
        app.worker_review_panel.apply_event(AgentEvent.make(
            AgentEventType.PLAN_PHASE,
            phase="Execution",
            actor="worker",
            status="running",
            message="two",
            metadata={"worker_id": "w2", "case": "case two"},
        ))
        app.action_next_worker()
        app.action_previous_worker()
        return app

    app = asyncio.run(run())

    assert calls == ["/plan-approve", "/plan-resume", "/plan-option A", "/plan-option B", "/plan-option N"]
    text = app.worker_review_panel.render_text()
    assert "/plan-approve: done" in text
    assert "case one" in text
    assert "case two" in text


def test_tui_streams_registry_plan_events_and_console_output(monkeypatch):
    from biobank_agent.cli.tui.main_screen import BiobankTuiApp
    import biobank_agent.cli_legacy as legacy_cli

    class FakeState:
        def __init__(self):
            self.custom_data = {}

    class FakeAgent:
        def __init__(self):
            self.state = FakeState()

    def fake_dispatch_line(text, agent, planner, token_usage):
        assert text == "/plan demo"
        assert callable(agent.state.custom_data.get("tui_event_callback"))
        assert callable(agent.state.custom_data.get("tui_confirm_fn"))
        legacy_cli._emit_tui_plan_event(
            agent,
            "Planning",
            "biobank",
            "running",
            "decomposing goal",
            {"source": "test"},
        )
        legacy_cli.console.print("Plan generated with 1 step.")

    monkeypatch.setattr(legacy_cli, "_handle_command", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("_handle_command should not be used")))
    monkeypatch.setattr(legacy_cli, "_dispatch_registered_command_line", fake_dispatch_line)

    app = BiobankTuiApp(settings=object())
    app.agent = FakeAgent()
    app.planner = object()

    output = app._run_legacy_command_capture("/plan demo", stream_to_log=True)

    assert output == ""
    assert "Planning | biobank | running | decomposing goal" in app.progress_panel.render_text()
    assert "Plan generated with 1 step." in "\n".join(app.log_lines)
    assert "tui_event_callback" not in app.agent.state.custom_data
    assert "tui_confirm_fn" not in app.agent.state.custom_data


def test_tui_dispatches_env_plugin_command_through_registry(monkeypatch):
    from biobank_agent.cli.commands.base import CommandContext, RegisteredCommand
    from biobank_agent.cli.commands.registry import EXTRA_COMMAND_MODULES_ENV
    from biobank_agent.cli.tui.main_screen import BiobankTuiApp

    module = types.ModuleType("biobank_tui_plugin_commands")

    def plugin_commands():
        def handle(ctx: CommandContext, arg: str) -> None:
            ctx.console.print(f"plugin handled {arg}")

        return [RegisteredCommand("/tui-plugin", "/tui-plugin <arg>", "TUI plugin demo", handle)]

    module.commands = plugin_commands
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv(EXTRA_COMMAND_MODULES_ENV, module.__name__)

    class FakeState:
        def __init__(self):
            self.custom_data = {}

    app = BiobankTuiApp(settings=object())
    app.agent = type("FakeAgent", (), {"state": FakeState()})()
    app.planner = object()

    output = app._run_registry_command_capture("/tui-plugin ok")

    assert "plugin handled ok" in output


def test_tui_app_falls_back_when_async_runtime_is_unsafe(monkeypatch):
    from biobank_agent.cli.tui.main_screen import BiobankTuiApp

    class FakeAsyncAgent:
        def __init__(self, agent):
            self.agent = agent

        @staticmethod
        def is_safe_for_async(agent):
            return False, "multi-model active"

    class FakeAgent:
        def run(self, text):
            return f"sync:{text}"

    import biobank_agent.core.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "AsyncAgent", FakeAsyncAgent)
    app = BiobankTuiApp(settings=object())
    app.agent = FakeAgent()

    output = asyncio.run(app._run_agent_stream("hello"))

    assert output == "sync:hello"
    assert "Streaming disabled for this turn: multi-model active" in "\n".join(app.log_lines)


def test_tui_textual_runtime_focuses_input_and_dispatches_command():
    if not textual_available():
        pytest.skip("Textual optional dependency is not installed")

    from biobank_agent.cli.tui.main_screen import BiobankTuiApp

    calls = []

    async def handler(text: str) -> str:
        calls.append(text)
        return f"handled:{text}"

    async def run() -> BiobankTuiApp:
        app = BiobankTuiApp(settings=object(), command_handler=handler)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            await pilot.press("/", "s", "t", "a", "t", "u", "s", "enter")
            await pilot.pause(0.2)
        return app

    app = asyncio.run(run())

    assert calls == ["/status"]
    assert "biobank > /status" in "\n".join(app.log_lines)
    assert "handled:/status" in "\n".join(app.log_lines)


def test_tui_textual_runtime_streams_registry_plan_command_events(monkeypatch):
    if not textual_available():
        pytest.skip("Textual optional dependency is not installed")

    from biobank_agent.cli.tui.main_screen import BiobankTuiApp
    import biobank_agent.cli_legacy as legacy_cli

    class FakeState:
        def __init__(self):
            self.custom_data = {}

    class FakeAgent:
        def __init__(self):
            self.state = FakeState()

    def fake_ensure_runtime(self):
        self.agent = FakeAgent()
        self.planner = object()

    def fake_dispatch_line(text, agent, planner, token_usage):
        assert text == "/plan-approve"
        legacy_cli._emit_tui_plan_event(
            agent,
            "Execution",
            "biobank",
            "running",
            "executing approved plan",
        )
        legacy_cli._emit_tui_plan_event(
            agent,
            "Report",
            "generate_report",
            "success",
            "dual report artifacts verified",
        )
        legacy_cli.console.print("Plan execution complete.")

    monkeypatch.setattr(BiobankTuiApp, "_ensure_runtime", fake_ensure_runtime)
    monkeypatch.setattr(legacy_cli, "_handle_command", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("_handle_command should not be used")))
    monkeypatch.setattr(legacy_cli, "_dispatch_registered_command_line", fake_dispatch_line)

    async def run() -> BiobankTuiApp:
        app = BiobankTuiApp(settings=object())
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            await pilot.press("/", "p", "l", "a", "n", "-", "a", "p", "p", "r", "o", "v", "e", "enter")
            await pilot.pause(0.5)
        return app

    app = asyncio.run(run())
    progress_text = app.progress_panel.render_text()

    assert "Execution | biobank | running | executing approved plan" in progress_text
    assert "Report | generate_report | success | dual report artifacts verified" in progress_text
    assert "Plan execution complete." in "\n".join(app.log_lines)


def test_tui_textual_runtime_plan_review_shortcut_keybinding():
    if not textual_available():
        pytest.skip("Textual optional dependency is not installed")

    from biobank_agent.cli.tui.main_screen import BiobankTuiApp

    calls = []

    async def handler(text: str) -> str:
        calls.append(text)
        return f"handled:{text}"

    async def run() -> BiobankTuiApp:
        app = BiobankTuiApp(settings=object(), command_handler=handler)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            await pilot.press("f5")
            await pilot.pause(0.2)
        return app

    app = asyncio.run(run())

    assert calls == ["/plan-approve"]
    assert "/plan-approve: done" in app.worker_review_panel.render_text()


def test_tui_textual_runtime_exports_nonblank_large_worker_screenshot(tmp_path):
    if not textual_available():
        pytest.skip("Textual optional dependency is not installed")

    from biobank_agent.cli.tui.main_screen import BiobankTuiApp

    async def run() -> tuple[BiobankTuiApp, str]:
        app = BiobankTuiApp(settings=object(), command_handler=lambda text: f"handled:{text}")
        async with app.run_test() as pilot:
            await pilot.resize_terminal(150, 48)
            await pilot.pause(0.1)
            for idx in range(14):
                app.handle_agent_event(AgentEvent.make(
                    AgentEventType.PLAN_PHASE,
                    phase="Execution" if idx % 2 else "External council",
                    actor="codex" if idx % 2 else "biobank",
                    status="running" if idx < 13 else "paused",
                    message=f"case-{idx:02d} processing long repair history",
                    metadata={
                        "worker_id": f"worker-{idx:02d}",
                        "case": f"case-{idx:02d}",
                        "transcript": f"/tmp/worker-{idx:02d}/transcript.clean.log",
                        "report_dir": f"/tmp/worker-{idx:02d}/reports/final",
                        "report_artifacts": [
                            "report.md",
                            "report_technical.md",
                            "report_nature.md",
                        ],
                        "choices": [
                            {"key": "A", "title": "Auto-repair blocked dependencies", "detail": "Update graph and resume."},
                            {"key": "B", "title": "Ask Codex/Claude", "detail": "Run external review hooks."},
                        ],
                    },
                ))
            app.handle_agent_event(AgentEvent.make(
                AgentEventType.TOOL_PROGRESS,
                skill="train_model",
                message="candidate 3/5 evaluating calibration",
                metadata={"worker_id": "worker-13", "case": "case-13"},
            ))
            await pilot.press("ctrl+j")
            await pilot.pause(0.1)
            return app, app.export_screenshot(title="Biobank TUI Stress", simplify=True)

    app, screenshot = asyncio.run(run())
    path = tmp_path / "biobank_tui_stress.svg"
    path.write_text(screenshot, encoding="utf-8")

    assert path.stat().st_size > 2000
    assert "<svg" in screenshot
    assert screenshot.count("<text") > 20
    assert "clip-path" in screenshot
    assert "Run dashboard" in app.progress_panel.render_text()
    assert "Report artifacts" in app.run_detail_panel.render_text()
    text = app.worker_review_panel.render_text()
    assert "case-13: running" in text
    assert "tool=train_model" in text
    assert "worker-00" not in text
    assert "Auto-repair blocked dependencies" in text


def test_core_memory_and_planning_facades_are_importable(tmp_path):
    from biobank_agent.core.memory import MemoryInjector, RolloutWriter
    from biobank_agent.core.planning import PlanExecutor, PlanMode, StudySpecCompiler

    assert PlanExecutor is not None
    assert PlanMode is not None
    assert StudySpecCompiler is not None

    (tmp_path / "README.md").write_text("project memory", encoding="utf-8")
    blocks = MemoryInjector().collect(tmp_path)
    assert blocks and blocks[0].scope == "project"

    rollout = tmp_path / "rollout.jsonl"
    RolloutWriter(rollout).append({"event": "ok"})
    assert '"event": "ok"' in rollout.read_text(encoding="utf-8")


def test_legacy_cli_dispatches_mcp_commands(monkeypatch):
    import biobank_agent.cli as cli

    calls = []
    agent = type("Agent", (), {"state": type("State", (), {"custom_data": {}})()})()
    planner = object()

    monkeypatch.setattr(cli, "_show_mcp_status", lambda agent_arg: calls.append(("list", agent_arg)))
    monkeypatch.setattr(cli, "_start_mcp", lambda agent_arg: calls.append(("start", agent_arg)))
    monkeypatch.setattr(cli, "_mcp_health", lambda agent_arg, raw_arg="": calls.append(("health", agent_arg, raw_arg)))
    monkeypatch.setattr(cli, "_call_mcp_tool", lambda agent_arg, raw_arg: calls.append(("call", agent_arg, raw_arg)))
    monkeypatch.setattr(cli, "_stop_mcp", lambda agent_arg: calls.append(("stop", agent_arg)))
    monkeypatch.setattr(cli, "_show_evolution_status", lambda agent_arg, raw_arg: calls.append(("evolve", agent_arg, raw_arg)))

    cli._handle_command("/mcp-list", agent, planner, {})
    cli._handle_command("/mcp-start", agent, planner, {})
    cli._handle_command("/mcp-health --repair", agent, planner, {})
    cli._handle_command('/mcp-call demo__echo {"text":"ok"}', agent, planner, {})
    cli._handle_command("/mcp-stop", agent, planner, {})
    cli._handle_command("/evolve --dry-run", agent, planner, {})

    assert [call[0] for call in calls] == ["list", "start", "health", "call", "stop", "evolve"]
    assert calls[2][2] == "--repair"
    assert calls[3][2] == 'demo__echo {"text":"ok"}'
    assert calls[-1][2] == "--dry-run"


def test_legacy_cli_dispatches_plan_commands_through_registry(monkeypatch):
    import biobank_agent.cli as cli

    calls = []
    agent = type("Agent", (), {"state": type("State", (), {"custom_data": {}})()})()
    planner = object()

    monkeypatch.setattr(cli, "_cmd_plan", lambda agent_arg, planner_arg, raw_arg: calls.append(("plan", raw_arg)))
    monkeypatch.setattr(cli, "_cmd_plan_approve", lambda agent_arg, planner_arg, usage: calls.append(("approve", dict(usage))))
    monkeypatch.setattr(cli, "_cmd_plan_edit", lambda agent_arg, planner_arg, raw_arg: calls.append(("edit", raw_arg)))
    monkeypatch.setattr(cli, "_cmd_plan_resume", lambda agent_arg, planner_arg, usage: calls.append(("resume", dict(usage))))
    monkeypatch.setattr(cli, "_cmd_plan_option", lambda agent_arg, planner_arg, raw_arg, usage: calls.append(("option", raw_arg)))

    cli._handle_command("/plan run E11", agent, planner, {"prompt_tokens": 1})
    cli._handle_command("/plan-approve", agent, planner, {"prompt_tokens": 2})
    cli._handle_command("/plan-edit add calibration", agent, planner, {})
    cli._handle_command("/plan-resume", agent, planner, {})
    cli._handle_command("/plan-option A", agent, planner, {})

    assert calls == [
        ("plan", "run E11"),
        ("approve", {"prompt_tokens": 2}),
        ("edit", "add calibration"),
        ("resume", {}),
        ("option", "A"),
    ]


def test_legacy_cli_mcp_manager_uses_settings_config_path(tmp_path):
    import biobank_agent.cli as cli

    cfg_path = tmp_path / "project_mcp.json"
    cfg_path.write_text('{"servers":[]}', encoding="utf-8")
    state = type("State", (), {"custom_data": {}})()
    agent = type(
        "Agent",
        (),
        {
            "state": state,
            "settings": type("Settings", (), {"mcp_config_path": str(cfg_path)})(),
            "registry": SkillRegistry(),
        },
    )()

    manager, _registry = cli._get_mcp_manager(agent)

    assert manager.config_path == cfg_path


def test_legacy_cli_dispatches_replay_through_registry(monkeypatch):
    import biobank_agent.cli as cli

    calls = []
    agent = type("Agent", (), {"state": type("State", (), {"custom_data": {}})()})()
    planner = object()

    monkeypatch.setattr(cli, "_replay_provenance", lambda agent_arg, raw_arg: calls.append((agent_arg, raw_arg)))

    cli._handle_command("/replay abc123 --dry-run", agent, planner, {})

    assert calls == [(agent, "abc123 --dry-run")]


def test_replicate_paper_loads_review_plan_for_approval(tmp_path):
    import biobank_agent.cli as cli
    from biobank_agent.planner import PlanMode, PlanState

    schema = {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Think",
            "parameters": {
                "type": "object",
                "properties": {"reasoning": {"type": "string"}},
                "required": ["reasoning"],
            },
        },
    }
    planner = PlanMode(
        plans_dir=tmp_path / "plans",
        available_skills=["think"],
        tool_schemas=[schema],
    )
    state = type("State", (), {"current_report_dir": tmp_path / "reports"})()
    agent = type(
        "Agent",
        (),
        {
            "state": state,
            "settings": type("Settings", (), {"plans_dir": tmp_path / "plans"})(),
        },
    )()
    result = {
        "plan": [
            {
                "id": "s1",
                "skill": "think",
                "args": {"reasoning": "replicate safely"},
                "description": "Review replication",
                "depends_on": [],
            }
        ]
    }

    cli._load_replication_plan_for_review(agent, planner, "paper", result)

    assert planner.state == PlanState.REVIEW
    assert planner.plan is not None
    assert planner.plan.steps[0].skill == "think"
    assert planner.validation_issues == []
    assert (tmp_path / "plans" / ".plan_checkpoint.json").exists()


def test_mcp_legacy_bridge_makes_remote_tools_callable():
    import biobank_agent.cli as cli

    class FakeHandler:
        name = "mcp_demo__echo"

        def spec(self):
            return ToolSpec(
                name=self.name,
                description="Echo MCP args",
                parameters={"text": {"type": "string"}},
                required=["text"],
            )

        def required_capabilities(self):
            return frozenset({Capability.NETWORK})

        async def handle(self, ctx: ToolContext):
            return {"echo": ctx.args["text"], "tool": ctx.name}

    agent = type(
        "Agent",
        (),
        {
            "registry": SkillRegistry(),
            "settings": object(),
            "dm": type("DM", (), {"conn": None})(),
            "catalog": None,
            "memory": None,
            "state": type("State", (), {"custom_data": {}})(),
        },
    )()

    assert cli._register_mcp_handlers_in_legacy(agent, [FakeHandler()]) == 1
    assert "mcp_demo__echo" in agent.registry
    assert agent.registry.execute("mcp_demo__echo", {"text": "ok"}) == {
        "echo": "ok",
        "tool": "mcp_demo__echo",
    }

    cli._unregister_mcp_legacy_tools(agent)
    assert "mcp_demo__echo" not in agent.registry


def test_mcp_call_command_executes_loaded_tool_and_records_result(tmp_path):
    import biobank_agent.cli as cli

    class FakeHandler:
        name = "mcp_demo__echo"

        def spec(self):
            return ToolSpec(
                name=self.name,
                description="Echo MCP args",
                parameters={"text": {"type": "string"}},
                required=["text"],
            )

        def required_capabilities(self):
            return frozenset({Capability.NETWORK})

        async def handle(self, ctx: ToolContext):
            return {"echo": ctx.args["text"], "tool": ctx.name}

    state = type("State", (), {"custom_data": {}, "records": []})()
    agent = type(
        "Agent",
        (),
        {
            "registry": SkillRegistry(),
            "settings": type("Settings", (), {"reports_dir": tmp_path})(),
            "dm": type("DM", (), {"conn": None})(),
            "catalog": None,
            "memory": None,
            "state": state,
        },
    )()

    assert cli._register_mcp_handlers_in_legacy(agent, [FakeHandler()]) == 1
    cli._call_mcp_tool(agent, 'demo__echo {"text":"ok"}')

    assert state.records[-1]["skill"] == "mcp_demo__echo"
    assert state.records[-1]["result"] == {"echo": "ok", "tool": "mcp_demo__echo"}


def test_external_planning_council_is_always_disabled():
    import biobank_agent.cli as cli

    settings = type(
        "Settings",
        (),
        {
            "plan_external_council_enabled": True,
            "plan_external_council_policy": "always",
        },
    )()

    assert cli._should_run_external_planning_council(settings, "run a short E11 report") is False
    assert cli._should_run_external_planning_council(settings, "any goal whatsoever") is False
