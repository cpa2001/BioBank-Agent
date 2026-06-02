"""Tests for the default runtime-backed interactive CLI."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from biobank_agent.cli.commands import CommandContext
from biobank_agent.cli.commands.registry import SlashCommandRegistry, build_core_registry
from biobank_agent.cli.interactive import InteractiveShell
from biobank_agent.core.events import AgentEvent, AgentEventType
from biobank_agent.runtime import PlanState, PlanStatus, PlanStep, ProviderResponse, ToolCall


class _Settings:
    def __init__(self, root: Path) -> None:
        self.llm_base_url = "http://fake.local/v1"
        self.llm_api_key = "fake"
        self.llm_model = "fake-model"
        self.preferred_multi_models = "fake-model,fake-model,fake-model"
        self.max_tool_rounds = 3
        self.reports_dir = root / "reports"
        self.memory_dir = root / "memory"
        self.tool_call_content_mode = "null"
        self.auto_compact_event_threshold = 120
        self.auto_compact_turn_threshold = 20

    def ensure_dirs(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)


# A canonical 4-step plan the council-aware fake returns. The ids match the
# downstream execution tracking (context/design/execute/verify) so existing
# approval/verify assertions stay valid while the real council machinery
# (candidates -> critique -> merge -> validate) is exercised.
_FAKE_PLAN = {
    "title": "Test plan",
    "summary": "Council-drafted test plan.",
    "steps": [
        {"id": "context", "title": "Gather context", "purpose": "Read inputs", "dependencies": [], "tool_scope": ["file_read", "search"], "file_scope": ["workspace"], "verification": ["evidence cited"], "risks": []},
        {"id": "design", "title": "Design approach", "purpose": "Decompose", "dependencies": ["context"], "tool_scope": [], "file_scope": [], "verification": ["criteria set"], "risks": []},
        {"id": "execute", "title": "Execute approved work", "purpose": "Run", "dependencies": ["design"], "tool_scope": ["shell", "file_edit"], "file_scope": [], "verification": ["events recorded"], "risks": ["blocked by approvals"]},
        {"id": "verify", "title": "Verify and repair", "purpose": "Check", "dependencies": ["execute"], "tool_scope": ["test_runner"], "file_scope": [], "verification": ["checks pass"], "risks": []},
    ],
    "risks": ["may need approvals"],
    "verification_plan": ["run targeted checks"],
    "required_approvals": ["approve before mutation"],
    "proposed_tool_scope": ["file_read", "search", "shell", "file_edit", "test_runner"],
    "proposed_file_scope": ["workspace"],
    "open_questions": [],
}


class _FakeProvider:
    def __init__(self) -> None:
        self.requests = []
        self.responses = [
            ProviderResponse(text="answer", provider="fake", model="fake-model", usage={"prompt_tokens": 2, "completion_tokens": 3})
        ]

    def complete(self, request):
        self.requests.append(request)
        role = getattr(request.role, "value", str(request.role))
        content = request.messages[-1].get("content", "") if request.messages else ""
        if role == "critic":
            return ProviderResponse(text='{"score": 88, "strengths": ["clear"], "weaknesses": [], "missing": []}', provider="fake", model=request.model or "fake-model")
        if role in ("planner", "summarizer"):
            if "clarifying questions" in content or '"questions"' in content:
                return ProviderResponse(text='{"questions": []}', provider="fake", model=request.model or "fake-model")
            return ProviderResponse(text=json.dumps(_FAKE_PLAN), provider="fake", model=request.model or "fake-model")
        return self.responses.pop(0) if self.responses else ProviderResponse(text="done", provider="fake", model="fake-model")


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

    def _build_ctx(self, report_dir):
        ctx = SimpleNamespace(
            settings=self.settings,
            state=self.state,
            report_dir=report_dir,
            memory=self.memory,
        )
        return ctx


class _FakeSubagentResponse:
    def __init__(self, text: str, model: str = "planner-model") -> None:
        self.text = text
        self.model = model


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.calls = []

    def dispatch_subagent(self, call, context_pack=None, model_id=None):
        self.calls.append((call.role.value, call.task, dict(context_pack or {}), model_id))
        return _FakeSubagentResponse(f"{call.role.value}:{call.task}", model=model_id or "planner-model")


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


class _FakeProgressTool(_FakeTool):
    @property
    def name(self):
        return "progress_tool"

    def spec(self):
        from biobank_agent.core.tools.protocol import ToolSpec

        return ToolSpec(name="progress_tool", description="progress demo", parameters={})

    async def handle(self, ctx):
        ctx.emit_progress("read data", "loading demo dataset 1/1", {"step": 1, "total": 1, "data_source": "demo.tsv"})
        return {"ok": True, "source": "demo.tsv"}


class _FakeExternalAgentTool:
    def __init__(self, name: str = "external_agent_status") -> None:
        self._name = name

    @property
    def name(self):
        return self._name

    def spec(self):
        from biobank_agent.core.tools.protocol import ToolSpec

        return ToolSpec(name=self._name, description="external agent demo", parameters={})

    def required_capabilities(self):
        from biobank_agent.core.tools.protocol import Capability

        return frozenset({Capability.CALL_REVIEWER})

    @property
    def is_mutating(self):
        return False

    async def handle(self, ctx):
        return {"received": dict(ctx.args), "tool": ctx.name}


class _FakeMcpTool:
    def __init__(self, name: str = "mcp_demo__echo") -> None:
        self._name = name

    @property
    def name(self):
        return self._name

    def spec(self):
        from biobank_agent.core.tools.protocol import ActionClass, SafetyClass, TrajectorySerialization, ToolSpec, WorkspaceScope

        return ToolSpec(
            name=self._name,
            description="demo mcp tool",
            parameters={"text": {"type": "string"}},
            required=["text"],
            safety_class=SafetyClass.NETWORK,
            approval_requirement="ask_before_network",
            workspace_scope=WorkspaceScope.EXTERNAL_READ,
            action_classes=(ActionClass.NETWORK,),
            trajectory_serialization=TrajectorySerialization.METADATA_ONLY,
        )

    def required_capabilities(self):
        from biobank_agent.core.tools.protocol import Capability

        return frozenset({Capability.NETWORK, Capability.READ_DATA})

    @property
    def is_mutating(self):
        return False

    async def handle(self, ctx):
        return {"echo": ctx.args["text"], "tool": ctx.name}


def _shell(tmp_path: Path) -> tuple[InteractiveShell, StringIO]:
    output = StringIO()
    shell = InteractiveShell(
        settings=_Settings(tmp_path),
        console=Console(file=output, force_terminal=False, width=120),
    )
    shell.legacy_agent_factory = _FakeLegacyAgent
    session = shell.initialize()
    fake = _FakeProvider()
    shell.runtime.provider_router.providers = {"fake-model": fake}
    shell.runtime.tool_registry.register(_FakeTool())
    shell.legacy_agent.orchestrator = _FakeOrchestrator()
    return shell, output


def test_interactive_startup_banner_contains_runtime_state(tmp_path):
    shell, output = _shell(tmp_path)

    shell.console.print(shell.startup_banner())
    text = output.getvalue()

    assert "BioBank Agent" in text
    assert "workspace" in text
    assert shell.session.session_id in text
    assert "provider roles" in text
    assert "permissions" in text
    assert "tools" in text
    assert "wgs" in text
    assert "mcp" in text


def test_slash_parser_accepts_whitespace_and_inline_args():
    parsed = SlashCommandRegistry().parse("   /PLAN    run a careful analysis  ")

    assert parsed.name == "/plan"
    assert parsed.arg == "run a careful analysis"
    assert parsed.argv == ("run", "a", "careful", "analysis")


def test_handle_slash_reports_error_with_markup_in_message_without_crashing(tmp_path):
    """The error reporter must not re-parse the exception text as Rich markup.
    A command raising an exception whose message contains '[/]' previously made
    `console.print(f"[red]Command failed:[/] {exc}")` raise a SECOND, uncaught
    MarkupError -> the live CLI died. The reporter must escape the exception
    text so reporting an error can never itself crash."""
    shell, output = _shell(tmp_path)

    def _boom(_text, _ctx):
        raise RuntimeError("closing tag '[/]' at position 452 has nothing to close")

    shell.slash_registry.dispatch = _boom  # type: ignore[assignment]
    shell.handle_slash("/anything please")  # must NOT raise
    text = output.getvalue()
    assert "Command failed:" in text
    assert "nothing to close" in text  # message shown (escaped), not swallowed


def test_handle_message_reports_error_with_markup_without_crashing(tmp_path):
    """Same hardening for the run-turn error path (interactive.py:325)."""
    shell, output = _shell(tmp_path)

    def _boom(_session, _text):
        raise RuntimeError("bad [/] markup [unclosed in turn")

    shell.runtime.run_turn = _boom  # type: ignore[assignment]
    shell.handle_message("do something")  # must NOT raise
    assert "Error:" in output.getvalue()


def test_failed_plan_natural_language_diagnosis_does_not_advance_steps(tmp_path):
    shell, output = _shell(tmp_path)
    plan = PlanState(
        objective="Run WGS analysis",
        title="WGS plan",
        status=PlanStatus.FAILED,
        steps=[
            PlanStep(id="probe", title="Probe VCF", status="failed", tool_scope=["vcf_sample_list"]),
            PlanStep(id="qc", title="QC", status="pending", dependencies=["probe"], tool_scope=["vcf_qc"]),
        ],
    )
    shell.session.state.plan = plan
    shell.runtime.run_turn = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("must not run model"))  # type: ignore[assignment]

    events = shell.handle_message("what is the problem?")

    assert plan.steps[0].status == "failed"
    assert plan.steps[1].status == "pending"
    assert "Plan Diagnosis" in output.getvalue()
    assert any(event.type == AgentEventType.PLAN_PHASE for event in events)


def test_failed_plan_continue_retries_from_failed_step(tmp_path):
    shell, output = _shell(tmp_path)
    plan = PlanState(
        objective="Implement non-WGS workflow",
        title="Retry plan",
        status=PlanStatus.FAILED,
        steps=[
            PlanStep(id="execute", title="Execute", status="failed"),
            PlanStep(id="verify", title="Verify", status="pending", dependencies=["execute"]),
        ],
    )
    shell.session.state.plan = plan

    shell.handle_message("continue")

    assert plan.status == PlanStatus.COMPLETED
    assert [step.status for step in plan.steps] == ["done", "done"]
    assert "Retrying plan from step" in output.getvalue()


def test_required_slash_commands_smoke(tmp_path):
    shell, output = _shell(tmp_path)

    for command in (
        "/help",
        "/status",
        "/plan draft a study",
        "/goal finish migration",
        "/resume",
        "/compact",
        "/diff",
        "/permissions readonly",
        "/doctor",
        "/agent",
        "/subagents",
        "/review check report",
        "/audit",
    ):
        shell.handle_line(command)

    shell.handle_line("/quit")
    text = output.getvalue()
    assert "Slash commands" in text
    assert "Runtime Status" in text
    assert "Goal" in text
    assert "finish migration" in text
    assert "Plan:" in text
    assert "Doctor" in text
    assert "Audit" in text
    assert shell.exit_requested is True
    assert any(e["type"] == "command_started" for e in shell.session.events)
    assert any(e["type"] == "command_finished" for e in shell.session.events)


def test_agent_command_can_switch_active_role_and_records_event(tmp_path):
    shell, output = _shell(tmp_path)

    shell.handle_line("/agent planner")

    assert shell.runtime.config.active_role == "planner"
    assert any(event["type"] == "plan_phase" and event["payload"].get("phase") == "Provider routing" for event in shell.session.events)
    assert "Active role" in output.getvalue()


def test_subagents_command_uses_orchestrator_and_records_action_graph(tmp_path):
    shell, output = _shell(tmp_path)

    shell.handle_line("/subagents verifier inspect the current plan")

    text = output.getvalue()
    assert "Subagent: verifier" in text
    assert shell.legacy_agent.orchestrator.calls
    assert shell.session.action_graph_refs
    assert shell.session.action_graph_refs[-1].node_type == "subagent"
    assert any(event["type"] == "plan_phase" and event["payload"].get("phase") == "Subagent" for event in shell.session.events)


def test_mcp_command_smoke_records_runtime_events_and_graph(tmp_path):
    shell, output = _shell(tmp_path)

    shell.mcp_manager.load_config = lambda: []
    shell.mcp_manager.status = lambda: []
    shell.mcp_manager.handlers = []

    shell.handle_line("/mcp")
    shell.handle_line("/mcp health")

    text = output.getvalue()
    assert "MCP Servers" in text
    assert "MCP Health" in text


def test_mcp_call_argument_parser_handles_json_and_key_values(tmp_path):
    shell, _output = _shell(tmp_path)
    tool_name, args = shell._parse_mcp_call_args('demo__echo {"query":"E11"}')
    assert tool_name == "demo__echo"
    assert args == {"query": "E11"}
    tool_name, args = shell._parse_mcp_call_args("demo__echo query=E11 limit=3")
    assert tool_name == "demo__echo"
    assert args == {"query": "E11", "limit": "3"}


def test_mcp_call_routes_through_runtime_tool_execution(tmp_path):
    shell, output = _shell(tmp_path)
    shell.runtime.tool_registry.register(_FakeMcpTool())

    shell.handle_line('/mcp call demo__echo {"text":"ok"}')

    text = output.getvalue()
    assert "MCP call: demo__echo" in text
    assert shell.session.turns == []
    assert any(event["type"] == "tool_call_requested" for event in shell.session.events)
    assert any(event["type"] == "tool_call_completed" for event in shell.session.events)
    assert any(ref.node_type == "tool" and ref.node_id.startswith("mcp:") for ref in shell.session.action_graph_refs)


def test_external_agent_slash_commands_route_through_runtime_tools(tmp_path):
    shell, _output = _shell(tmp_path)
    shell.runtime.tool_registry.register(_FakeExternalAgentTool("external_agent_status"))
    shell.runtime.tool_registry.register(_FakeExternalAgentTool("codex_check_execution"))

    status_events = shell.handle_line("/external-agents codex")
    review_events = shell.handle_line("/codex-check report quality")

    assert any(event.type == AgentEventType.TOOL_CALL_REQUESTED for event in status_events)
    assert any(event.type == AgentEventType.TOOL_CALL_COMPLETED for event in review_events)
    assert any(ref.node_type == "tool" and ref.node_id.startswith("external:external_agent_status:") for ref in shell.session.action_graph_refs)
    assert any(ref.node_type == "tool" and ref.node_id.startswith("external:codex_check_execution:") for ref in shell.session.action_graph_refs)
    assert shell.session.state.custom_data["external_agent_requests"] == [
        {"skill": "external_agent_status", "args": {"agent": "codex"}},
        {"skill": "codex_check_execution", "args": {"focus": "report quality"}},
    ]
    finished = [
        event for event in review_events
        if event.type == AgentEventType.COMMAND_FINISHED and event.payload.get("command") == "/codex-check"
    ]
    assert finished
    assert finished[-1].payload["result"]["status"] == "done"
    assert finished[-1].payload["result"]["result"]["received"]["focus"] == "report quality"


def test_permissions_command_rebinds_runtime_policy(tmp_path):
    shell, _output = _shell(tmp_path)

    shell.handle_line("/permissions read_only")

    assert shell.runtime.config.approval_profile == "read_only"
    assert shell.runtime.approval_policy.profile.name == "read_only"
    assert shell.runtime._scheduler.policy.profile.name == "read_only"


def test_plan_goal_compact_and_resume_restore_structured_state(tmp_path):
    shell, output = _shell(tmp_path)

    shell.handle_line("/goal Deliver resumable long task")
    shell.handle_line("/plan implement stateful workflow")
    assert shell.session.state.goal.objective == "Deliver resumable long task"
    assert shell.session.state.plan.objective == "implement stateful workflow"
    assert shell.runtime.config.approval_profile == "plan"

    shell.handle_line("/plan-approve")
    # Approve now autonomously executes the plan (the fake provider converges
    # with no tools), so it ends COMPLETED, and the user's real profile is
    # restored (NOT forced to ask_before_edits) so execution runs autonomously.
    assert shell.session.state.plan.status.value == "completed"
    assert shell.runtime.config.approval_profile == shell.session.config.approval_profile
    assert shell.runtime.config.approval_profile != "ask_before_edits"
    assert any(item.startswith("git_status=") for item in shell.session.state.plan.context_gathering)

    shell.handle_line("/plan-edit refine the checkpoints")
    assert shell.session.state.plan.revision == 2
    assert any("Refinement requested by user" in risk for risk in shell.session.state.plan.risks)

    provider = shell.runtime.provider_router.providers["fake-model"]
    provider.responses = [
        ProviderResponse(
            text="",
            provider="fake",
            model="fake-model",
            tool_calls=[ToolCall(id="call_1", name="demo_tool", args={})],
        ),
        ProviderResponse(text="tool complete", provider="fake", model="fake-model"),
    ]
    shell.handle_line("use a tool during the session")
    assert shell.session.action_graph_refs
    assert any(ref.node_type == "tool" and ref.node_id == "call_1" for ref in shell.session.action_graph_refs)

    shell.handle_line("/compact")
    session_id = shell.session.session_id
    assert shell.session.state.compacted_summary["goal"]["objective"] == "Deliver resumable long task"
    assert shell.session.state.compacted_summary["plan"]["objective"] == "implement stateful workflow"
    assert shell.session.state.compacted_summary["action_graph_refs"]

    shell.handle_line("/new next")
    shell.handle_line(f"/resume {session_id}")

    assert shell.session.session_id == session_id
    assert shell.session.state.goal.objective == "Deliver resumable long task"
    assert shell.session.state.plan.objective == "implement stateful workflow"
    assert shell.session.action_graph_refs
    assert any(ref.node_type == "tool" and ref.node_id == "call_1" for ref in shell.session.action_graph_refs)
    assert shell.runtime.config.approval_profile == shell.session.config.approval_profile
    assert any(event["type"] == "resume_completed" for event in shell.session.events)
    text = output.getvalue()
    assert "Plan approved" in text
    assert "Compacted session context" in text
    assert "Resumed session" in text


def test_plan_command_prints_progress_and_flowchart(tmp_path):
    shell, output = _shell(tmp_path)

    shell.handle_line("/plan build a visible progress workflow")

    text = output.getvalue()
    # Concision pass: the "Plan Formulation Progress" preamble and the static
    # council-activity summary are no longer reprinted; the plan + the workflow
    # diagram are shown once, followed by the review notice.
    assert "Plan Formulation Progress" not in text
    assert "Workflow Diagram" in text
    assert "Plan Flowchart" in text
    assert "Plan Review" in text
    assert "Awaiting Approval" in text
    assert "Approve and implement plan" in text
    assert "Non-interactive mode" in text
    assert "awaiting approval" in text
    assert "context" in text
    assert "execute" in text
    assert shell.session.state.plan.steps[0].status == "planned"
    assert any(ref.node_type == "plan" for ref in shell.session.action_graph_refs)
    assert sum(1 for ref in shell.session.action_graph_refs if ref.node_type == "plan_step") == 4
    assert any(ref.node_type == "plan_review" for ref in shell.session.action_graph_refs)
    assert any(event["type"] == "plan_phase" for event in shell.session.events)


def test_plan_review_provider_can_approve_without_extra_command(tmp_path):
    shell, output = _shell(tmp_path)
    shell.plan_review_choice_provider = lambda choices: "approve"

    shell.handle_line("/plan approve from review menu")

    text = output.getvalue()
    assert "Plan approved" in text
    # Approving from the review menu now executes the plan autonomously.
    assert shell.session.state.plan.status.value == "completed"
    assert shell.session.state.plan.steps[0].status == "done"
    assert any(
        event["type"] == "plan_phase"
        and event["payload"].get("phase") == "Plan review"
        and event["payload"].get("message") == "selected approve and implement"
        for event in shell.session.events
    )
    assert any(
        ref.node_type == "plan_review" and ref.payload.get("action") == "approve"
        for ref in shell.session.action_graph_refs
    )


def test_plan_review_provider_can_refine_then_exit(tmp_path):
    shell, output = _shell(tmp_path)
    choices = iter(["fix", "exit"])
    shell.plan_review_choice_provider = lambda _choices: next(choices)
    shell._read_plan_refinement = lambda: "add stronger WGS QC gates"

    shell.handle_line("/plan refine from review menu")

    text = output.getvalue()
    assert "add stronger WGS QC gates" in "\n".join(shell.session.state.plan.risks)
    assert shell.session.state.plan.revision == 2
    assert shell.session.state.plan.status.value == "draft"
    assert shell.session.state.custom_data["plan_mode_exited"] is True
    assert "Plan mode exit recorded" in text
    assert any(
        ref.node_type == "plan_review" and ref.payload.get("action") == "refine"
        for ref in shell.session.action_graph_refs
    )


def test_plan_approve_executes_plan_autonomously_with_summary(tmp_path):
    shell, output = _shell(tmp_path)

    shell.handle_line("/plan build a visible execution workflow")
    shell.handle_line("/plan-approve")

    text = output.getvalue()
    assert "Plan approved" in text
    assert "Executing autonomously" in text       # no per-step prompt
    assert "Execution complete" in text
    assert "Plan Execution Summary" in text
    assert "awaiting executable work" not in text  # the old static dead-end is gone
    # context+design pre-marked done; execute+verify executed via run_turn (the
    # fake provider converges with no tools) -> all 4 steps terminal.
    assert all(step.status in {"done", "completed"} for step in shell.session.state.plan.steps)
    assert shell.session.state.plan.status.value == "completed"


def test_plan_approve_restores_real_profile_not_plan_or_ask(tmp_path):
    shell, output = _shell(tmp_path)
    shell.handle_line("/plan keep yolo autonomous")
    # During plan mode the runtime profile is the read-only "plan".
    assert shell.runtime.config.approval_profile == "plan"
    shell.handle_line("/plan-approve")
    # After approve, execution must run under the user's REAL autonomous profile —
    # NOT the read-only "plan" (which would deny write/shell tools, blocking the
    # whole autonomous run) and NOT ask_before_edits.
    assert shell.runtime.config.approval_profile not in {"plan", "ask_before_edits"}
    assert "ask_before_edits" not in output.getvalue()


def test_plan_execution_reads_no_per_step_input(tmp_path, monkeypatch):
    shell, _output = _shell(tmp_path)
    shell.handle_line("/plan no input during execution")

    def _boom(*_a, **_k):
        raise AssertionError("execution must not read user input")

    monkeypatch.setattr(shell, "_read_line", _boom)
    monkeypatch.setattr(shell, "_read_single_key", _boom)
    shell.handle_line("/plan-approve")  # must complete without reading input
    assert shell.session.state.plan.status.value == "completed"


def test_ctrlc_mid_execution_halts_cleanly(tmp_path):
    """A genuine Ctrl-C during autonomous execution must stop the loop, mark the
    plan paused, print a cancellation line, and NOT escape handle_line."""
    shell, output = _shell(tmp_path)
    shell.handle_line("/plan halt me midway")
    calls = {"n": 0}
    real = shell.runtime.run_turn

    def _spy(session, text):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt  # interrupt the first executable step
        return real(session, text)

    shell.runtime.run_turn = _spy  # type: ignore[assignment]
    shell.handle_line("/plan-approve")  # must NOT raise out
    text = output.getvalue()
    assert "cancelled at step" in text
    assert shell.session.state.plan.status.value == "paused"


def test_execution_drives_view_start_record_stop(tmp_path, monkeypatch):
    """The execution loop must drive a live view (start -> record... -> stop) and
    ALWAYS stop it (the Live must be torn down)."""
    shell, _output = _shell(tmp_path)
    shell.handle_line("/plan drive the view")
    events: list[str] = []

    class _RecView:
        def start(self, *a, **k):
            events.append("start")
        def record(self, *a, **k):
            events.append("record")
        def stop(self, *a, **k):
            events.append("stop")

    monkeypatch.setattr(shell, "_make_execution_view", lambda: _RecView())
    shell.handle_line("/plan-approve")
    assert events and events[0] == "start" and events[-1] == "stop"
    assert "record" in events


def test_runtime_tool_progress_and_action_graph_are_printed_in_cli(tmp_path):
    shell, output = _shell(tmp_path)
    shell.runtime.tool_registry.register(_FakeProgressTool())
    shell.handle_line("/plan run visible progress tool")
    shell.handle_line("/plan-approve")
    provider = shell.runtime.provider_router.providers["fake-model"]
    provider.responses = [
        ProviderResponse(
            text="",
            provider="fake",
            model="fake-model",
            tool_calls=[ToolCall(id="progress_call", name="progress_tool", args={})],
        ),
        ProviderResponse(text="finished progress tool", provider="fake", model="fake-model"),
    ]

    shell.handle_line("run visible progress tool")

    text = output.getvalue()
    assert "Execution Progress" in text
    assert "loading demo dataset 1/1" in text
    assert "Tool completed" in text
    assert "Action Graph Flowchart" in text
    assert "tool:progress_call" in text
    # (Approve already executed the plan steps autonomously; this turn exercises
    # the per-turn tool-progress + action-graph rendering.)
    assert next(step for step in shell.session.state.plan.steps if step.id == "execute").status == "done"
    assert any(event["type"] == "tool_progress" for event in shell.session.events)
    assert any(ref.node_id == "progress_call" for ref in shell.session.action_graph_refs)


def test_verify_updates_plan_execution_progress(tmp_path):
    shell, output = _shell(tmp_path)

    shell.handle_line("/plan verify visible progress")
    shell.handle_line("/plan-approve")
    shell.handle_line("/verify python -c 'print(123)'")

    text = output.getvalue()
    assert "Verification passed" in text
    # Approve already executed the plan (all 4 steps terminal); verify stays done.
    assert next(step for step in shell.session.state.plan.steps if step.id == "verify").status == "done"


def test_graph_command_prints_current_action_graph(tmp_path):
    shell, output = _shell(tmp_path)

    shell.handle_line("/plan graph-ready workflow")
    shell.handle_line("/agent planner")
    shell.handle_line("/graph")

    text = output.getvalue()
    assert "Action Graph Flowchart" in text
    assert "plan_step" in text
    assert "provider_role:planner" in text
    assert "session:" in text


def test_audit_prints_action_graph_flowchart(tmp_path):
    shell, output = _shell(tmp_path)

    shell.handle_line("/agent critic")
    shell.handle_line("/audit")

    text = output.getvalue()
    assert "Audit" in text
    assert "Action Graph Flowchart" in text
    assert "provider_role:critic" in text


def test_goal_updates_and_resume_latest_restores_most_recent_session(tmp_path):
    shell, _output = _shell(tmp_path)

    shell.handle_line("/goal First objective")
    first_session = shell.session.session_id
    assert shell.session.state.goal.revision == 1

    shell.handle_line("/goal Updated objective")
    assert shell.session.state.goal.objective == "Updated objective"
    assert shell.session.state.goal.revision == 2
    assert sum(1 for event in shell.session.events if event["type"] == "goal_updated") == 2

    shell.handle_line("/new second")
    shell.handle_line("/goal Latest objective")
    latest_session = shell.session.session_id
    assert latest_session != first_session

    shell.handle_line("/resume --last")

    assert shell.session.session_id == latest_session
    assert shell.session.state.goal.objective == "Latest objective"
    assert any(event["type"] == "resume_completed" and event["payload"].get("mode") == "latest" for event in shell.session.events)


def test_goal_update_preserves_existing_state_fields(tmp_path):
    shell, _output = _shell(tmp_path)

    shell.handle_line("/goal First objective")
    shell.session.state.goal.tasks[0]["status"] = "done"
    shell.session.state.goal.verification_commands.append("pytest -q")
    shell.session.state.goal.open_questions.append("Need confirmation")
    shell.session.state.goal.risks.append("Extra risk")

    shell.handle_line("/goal Updated objective")

    assert shell.session.state.goal.objective == "Updated objective"
    assert shell.session.state.goal.tasks[0]["status"] == "done"
    assert "pytest -q" in shell.session.state.goal.verification_commands
    assert "Need confirmation" in shell.session.state.goal.open_questions
    assert "Extra risk" in shell.session.state.goal.risks


def test_resume_without_args_lists_saved_sessions(tmp_path):
    shell, output = _shell(tmp_path)

    shell.handle_line("/goal List sessions")
    shell.handle_line("/new second")
    shell.handle_line("/goal Another session")

    shell.handle_line("/resume")

    text = output.getvalue()
    assert "Saved Sessions" in text
    assert shell.session.session_id is not None


def test_plan_reject_and_verify_failure_are_recorded(tmp_path):
    shell, output = _shell(tmp_path)

    shell.handle_line("/plan rejectable task")
    shell.handle_line("/plan-reject not the right direction")
    assert shell.session.state.plan.status.value == "rejected"
    assert any("Rejected by user" in risk for risk in shell.session.state.plan.risks)

    shell.handle_line("/verify python -c 'import sys; sys.exit(1)'")
    assert shell.session.state.verification_results
    assert shell.session.state.repair_attempts
    assert "Verification failed" in output.getvalue()


def test_resume_warns_when_workspace_changed_since_checkpoint(tmp_path, monkeypatch):
    shell, output = _shell(tmp_path)
    shell.handle_line("/goal Watch workspace")
    session_id = shell.session.session_id
    shell.session.state.checkpoints[-1].workspace_fingerprint = "old"
    shell.runtime.save_session(shell.session)
    monkeypatch.setattr(shell.runtime, "workspace_fingerprint", lambda _cwd: "new")

    shell.handle_line(f"/resume {session_id}")

    assert "Workspace has changed" in output.getvalue()


def test_normal_turn_auto_compacts_when_threshold_exceeded(tmp_path):
    shell, _output = _shell(tmp_path)
    shell.settings.auto_compact_event_threshold = 1
    shell.settings.auto_compact_turn_threshold = 99

    shell.handle_line("trigger compaction")

    assert shell.session.state.compacted_summary
    assert any(event["type"] == "compact_completed" for event in shell.session.events)
    restored = json.loads(shell.runtime.session_store.session_file(shell.session.session_id).read_text(encoding="utf-8"))
    assert restored["state"]["compacted_summary"]


def test_normal_message_routes_through_agent_runtime_and_saves_session(tmp_path):
    shell, output = _shell(tmp_path)

    events = shell.handle_line("summarize this cohort")

    assert "answer" in output.getvalue()
    assert shell.session.turns[0].content == "summarize this cohort"
    assert shell.runtime.provider_router.providers["fake-model"].requests
    assert any(event.type == AgentEventType.USER_TURN_STARTED for event in events)
    session_file = shell.runtime.session_store.session_file(shell.session.session_id)
    trajectory_file = shell.runtime.session_store.rollout_file(shell.session.session_id)
    assert session_file.exists()
    assert trajectory_file.exists()
    restored = json.loads(session_file.read_text(encoding="utf-8"))
    assert restored["turns"][0]["content"] == "summarize this cohort"


def test_action_graph_and_trajectory_recorded_for_tool_call(tmp_path):
    shell, _output = _shell(tmp_path)
    provider = shell.runtime.provider_router.providers["fake-model"]
    provider.responses = [
        ProviderResponse(
            text="",
            provider="fake",
            model="fake-model",
            tool_calls=[ToolCall(id="call_1", name="demo_tool", args={})],
        ),
        ProviderResponse(text="final", provider="fake", model="fake-model"),
    ]

    shell.handle_line("use a tool")

    assert shell.session.action_graph_refs
    assert any(ref.node_type == "tool" and ref.node_id == "call_1" for ref in shell.session.action_graph_refs)
    session_json = json.loads(shell.runtime.session_store.session_file(shell.session.session_id).read_text(encoding="utf-8"))
    assert any(ref["node_type"] == "tool" and ref["node_id"] == "call_1" for ref in session_json["action_graph_refs"])
    assert shell.legacy_agent.state.trajectory.records


def test_event_serialization_for_command_event(tmp_path):
    shell, _output = _shell(tmp_path)
    events = shell.handle_line("/status")

    payload = events[0].to_dict()
    restored = AgentEvent.from_dict(payload)
    assert restored.type == AgentEventType.COMMAND_STARTED
    assert restored.payload["command"] == "/status"


def test_session_save_round_trip_preserves_runtime_metadata(tmp_path):
    shell, _output = _shell(tmp_path)

    shell.handle_line("/agent critic")
    shell.handle_line("/subagents verifier inspect the session")
    shell.runtime.save_session(shell.session)

    restored = json.loads(shell.runtime.session_store.session_file(shell.session.session_id).read_text(encoding="utf-8"))
    assert restored["config"]["active_role"] == "critic"
    assert restored["action_graph_refs"]
    assert any(item["node_type"] == "provider_role" for item in restored["action_graph_refs"])
    assert any(item["node_type"] == "subagent" for item in restored["action_graph_refs"])


def test_registry_dispatch_returns_handler_result():
    registry = SlashCommandRegistry(build_core_registry())
    calls = []
    ctx = CommandContext(
        agent=None,
        planner=None,
        token_usage={},
        console=SimpleNamespace(print=lambda *args, **kwargs: None),
        actions={"status": lambda: calls.append("status")},
    )

    registry.dispatch(" /status   ", ctx)

    assert calls == ["status"]


def test_production_code_does_not_reference_cli_fixture_prompts():
    production_files = list(Path("biobank_agent").rglob("*.py"))
    forbidden = ("summarize this cohort", "finish migration", "draft a study", "call_1")
    hits = []
    for path in production_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                hits.append((str(path), token))
    assert hits == []
