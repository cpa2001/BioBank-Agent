"""Tests for the default runtime-backed interactive CLI."""

from __future__ import annotations

import json
import time
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from biobank_agent.cli.commands import CommandContext
from biobank_agent.cli.commands.registry import SlashCommandRegistry, build_core_registry
from biobank_agent.cli.interactive import InteractiveShell, PlanBuildTimeoutError, PlanStepTimeoutError
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


def test_planning_failure_diagnosis_without_active_plan(tmp_path):
    shell, output = _shell(tmp_path)
    shell.session.state.plan = None
    shell.session.state.custom_data["plan_last_diagnosis"] = {
        "status": "failed",
        "step_id": "planning",
        "step_title": "Plan generation",
        "step_status": "failed",
        "reason": "planner returned invalid JSON",
        "objective": "complex task",
        "blocked": True,
        "repair_options": ["retry with a narrower task"],
    }

    shell.handle_message("what is the problem?")

    assert "Plan Diagnosis" in output.getvalue()
    assert "planner returned invalid JSON" in output.getvalue()


def test_plan_timeout_errors_are_not_builtin_timeout_subclasses():
    assert not issubclass(PlanBuildTimeoutError, TimeoutError)
    assert not issubclass(PlanStepTimeoutError, TimeoutError)


def test_plan_generation_timeout_saves_natural_language_diagnosis(tmp_path):
    shell, output = _shell(tmp_path)
    shell.settings.plan_build_timeout_s = 0.1

    def slow_draft(_objective):
        time.sleep(5)

    shell._draft_plan_with_live_progress = slow_draft

    result = shell._cmd_plan("complex WGS analysis")

    assert result["status"] == "failed"
    assert result["step_status"] == "timeout"
    diagnosis = shell.session.state.custom_data["plan_last_diagnosis"]
    assert diagnosis["step_id"] == "planning"
    assert diagnosis["step_status"] == "timeout"
    assert "plan_build_timeout_s=0.1s" in diagnosis["reason"]
    assert "Planning timed out" in output.getvalue()

    shell.handle_message("what is the problem?")

    text = output.getvalue()
    assert "Plan Diagnosis" in text
    assert "Plan generation stalled" in text


def test_plan_build_timeout_restores_approval_profile(tmp_path):
    shell, _output = _shell(tmp_path)
    shell.settings.plan_build_timeout_s = 0.1
    runtime = shell._require_runtime()
    runtime.set_approval_profile("yolo")  # the user's real profile (runtime normalizes it)
    user_profile = runtime.config.approval_profile

    def slow_draft_that_flips(_objective):
        runtime.set_approval_profile("plan")  # drafting flips to the read-only "plan" profile
        time.sleep(5)

    shell._draft_plan_with_live_progress = slow_draft_that_flips
    result = shell._cmd_plan("complex WGS analysis")

    assert result["status"] == "failed"
    # A timed-out build must NOT leave the runtime stuck in read-only plan mode.
    assert runtime.config.approval_profile != "plan"
    assert runtime.config.approval_profile == user_profile


def test_failed_plan_build_does_not_widen_plan_permission(tmp_path):
    shell, _output = _shell(tmp_path)
    shell.settings.plan_build_timeout_s = 0.1
    runtime = shell._require_runtime()
    runtime.set_approval_profile("plan")  # the user is ALREADY in read-only plan mode

    def slow_draft(_objective):
        time.sleep(5)

    shell._draft_plan_with_live_progress = slow_draft
    shell._cmd_plan("complex WGS analysis")

    # A failed draft must NOT escalate a user who was already in plan mode to a wider profile.
    assert runtime.config.approval_profile == "plan"


def test_plan_edit_respects_build_timeout_and_preserves_plan(tmp_path):
    shell, _output = _shell(tmp_path)
    shell.settings.plan_build_timeout_s = 0.1
    runtime = shell._require_runtime()
    runtime.set_approval_profile("yolo")
    user_profile = runtime.config.approval_profile
    shell.session.state.plan = PlanState(objective="vitiligo WGS analysis", title="WGS", steps=[], revision=1)
    shell.session.state.pending_work = [{"step": "original-qc"}]

    def slow_draft_that_mutates(_objective, refinement=""):
        # the runtime mutates session.state (plan + pending_work) and flips to read-only "plan", then stalls
        runtime.set_approval_profile("plan")
        shell.session.state.plan.revision = 99
        shell.session.state.plan.objective = "MUTATED"
        shell.session.state.pending_work = [{"step": "half-mutated"}]
        time.sleep(5)

    shell._draft_plan_with_live_progress = slow_draft_that_mutates
    result = shell._cmd_plan_edit("simplify the QC step")

    # /plan-edit goes through the build-timeout guard (no hang) AND leaves no half-mutated state behind.
    assert result["status"] == "failed"
    assert result.get("step_status") == "timeout"
    assert shell.session.state.plan.revision == 1
    assert shell.session.state.plan.objective == "vitiligo WGS analysis"
    assert shell.session.state.pending_work == [{"step": "original-qc"}]
    assert runtime.config.approval_profile == user_profile  # profile restored, not left at "plan"


def test_plan_generation_temporarily_caps_provider_request_timeout(tmp_path):
    shell, _output = _shell(tmp_path)
    shell.settings.plan_build_timeout_s = 1.5
    plan = PlanState(objective="short task", title="Short", steps=[])

    class TimeoutAwareProvider:
        def __init__(self) -> None:
            self.request_timeout_s = 60.0
            self.max_retries = 3
            self.observed_during_plan: float | None = None
            self.observed_retries_during_plan: int | None = None

        def set_llm_policy(self, *, request_timeout_s=None, max_retries=None):
            previous = (self.request_timeout_s, self.max_retries)
            if request_timeout_s is not None:
                self.request_timeout_s = float(request_timeout_s)
            if max_retries is not None:
                self.max_retries = int(max_retries)
            return previous

    provider = TimeoutAwareProvider()
    shell.runtime.provider_router.providers = {"fake-model": provider}

    def draft(_objective):
        provider.observed_during_plan = provider.request_timeout_s
        provider.observed_retries_during_plan = provider.max_retries
        return plan

    shell._draft_plan_with_live_progress = draft
    shell._run_plan_review_loop = lambda _plan, previous_mode: None

    result = shell._cmd_plan("short task")

    assert result["status"] == "draft"
    assert provider.observed_during_plan == 1.5
    assert provider.observed_retries_during_plan == 0
    assert provider.request_timeout_s == 60.0
    assert provider.max_retries == 3


def test_wgs_preflight_ignores_tabular_gwas_summary_csv(tmp_path):
    plan = PlanState(
        objective="Read file all_traits_5e-11_gwas_results.csv and classify traits",
        title="Tabular GWAS summary analysis",
        steps=[],
        audit_summary="Tabular-analysis fallback plan",
    )
    step = PlanStep(
        id="load",
        title="Load and profile the file",
        purpose="Use python_exec to load all_traits_5e-11_gwas_results.csv as a table.",
        tool_scope=["python_exec"],
        file_scope=["all_traits_5e-11_gwas_results.csv"],
    )

    assert InteractiveShell._plan_step_mentions_wgs(plan, step) is False


def test_wgs_preflight_detects_real_variant_inputs(tmp_path):
    plan = PlanState(objective="Run WGS association", title="WGS", steps=[])
    step = PlanStep(
        id="assoc",
        title="Run association",
        purpose="Run VCF association with PCA covariates",
        tool_scope=["vcf_association"],
        file_scope=["cohort.vcf.gz"],
    )

    assert InteractiveShell._plan_step_mentions_wgs(plan, step) is True


def test_wgs_preflight_does_not_block_non_variant_context_step_in_wgs_plan(tmp_path):
    plan = PlanState(objective="Run WGS association for vitiligo", title="WGS analysis", steps=[])
    step = PlanStep(
        id="context",
        title="Read phenotype table",
        purpose="Inspect phenotype.tsv columns and summarize case/control labels.",
        tool_scope=["python_exec"],
        file_scope=["phenotype.tsv"],
    )

    assert InteractiveShell._plan_step_mentions_wgs(plan, step) is False


def test_collect_artifact_index_from_step_outputs(tmp_path):
    shell, _output = _shell(tmp_path)
    plan = PlanState(
        objective="collect outputs",
        title="outputs",
        steps=[PlanStep(id="s1", title="Make outputs")],
    )
    outputs = {
        "s1": {
            "tool_results": [
                ("python_exec", {
                    "report_dir": str(tmp_path / "reports" / "s1"),
                    "output": str(tmp_path / "reports" / "s1" / "single.tsv"),
                    "outputs": [str(tmp_path / "reports" / "s1" / "table.tsv")],
                    "run": {"log_path": str(tmp_path / "logs" / "tool.log")},
                })
            ],
            "text": "done",
        }
    }

    artifacts = shell._collect_artifact_index(plan, outputs)

    paths = {item["path"] for item in artifacts}
    assert str(tmp_path / "reports" / "s1") in paths
    assert str(tmp_path / "reports" / "s1" / "single.tsv") in paths
    assert str(tmp_path / "reports" / "s1" / "table.tsv") in paths
    assert str(tmp_path / "logs" / "tool.log") in paths


def test_finalize_execution_persists_artifact_index(tmp_path):
    shell, _output = _shell(tmp_path)
    plan = PlanState(
        objective="collect outputs",
        title="outputs",
        steps=[PlanStep(id="s1", title="Make outputs")],
    )
    outputs = {
        "s1": {
            "tool_results": [
                ("python_exec", {"log_path": str(tmp_path / "logs" / "tool.log")})
            ],
            "text": "done",
        }
    }

    result = shell._finalize_execution(plan, 1, 0, None, 1, outputs)

    assert result["artifacts"]
    assert shell.session.state.custom_data["last_artifact_index"] == result["artifacts"]
    reloaded = shell.runtime.session_store.load(shell.session.session_id)
    assert reloaded is not None
    assert reloaded.state.custom_data["last_artifact_index"] == result["artifacts"]


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
        "/tools",
        "/skills",
        "/artifacts",
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
    assert "Execution Plan" in text
    assert "Doctor" in text
    assert "Tool Registry" in text
    assert "Skill Tree" in text
    assert "No captured plan artifacts yet" in text
    assert "Audit" in text
    assert "Command failed" not in text
    assert shell.exit_requested is True
    assert any(e["type"] == "command_started" for e in shell.session.events)
    assert any(e["type"] == "command_finished" for e in shell.session.events)


def test_tools_command_returns_structured_payload(tmp_path):
    shell, output = _shell(tmp_path)

    result = shell._cmd_tools()

    assert result["status"] == "ok"
    assert result["total_tools"] >= 1
    assert "tools" in result and any(row["name"] == "demo_tool" for row in result["tools"])
    assert "Tool Registry" in output.getvalue()


def test_skills_command_returns_structured_payload(tmp_path):
    shell, output = _shell(tmp_path)

    result = shell._cmd_skills()

    assert result["status"] == "ok"
    assert result["total_tools"] >= result["direct_tools"] >= 1
    assert "domains" in result and result["domains"]
    assert "Skill Tree" in output.getvalue()


def test_artifacts_command_renders_last_index(tmp_path):
    shell, output = _shell(tmp_path)
    path = str(tmp_path / "reports" / "table.tsv")
    shell.session.state.custom_data["last_artifact_index"] = [{
        "step_id": "s1",
        "step_title": "Make table",
        "tool": "python_exec",
        "kind": "output_path",
        "path": path,
    }]

    result = shell._cmd_artifacts()

    assert result["status"] == "ok"
    assert result["count"] == 1
    assert result["artifacts"][0]["path"] == path
    assert "Output Artifacts" in output.getvalue()
    assert path in output.getvalue()


def test_doctor_uses_session_workspace_and_reports_root(tmp_path):
    shell, output = _shell(tmp_path)
    workspace = tmp_path / "analysis_workspace"
    workspace.mkdir()
    shell.session.cwd = str(workspace)

    result = shell._cmd_doctor()

    assert result["workspace"] == str(workspace)
    assert result["reports_dir"].endswith(str(workspace / "reports"))
    text = output.getvalue()
    assert str(workspace) in text
    assert str(Path.cwd()) not in result["workspace"]


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


def test_plan_command_prints_unified_plan_surface(tmp_path):
    shell, output = _shell(tmp_path)

    shell.handle_line("/plan build a visible progress workflow")

    text = output.getvalue()
    # The plan now renders as ONE unified surface (objective + steps + linear flow + detail),
    # replacing the old fields-table + steps-table + separate flowchart panel.
    assert "Plan Formulation Progress" not in text
    assert "Execution Plan" in text
    assert "Steps" in text
    assert "Plan Review" in text
    assert "Awaiting Approval" in text
    assert "Approve and implement plan" in text
    assert "Revise plan" in text  # the old "Approve and fix plan" was misleading — it regenerates
    assert "Non-interactive mode" in text
    assert "awaiting approval" in text
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
    # context+design pre-marked done; execute+verify executed via run_turn (the fake
    # provider converges with NO tools, so under the Evidence Contract those
    # declared-verification steps end 'unverified', not 'done') -> all 4 steps terminal.
    assert all(step.status in {"done", "completed", "unverified"} for step in shell.session.state.plan.steps)
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


def test_step_timeout_auto_resumes_instead_of_pausing(tmp_path):
    """A step that stalls (PlanStepTimeoutError) must AUTO-RESUME via the bounded retry loop — re-run
    with full session context — instead of pausing for user input on the first stall. Only an exhausted
    retry budget escalates to needs_input. (User requirement: a timeout must not stop a recoverable task.)"""
    shell, _output = _shell(tmp_path)
    shell.handle_line("/plan stall once then resume")
    calls = {"n": 0}
    real = shell.runtime.run_turn

    def _spy(session, text):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PlanStepTimeoutError("went silent")  # first attempt stalls
        return real(session, text)  # subsequent attempts succeed normally

    shell.runtime.run_turn = _spy  # type: ignore[assignment]
    shell.handle_line("/plan-approve")  # must auto-resume, NOT pause for input
    assert calls["n"] >= 2, "the step must auto-retry after a stall instead of pausing immediately"
    assert shell.session.state.plan.status.value == "completed"


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
    # The execute step ran via run_turn with the fake provider (no tools), so under the
    # Evidence Contract it ends terminal as 'unverified' rather than 'done'.
    assert next(step for step in shell.session.state.plan.steps if step.id == "execute").status in {"done", "unverified"}
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


def test_cmd_cd_switches_workspace_and_keeps_prior_readable(tmp_path):
    shell, _ = _shell(tmp_path)
    proj_a = tmp_path / "projA"
    proj_a.mkdir()
    proj_b = tmp_path / "projB"
    proj_b.mkdir()
    start_cwd = shell.session.cwd

    res = shell._cmd_cd(str(proj_a))
    assert res["status"] == "ok"
    assert shell.session.cwd == str(proj_a.resolve())
    # the previous workspace is kept as an extra readable root
    assert start_cwd in shell.session.state.custom_data["workspace_extra_roots"]

    res = shell._cmd_cd(str(proj_b))
    assert shell.session.cwd == str(proj_b.resolve())
    extras = shell.session.state.custom_data["workspace_extra_roots"]
    assert str(proj_a.resolve()) in extras and start_cwd in extras

    # a non-existent target is rejected and leaves the workspace unchanged
    res = shell._cmd_cd(str(tmp_path / "does_not_exist"))
    assert res["status"] == "error"
    assert shell.session.cwd == str(proj_b.resolve())


def test_cmd_cd_no_arg_reports_current_workspace(tmp_path):
    shell, output = _shell(tmp_path)
    res = shell._cmd_cd("")
    assert res["status"] == "ok"
    assert res["workspace"] == shell.session.cwd


def test_evidence_contract_unverified_without_artifact(tmp_path):
    import types
    shell, _ = _shell(tmp_path)
    step = types.SimpleNamespace(verification=["a results CSV is written under reports/"], title="t", id="s1")
    ok, reason = shell._assess_step_evidence(step, {"tool_results": [], "text": "All done!"})
    assert ok is False and "no" in reason.lower()
    ok2, _ = shell._assess_step_evidence(step, {"tool_results": [("python_exec", {"path": "/x/out.csv"})], "text": ""})
    assert ok2 is True
    ok3, _ = shell._assess_step_evidence(step, {"tool_results": [("read", {"columns": ["a", "b"]})], "text": ""})
    assert ok3 is True
    ok4, _ = shell._assess_step_evidence(step, {"tool_results": [("shell", {"stdout": "x" * 60})], "text": ""})
    assert ok4 is True


def test_evidence_contract_exempts_steps_without_declared_verification(tmp_path):
    import types
    shell, _ = _shell(tmp_path)
    step = types.SimpleNamespace(verification=[], title="reason about approach", id="s1")
    ok, reason = shell._assess_step_evidence(step, {"tool_results": [], "text": ""})
    assert ok is True and "no verification contract" in reason


def test_evidence_contract_can_be_disabled(tmp_path):
    import types
    shell, _ = _shell(tmp_path)
    shell.settings.evidence_contract_enabled = False
    step = types.SimpleNamespace(verification=["must write a file"], title="t", id="s1")
    ok, _ = shell._assess_step_evidence(step, {"tool_results": [], "text": ""})
    assert ok is True


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
