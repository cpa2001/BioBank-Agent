"""Tests for the runtime completion gates."""

from __future__ import annotations

from biobank_agent.runtime.completion import CompletionGate
from biobank_agent.runtime.engine import ProviderRouter
from biobank_agent.runtime.types import PlanState, PlanStep, ProviderResponse, RuntimeConfig


class _FakeProvider:
    def __init__(self, handler):
        self.handler = handler

    def complete(self, request):
        return ProviderResponse(text=self.handler(request), provider="fake", model=request.model or "m")


def _router(handler):
    cfg = RuntimeConfig(primary_model="m", planner_model="m", critic_model="m", summarizer_model="m", safety_reviewer_model="m")
    return ProviderRouter({"m": _FakeProvider(handler)}, cfg)


def _plan(with_report=True):
    steps = [
        PlanStep(id="qc", title="QC the VCF", status="done", tool_scope=["vcf_qc"]),
        PlanStep(id="assoc", title="Association test", status="done", tool_scope=["vcf_association"]),
    ]
    if with_report:
        steps.append(PlanStep(id="reporting", title="Compile report", status="pending", tool_scope=["report"]))
    return PlanState(objective="vitiligo WGS analysis", steps=steps)


def test_accepts_when_goal_met_and_report_present():
    gate = CompletionGate(_router(lambda req: '{"accepted": true, "missing": [], "reasons": "all steps complete and report produced"}'))
    a = gate.assess("vitiligo WGS analysis", _plan(with_report=True), evidence_summary="Tools run: vcf_qc:ok, report:ok", report_present=True)
    assert a.accepted is True
    assert a.goal_checked is True


def test_report_contract_blocks_acceptance_when_report_missing():
    # Judge says accepted, but the plan promised a report and none exists -> blocked.
    gate = CompletionGate(_router(lambda req: '{"accepted": true, "missing": [], "reasons": "looks done"}'))
    a = gate.assess("vitiligo WGS analysis", _plan(with_report=True), evidence_summary="Tools run: vcf_qc:ok", report_present=False)
    assert a.accepted is False
    assert a.needs_report is True and a.report_present is False
    assert any("report" in w.lower() for w in a.warnings)


def test_not_accepted_surfaces_missing_items():
    gate = CompletionGate(_router(lambda req: '{"accepted": false, "missing": ["association test not run"], "reasons": "incomplete"}'))
    a = gate.assess("obj", _plan(with_report=False), evidence_summary="Tools run: vcf_qc:ok", report_present=False)
    assert a.accepted is False
    assert "association test not run" in a.missing


def test_judge_unavailable_does_not_auto_accept():
    gate = CompletionGate(_router(lambda req: "not json at all"))
    a = gate.assess("obj", _plan(with_report=False), evidence_summary="x", report_present=False)
    assert a.accepted is False
    assert a.goal_checked is False


def test_unfinished_steps_warn():
    gate = CompletionGate(_router(lambda req: '{"accepted": false, "missing": [], "reasons": ""}'))
    plan = _plan(with_report=False)
    plan.steps[1].status = "pending"
    a = gate.assess("obj", plan, evidence_summary="Tools run: vcf_qc:ok", report_present=False)
    assert any("not marked complete" in w for w in a.warnings)


def test_string_accepted_false_does_not_accept():
    # A judge that returns the STRING "false" (truthy) must not be accepted.
    gate = CompletionGate(_router(lambda req: '{"accepted": "false", "missing": [], "reasons": "x"}'))
    a = gate.assess("obj", _plan(with_report=False), evidence_summary="x", report_present=False)
    assert a.accepted is False


def test_failed_report_tool_does_not_satisfy_contract(tmp_path):
    from biobank_agent.core.events import AgentEvent, AgentEventType

    from tests.test_interactive_cli_runtime import _shell

    shell, _ = _shell(tmp_path)
    session = shell.session
    session.events.append(
        AgentEvent.make(AgentEventType.TOOL_CALL_COMPLETED, session_id=session.session_id, tool="generate_report", state="failed").to_dict()
    )
    _summary, report_present = shell._build_completion_evidence(session)
    assert report_present is False  # failed report tool must NOT satisfy the contract
    session.events.append(
        AgentEvent.make(AgentEventType.TOOL_CALL_COMPLETED, session_id=session.session_id, tool="generate_report", state="done").to_dict()
    )
    _summary2, report_present2 = shell._build_completion_evidence(session)
    assert report_present2 is True  # a successful report tool does


def test_repeated_verify_respects_repair_budget(tmp_path):
    from tests.test_interactive_cli_runtime import _shell

    shell, output = _shell(tmp_path)
    shell.handle_line("/plan do a small analysis")
    for _ in range(4):
        shell.handle_line("/verify")  # never accepted (fake judge returns no boolean accept)
    assert len(shell.session.state.repair_attempts) <= 2  # bounded budget
    assert "budget exhausted" in output.getvalue().lower()


def test_verify_no_arg_runs_completion_assessment(tmp_path):
    from tests.test_interactive_cli_runtime import _shell

    shell, output = _shell(tmp_path)
    shell.handle_line("/plan do a small analysis")
    shell.handle_line("/verify")  # no command -> completion assessment, not "usage"
    text = output.getvalue()
    assert "Completion Assessment" in text
    assert "Usage: /verify" not in text


def test_propose_repair_returns_text():
    def handler(req):
        content = req.messages[-1]["content"]
        if "Outstanding gaps" in content:
            return "1. Run the association test.\n2. Generate the report."
        return '{"accepted": false, "missing": ["report"], "reasons": "no report"}'

    gate = CompletionGate(_router(handler))
    plan = _plan(with_report=True)
    a = gate.assess("obj", plan, evidence_summary="x", report_present=False)
    proposal = gate.propose_repair("obj", plan, a)
    assert "association test" in proposal.lower() or "report" in proposal.lower()
