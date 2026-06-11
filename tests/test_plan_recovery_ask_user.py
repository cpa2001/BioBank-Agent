"""Tests for autonomous step recovery + the ask_user clarification channel.

Covers the runtime-backed ``InteractiveShell`` plan loop additions:
- ``_pending_ask_user_question`` sentinel detection,
- ``_plan_answer_suffix`` context folding,
- ``_run_step_with_recovery`` retry -> augmented-retry -> needs_input/failed ladder,
- ``_handle_plan_natural_language`` treating a reply as the answer to a pending question.
"""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rich.console import Console

from biobank_agent.cli.interactive import InteractiveShell
from biobank_agent.core.events import AgentEventType
from biobank_agent.core.tools.native import build_native_tools

TCC = AgentEventType.TOOL_CALL_COMPLETED.value
ERR = AgentEventType.ERROR.value


# ── fakes ────────────────────────────────────────────────────


class FakeToolResult:
    def __init__(self, name, result=None, error=""):
        self.name = name
        self.result = result or {}
        self.error = error


class FakeTurn:
    def __init__(self, status, tool_results=None):
        self.status = status
        self.tool_results = list(tool_results or [])


def _session():
    return SimpleNamespace(
        turns=[],
        events=[],
        state=SimpleNamespace(custom_data={}, pending_work=[]),
    )


def _step():
    return SimpleNamespace(
        id="s1", title="Probe VCF", purpose="probe the vcf state",
        tool_scope=["vcf_qc"], file_scope=["*.vcf.gz"], verification=["counts logged"],
    )


def _shell(session, run_turn_script, *, max_retries=2):
    shell = InteractiveShell(
        settings=SimpleNamespace(plan_step_max_retries=max_retries),
        console=Console(file=StringIO(), force_terminal=True),
    )
    shell.session = session
    rt = MagicMock()
    rt.run_turn.side_effect = run_turn_script
    shell.runtime = rt
    shell._recent_plan_error_details = lambda: "stub failure reason"
    return shell


# ── ask_user / native tool ───────────────────────────────────


def test_ask_user_tool_registered_and_safe():
    tools = {t.name: t for t in build_native_tools()}
    assert "ask_user" in tools
    au = tools["ask_user"]
    assert au.is_mutating is False
    assert set(au.required_capabilities()) == set()  # asking != acting → no caps


def test_pending_ask_user_question_detected_when_last_call():
    s = _session()
    s.turns.append(FakeTurn("completed", [
        FakeToolResult("vcf_qc", {"status": "ok"}),
        FakeToolResult("ask_user", {"awaiting_user": True, "question": "Where are the VCFs?"}),
    ]))
    assert InteractiveShell._pending_ask_user_question(s) == "Where are the VCFs?"


def test_pending_ask_user_question_ignored_when_not_last():
    # Agent asked then self-resolved (kept working) → not treated as blocked.
    s = _session()
    s.turns.append(FakeTurn("completed", [
        FakeToolResult("ask_user", {"awaiting_user": True, "question": "x?"}),
        FakeToolResult("shell", {"status": "ok"}),
    ]))
    assert InteractiveShell._pending_ask_user_question(s) == ""


def test_pending_ask_user_question_empty_session():
    assert InteractiveShell._pending_ask_user_question(_session()) == ""


def test_plan_answer_suffix():
    s = _session()
    assert InteractiveShell._plan_answer_suffix(s) == ""
    s.state.custom_data["plan_user_answers"] = [{"q": "Where?", "a": "/data/vc"}]
    suffix = InteractiveShell._plan_answer_suffix(s)
    assert "/data/vc" in suffix and "Where?" in suffix


# ── recovery ladder ──────────────────────────────────────────


def test_recovery_ok_on_first_attempt():
    s = _session()

    def script(session, instruction):
        session.events.append({"type": TCC})
        session.turns.append(FakeTurn("completed", [FakeToolResult("vcf_qc", {"status": "ok"})]))

    shell = _shell(s, script)
    with patch("biobank_agent.cli.interactive.time.sleep"):
        status, reason, calls, q = shell._run_step_with_recovery(MagicMock(), _step(), 1, 9, MagicMock())
    assert status == "ok"
    assert shell.runtime.run_turn.call_count == 1


def test_recovery_retries_then_succeeds():
    s = _session()
    state = {"n": 0}

    def script(session, instruction):
        state["n"] += 1
        if state["n"] < 3:  # attempts 0,1 fail
            session.events.append({"type": ERR, "message": "boom"})
            session.turns.append(FakeTurn("failed", [FakeToolResult("vcf_qc", {"status": "error"}, "boom")]))
        else:  # attempt 2 succeeds
            session.events.append({"type": TCC})
            session.turns.append(FakeTurn("completed", [FakeToolResult("vcf_qc", {"status": "ok"})]))

    shell = _shell(s, script, max_retries=2)
    with patch("biobank_agent.cli.interactive.time.sleep"):
        status, reason, calls, q = shell._run_step_with_recovery(MagicMock(), _step(), 1, 9, MagicMock())
    assert status == "ok"
    assert shell.runtime.run_turn.call_count == 3  # 1 original + 2 retries


def test_recovery_exhausts_then_fails():
    s = _session()

    def script(session, instruction):
        session.events.append({"type": ERR, "message": "always"})
        session.turns.append(FakeTurn("failed", [FakeToolResult("vcf_qc", {"status": "error"}, "always")]))

    shell = _shell(s, script, max_retries=2)
    with patch("biobank_agent.cli.interactive.time.sleep"):
        status, reason, calls, q = shell._run_step_with_recovery(MagicMock(), _step(), 1, 9, MagicMock())
    assert status == "failed"
    assert reason == "stub failure reason"
    assert shell.runtime.run_turn.call_count == 3  # original + 2 retries, no infinite loop


def test_recovery_stops_to_ask_user():
    s = _session()

    def script(session, instruction):
        session.events.append({"type": TCC})
        session.turns.append(FakeTurn("completed", [
            FakeToolResult("ask_user", {"awaiting_user": True, "question": "Which cohort is the case group?"}),
        ]))

    shell = _shell(s, script, max_retries=2)
    with patch("biobank_agent.cli.interactive.time.sleep"):
        status, reason, calls, q = shell._run_step_with_recovery(MagicMock(), _step(), 1, 9, MagicMock())
    assert status == "needs_input"
    assert q == "Which cohort is the case group?"
    assert shell.runtime.run_turn.call_count == 1  # no retries once blocked on user input


# ── pending-answer routing ───────────────────────────────────


def _router_shell(session):
    shell = InteractiveShell(
        settings=SimpleNamespace(plan_step_max_retries=2),
        console=Console(file=StringIO(), force_terminal=True),
    )
    shell.session = session
    shell.runtime = MagicMock()
    shell._cmd_plan_retry = MagicMock()
    shell._cmd_plan_use = MagicMock()
    shell._cmd_plan_diagnose = MagicMock()
    shell._cmd_plan_edit = MagicMock()
    return shell


def test_reply_to_pending_question_resumes_with_answer():
    s = _session()
    s.state.custom_data["plan_pending_question"] = "Where are the VCFs?"
    shell = _router_shell(s)
    shell._handle_plan_natural_language("the data is in data/vc_wgs_vcf")
    answers = s.state.custom_data.get("plan_user_answers")
    assert answers and answers[-1]["a"] == "the data is in data/vc_wgs_vcf"
    assert "plan_pending_question" not in s.state.custom_data
    shell._cmd_plan_retry.assert_called_once()


def test_reply_with_path_also_updates_context():
    s = _session()
    s.state.custom_data["plan_pending_question"] = "Where are the VCFs?"
    shell = _router_shell(s)
    # "vcf_dir=..." classifies as "use" → also folded into plan context.
    shell._handle_plan_natural_language("vcf_dir=data/vc_wgs_vcf")
    shell._cmd_plan_use.assert_called_once()
    shell._cmd_plan_retry.assert_called_once()
    assert "plan_pending_question" not in s.state.custom_data


def test_diagnose_intent_does_not_consume_pending_question():
    s = _session()
    s.state.custom_data["plan_pending_question"] = "Where are the VCFs?"
    shell = _router_shell(s)
    shell._handle_plan_natural_language("why did it fail?")
    # Explicit diagnose must NOT be swallowed as an answer.
    shell._cmd_plan_diagnose.assert_called_once()
    shell._cmd_plan_retry.assert_not_called()
    assert s.state.custom_data.get("plan_pending_question") == "Where are the VCFs?"
