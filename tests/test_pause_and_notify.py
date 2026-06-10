"""Tests for the pause-and-ask gate and notifications (issues #3, #1)."""

from __future__ import annotations

import json
import types
from pathlib import Path

from biobank_agent.cli.interactive import InteractiveShell
from biobank_agent.core.tools.registry import ToolRegistry
from biobank_agent.runtime.notify import notify
from biobank_agent.skills.pause_and_ask import pause_and_ask


def test_pause_and_ask_returns_awaiting_user_sentinel():
    out = pause_and_ask(
        question="VCF has no FORMAT/AD — convert it or skip the burden test?",
        reason="burden test needs AD",
        category="data_format",
        options="convert with bcftools, skip burden",
    )
    assert out["awaiting_user"] is True
    assert out["category"] == "data_format"
    assert out["options"] == ["convert with bcftools", "skip burden"]
    assert "do not write a simplified replacement" in out["message"].lower()


def test_pause_and_ask_requires_question():
    assert pause_and_ask(question="")["status"] == "error"


def test_pause_and_ask_normalizes_bad_category():
    assert pause_and_ask(question="q", category="nonsense")["category"] == "ambiguous_input"


def test_pause_and_ask_registered_as_runtime_tool():
    reg = ToolRegistry()
    reg.hydrate_from_legacy()
    assert "pause_and_ask" in reg._handlers


def _session_with_last_result(name, result):
    tr = types.SimpleNamespace(name=name, result=result)
    turn = types.SimpleNamespace(tool_results=[tr], assistant_messages=[])
    return types.SimpleNamespace(turns=[turn])


def test_detector_recognizes_pause_and_ask():
    session = _session_with_last_result("pause_and_ask", {
        "awaiting_user": True, "question": "Convert the VCF?",
        "reason": "needs AD", "options": ["convert", "skip"],
    })
    q = InteractiveShell._pending_ask_user_question(session)
    assert "Convert the VCF?" in q and "needs AD" in q and "convert" in q


def test_detector_still_recognizes_ask_user():
    session = _session_with_last_result("ask_user", {"awaiting_user": True, "question": "Where is the data?"})
    assert InteractiveShell._pending_ask_user_question(session) == "Where is the data?"


def test_detector_ignores_non_awaiting_results():
    session = _session_with_last_result("shell", {"status": "ok", "stdout": "done"})
    assert InteractiveShell._pending_ask_user_question(session) == ""


def test_notify_writes_jsonl(tmp_path):
    class S:
        notify_enabled = True
        notify_command = ""
        memory_dir = str(tmp_path)
    notify("job_done", title="t", body="b", settings=S())
    log = tmp_path / "notifications.jsonl"
    assert log.exists()
    rec = json.loads(log.read_text().strip())
    assert rec["event"] == "job_done" and rec["title"] == "t"


def test_notify_respects_disabled(tmp_path):
    class S:
        notify_enabled = False
        notify_command = ""
        memory_dir = str(tmp_path)
    notify("x", settings=S())
    assert not (tmp_path / "notifications.jsonl").exists()


def test_notify_never_raises_on_bad_settings():
    # no settings at all — must not raise (writes to default memory dir, best-effort)
    notify("evt", title="t", body="b")
