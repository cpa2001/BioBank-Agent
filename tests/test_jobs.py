"""Tests for the background JobManager and job tools (issue #1: long tasks)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from biobank_agent.runtime.jobs import JobManager
from biobank_agent.core.tools.native import (
    RunJobTool, JobStatusTool, JobWaitTool, JobListTool, JobCancelTool,
)
from biobank_agent.core.tools.protocol import ToolContext


def _jm(tmp_path) -> JobManager:
    return JobManager(tmp_path / ".biobank_jobs")


def test_job_completes_with_output(tmp_path):
    jm = _jm(tmp_path)
    rec = jm.submit(["bash", "-c", "echo hello; echo world"], cwd=str(tmp_path))
    assert rec.state == "running" and rec.pid
    final = jm.wait(rec.job_id, timeout=10)
    assert final.state == "done" and final.returncode == 0
    assert "hello" in final.tail_stdout and "world" in final.tail_stdout
    assert Path(final.log_path).exists()


def test_job_failure_records_returncode(tmp_path):
    jm = _jm(tmp_path)
    rec = jm.submit(["bash", "-c", "echo oops 1>&2; exit 3"], cwd=str(tmp_path))
    final = jm.wait(rec.job_id, timeout=10)
    assert final.state == "failed" and final.returncode == 3


def test_job_timeout(tmp_path):
    jm = _jm(tmp_path)
    rec = jm.submit(["bash", "-c", "sleep 30"], cwd=str(tmp_path), timeout=1)
    final = jm.wait(rec.job_id, timeout=10)
    assert final.state == "timeout"


def test_job_cancel_kills_process(tmp_path):
    import os
    jm = _jm(tmp_path)
    rec = jm.submit(["bash", "-c", "sleep 30"], cwd=str(tmp_path))
    cancelled = jm.cancel(rec.job_id)
    assert cancelled.state == "cancelled"
    # the process group is dead
    import time as _t
    _t.sleep(0.3)
    try:
        os.kill(rec.pid, 0)
        alive = True
    except OSError:
        alive = False
    assert alive is False


def test_reattach_marks_dead_orphan_terminal(tmp_path):
    jm = _jm(tmp_path)
    orphan = tmp_path / ".biobank_jobs" / "orphan"
    orphan.mkdir(parents=True)
    (orphan / "state.json").write_text(json.dumps({
        "job_id": "orphan", "cmd": "x", "cwd": ".", "state": "running", "pid": 999999,
    }))
    recovered = jm.reattach()
    rec = next(r for r in recovered if r.job_id == "orphan")
    assert rec.is_terminal  # not left as a phantom "running"


def test_notifier_fires_on_completion(tmp_path):
    notes = []
    jm = JobManager(tmp_path / ".biobank_jobs",
                    notifier=lambda ev, title, body: notes.append(ev))
    rec = jm.submit(["bash", "-c", "echo ok"], cwd=str(tmp_path))
    jm.wait(rec.job_id, timeout=10)
    assert "job_done" in notes


class _Settings:
    jobs_dir_name = ".biobank_jobs"
    exec_default_timeout_s = 120
    exec_long_tool_timeout_s = 7200
    exec_hard_ceiling_s = 86400
    exec_allow_no_timeout = True
    exec_background_threshold_s = 600


def _ctx(tmp_path, jm, args):
    ctx = ToolContext(name="x", args=args, capabilities=set())
    ctx.workspace_root = tmp_path
    ctx.settings = _Settings()
    ctx.job_manager = jm
    ctx.tool_call_id = "call_1"
    return ctx


def test_run_job_and_wait_tools(tmp_path):
    jm = _jm(tmp_path)
    out = asyncio.run(RunJobTool().handle(_ctx(tmp_path, jm, {"command": "echo hi; sleep 1", "label": "demo"})))
    assert out["status"] == "ok" and out["job_id"]
    assert "background" in out["message"]
    final = asyncio.run(JobWaitTool().handle(_ctx(tmp_path, jm, {"job_id": out["job_id"], "timeout_s": 10})))
    assert final["state"] == "done"
    listing = asyncio.run(JobListTool().handle(_ctx(tmp_path, jm, {})))
    assert listing["count"] == 1


def test_job_tools_degrade_without_manager(tmp_path):
    ctx = ToolContext(name="x", args={"command": "echo x"}, capabilities=set())
    ctx.workspace_root = tmp_path
    ctx.settings = _Settings()
    out = asyncio.run(RunJobTool().handle(ctx))
    assert out["status"] == "error" and "not available" in out["error"]


def test_cancel_tool(tmp_path):
    jm = _jm(tmp_path)
    rec = jm.submit(["bash", "-c", "sleep 30"], cwd=str(tmp_path))
    out = asyncio.run(JobCancelTool().handle(_ctx(tmp_path, jm, {"job_id": rec.job_id})))
    assert out["state"] == "cancelled"
