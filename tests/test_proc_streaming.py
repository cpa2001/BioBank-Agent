"""Tests for the streaming subprocess substrate (issue #2: live + persisted output)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from biobank_agent.runtime.proc import run_streaming
from biobank_agent.core.tools.native import ShellTool
from biobank_agent.core.tools.protocol import ToolContext


def test_run_streaming_captures_streams_and_log(tmp_path):
    lines = []
    log = tmp_path / "exec.log"
    result = run_streaming(
        ["bash", "-c", "echo out1; echo err1 1>&2; echo out2"],
        line_sink=lambda stream, line: lines.append((stream, line)),
        log_path=log,
    )
    assert result.ok and result.returncode == 0 and not result.timed_out
    assert "out1" in result.stdout_tail and "out2" in result.stdout_tail
    assert "err1" in result.stderr_tail
    assert ("stdout", "out1") in lines and ("stderr", "err1") in lines
    assert log.exists()
    body = log.read_text()
    assert "out1" in body and "err1" in body and "out2" in body
    assert result.log_path == str(log)


def test_run_streaming_times_out_and_kills(tmp_path):
    start = time.monotonic()
    result = run_streaming(["bash", "-c", "echo go; sleep 10; echo never"], timeout=1)
    elapsed = time.monotonic() - start
    assert result.timed_out is True
    assert result.ok is False
    assert elapsed < 6  # killed promptly, not after the full 10s
    assert "Timed out after 1s" in result.stderr_tail


def test_run_streaming_nonexistent_command():
    result = run_streaming(["this_command_does_not_exist_xyz_123"])
    assert result.ok is False
    assert result.returncode is None


def test_run_streaming_tail_is_bounded(tmp_path):
    result = run_streaming(["bash", "-c", "for i in $(seq 1 2000); do echo line_$i; done"], tail_chars=200)
    assert len(result.stdout_tail) <= 200
    assert "line_2000" in result.stdout_tail   # tail keeps the END
    assert "line_1\n" not in result.stdout_tail  # head dropped


class _Settings:
    jobs_dir_name = ".biobank_jobs"
    exec_stream_tail_chars = 5000
    exec_default_timeout_s = 120
    exec_long_tool_timeout_s = 7200
    exec_hard_ceiling_s = 86400
    exec_allow_no_timeout = True


def _shell_ctx(tmp_path, command):
    ctx = ToolContext(name="shell", args={"command": command}, capabilities=set())
    ctx.workspace_root = tmp_path
    ctx.settings = _Settings()
    ctx.tool_call_id = "call_test"
    return ctx


def test_shell_tool_streams_and_logs(tmp_path):
    seen = []
    ctx = _shell_ctx(tmp_path, "echo hello; echo oops 1>&2")
    ctx.emit_line = lambda stream, line: seen.append((stream, line))
    out = asyncio.run(ShellTool().handle(ctx))
    assert out["status"] == "ok"
    assert "hello" in out["stdout"]
    assert ("stdout", "hello") in seen
    log_path = out.get("log_path")
    assert log_path and Path(log_path).exists()
    assert ".biobank_jobs" in log_path
    assert "hello" in Path(log_path).read_text()


def test_shell_tool_nonzero_exit_surfaces_error(tmp_path):
    ctx = _shell_ctx(tmp_path, "echo boom 1>&2; exit 3")
    out = asyncio.run(ShellTool().handle(ctx))
    assert out["status"] == "error"
    assert out["returncode"] == 3
    assert "command exited 3" in out["error"]
