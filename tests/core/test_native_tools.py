"""Tests for native workspace tools and their runtime wiring."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from biobank_agent.core.events import AgentEvent, AgentEventType
from biobank_agent.core.tools.approval import ApprovalPolicy, builtin_profile
from biobank_agent.core.tools.native import (
    ApplyPatchTool,
    EditFileTool,
    GitDiffTool,
    GitStatusTool,
    ReadFileTool,
    SearchTool,
    ShellTool,
    WriteFileTool,
    TestRunnerTool,
)
from biobank_agent.core.tools.protocol import Capability, ToolContext
from biobank_agent.core.tools.scheduler import ToolRequest, ToolScheduler
from biobank_agent.core.tools.registry import ToolRegistry
from biobank_agent.core.events import AgentEventBus


class _Recorder:
    def __init__(self) -> None:
        self.trajectory: list[dict] = []
        self.graph: list[dict] = []

    def record_trajectory(self, payload):
        self.trajectory.append(dict(payload))

    def record_action_graph(self, payload):
        self.graph.append(dict(payload))


def _ctx(tmp_path: Path, **kwargs):
    rec = kwargs.pop("recorder", None) or _Recorder()
    ctx = ToolContext(
        name="tool",
        args={},
        capabilities=frozenset({Capability.READ_DATA, Capability.WRITE_REPORTS, Capability.SHELL_EXEC}),
        workspace_root=tmp_path,
        permission_mode="workspace_write",
        record_trajectory=rec.record_trajectory,
        record_action_graph=rec.record_action_graph,
        **kwargs,
    )
    return ctx, rec


@pytest.mark.asyncio
async def test_file_read_write_edit_and_path_guard(tmp_path):
    (tmp_path / "demo.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    ctx, rec = _ctx(tmp_path)

    read = ReadFileTool()
    ctx.args = {"path": "demo.txt", "max_bytes": 16}
    out = await read.handle(ctx)
    assert out["status"] == "ok"
    assert "alpha" in out["content"]

    write = WriteFileTool()
    ctx.args = {"path": "written.txt", "content": "hello"}
    out = await write.handle(ctx)
    assert out["status"] == "ok"
    assert (tmp_path / "written.txt").read_text(encoding="utf-8") == "hello"

    edit = EditFileTool()
    ctx.args = {"path": "demo.txt", "replacements": [{"find": "alpha", "replace": "gamma"}]}
    out = await edit.handle(ctx)
    assert out["status"] == "ok"
    assert "gamma" in (tmp_path / "demo.txt").read_text(encoding="utf-8")

    ctx.args = {"path": "../escape.txt"}
    out = await read.handle(ctx)
    assert out["status"] == "error"

    assert rec.trajectory
    assert rec.graph


@pytest.mark.asyncio
async def test_file_read_refuses_secrets_and_redacts_values(tmp_path):
    """file_read must refuse .env / secret files outright and redact secret
    VALUES (not just key names) in any readable file."""
    (tmp_path / ".env").write_text("LLM_API_KEY=sk-or-secret123456\n")
    (tmp_path / "cfg.txt").write_text("API_KEY=sk-or-leak9999999\nplain=ok\n")
    tool = ReadFileTool()
    ctx, _ = _ctx(tmp_path)

    ctx.args = {"path": ".env"}
    out = await tool.handle(ctx)
    assert out["status"] == "error" and "sensitive" in out["error"].lower()

    ctx.args = {"path": "cfg.txt"}
    out = await tool.handle(ctx)
    assert out["status"] == "ok"
    assert "[REDACTED]" in out["content"]
    assert "sk-or-leak" not in out["content"]
    assert "plain=ok" in out["content"]


@pytest.mark.asyncio
async def test_shell_nonzero_exit_populates_error_field(tmp_path):
    """A nonzero exit must populate `error` (not just status) so the scheduler,
    loop guard, and audit treat it as a failed tool call."""
    shell = ShellTool()
    ctx, _ = _ctx(tmp_path)
    ctx.args = {"command": "python -c 'import sys; sys.stderr.write(\"boom\\n\"); sys.exit(3)'"}
    out = await shell.handle(ctx)
    assert out["status"] == "error"
    assert out.get("error"), "nonzero exit must set an error field"
    assert "3" in out["error"]


@pytest.mark.asyncio
async def test_shell_redaction_timeout_and_classification(tmp_path):
    shell = ShellTool()
    ctx, _ = _ctx(tmp_path)

    ctx.args = {"command": "python -c 'print(\"api_key=secret\")'"}
    out = await shell.handle(ctx)
    assert out["status"] == "ok"
    assert "[REDACTED]" in out["stdout"] or "api_key" not in out["stdout"]

    ctx.args = {"command": "python -c 'import time; time.sleep(1)'", "timeout_s": 1}
    out = await shell.handle(ctx)
    assert out["status"] == "error"

    ctx.args = {"command": "curl https://example.com"}
    out = await shell.handle(ctx)
    assert out["safety_class"] in {"network", "shell_safe"}


@pytest.mark.asyncio
async def test_search_and_git_and_test_runner(tmp_path):
    (tmp_path / "a.py").write_text("needle = 1\n", encoding="utf-8")
    ctx, _ = _ctx(tmp_path)

    search = SearchTool()
    ctx.args = {"pattern": "needle", "path": ".", "limit": 5}
    out = await search.handle(ctx)
    assert out["status"] == "ok"
    assert out["matches"]

    status = GitStatusTool()
    diff = GitDiffTool()
    out = await status.handle(ctx)
    assert out["status"] in {"ok", "error"}
    out = await diff.handle(ctx)
    assert out["status"] in {"ok", "error"}

    runner = TestRunnerTool()
    ctx.args = {"command": "python -c 'print(1)'", "timeout_s": 10}
    out = await runner.handle(ctx)
    assert out["status"] == "ok"


@pytest.mark.asyncio
async def test_apply_patch_rejects_path_traversal_on_dry_run(tmp_path):
    tool = ApplyPatchTool()
    ctx, _ = _ctx(tmp_path)
    ctx.args = {"path": "../escape.py", "diff": "print('x')", "dry_run": True}

    out = await tool.handle(ctx)

    assert out["status"] == "error"
    # Actionable denial message: states the path is outside the workspace.
    assert "outside the workspace" in out["error"] and "workspace root" in out["error"]


@pytest.mark.asyncio
async def test_scheduler_records_trajectory_and_action_graph(tmp_path):
    class _Handler:
        name = "demo"

        def spec(self):
            from biobank_agent.core.tools.protocol import ToolSpec

            return ToolSpec(name="demo", description="", parameters={})

        def required_capabilities(self):
            return frozenset({Capability.READ_DATA})

        @property
        def is_mutating(self):
            return False

        async def handle(self, ctx):
            return {"ok": True}

    bus = AgentEventBus()
    scheduler = ToolScheduler(policy=ApprovalPolicy(builtin_profile("workspace_write")), bus=bus)
    recorder = _Recorder()

    def factory(req):
        return ToolContext(
            name=req.handler.name,
            args=dict(req.args),
            capabilities=req.handler.required_capabilities(),
            workspace_root=tmp_path,
            record_trajectory=recorder.record_trajectory,
            record_action_graph=recorder.record_action_graph,
        )

    outcome = await scheduler.run(
        ToolRequest(call_id="c1", handler=_Handler(), args={}, turn_id="t1", context_factory=factory)
    )
    assert outcome.result["ok"] is True
    assert any(item["phase"] == "tool_started" for item in recorder.trajectory)
    assert any(item["phase"] == "tool_finished" for item in recorder.trajectory)


def test_tool_registry_hydrates_native_tools_with_legacy():
    registry = ToolRegistry()
    registry.hydrate_from_legacy()
    names = {handler.name for handler in registry.list_handlers()}
    assert {"shell", "file_read", "file_write", "file_edit", "apply_patch", "search", "git_status", "git_diff", "test_runner"} <= names
