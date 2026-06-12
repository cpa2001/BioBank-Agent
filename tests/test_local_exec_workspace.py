"""Workspace-root behavior for legacy direct execution skills."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from biobank_agent.core.tools.protocol import Capability, LegacySkillToolHandler, ToolContext, ToolSpec
from biobank_agent.skills.local_exec import python_exec, shell_exec


def test_legacy_handler_passes_ctx_through_var_kwargs(tmp_path):
    seen = {}

    def invoke(**kwargs):
        seen.update(kwargs)
        return {"workspace": str(kwargs["ctx"].workspace_root)}

    handler = LegacySkillToolHandler(
        name="legacy_probe",
        spec=ToolSpec(name="legacy_probe", description="probe", parameters={}),
        callable_=invoke,
        capabilities={Capability.READ_DATA},
    )
    ctx = ToolContext(
        name="legacy_probe",
        args={"value": "x"},
        capabilities={Capability.READ_DATA},
        workspace_root=tmp_path,
    )

    out = asyncio.run(handler.handle(ctx))

    assert seen["ctx"] is ctx
    assert out["workspace"] == str(tmp_path)


def test_python_exec_defaults_to_ctx_workspace_root(tmp_path):
    ctx = SimpleNamespace(workspace_root=tmp_path, settings=SimpleNamespace(), tool_call_id="py_probe")

    out = python_exec(
        "import os; print(os.getcwd())",
        timeout_s=10,
        ctx=ctx,
    )

    assert out["status"] == "success"
    assert out["cwd"] == str(tmp_path)
    assert str(tmp_path) in out["stdout"]
    assert out["log_path"].endswith(".biobank_jobs/inline/py_probe.log")
    assert "returncode=0" in tmp_path.joinpath(".biobank_jobs/inline/py_probe.log").read_text(encoding="utf-8")


def test_python_exec_relative_cwd_resolves_under_workspace(tmp_path):
    subdir = tmp_path / "analysis"
    subdir.mkdir()
    ctx = SimpleNamespace(workspace_root=tmp_path, settings=SimpleNamespace())

    out = python_exec(
        "import os; print(os.getcwd())",
        cwd="analysis",
        timeout_s=10,
        ctx=ctx,
    )

    assert out["status"] == "success"
    assert out["cwd"] == str(subdir)
    assert str(subdir) in out["stdout"]


def test_python_exec_rejects_cwd_outside_workspace(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    ctx = SimpleNamespace(workspace_root=tmp_path, settings=SimpleNamespace())

    out = python_exec(
        "print('should not run')",
        cwd=str(outside),
        timeout_s=10,
        ctx=ctx,
    )

    assert out["status"] == "error"
    assert "outside allowed workspace roots" in out["error"]


def test_shell_exec_rejects_parent_escape_from_workspace(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    ctx = SimpleNamespace(workspace_root=tmp_path, settings=SimpleNamespace())

    out = shell_exec("pwd", cwd=f"../{outside.name}", timeout_s=10, ctx=ctx)

    assert out["status"] == "error"
    assert "outside allowed workspace roots" in out["error"]


def test_shell_exec_allows_configured_extra_root(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    ctx = SimpleNamespace(workspace_root=tmp_path, extra_roots=[str(outside)], settings=SimpleNamespace())

    out = shell_exec("pwd", cwd=str(outside), timeout_s=10, ctx=ctx)

    assert out["status"] == "success"
    assert out["cwd"] == str(outside.resolve())
    assert str(outside.resolve()) in out["stdout"]


def test_shell_exec_defaults_to_ctx_workspace_root(tmp_path):
    ctx = SimpleNamespace(workspace_root=tmp_path, settings=SimpleNamespace())

    out = shell_exec("pwd", timeout_s=10, ctx=ctx)

    assert out["status"] == "success"
    assert out["cwd"] == str(tmp_path)
    assert str(tmp_path) in out["stdout"]
    assert out["log_path"].endswith(".log")


def test_python_exec_timeout_returns_structured_error_and_log(tmp_path):
    ctx = SimpleNamespace(workspace_root=tmp_path, settings=SimpleNamespace(), tool_call_id="py_timeout")

    out = python_exec(
        "import time; print('start'); time.sleep(5)",
        timeout_s=1,
        ctx=ctx,
    )

    assert out["status"] == "error"
    assert out["returncode"] is None
    assert "Timed out" in out["stderr"]
    assert out["log_path"].endswith(".biobank_jobs/inline/py_timeout.log")
    assert "Timed out" in tmp_path.joinpath(".biobank_jobs/inline/py_timeout.log").read_text(encoding="utf-8")
