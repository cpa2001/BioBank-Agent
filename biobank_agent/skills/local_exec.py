"""Controlled local execution skills for diagnosis and bounded repair."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from biobank_agent.registry import skill


_MUTATING_HINTS = (
    " pip install",
    "python -m pip",
    " rm ",
    " mv ",
    " cp ",
    " chmod ",
    " chown ",
    " >",
    "tee ",
    "apply_patch",
)


def _workspace(ctx) -> Path:
    settings = getattr(ctx, "settings", None)
    root = getattr(settings, "project_root", "") if settings else ""
    return Path(root).expanduser() if root else Path.cwd()


def _requires_confirmation(command: str, write_policy: str, confirmed: bool) -> str:
    policy = (write_policy or "read_only").strip().lower()
    padded = f" {command.strip()} "
    mutating = policy not in {"read_only", "readonly", "diagnostic"} or any(hint in padded for hint in _MUTATING_HINTS)
    if mutating and not confirmed:
        return "Command may mutate the environment or workspace; rerun with confirmed=true after user approval."
    return ""


@skill(
    name="shell_exec",
    description=(
        "Run a bounded local shell command for Biobank Agent diagnosis or approved repair. "
        "Read-only probes may run directly; installs or workspace mutations require confirmed=true."
    ),
    parameters={
        "command": {"type": "string", "description": "Shell command to run"},
        "purpose": {"type": "string", "description": "Why this command is needed", "default": ""},
        "cwd": {"type": "string", "description": "Working directory; defaults to project root", "default": ""},
        "timeout_s": {"type": "integer", "description": "Timeout in seconds; 0 = auto (scales for long tools like plink/gatk/bcftools)", "default": 0},
        "write_policy": {"type": "string", "description": "read_only, workspace_write, or environment_write", "default": "read_only"},
        "confirmed": {"type": "boolean", "description": "Required before mutation/install commands", "default": False},
    },
    required=["command"],
)
def shell_exec(
    command: str,
    purpose: str = "",
    cwd: str = "",
    timeout_s: int = 0,
    write_policy: str = "read_only",
    confirmed: bool = False,
    *,
    ctx=None,
) -> dict:
    from biobank_agent.utils.exec_policy import resolve_timeout

    command = str(command or "").strip()
    if not command:
        return {"error": "command is required"}
    confirmation_error = _requires_confirmation(command, write_policy, confirmed)
    if confirmation_error:
        return {
            "status": "requires_confirmation",
            "error": confirmation_error,
            "command": command,
            "purpose": purpose,
            "write_policy": write_policy,
        }
    workdir = Path(cwd).expanduser() if cwd else _workspace(ctx)
    if hasattr(ctx, "emit_progress"):
        ctx.emit_progress("shell", f"running: {command[:120]}", {"cwd": str(workdir), "purpose": purpose})
    eff_timeout = max(1, int(resolve_timeout(command=command, override=timeout_s,
                                             settings=getattr(ctx, "settings", None)) or 120))
    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=eff_timeout,
        check=False,
    )
    return {
        "status": "success" if proc.returncode == 0 else "error",
        "command": command,
        "cwd": str(workdir),
        "purpose": purpose,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-8000:],
        "stderr": (proc.stderr or "")[-8000:],
    }


@skill(
    name="python_exec",
    description=(
        "Run bounded Python code for Biobank Agent diagnosis or approved repair. "
        "Use for quick data/code probes; workspace or environment writes require confirmed=true."
    ),
    parameters={
        "code": {"type": "string", "description": "Python code to execute with python -c"},
        "purpose": {"type": "string", "description": "Why this code is needed", "default": ""},
        "cwd": {"type": "string", "description": "Working directory; defaults to project root", "default": ""},
        "timeout_s": {"type": "integer", "description": "Timeout in seconds", "default": 120},
        "write_policy": {"type": "string", "description": "read_only, workspace_write, or environment_write", "default": "read_only"},
        "confirmed": {"type": "boolean", "description": "Required before mutation/install code", "default": False},
    },
    required=["code"],
)
def python_exec(
    code: str,
    purpose: str = "",
    cwd: str = "",
    timeout_s: int = 120,
    write_policy: str = "read_only",
    confirmed: bool = False,
    *,
    ctx=None,
) -> dict:
    code = str(code or "")
    if not code.strip():
        return {"error": "code is required"}
    confirmation_error = _requires_confirmation(code, write_policy, confirmed)
    if confirmation_error:
        return {
            "status": "requires_confirmation",
            "error": confirmation_error,
            "purpose": purpose,
            "write_policy": write_policy,
        }
    workdir = Path(cwd).expanduser() if cwd else _workspace(ctx)
    if hasattr(ctx, "emit_progress"):
        ctx.emit_progress("python", f"running Python probe: {purpose or code[:80]}", {"cwd": str(workdir)})
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=max(1, int(timeout_s or 120)),
        check=False,
    )
    return {
        "status": "success" if proc.returncode == 0 else "error",
        "cwd": str(workdir),
        "purpose": purpose,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-8000:],
        "stderr": (proc.stderr or "")[-8000:],
    }
