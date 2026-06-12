"""Controlled local execution skills for diagnosis and bounded repair."""

from __future__ import annotations

import subprocess
import sys
import uuid
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
    root = getattr(ctx, "workspace_root", None) if ctx is not None else None
    if root:
        return Path(root).expanduser()
    settings = getattr(ctx, "settings", None)
    root = getattr(settings, "project_root", "") if settings else ""
    return Path(root).expanduser() if root else Path.cwd()


def _workdir(ctx, cwd: str = "") -> Path:
    workspace = _workspace(ctx).resolve()
    extra_roots = []
    if ctx is not None:
        try:
            extra_roots = list(getattr(ctx, "extra_roots", []) or [])
        except Exception:
            extra_roots = []
    allowed_roots = [workspace]
    for root in extra_roots:
        try:
            allowed_roots.append(Path(root).expanduser().resolve())
        except OSError:
            continue
    if str(cwd or "").strip():
        path = Path(str(cwd)).expanduser()
        candidate = path if path.is_absolute() else workspace / path
    else:
        candidate = workspace
    resolved = candidate.resolve()
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise PermissionError(f"cwd is outside allowed workspace roots: {resolved} (allowed: {roots})")
    return resolved


def _inline_log_path(ctx, tool: str = "shell_exec") -> Path | None:
    if ctx is None:
        return None
    settings = getattr(ctx, "settings", None)
    jobs_dir = str(getattr(settings, "jobs_dir_name", ".biobank_jobs") or ".biobank_jobs")
    call_id = str(getattr(ctx, "tool_call_id", "") or f"{tool}_{uuid.uuid4().hex[:12]}").replace("/", "_")
    path = _workspace(ctx) / jobs_dir / "inline" / f"{call_id}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_inline_log(path: Path | None, command: str, returncode, stdout: str, stderr: str) -> str | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            f"$ {command}",
            f"returncode={returncode}",
            "",
            "[stdout]",
            stdout or "",
            "",
            "[stderr]",
            stderr or "",
        ]),
        encoding="utf-8",
        errors="replace",
    )
    return str(path)


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
    from biobank_agent.runtime.proc import run_streaming
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
    try:
        workdir = _workdir(ctx, cwd)
    except PermissionError as exc:
        return {"status": "error", "error": str(exc), "command": command, "purpose": purpose, "returncode": None}
    if hasattr(ctx, "emit_progress"):
        ctx.emit_progress("shell", f"running: {command[:120]}", {"cwd": str(workdir), "purpose": purpose})
    eff_timeout = max(1, int(resolve_timeout(command=command, override=timeout_s,
                                             settings=getattr(ctx, "settings", None)) or 120))
    result = run_streaming(
        ["bash", "-lc", command],
        cwd=str(workdir),
        timeout=eff_timeout,
        line_sink=getattr(ctx, "emit_line", None),
        log_path=_inline_log_path(ctx, "shell_exec"),
        tail_chars=8000,
    )
    return {
        "status": "success" if result.ok else "error",
        "command": command,
        "cwd": str(workdir),
        "purpose": purpose,
        "returncode": result.returncode,
        "stdout": (result.stdout_tail or "")[-8000:],
        "stderr": (result.stderr_tail or "")[-8000:],
        "log_path": result.log_path,
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
    try:
        workdir = _workdir(ctx, cwd)
    except PermissionError as exc:
        return {"status": "error", "error": str(exc), "purpose": purpose, "returncode": None}
    if hasattr(ctx, "emit_progress"):
        ctx.emit_progress("python", f"running Python probe: {purpose or code[:80]}", {"cwd": str(workdir)})
    timeout = max(1, int(timeout_s or 120))
    log_path = _inline_log_path(ctx, "python_exec")
    command_label = f"{sys.executable} -c <python>"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = str(exc.stdout or "")
        stderr = f"Timed out after {timeout}s. {exc.stderr or ''}"
        written_log = _write_inline_log(log_path, command_label, None, stdout, stderr)
        return {
            "status": "error",
            "cwd": str(workdir),
            "purpose": purpose,
            "returncode": None,
            "stdout": stdout[-8000:],
            "stderr": stderr[-8000:],
            "log_path": written_log,
        }
    written_log = _write_inline_log(log_path, command_label, proc.returncode, proc.stdout or "", proc.stderr or "")
    return {
        "status": "success" if proc.returncode == 0 else "error",
        "cwd": str(workdir),
        "purpose": purpose,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-8000:],
        "stderr": (proc.stderr or "")[-8000:],
        "log_path": written_log,
    }
