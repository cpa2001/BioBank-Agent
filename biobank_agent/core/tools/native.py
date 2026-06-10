"""Native workspace tools for the runtime-backed BioBank shell."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from biobank_agent.core.evolution.auto_merger import AutoMerger, Patch
from biobank_agent.core.evolution.patch_classifier import classify as classify_patch
from biobank_agent.runtime.proc import run_streaming
from biobank_agent.utils.exec_policy import resolve_timeout, tool_key

from .protocol import (
    ActionClass,
    Capability,
    SafetyClass,
    TrajectorySerialization,
    ToolContext,
    ToolSpec,
    WorkspaceScope,
    _BaseHandler,
)


# (pattern, replacement) pairs. The KEY=VALUE pattern keeps the key name but
# redacts the VALUE (the old patterns masked only the key word, leaking the
# secret value, e.g. LLM_API_KEY=sk-...). The token patterns mask secrets
# wherever they appear in captured output.
_SECRET_SUBS = (
    (re.compile(r'(?im)^(\s*[\w.\-]*?(?:api[_-]?key|secret|token|password|passwd|auth|credential)[\w.\-]*\s*[=:]\s*)(["\']?)\S+'), r"\1\2[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b"), "[REDACTED]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{8,}\b"), "[REDACTED]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"), "bearer [REDACTED]"),
)

# Files whose contents must never be returned by the read/file tools.
_SENSITIVE_FILE_RE = re.compile(r"(?i)(^\.env($|\.)|(^|/)\.env($|\.)|\.pem$|\.key$|(^|/)id_rsa|(^|/)credentials($|\.)|\.secrets?($|\.))")


def _workspace_root(ctx: ToolContext) -> Path:
    root = getattr(ctx, "workspace_root", None)
    if root:
        return Path(root).expanduser().resolve()
    settings = getattr(ctx, "settings", None)
    candidate = getattr(settings, "project_root", None)
    if candidate:
        return Path(candidate).expanduser().resolve()
    return Path.cwd().resolve()


def _reports_root(ctx: ToolContext) -> Path:
    report_dir = getattr(ctx, "report_dir", None)
    if report_dir:
        return Path(report_dir).expanduser().resolve()
    settings = getattr(ctx, "settings", None)
    reports_dir = getattr(settings, "reports_dir", None)
    if reports_dir:
        return Path(reports_dir).expanduser().resolve()
    return _workspace_root(ctx) / "reports"


def _allowed_roots(ctx: ToolContext) -> list[Path]:
    """The workspace root plus any extra roots the user opted into (e.g. via
    ``/cd`` keeping the previous project dir readable). The primary root is
    always first; relative paths resolve under it."""
    roots: list[Path] = [_workspace_root(ctx)]
    for extra in (getattr(ctx, "extra_roots", None) or []):
        try:
            resolved = Path(extra).expanduser().resolve()
        except Exception:
            continue
        if resolved not in roots:
            roots.append(resolved)
    return roots


def _inline_log_path(ctx: ToolContext) -> Path:
    """Per-call log file for a streamed foreground command, kept under the
    workspace jobs dir so it passes the sandbox and is easy to tail/audit."""
    settings = getattr(ctx, "settings", None)
    jobs_dir = str(getattr(settings, "jobs_dir_name", ".biobank_jobs") or ".biobank_jobs")
    call_id = str(getattr(ctx, "tool_call_id", None) or "inline").replace("/", "_")
    return _workspace_root(ctx) / jobs_dir / "inline" / f"{call_id}.log"


def _safe_resolve(roots: Path | list[Path], path: str | Path) -> Path:
    """Resolve ``path`` and confine it to an allowed root.

    ``roots`` may be a single Path (the common case) or a list of allowed roots
    (workspace + opted-in extras). Relative paths resolve under the FIRST root;
    the resolved path is accepted if it lives under ANY allowed root."""
    root_list = [roots] if isinstance(roots, Path) else [r for r in roots if r]
    if not root_list:
        raise PermissionError("no workspace root is configured")
    base = root_list[0]
    candidate = (base / Path(path)).expanduser()
    resolved = candidate.resolve()
    for root in root_list:
        if resolved == root or root in resolved.parents:
            return resolved
    # Actionable message: tell the model exactly what is allowed so it adapts
    # (use a path inside an allowed root) instead of retrying the rejected path.
    allowed = " ; ".join(str(r) for r in root_list)
    raise PermissionError(
        f"path {str(path)!r} is outside the workspace (and any allowed root) and is not allowed. "
        f"File and shell paths must stay INSIDE an allowed workspace root: {allowed}. "
        f"Use a path under the workspace, e.g. '{base}/scratch/<name>' or a relative path "
        f"like 'scratch/<name>'."
    )


def _truncate(text: str, limit: int = 8000) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _redact_text(text: str) -> str:
    if not text:
        return ""
    redacted = str(text)
    for pattern, repl in _SECRET_SUBS:
        redacted = pattern.sub(repl, redacted)
    return redacted


# Run shell/test commands through a real shell so pipes, redirects, globs, and
# && / || work (the agent naturally composes such commands). Confinement comes
# from the approval profile (shell is denied in read_only/plan), the workspace
# cwd, the sanitized env (no secrets), and output redaction — not from arg
# splitting (shlex never jailed command *contents* either).
_SHELL_BIN = shutil.which("bash") or shutil.which("sh") or "/bin/sh"


def _shell_argv(command: str) -> list[str]:
    return [_SHELL_BIN, "-c", command]


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _env_sanitized(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env: dict[str, str] = {}
    allowed = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PYTHONPATH",
        "PYTHONUTF8",
    }
    for key in allowed:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    if extra_env:
        for key, value in extra_env.items():
            if not key or any(token in key.lower() for token in ("secret", "token", "key", "auth")):
                continue
            env[str(key)] = str(value)
    return env


def _classify_shell(command: str) -> tuple[SafetyClass, tuple[ActionClass, ...]]:
    lower = f" {command.lower()} "
    destructive_hints = (" rm ", " git push ", " git reset ", " chmod ", " chown ", " mkfs ", " dd ", " sudo ")
    network_hints = (" curl ", " wget ", " http://", " https://", " pip install ", " conda install ")
    if any(hint in lower for hint in destructive_hints):
        return SafetyClass.SHELL_DESTRUCTIVE, (ActionClass.SHELL_DESTRUCTIVE,)
    if any(hint in lower for hint in network_hints):
        return SafetyClass.NETWORK, (ActionClass.NETWORK, ActionClass.SHELL_SAFE)
    return SafetyClass.SHELL_SAFE, (ActionClass.SHELL_SAFE,)


def _default_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "returncode": {"type": "integer"},
            "path": {"type": "string"},
            "paths": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
            "error": {"type": "string"},
            "changed": {"type": "boolean"},
        },
    }


def _record_tool_artifacts(ctx: ToolContext, payload: dict[str, Any]) -> None:
    if callable(getattr(ctx, "record_trajectory", None)):
        try:
            ctx.record_trajectory(payload)
        except Exception:
            pass
    if callable(getattr(ctx, "record_action_graph", None)):
        try:
            ctx.record_action_graph(payload)
        except Exception:
            pass


class _WorkspaceTool(_BaseHandler):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        required: list[str],
        capabilities: Iterable[Capability],
        safety_class: SafetyClass,
        approval_requirement: str,
        workspace_scope: WorkspaceScope,
        action_classes: tuple[ActionClass, ...],
        trajectory_serialization: TrajectorySerialization,
        output_schema: dict[str, Any] | None = None,
        is_mutating: bool = False,
    ) -> None:
        spec = ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            required=required,
            output_schema=output_schema or _default_output_schema(),
            safety_class=safety_class,
            approval_requirement=approval_requirement,
            workspace_scope=workspace_scope,
            action_classes=action_classes,
            trajectory_serialization=trajectory_serialization,
        )
        super().__init__(_name=name, _spec=spec, _caps=frozenset(capabilities), _is_mutating=is_mutating)


class ShellTool(_WorkspaceTool):
    def __init__(self) -> None:
        super().__init__(
            name="shell",
            description="Run a bounded shell command in the workspace.",
            parameters={
                "command": {"type": "string", "description": "Shell command to run"},
                "cwd": {"type": "string", "description": "Working directory relative to workspace root", "default": ""},
                "timeout_s": {"type": "integer", "description": "Timeout in seconds; 0 = auto (scales for long tools like plink/gatk/bcftools)", "default": 0},
                "env": {"type": "object", "additionalProperties": {"type": "string"}, "default": {}},
            },
            required=["command"],
            capabilities={Capability.SHELL_EXEC},
            safety_class=SafetyClass.SHELL_SAFE,
            approval_requirement="ask_before_shell",
            workspace_scope=WorkspaceScope.WORKSPACE,
            action_classes=(ActionClass.SHELL_SAFE,),
            trajectory_serialization=TrajectorySerialization.TRUNCATED,
            is_mutating=False,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        command = str(ctx.args.get("command") or "").strip()
        if not command:
            return {"status": "error", "error": "command is required"}
        cwd = str(ctx.args.get("cwd") or "").strip()
        timeout_s = resolve_timeout(command=command, override=ctx.args.get("timeout_s"),
                                    settings=getattr(ctx, "settings", None))
        timeout_s = max(1, int(timeout_s or 120))
        env = _env_sanitized(dict(ctx.args.get("env") or {}))
        workdir = _safe_resolve(_allowed_roots(ctx), cwd) if cwd else _workspace_root(ctx)
        safety, actions = _classify_shell(command)
        tail_chars = int(getattr(getattr(ctx, "settings", None), "exec_stream_tail_chars", 8000) or 8000)
        # Stream stdout/stderr live to the display and a persisted log so a long
        # command's progress is visible in real time (issue #2), instead of only a
        # truncated dump at the end. Own process group => timeout reaps children.
        result = await asyncio.to_thread(
            run_streaming,
            _shell_argv(command),
            cwd=str(workdir),
            env=env,
            timeout=timeout_s,
            line_sink=getattr(ctx, "emit_line", None),
            log_path=_inline_log_path(ctx),
            tail_chars=tail_chars,
        )
        if result.timed_out:
            payload = {
                "status": "error",
                "error": f"command timed out after {timeout_s}s",
                "command": command,
                "cwd": str(workdir),
                "safety_class": safety.value,
                "action_classes": [action.value for action in actions],
                "log_path": result.log_path,
            }
            _record_tool_artifacts(ctx, payload)
            return payload
        payload = {
            "status": "ok" if result.ok else "error",
            "command": command,
            "cwd": str(workdir),
            "returncode": result.returncode,
            "stdout": _truncate(_redact_text(result.stdout_tail or "")),
            "stderr": _truncate(_redact_text(result.stderr_tail or "")),
            "safety_class": safety.value,
            "action_classes": [action.value for action in actions],
            "log_path": result.log_path,
        }
        # A nonzero exit is a real failure: surface it in the `error` field so the
        # scheduler/loop-guard/audit (which key on `error`) treat it as a failed
        # tool call rather than a silent success.
        if result.returncode != 0:
            stderr_tail = _truncate(_redact_text(result.stderr_tail or ""), 400).strip()
            payload["error"] = f"command exited {result.returncode}" + (f": {stderr_tail}" if stderr_tail else "")
        _record_tool_artifacts(ctx, payload)
        return payload


class ReadFileTool(_WorkspaceTool):
    def __init__(self) -> None:
        super().__init__(
            name="file_read",
            description="Read a text file inside the workspace.",
            parameters={
                "path": {"type": "string", "description": "Path relative to workspace root"},
                "max_bytes": {"type": "integer", "description": "Maximum bytes to read", "default": 200000},
            },
            required=["path"],
            capabilities={Capability.READ_DATA},
            safety_class=SafetyClass.READ,
            approval_requirement="allow",
            workspace_scope=WorkspaceScope.WORKSPACE,
            action_classes=(ActionClass.READ,),
            trajectory_serialization=TrajectorySerialization.TRUNCATED,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        try:
            raw = str(ctx.args.get("path") or "")
            # Refuse to read secret-bearing files (.env, keys, credentials) so the
            # agent cannot exfiltrate the API key etc. via file_read.
            if _SENSITIVE_FILE_RE.search(raw.replace("\\", "/")):
                payload = {"status": "error", "error": f"refusing to read sensitive file {raw!r} (secrets are not readable)", "path": raw}
                _record_tool_artifacts(ctx, payload)
                return payload
            path = _safe_resolve(_allowed_roots(ctx), raw)
            if _SENSITIVE_FILE_RE.search(str(path).replace("\\", "/")):
                payload = {"status": "error", "error": f"refusing to read sensitive file {raw!r} (secrets are not readable)", "path": str(path)}
                _record_tool_artifacts(ctx, payload)
                return payload
            max_bytes = max(1, int(ctx.args.get("max_bytes") or 200000))
            data = await asyncio.to_thread(path.read_bytes)
            # Defense-in-depth: redact any secret-looking values in the content.
            content = _redact_text(data[:max_bytes].decode("utf-8", errors="replace"))
            payload = {"status": "ok", "path": str(path), "content": content, "truncated": len(data) > max_bytes}
        except Exception as exc:
            payload = {"status": "error", "error": str(exc), "path": str(ctx.args.get("path") or "")}
        _record_tool_artifacts(ctx, payload)
        return payload


class WriteFileTool(_WorkspaceTool):
    def __init__(self) -> None:
        super().__init__(
            name="file_write",
            description="Write a file inside the workspace.",
            parameters={
                "path": {"type": "string", "description": "Path relative to workspace root"},
                "content": {"type": "string", "description": "File content"},
            },
            required=["path", "content"],
            capabilities={Capability.WRITE_REPORTS},
            safety_class=SafetyClass.WRITE,
            approval_requirement="ask_before_edits",
            workspace_scope=WorkspaceScope.WORKSPACE,
            action_classes=(ActionClass.WRITE,),
            trajectory_serialization=TrajectorySerialization.REDACTED,
            is_mutating=True,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        try:
            path = _safe_resolve(_allowed_roots(ctx), str(ctx.args.get("path") or ""))
            content = str(ctx.args.get("content") or "")
            path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(path.write_text, content, encoding="utf-8")
            payload = {"status": "ok", "path": str(path), "changed": True, "bytes": len(content.encode("utf-8"))}
        except Exception as exc:
            payload = {"status": "error", "error": str(exc), "path": str(ctx.args.get("path") or "")}
        _record_tool_artifacts(ctx, payload)
        return payload


class EditFileTool(_WorkspaceTool):
    def __init__(self) -> None:
        super().__init__(
            name="file_edit",
            description="Edit a file by replacing one or more literal snippets.",
            parameters={
                "path": {"type": "string", "description": "Path relative to workspace root"},
                "replacements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "find": {"type": "string"},
                            "replace": {"type": "string"},
                        },
                        "required": ["find", "replace"],
                    },
                },
                "dry_run": {"type": "boolean", "default": False},
            },
            required=["path", "replacements"],
            capabilities={Capability.WRITE_REPORTS},
            safety_class=SafetyClass.SELF_MODIFICATION,
            approval_requirement="ask_before_edits",
            workspace_scope=WorkspaceScope.WORKSPACE,
            action_classes=(ActionClass.WRITE, ActionClass.SELF_MODIFICATION),
            trajectory_serialization=TrajectorySerialization.REDACTED,
            is_mutating=True,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        try:
            path = _safe_resolve(_allowed_roots(ctx), str(ctx.args.get("path") or ""))
            replacements = list(ctx.args.get("replacements") or [])
            dry_run = bool(ctx.args.get("dry_run"))
            original = await asyncio.to_thread(path.read_text, encoding="utf-8")
            updated = original
            for item in replacements:
                find = str(item.get("find") or "")
                replace = str(item.get("replace") or "")
                if find not in updated:
                    payload = {"status": "error", "error": f"missing hunk: {find!r}", "path": str(path)}
                    _record_tool_artifacts(ctx, payload)
                    return payload
                updated = updated.replace(find, replace)
            if not dry_run:
                await asyncio.to_thread(path.write_text, updated, encoding="utf-8")
            payload = {
                "status": "ok",
                "path": str(path),
                "changed": updated != original,
                "dry_run": dry_run,
                "preview": _truncate(updated),
            }
        except Exception as exc:
            payload = {"status": "error", "error": str(exc), "path": str(ctx.args.get("path") or "")}
        _record_tool_artifacts(ctx, payload)
        return payload


class ApplyPatchTool(_WorkspaceTool):
    def __init__(self) -> None:
        super().__init__(
            name="apply_patch",
            description="Apply a unified diff patch inside the workspace.",
            parameters={
                "path": {"type": "string", "description": "Primary target path relative to workspace root"},
                "diff": {"type": "string", "description": "Unified diff text"},
                "dry_run": {"type": "boolean", "default": False},
            },
            required=["path", "diff"],
            capabilities={Capability.WRITE_REPORTS},
            safety_class=SafetyClass.SELF_MODIFICATION,
            approval_requirement="ask_before_edits",
            workspace_scope=WorkspaceScope.WORKSPACE,
            action_classes=(ActionClass.WRITE, ActionClass.SELF_MODIFICATION),
            trajectory_serialization=TrajectorySerialization.REDACTED,
            is_mutating=True,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        try:
            path = str(ctx.args.get("path") or "")
            diff = str(ctx.args.get("diff") or "")
            dry_run = bool(ctx.args.get("dry_run"))
            repo_root = _workspace_root(ctx)
            resolved = _safe_resolve(repo_root, path)
            path = str(resolved.relative_to(repo_root))
            assessment = classify_patch(diff)
            if dry_run:
                payload = {
                    "status": "ok",
                    "path": path,
                    "risk": assessment.risk.value,
                    "reason": assessment.reason,
                    "dry_run": True,
                }
                _record_tool_artifacts(ctx, payload)
                return payload
            merger = AutoMerger(repo_root=repo_root)
            outcome = await merger.review_and_merge(Patch(target_path=path, unified_diff=diff, summary="runtime patch"))
            payload = {
                "status": outcome.status,
                "path": path,
                "risk": outcome.risk.value,
                "branch": outcome.branch,
                "pr_url": outcome.pr_url,
                "error": outcome.error,
                "changed": outcome.status in {"merged", "pr_opened", "pr_ready"},
            }
        except Exception as exc:
            payload = {"status": "error", "error": str(exc), "path": str(ctx.args.get("path") or "")}
        _record_tool_artifacts(ctx, payload)
        return payload


class SearchTool(_WorkspaceTool):
    def __init__(self) -> None:
        super().__init__(
            name="search",
            description="Search the workspace with ripgrep, falling back to Python scanning.",
            parameters={
                "pattern": {"type": "string", "description": "Search pattern"},
                "path": {"type": "string", "description": "Relative root to search", "default": "."},
                "limit": {"type": "integer", "default": 50},
            },
            required=["pattern"],
            capabilities={Capability.READ_DATA},
            safety_class=SafetyClass.READ,
            approval_requirement="allow",
            workspace_scope=WorkspaceScope.WORKSPACE,
            action_classes=(ActionClass.READ,),
            trajectory_serialization=TrajectorySerialization.TRUNCATED,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        try:
            pattern = str(ctx.args.get("pattern") or "")
            root = _safe_resolve(_allowed_roots(ctx), str(ctx.args.get("path") or "."))
            limit = max(1, int(ctx.args.get("limit") or 50))
            rg = shutil.which("rg")
            matches: list[dict[str, Any]] = []
            if rg:
                proc = await asyncio.to_thread(
                    subprocess.run,
                    [rg, "--line-number", "--with-filename", "--hidden", "--glob", "!.git", "--", pattern, str(root)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
                for line in lines[:limit]:
                    matches.append({"line": line})
            else:
                for file in root.rglob("*"):
                    if not file.is_file():
                        continue
                    try:
                        text = file.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        continue
                    if pattern in text:
                        matches.append({"path": str(file), "preview": _truncate(text, 400)})
                        if len(matches) >= limit:
                            break
            payload = {"status": "ok", "pattern": pattern, "root": str(root), "matches": matches}
        except Exception as exc:
            payload = {"status": "error", "error": str(exc), "pattern": str(ctx.args.get("pattern") or "")}
        _record_tool_artifacts(ctx, payload)
        return payload


class GitStatusTool(_WorkspaceTool):
    def __init__(self) -> None:
        super().__init__(
            name="git_status",
            description="Show git status for the workspace.",
            parameters={},
            required=[],
            capabilities={Capability.READ_DATA},
            safety_class=SafetyClass.READ,
            approval_requirement="allow",
            workspace_scope=WorkspaceScope.WORKSPACE,
            action_classes=(ActionClass.READ,),
            trajectory_serialization=TrajectorySerialization.TRUNCATED,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        try:
            root = _workspace_root(ctx)
            proc = await asyncio.to_thread(subprocess.run, ["git", "status", "--short"], cwd=str(root), capture_output=True, text=True, timeout=30, check=False)
            payload = {
                "status": "ok" if proc.returncode == 0 else "error",
                "returncode": proc.returncode,
                "stdout": _truncate(proc.stdout or ""),
                "stderr": _truncate(proc.stderr or ""),
            }
        except Exception as exc:
            payload = {"status": "error", "error": str(exc)}
        _record_tool_artifacts(ctx, payload)
        return payload


class GitDiffTool(_WorkspaceTool):
    def __init__(self) -> None:
        super().__init__(
            name="git_diff",
            description="Show git diff for the workspace.",
            parameters={},
            required=[],
            capabilities={Capability.READ_DATA},
            safety_class=SafetyClass.READ,
            approval_requirement="allow",
            workspace_scope=WorkspaceScope.WORKSPACE,
            action_classes=(ActionClass.READ,),
            trajectory_serialization=TrajectorySerialization.TRUNCATED,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        try:
            root = _workspace_root(ctx)
            proc = await asyncio.to_thread(subprocess.run, ["git", "diff", "--"], cwd=str(root), capture_output=True, text=True, timeout=30, check=False)
            payload = {
                "status": "ok" if proc.returncode == 0 else "error",
                "returncode": proc.returncode,
                "stdout": _truncate(proc.stdout or ""),
                "stderr": _truncate(proc.stderr or ""),
            }
        except Exception as exc:
            payload = {"status": "error", "error": str(exc)}
        _record_tool_artifacts(ctx, payload)
        return payload


class TestRunnerTool(_WorkspaceTool):
    __test__ = False

    def __init__(self) -> None:
        super().__init__(
            name="test_runner",
            description="Run a bounded test command in the workspace.",
            parameters={
                "command": {"type": "string", "description": "Test command to run"},
                "cwd": {"type": "string", "description": "Working directory relative to workspace root", "default": ""},
                "timeout_s": {"type": "integer", "description": "Timeout in seconds", "default": 1800},
            },
            required=["command"],
            capabilities={Capability.SHELL_EXEC},
            safety_class=SafetyClass.SHELL_SAFE,
            approval_requirement="ask_before_edits",
            workspace_scope=WorkspaceScope.WORKSPACE,
            action_classes=(ActionClass.SHELL_SAFE,),
            trajectory_serialization=TrajectorySerialization.TRUNCATED,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        command = str(ctx.args.get("command") or "").strip()
        if not command:
            return {"status": "error", "error": "command is required"}
        cwd = str(ctx.args.get("cwd") or "").strip()
        timeout_s = max(1, int(ctx.args.get("timeout_s") or 1800))
        workdir = _safe_resolve(_allowed_roots(ctx), cwd) if cwd else _workspace_root(ctx)
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                _shell_argv(command),
                cwd=str(workdir),
                env=_env_sanitized(),  # strip secrets (was inheriting full env, leaking the API key)
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            payload = {
                "status": "ok" if proc.returncode == 0 else "error",
                "command": command,
                "cwd": str(workdir),
                "returncode": proc.returncode,
                "stdout": _truncate(_redact_text(proc.stdout or "")),
                "stderr": _truncate(_redact_text(proc.stderr or "")),
            }
            # Nonzero exit (e.g. failing tests) must surface as a tool error so it
            # is visible to the loop-guard / audit / completion feedback.
            if proc.returncode != 0:
                tail = _truncate(_redact_text((proc.stderr or "") or (proc.stdout or "")), 400).strip()
                payload["error"] = f"tests exited {proc.returncode}" + (f": {tail}" if tail else "")
        except subprocess.TimeoutExpired:
            payload = {
                "status": "error",
                "error": f"command timed out after {timeout_s}s",
                "command": command,
                "cwd": str(workdir),
            }
        _record_tool_artifacts(ctx, payload)
        return payload


class SelfEvolveTool(_WorkspaceTool):
    """Let the agent durably improve code with a VERIFICATION GATE: the patch is
    applied in an isolated git worktree, its tests run there, and it is committed
    only if they pass (auto-rollback on failure). Edits to core/runtime/tests are
    never auto-merged — they go to a review branch. This is the agent-facing half
    of self-evolution-through-harness."""

    def __init__(self) -> None:
        super().__init__(
            name="self_evolve",
            description=(
                "Durably improve a file in the workspace, gated by tests. Provide a unified diff "
                "(or full new file content) and the test command(s) that must pass. The change is "
                "applied in an isolated git worktree, the tests run there, and it is committed ONLY "
                "if every test passes (otherwise rolled back). Edits to core/runtime/tests are kept "
                "on a review branch, never auto-merged. Use this to self-evolve skills safely."
            ),
            parameters={
                "target_path": {"type": "string", "description": "Workspace-relative file the patch targets"},
                "diff": {"type": "string", "description": "A unified diff, OR the full new file content"},
                "test_commands": {"type": "array", "items": {"type": "string"}, "description": "Test/lint runner commands (e.g. 'pytest path -q') that must pass to apply"},
                "summary": {"type": "string", "description": "Short change summary", "default": ""},
            },
            required=["target_path", "diff", "test_commands"],
            capabilities={Capability.WRITE_REPORTS, Capability.SHELL_EXEC},
            safety_class=SafetyClass.SHELL_SAFE,
            approval_requirement="ask_before_edits",
            workspace_scope=WorkspaceScope.WORKSPACE,
            action_classes=(ActionClass.SHELL_SAFE,),
            trajectory_serialization=TrajectorySerialization.TRUNCATED,
            is_mutating=True,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        from biobank_agent.runtime.self_evolve import apply_patch_transactionally

        tc = ctx.args.get("test_commands") or []
        if isinstance(tc, str):
            tc = [tc]
        result = await asyncio.to_thread(
            apply_patch_transactionally,
            repo_root=_workspace_root(ctx),
            target_path=str(ctx.args.get("target_path") or "").strip(),
            diff=str(ctx.args.get("diff") or ""),
            test_commands=[str(x) for x in tc],
            summary=str(ctx.args.get("summary") or ""),
        )
        payload = result.to_dict()
        payload["evolve_status"] = result.status
        # applied / review_branch are successes; tests_failed / rejected / error
        # surface as a tool error so the model sees the reason and can adapt.
        payload["status"] = "ok" if result.status in ("applied", "review_branch") else "error"
        if payload["status"] == "error" and not payload.get("error"):
            payload["error"] = f"self_evolve {result.status}"
        _record_tool_artifacts(ctx, payload)
        return payload


class AskUserTool(_WorkspaceTool):
    """Pause the run and ask the human ONE concrete question.

    Use ONLY when genuinely blocked on a fact that only the user can provide
    (e.g. where the input data lives, an ambiguous case/control definition).
    Asking is not acting, so this tool is allowed under every approval profile
    including ``yolo``/``full_auto``. It does not block inside the model turn:
    it records the question and the plan loop detects this sentinel in
    ``turn.tool_results``, pauses the plan, surfaces the question at the prompt,
    and resumes the step once the user answers. Never use it to confirm actions
    you can take yourself.
    """

    def __init__(self) -> None:
        super().__init__(
            name="ask_user",
            description=(
                "Pause and ask the human ONE concrete question when you are genuinely "
                "blocked on a fact only they can provide (a missing data path, an ambiguous "
                "case/control definition). The user answers and the step resumes with their "
                "answer. Use sparingly — never to confirm actions you can perform yourself, "
                "and never speculatively."
            ),
            parameters={
                "question": {
                    "type": "string",
                    "description": "The single concrete question to ask the user.",
                },
            },
            required=["question"],
            capabilities=set(),
            safety_class=SafetyClass.READ,
            approval_requirement="allow",
            workspace_scope=WorkspaceScope.NONE,
            action_classes=(ActionClass.READ,),
            trajectory_serialization=TrajectorySerialization.FULL,
            is_mutating=False,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        question = str(ctx.args.get("question") or "").strip()
        if not question:
            return {"status": "error", "error": "question is required"}
        # Signal-only: the plan loop detects this sentinel (last tool result named
        # ``ask_user`` with ``awaiting_user``), pauses, and surfaces the question.
        # The model should stop here and end its turn rather than keep calling tools.
        return {
            "status": "ok",
            "awaiting_user": True,
            "question": question,
            "message": (
                "Question recorded. Stop calling tools and end your turn with a brief note "
                "that you are waiting for the user's answer; the step resumes once they reply."
            ),
        }


def _job_summary(rec, *, with_tail: bool = True) -> dict[str, Any]:
    """Compact, JSON-safe view of a JobRecord for tool output."""
    if rec is None:
        return {"status": "error", "error": "job not found"}
    now = time.time()
    elapsed = round((rec.ended_at or now) - (rec.started_at or now), 1)
    out: dict[str, Any] = {
        "status": "ok",
        "job_id": rec.job_id,
        "state": rec.state,
        "returncode": rec.returncode,
        "label": rec.label or rec.tool,
        "log_path": rec.log_path,
        "elapsed_s": elapsed,
        "running": rec.state == "running",
    }
    if with_tail and rec.tail_stdout:
        out["stdout_tail"] = _truncate(_redact_text(rec.tail_stdout), 4000)
    if with_tail and rec.tail_stderr:
        out["stderr_tail"] = _truncate(_redact_text(rec.tail_stderr), 1500)
    return out


class RunJobTool(_WorkspaceTool):
    """Launch a long command as a DETACHED background job (issue #1).

    Use for commands that take many minutes (plink/gatk/bcftools, a WGS pipeline)
    so the run is not blocked and survives interrupts. Returns a ``job_id``
    immediately; poll with ``job_status``/``job_wait`` and read the log at the
    returned ``log_path``."""

    def __init__(self) -> None:
        super().__init__(
            name="run_job",
            description=(
                "Run a long shell command as a detached background job (for multi-minute "
                "bioinformatics tools). Returns a job_id immediately; do NOT block — poll "
                "with job_wait/job_status and tail the log_path. timeout_s=0 means auto "
                "(scales for long tools); negative means no cap."
            ),
            parameters={
                "command": {"type": "string", "description": "Shell command to run in the background"},
                "cwd": {"type": "string", "description": "Working directory relative to workspace root", "default": ""},
                "timeout_s": {"type": "integer", "description": "Timeout seconds; 0 = auto, negative = no cap", "default": 0},
                "expected_duration_s": {"type": "integer", "description": "Rough expected runtime (hint)", "default": 0},
                "label": {"type": "string", "description": "Short human label for the job", "default": ""},
                "env": {"type": "object", "additionalProperties": {"type": "string"}, "default": {}},
                "backend": {"type": "string", "description": "Execution backend: '' = auto (local, or a cluster scheduler if configured), or local|slurm|sge|lsf|dxrun", "default": ""},
            },
            required=["command"],
            capabilities={Capability.SHELL_EXEC},
            safety_class=SafetyClass.SHELL_SAFE,
            approval_requirement="ask_before_shell",
            workspace_scope=WorkspaceScope.WORKSPACE,
            action_classes=(ActionClass.SHELL_SAFE,),
            trajectory_serialization=TrajectorySerialization.TRUNCATED,
            is_mutating=False,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        jm = getattr(ctx, "job_manager", None)
        if jm is None:
            return {"status": "error", "error": "background jobs are not available in this context"}
        command = str(ctx.args.get("command") or "").strip()
        if not command:
            return {"status": "error", "error": "command is required"}
        cwd_arg = str(ctx.args.get("cwd") or "").strip()
        workdir = _safe_resolve(_allowed_roots(ctx), cwd_arg) if cwd_arg else _workspace_root(ctx)
        settings = getattr(ctx, "settings", None)
        timeout = resolve_timeout(command=command, override=ctx.args.get("timeout_s"),
                                  settings=settings, background=True)
        env = _env_sanitized(dict(ctx.args.get("env") or {}))
        label = str(ctx.args.get("label") or "").strip() or command[:60]

        # Scheduler backend: when a cluster scheduler is configured/detected (and
        # available), submit there instead of running locally — that's where real GWAS
        # jobs belong. Falls back to the local detached job otherwise.
        from biobank_agent.runtime.environments import detect_backend, get_backend, JobResources

        backend_arg = str(ctx.args.get("backend") or "").strip()
        backend = get_backend(backend_arg) if backend_arg else detect_backend(settings)
        if getattr(backend, "is_scheduler", False) and backend.available():
            jobs_root = _workspace_root(ctx) / str(getattr(settings, "jobs_dir_name", ".biobank_jobs") or ".biobank_jobs") / "scheduler"
            jobs_root.mkdir(parents=True, exist_ok=True)
            call_id = str(getattr(ctx, "tool_call_id", "") or "job").replace("/", "_")
            script_path = jobs_root / f"{call_id}.sh"
            resources = JobResources(
                cpus=int(getattr(settings, "scheduler_cpus", 1) or 1),
                mem_mb=int(getattr(settings, "scheduler_mem_mb", 4096) or 4096),
                time_min=int(getattr(settings, "scheduler_time_min", 240) or 240),
                partition=str(getattr(settings, "scheduler_partition", "") or ""),
            )
            script_path.write_text(backend.build_script(command, cwd=str(workdir), job_name=label, resources=resources), encoding="utf-8")
            submit_argv = backend.submit_argv(str(script_path), job_name=label, resources=resources)
            sub = await asyncio.to_thread(run_streaming, submit_argv, cwd=str(workdir), timeout=120, log_path=str(jobs_root / f"{call_id}.submit.log"))
            job_id = backend.parse_job_id(sub.stdout_tail or "")
            if not job_id:
                return {"status": "error", "backend": backend.name,
                        "error": f"{backend.name} submit failed: {(sub.stderr_tail or sub.stdout_tail or 'no job id returned')[:500]}",
                        "script_path": str(script_path)}
            out = {
                "status": "ok", "backend": backend.name, "scheduler_job_id": job_id,
                "script_path": str(script_path),
                "status_command": " ".join(backend.status_argv(job_id)),
                "cancel_command": " ".join(backend.cancel_argv(job_id)),
                "message": (f"Submitted to {backend.name} as job {job_id}. Poll with: "
                            f"{' '.join(backend.status_argv(job_id))} ; cancel with: {' '.join(backend.cancel_argv(job_id))}."),
            }
            _record_tool_artifacts(ctx, out)
            return out

        rec = await asyncio.to_thread(
            jm.submit, _shell_argv(command), cwd=str(workdir), env=env, timeout=timeout,
            tool=tool_key(command=command), label=label, shell=False,
        )
        if rec.state == "failed" and rec.pid is None:
            return {"status": "error", "error": rec.tail_stderr or "failed to launch job", "job_id": rec.job_id}
        out = _job_summary(rec, with_tail=False)
        out["message"] = (
            f"Job {rec.job_id} running in background (pid {rec.pid}). Do not block: call "
            f"job_wait to await it or job_status to poll; full output streams to {rec.log_path}."
        )
        _record_tool_artifacts(ctx, out)
        return out


class JobStatusTool(_WorkspaceTool):
    def __init__(self) -> None:
        super().__init__(
            name="job_status",
            description="Check a background job's state, return code, elapsed time and output tail. Use job_wait to BLOCK until done rather than spin-polling this.",
            parameters={"job_id": {"type": "string", "description": "The job id from run_job"}},
            required=["job_id"],
            capabilities=set(),
            safety_class=SafetyClass.READ,
            approval_requirement="allow",
            workspace_scope=WorkspaceScope.NONE,
            action_classes=(ActionClass.READ,),
            trajectory_serialization=TrajectorySerialization.TRUNCATED,
            is_mutating=False,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        jm = getattr(ctx, "job_manager", None)
        if jm is None:
            return {"status": "error", "error": "background jobs are not available in this context"}
        rec = await asyncio.to_thread(jm.status, str(ctx.args.get("job_id") or ""))
        return _job_summary(rec)


class JobListTool(_WorkspaceTool):
    def __init__(self) -> None:
        super().__init__(
            name="job_list",
            description="List background jobs (running and recent) with their state.",
            parameters={},
            required=[],
            capabilities=set(),
            safety_class=SafetyClass.READ,
            approval_requirement="allow",
            workspace_scope=WorkspaceScope.NONE,
            action_classes=(ActionClass.READ,),
            trajectory_serialization=TrajectorySerialization.TRUNCATED,
            is_mutating=False,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        jm = getattr(ctx, "job_manager", None)
        if jm is None:
            return {"status": "error", "error": "background jobs are not available in this context"}
        recs = await asyncio.to_thread(jm.list)
        return {"status": "ok", "jobs": [_job_summary(r, with_tail=False) for r in recs], "count": len(recs)}


class JobWaitTool(_WorkspaceTool):
    def __init__(self) -> None:
        super().__init__(
            name="job_wait",
            description="Block until a background job finishes (or timeout_s elapses), then return its result + output tail.",
            parameters={
                "job_id": {"type": "string", "description": "The job id from run_job"},
                "timeout_s": {"type": "integer", "description": "Max seconds to wait; 0 = wait indefinitely", "default": 0},
            },
            required=["job_id"],
            capabilities=set(),
            safety_class=SafetyClass.READ,
            approval_requirement="allow",
            workspace_scope=WorkspaceScope.NONE,
            action_classes=(ActionClass.READ,),
            trajectory_serialization=TrajectorySerialization.TRUNCATED,
            is_mutating=False,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        jm = getattr(ctx, "job_manager", None)
        if jm is None:
            return {"status": "error", "error": "background jobs are not available in this context"}
        try:
            timeout = int(ctx.args.get("timeout_s") or 0)
        except (TypeError, ValueError):
            timeout = 0
        rec = await asyncio.to_thread(jm.wait, str(ctx.args.get("job_id") or ""),
                                      timeout=(timeout if timeout > 0 else None))
        out = _job_summary(rec)
        if rec is not None and not rec.is_terminal:
            out["message"] = "job still running (wait timed out); poll again or increase timeout_s"
        return out


class JobCancelTool(_WorkspaceTool):
    def __init__(self) -> None:
        super().__init__(
            name="job_cancel",
            description="Terminate a running background job (and its child processes).",
            parameters={"job_id": {"type": "string", "description": "The job id from run_job"}},
            required=["job_id"],
            capabilities={Capability.SHELL_EXEC},
            safety_class=SafetyClass.SHELL_SAFE,
            approval_requirement="ask_before_shell",
            workspace_scope=WorkspaceScope.WORKSPACE,
            action_classes=(ActionClass.SHELL_SAFE,),
            trajectory_serialization=TrajectorySerialization.TRUNCATED,
            is_mutating=True,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        jm = getattr(ctx, "job_manager", None)
        if jm is None:
            return {"status": "error", "error": "background jobs are not available in this context"}
        rec = await asyncio.to_thread(jm.cancel, str(ctx.args.get("job_id") or ""))
        return _job_summary(rec, with_tail=False)


class SkillSearchTool(_WorkspaceTool):
    """Find and load a domain skill that isn't in your current tool list.

    To keep context lean, only high-frequency skills are offered each turn; the
    rest of the ~100 domain skills (GWAS/QC/PheWAS/survival/literature/report/...)
    are loaded on demand. Call this with a short intent; matching skills are added
    to your tool list for the NEXT step, where you can call them directly.
    """

    def __init__(self) -> None:
        super().__init__(
            name="skill_search",
            description=(
                "Search for and load domain skills not currently in your tool list. Only "
                "high-frequency tools are loaded each turn; call skill_search('<intent>') to "
                "load specialists on demand (e.g. 'GWAS association with PCA covariates', "
                "'survival analysis', 'pathway enrichment', 'read a paper'). The matched "
                "skills become callable on your next step."
            ),
            parameters={
                "query": {"type": "string", "description": "What you need to do (intent / keywords)."},
                "k": {"type": "integer", "description": "Max skills to load (default 8).", "default": 8},
            },
            required=["query"],
            capabilities=set(),
            safety_class=SafetyClass.READ,
            approval_requirement="allow",
            workspace_scope=WorkspaceScope.NONE,
            action_classes=(ActionClass.READ,),
            trajectory_serialization=TrajectorySerialization.FULL,
            is_mutating=False,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        query = str(ctx.args.get("query") or "").strip()
        if not query:
            return {"status": "error", "error": "query is required"}
        try:
            k = max(1, int(ctx.args.get("k") or 8))
        except (TypeError, ValueError):
            k = 8
        registry = getattr(ctx, "tool_registry", None)
        if registry is None or not hasattr(registry, "search"):
            return {"status": "error", "error": "skill registry unavailable"}
        hits = registry.search(query, k=k)
        names = [h["name"] for h in hits]
        activate = getattr(ctx, "activate_skills", None)
        if names and callable(activate):
            try:
                activate(names)
            except Exception:
                pass
        if not names:
            return {"status": "ok", "query": query, "loaded": [], "results": [],
                    "message": "No matching skills. Rephrase the intent or use the core tools."}
        return {
            "status": "ok",
            "query": query,
            "loaded": names,
            "results": hits,
            "message": (
                f"Loaded {len(names)} skill(s) for your next step: {', '.join(names)}. "
                "Call the one you need directly now."
            ),
        }


class NavigateSkillTreeTool(_WorkspaceTool):
    """Browse the hierarchical skill tree one level at a time and load a leaf's skills.

    Complements skill_search: search loads by intent keywords, navigation lets the model
    explore what exists by category. Entering a leaf activates its skills for the next step.
    """

    def __init__(self) -> None:
        super().__init__(
            name="navigate_skill_tree",
            description=(
                "Browse the skill tree by category instead of by keyword. Call with no node_id "
                "to list top-level categories, then pass a node_id to open its sub-categories "
                "and skills. Opening a leaf loads its skills so they become callable on your "
                "next step. Use skill_search when you already know the intent; use this to "
                "explore what exists."
            ),
            parameters={
                "node_id": {"type": "string", "description": "Tree node to open; omit for the root level."},
            },
            required=[],
            capabilities=set(),
            safety_class=SafetyClass.READ,
            approval_requirement="allow",
            workspace_scope=WorkspaceScope.NONE,
            action_classes=(ActionClass.READ,),
            trajectory_serialization=TrajectorySerialization.FULL,
            is_mutating=False,
        )

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        from biobank_agent.skills import skill_tree

        node_id = str(ctx.args.get("node_id") or "").strip() or None
        registry = getattr(ctx, "tool_registry", None)

        def _describe(name: str) -> str:
            try:
                handler = registry.get(name) if registry is not None else None
                return handler.spec().description if handler else ""
            except Exception:
                return ""

        payload = skill_tree.navigate_tree(node_id, describe=_describe)
        leaf_skills = [s["name"] for s in payload.get("skills", [])]
        activate = getattr(ctx, "activate_skills", None)
        loaded: list[str] = []
        if leaf_skills and callable(activate):
            try:
                activate(leaf_skills)
                loaded = leaf_skills
            except Exception:
                loaded = []
        return {
            "status": "ok",
            "node_id": payload.get("node_id"),
            "summary": payload.get("summary", ""),
            "children": payload.get("children", []),
            "skills": payload.get("skills", []),
            "loaded": loaded,
            "message": (
                f"Loaded {len(loaded)} skill(s) for your next step: {', '.join(loaded)}."
                if loaded else
                "Open a child node to drill down, or call skill_search('<intent>') to load by intent."
            ),
        }


def build_native_tools() -> list[_WorkspaceTool]:
    return [
        ShellTool(),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        ApplyPatchTool(),
        SearchTool(),
        GitStatusTool(),
        GitDiffTool(),
        TestRunnerTool(),
        SelfEvolveTool(),
        AskUserTool(),
        RunJobTool(),
        JobStatusTool(),
        JobListTool(),
        JobWaitTool(),
        JobCancelTool(),
        SkillSearchTool(),
        NavigateSkillTreeTool(),
    ]
