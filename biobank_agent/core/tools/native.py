"""Native workspace tools for the runtime-backed BioBank shell."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from biobank_agent.core.evolution.auto_merger import AutoMerger, Patch
from biobank_agent.core.evolution.patch_classifier import classify as classify_patch

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


def _safe_resolve(root: Path, path: str | Path) -> Path:
    candidate = (root / Path(path)).expanduser()
    resolved = candidate.resolve()
    if resolved == root or root in resolved.parents:
        return resolved
    # Actionable message: tell the model EXACTLY what is allowed so it adapts
    # (use a path inside the workspace) instead of retrying the rejected path.
    raise PermissionError(
        f"path {str(path)!r} is outside the workspace and is not allowed. "
        f"File and shell paths must stay INSIDE the workspace root {root} "
        f"(or its subdirectories). Use a path under the workspace, e.g. "
        f"'{root}/scratch/<name>' or a relative path like 'scratch/<name>', "
        f"NOT an external path such as /tmp."
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
                "timeout_s": {"type": "integer", "description": "Timeout in seconds", "default": 120},
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
        timeout_s = max(1, int(ctx.args.get("timeout_s") or 120))
        env = _env_sanitized(dict(ctx.args.get("env") or {}))
        workdir = _safe_resolve(_workspace_root(ctx), cwd) if cwd else _workspace_root(ctx)
        safety, actions = _classify_shell(command)
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                _shell_argv(command),
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            payload = {
                "status": "error",
                "error": f"command timed out after {timeout_s}s",
                "command": command,
                "cwd": str(workdir),
                "safety_class": safety.value,
                "action_classes": [action.value for action in actions],
            }
            _record_tool_artifacts(ctx, payload)
            return payload
        payload = {
            "status": "ok" if proc.returncode == 0 else "error",
            "command": command,
            "cwd": str(workdir),
            "returncode": proc.returncode,
            "stdout": _truncate(_redact_text(proc.stdout or "")),
            "stderr": _truncate(_redact_text(proc.stderr or "")),
            "safety_class": safety.value,
            "action_classes": [action.value for action in actions],
        }
        # A nonzero exit is a real failure: surface it in the `error` field so the
        # scheduler/loop-guard/audit (which key on `error`) treat it as a failed
        # tool call rather than a silent success.
        if proc.returncode != 0:
            stderr_tail = _truncate(_redact_text(proc.stderr or ""), 400).strip()
            payload["error"] = f"command exited {proc.returncode}" + (f": {stderr_tail}" if stderr_tail else "")
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
            path = _safe_resolve(_workspace_root(ctx), raw)
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
            path = _safe_resolve(_workspace_root(ctx), str(ctx.args.get("path") or ""))
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
            path = _safe_resolve(_workspace_root(ctx), str(ctx.args.get("path") or ""))
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
            root = _safe_resolve(_workspace_root(ctx), str(ctx.args.get("path") or "."))
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
        workdir = _safe_resolve(_workspace_root(ctx), cwd) if cwd else _workspace_root(ctx)
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
    ]
