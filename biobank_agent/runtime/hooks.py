"""General runtime hook registry with gated third-party execution.

This generalises the single-purpose ``core/safety/evidence_hooks.py`` pattern
(a defensive callback the runtime fires when a tool finishes) into a small
registry keyed by lifecycle event. First-party (``trust="builtin"``) hooks
always run; hooks contributed by installed plugins or user config
(``trust="external"``) are recorded but **never run** unless the owning source
is explicitly opted in — either globally (``plugin_allow_hooks``) or per plugin
(``allowed_plugins``). This keeps the plugin subsystem's promise that
third-party code does not execute on install.

Every hook runs behind a broad ``except``: a raising hook is logged at debug and
skipped, never propagated. Firing hooks is observability/extension, not
load-bearing control flow, so ``emit`` cannot fail a turn.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


# Lifecycle events emitted by the runtime. Plugins/user hooks bind to these names.
ON_TURN_START = "on_turn_start"
PRE_TOOL = "pre_tool"
POST_TOOL = "post_tool"
ON_STEP = "on_step"
ON_ERROR = "on_error"
ON_COMPLETE = "on_complete"

LIFECYCLE_EVENTS = frozenset(
    {ON_TURN_START, PRE_TOOL, POST_TOOL, ON_STEP, ON_ERROR, ON_COMPLETE}
)

TRUST_BUILTIN = "builtin"
TRUST_EXTERNAL = "external"

# Map Claude-Code plugin hook event names onto our lifecycle events so an installed plugin's
# hooks bind to the point the runtime actually emits. Unknown names pass through unchanged
# (registered + gated, but never fired since the runtime emits no such event).
CLAUDE_CODE_EVENT_MAP = {
    "pretooluse": PRE_TOOL,
    "posttooluse": POST_TOOL,
    "userpromptsubmit": ON_TURN_START,
    "sessionstart": ON_TURN_START,
    "stop": ON_COMPLETE,
    "subagentstop": ON_COMPLETE,
    "notification": ON_ERROR,
    "error": ON_ERROR,
}


def map_plugin_event(name: str) -> str:
    """Normalise a plugin hook event name to a runtime lifecycle event."""
    key = str(name or "").strip().lower().replace("_", "").replace("-", "")
    return CLAUDE_CODE_EVENT_MAP.get(key, str(name or ""))


@dataclass
class HookHandle:
    """One registered callback for one lifecycle event."""

    event: str
    fn: Callable[..., Any]
    name: str = ""
    trust: str = TRUST_BUILTIN
    # Owning plugin (for external hooks); matched against the per-plugin allowlist.
    plugin: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = getattr(self.fn, "__name__", None) or f"{self.event}_hook"


@dataclass
class HookOutcome:
    """Result of attempting one hook during an ``emit``."""

    name: str
    event: str
    status: str  # "fired" | "skipped" | "error"
    value: Any = None
    error: str = ""


class HookRegistry:
    """Event-keyed store of hook callbacks with trust-based gating."""

    def __init__(self) -> None:
        self._hooks: dict[str, list[HookHandle]] = {}

    def register(
        self,
        event: str,
        fn: Callable[..., Any],
        *,
        name: str = "",
        trust: str = TRUST_BUILTIN,
        plugin: str = "",
    ) -> HookHandle:
        ev = str(event)
        handle = HookHandle(event=ev, fn=fn, name=name, trust=trust, plugin=plugin)
        self._hooks.setdefault(ev, []).append(handle)
        return handle

    def hook(
        self,
        event: str,
        *,
        name: str = "",
        trust: str = TRUST_BUILTIN,
        plugin: str = "",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form of :meth:`register`."""

        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(event, fn, name=name, trust=trust, plugin=plugin)
            return fn

        return _decorator

    def handles(self, event: str | None = None) -> list[HookHandle]:
        if event is None:
            return [h for handles in self._hooks.values() for h in handles]
        return list(self._hooks.get(str(event), ()))

    def unregister_plugin(self, plugin: str) -> int:
        """Drop every hook owned by ``plugin``. Returns the count removed."""
        removed = 0
        for ev, handles in self._hooks.items():
            kept = [h for h in handles if h.plugin != plugin]
            removed += len(handles) - len(kept)
            self._hooks[ev] = kept
        return removed

    @staticmethod
    def _is_allowed(
        handle: HookHandle, allow_external: bool, allowed_plugins: Iterable[str]
    ) -> bool:
        if handle.trust == TRUST_BUILTIN:
            return True
        # External (plugin/user) hooks are gated: run only under explicit opt-in.
        return bool(allow_external) or handle.plugin in set(allowed_plugins or ())

    def emit(
        self,
        event: str,
        *,
        allow_external: bool = False,
        allowed_plugins: Iterable[str] = (),
        **payload: Any,
    ) -> list[HookOutcome]:
        """Fire all hooks bound to ``event`` and return per-hook outcomes.

        Gated hooks that are not opted in are reported ``status="skipped"`` and
        never invoked. A hook that raises is reported ``status="error"``; it does
        not stop the remaining hooks and never propagates out of ``emit``.
        """
        outcomes: list[HookOutcome] = []
        allowed = set(allowed_plugins or ())
        for handle in self._hooks.get(str(event), ()):
            if not self._is_allowed(handle, allow_external, allowed):
                outcomes.append(HookOutcome(handle.name, handle.event, "skipped"))
                continue
            try:
                value = handle.fn(**payload)
                outcomes.append(HookOutcome(handle.name, handle.event, "fired", value=value))
            except Exception as exc:  # a hook must never break the turn
                logger.debug("hook %s (%s) failed: %s", handle.name, handle.event, exc)
                outcomes.append(HookOutcome(handle.name, handle.event, "error", error=str(exc)))
        return outcomes

    def clear(self) -> None:
        self._hooks.clear()


def _json_safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Reduce an emit payload to a JSON-serialisable dict for command hooks.

    Runtime objects (``AgentSession`` etc.) are not serialisable; keep their id
    and drop the object, and stringify anything else that will not encode.
    """
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "session":
            safe["session_id"] = getattr(value, "session_id", "") or ""
            continue
        try:
            json.dumps(value)
            safe[key] = value
        except (TypeError, ValueError):
            safe[key] = str(value)
    return safe


def matcher_matches(matcher: str, tool: str) -> bool:
    """Whether a Claude-Code hook ``matcher`` scopes to ``tool`` (regex over the tool name).

    Empty or ``*`` matches everything. A tool-scoped matcher on a non-tool event (no ``tool`` in the
    payload) does not apply and matches. A malformed regex falls back to an exact-name compare rather
    than silently matching everything.
    """
    m = str(matcher or "").strip()
    if not m or m == "*":
        return True
    if not tool:
        return True
    try:
        return re.search(m, str(tool)) is not None
    except re.error:
        return m == str(tool)


def make_command_hook(
    command: str,
    *,
    event: str,
    plugin: str = "",
    matcher: str = "",
    timeout_s: float = 30.0,
    runner: Callable[[str, str], "tuple[int, str]"] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Build a callback that runs a plugin's shell ``command`` for a lifecycle event.

    The returned callable executes third-party shell and is therefore only ever
    invoked by :meth:`HookRegistry.emit` when the owning plugin is opted in. The
    emit payload is passed to the command as JSON on stdin (Claude-Code hook
    convention). ``matcher`` scopes the hook to matching tools (see
    :func:`matcher_matches`) so a ``PreToolUse`` hook fires only for its declared
    tools, not every tool. ``runner`` is injectable so tests never spawn a real shell.
    """

    def _default_runner(cmd: str, stdin: str) -> "tuple[int, str]":
        proc = subprocess.run(
            cmd,
            shell=True,  # plugin hooks are shell command strings; gated behind opt-in
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    run = runner or _default_runner

    def _command_hook(**payload: Any) -> dict[str, Any]:
        if not matcher_matches(matcher, str(payload.get("tool", ""))):
            return {"command": command, "matched": False, "skipped": "matcher"}
        stdin = json.dumps({"event": event, "plugin": plugin, **_json_safe_payload(payload)}, default=str)
        code, output = run(command, stdin)
        return {"command": command, "returncode": code, "output": output, "matched": True}

    _command_hook.__name__ = f"cmd_hook:{plugin}:{event}"
    return _command_hook


_DEFAULT_REGISTRY: HookRegistry | None = None


def default_registry() -> HookRegistry:
    """Process-wide registry backing the module-level ``hook``/``emit_hook`` API.

    User- and plugin-contributed hooks share this instance so the runtime sees
    them; tests can pass a fresh :class:`HookRegistry` into the runtime instead.
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = HookRegistry()
    return _DEFAULT_REGISTRY


def hook(
    event: str, *, name: str = "", trust: str = TRUST_BUILTIN, plugin: str = ""
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a hook on the default registry (decorator)."""
    return default_registry().hook(event, name=name, trust=trust, plugin=plugin)


def emit_hook(event: str, **kwargs: Any) -> list[HookOutcome]:
    """Emit ``event`` on the default registry. See :meth:`HookRegistry.emit`."""
    allow_external = bool(kwargs.pop("allow_external", False))
    allowed_plugins = kwargs.pop("allowed_plugins", ())
    return default_registry().emit(
        event, allow_external=allow_external, allowed_plugins=allowed_plugins, **kwargs
    )


__all__ = [
    "ON_TURN_START",
    "PRE_TOOL",
    "POST_TOOL",
    "ON_STEP",
    "ON_ERROR",
    "ON_COMPLETE",
    "LIFECYCLE_EVENTS",
    "TRUST_BUILTIN",
    "TRUST_EXTERNAL",
    "CLAUDE_CODE_EVENT_MAP",
    "map_plugin_event",
    "HookHandle",
    "HookOutcome",
    "HookRegistry",
    "make_command_hook",
    "matcher_matches",
    "default_registry",
    "hook",
    "emit_hook",
]
