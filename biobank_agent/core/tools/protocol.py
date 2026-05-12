"""ToolHandler protocol — the v3 successor to ``@skill``.

The legacy registry (``biobank_agent/registry.py``) treats every skill
as a free function that receives the entire ``ctx`` (``agent.state``,
``dm``, ``catalog``, ``settings``, ``memory``). That gives every tool
unbounded read/write access to the agent — a problem for the v3 self-
evolution loop where untrusted skill bodies may be auto-merged.

``ToolHandler`` is the new contract:

* ``spec()`` returns the OpenAI-style schema (so registry tool listing
  remains compatible).
* ``required_capabilities()`` declares what the tool *needs* (READ_DATA,
  WRITE_REPORTS, NETWORK, EXPORT_AGGREGATE, EXPORT_PII, CALL_REVIEWER).
* ``is_mutating`` flag drives the ``ApprovalPolicy``.
* ``handle(args, ctx)`` is async by default — CPU-bound work goes
  through ``asyncio.to_thread`` inside the implementation.

Existing ``@skill`` functions are wrapped with
``LegacySkillToolHandler`` so we keep all 56 production skills
working out-of-the-box.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Iterable, Optional, Protocol, runtime_checkable


class Capability(str, Enum):
    """What a tool is allowed to touch.

    Maps onto the ``ApprovalPolicy`` 4-tier model (``approval.py``):
        READ_DATA          -> readonly query against DuckDB / parquet
        WRITE_REPORTS      -> create files under ``reports_dir``
        NETWORK            -> outbound HTTP(s)
        EXPORT_AGGREGATE   -> emit aggregate stats only (k-anonymous)
        EXPORT_PII         -> emit data that may contain PII
        CALL_REVIEWER      -> spawn external Codex / Claude reviewer
        MUTATE_MEMORY      -> persist into LongTermMemory beyond audit
        SHELL_EXEC         -> arbitrary subprocess execution
    """

    READ_DATA = "read_data"
    WRITE_REPORTS = "write_reports"
    NETWORK = "network"
    EXPORT_AGGREGATE = "export_aggregate"
    EXPORT_PII = "export_pii"
    CALL_REVIEWER = "call_reviewer"
    MUTATE_MEMORY = "mutate_memory"
    SHELL_EXEC = "shell_exec"


_DEFAULT_CAPS = frozenset({Capability.READ_DATA, Capability.WRITE_REPORTS})


@dataclass
class ToolSpec:
    """The schema the LLM sees for this tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # OpenAI-style JSON schema dict
    required: list[str] = field(default_factory=list)

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": dict(self.parameters),
                    "required": list(self.required),
                },
            },
        }


# ── Tool execution context (capability-restricted) ──────────


@dataclass
class ToolContext:
    """Capability-scoped facade injected into ``ToolHandler.handle``.

    The full legacy ctx is *not* exposed here — instead, fields are
    populated based on the tool's declared ``required_capabilities``.
    Tools that ask for ``READ_DATA`` get a DuckDB cursor; tools that
    ask for ``WRITE_REPORTS`` additionally get ``report_dir``; tools
    that don't ask for ``MUTATE_MEMORY`` see a read-only view of
    ``LongTermMemory``.

    This is the M2 enforcement of Codex review point H from M1.
    """

    name: str
    args: dict[str, Any]
    capabilities: frozenset[Capability]
    settings: Any = None
    duckdb_conn: Any = None
    catalog: Any = None
    state: Any = None
    memory: Any = None
    report_dir: Any = None
    emit_progress: Optional[Callable[..., None]] = None
    turn_id: Optional[str] = None
    tool_call_id: Optional[str] = None

    def has(self, cap: Capability) -> bool:
        return cap in self.capabilities

    def require(self, cap: Capability) -> None:
        if cap not in self.capabilities:
            raise PermissionError(
                f"Tool {self.name!r} attempted {cap.value} without declaring it"
            )


# ── ToolHandler protocol ─────────────────────────────────────


@runtime_checkable
class ToolHandler(Protocol):
    """Contract every v3 tool must satisfy."""

    @property
    def name(self) -> str: ...

    def spec(self) -> ToolSpec: ...

    def required_capabilities(self) -> frozenset[Capability]: ...

    @property
    def is_mutating(self) -> bool: ...

    async def handle(self, ctx: ToolContext) -> dict[str, Any]: ...


# ── Concrete base + legacy adapter ──────────────────────────


@dataclass
class _BaseHandler:
    """Convenience base class — implements the read-only descriptors."""

    _name: str
    _spec: ToolSpec
    _caps: frozenset[Capability] = field(default_factory=lambda: _DEFAULT_CAPS)
    _is_mutating: bool = False

    @property
    def name(self) -> str:
        return self._name

    def spec(self) -> ToolSpec:
        return self._spec

    def required_capabilities(self) -> frozenset[Capability]:
        return self._caps

    @property
    def is_mutating(self) -> bool:
        return self._is_mutating


class LegacySkillToolHandler(_BaseHandler):
    """Adapter that exposes a legacy ``@skill`` function as a ToolHandler.

    The legacy callable is invoked synchronously inside
    ``asyncio.to_thread`` so the runtime's event loop stays
    responsive. The ``ctx`` argument provided to the legacy function
    is the capability-scoped ``ToolContext`` augmented with
    backwards-compatible attributes (``ctx.dm``, ``ctx.state``,
    ``ctx.memory``, ``ctx.settings``, ``ctx.catalog``,
    ``ctx.report_dir``, ``ctx.emit_progress``).
    """

    def __init__(
        self,
        *,
        name: str,
        spec: ToolSpec,
        callable_: Callable[..., Any],
        capabilities: Iterable[Capability] = _DEFAULT_CAPS,
        is_mutating: bool = False,
    ) -> None:
        super().__init__(
            _name=name,
            _spec=spec,
            _caps=frozenset(capabilities),
            _is_mutating=is_mutating,
        )
        self._callable = callable_

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        # Pass through legacy arg names: skill(*, ctx, **args).
        sig = inspect.signature(self._callable)
        kwargs = dict(ctx.args or {})
        if "ctx" in sig.parameters:
            kwargs["ctx"] = ctx
        result = await asyncio.to_thread(self._callable, **kwargs)
        if not isinstance(result, dict):
            return {"output": result}
        return result


__all__ = [
    "Capability",
    "ToolSpec",
    "ToolContext",
    "ToolHandler",
    "LegacySkillToolHandler",
    "_BaseHandler",
]
