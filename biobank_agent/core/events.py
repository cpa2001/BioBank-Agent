"""Agent event model and async event bus.

The runtime emits AgentEvent objects through an asyncio Queue so renderers,
persistence (rollout.jsonl), telemetry, and PlanExecutor's existing
``event_sink`` callback can all consume the same stream.

Compatibility with legacy plan_executor.py:
    plan_executor.py:149 declares
        event_sink: Callable[[str, str, str, str, dict | None], None]
    with the 5-tuple ``(phase, actor, status, message, metadata)``.
    AgentEvent.to_legacy_event_sink_args() bridges to that protocol so a
    PlanExecutor wired to an AgentEventBus stays functional.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


# ── Event taxonomy ────────────────────────────────────────────


class AgentEventType(str, Enum):
    """Event taxonomy.

    Core lifecycle (always emitted):
        TURN_STARTED / TURN_FINISHED / CANCELLED / ERROR

    LLM streaming:
        MESSAGE_DELTA  - partial assistant text
        MESSAGE_COMPLETE - final assistant message landed
        THOUGHT - extended-thinking trace
        RETRY  - LLM API retry attempted

    Tool lifecycle:
        TOOL_REQUEST          - LLM asked for a tool
        TOOL_ARG_DELTA        - streaming JSON arg patch (codex-style)
        TOOL_STARTED          - scheduler began executing
        TOOL_PROGRESS         - mid-execution progress (bridges legacy emit_progress)
        TOOL_RESULT           - success
        TOOL_ERROR            - failure
        TOOL_CONFIRMATION_REQ - waiting for approval
        TOOL_CONFIRMATION_DONE

    Context/memory:
        MEMORY_INJECTED       - hierarchical memory loaded into prompt
        COMPACTED             - context compacted
        CONTEXT_WINDOW_WARNING - approaching token limit

    Plan integration:
        PLAN_PHASE - bridges plan_executor.py:149 event_sink protocol

    Orchestration:
        ORCHESTRATION_STRATEGY - SINGLE / DEBATE / ENSEMBLE / SUPERVISOR chosen

    Domain (optional payload extension layer):
        STUDY_SPEC_COMPILED        - StudySpec compiler succeeded
        SCHEMA_VIOLATION           - StudySpec gate refused a step/skill
        DISCLOSURE_LAYER           - 4-tier progressive disclosure landed
        EVIDENCE_LINKED            - Evidence Lattice link created
        REPRODUCIBILITY_CHECKPOINT - SHA-256 checkpoint persisted
    """

    TURN_STARTED = "turn_started"
    TURN_FINISHED = "turn_finished"
    CANCELLED = "cancelled"
    ERROR = "error"

    MESSAGE_DELTA = "message_delta"
    MESSAGE_COMPLETE = "message_complete"
    THOUGHT = "thought"
    RETRY = "retry"

    TOOL_REQUEST = "tool_request"
    TOOL_ARG_DELTA = "tool_arg_delta"
    TOOL_STARTED = "tool_started"
    TOOL_PROGRESS = "tool_progress"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    TOOL_CONFIRMATION_REQ = "tool_confirmation_req"
    TOOL_CONFIRMATION_DONE = "tool_confirmation_done"

    MEMORY_INJECTED = "memory_injected"
    COMPACTED = "compacted"
    CONTEXT_WINDOW_WARNING = "context_window_warning"

    PLAN_PHASE = "plan_phase"
    ORCHESTRATION_STRATEGY = "orchestration_strategy"

    STUDY_SPEC_COMPILED = "study_spec_compiled"
    SCHEMA_VIOLATION = "schema_violation"
    DISCLOSURE_LAYER = "disclosure_layer"
    EVIDENCE_LINKED = "evidence_linked"
    REPRODUCIBILITY_CHECKPOINT = "reproducibility_checkpoint"


# ── Event dataclass ──────────────────────────────────────────


@dataclass
class AgentEvent:
    """One event emitted by the runtime.

    payload schema is per-type; consumers should treat unknown keys as
    informational, never required. Stable keys per type are documented in
    docs/architecture/V3_EVENTS.md (created in M1.11).
    """

    type: AgentEventType
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    turn_id: Optional[str] = None
    tool_call_id: Optional[str] = None
    model_id: Optional[str] = None
    schema_version: int = 1

    @classmethod
    def make(
        cls,
        type: AgentEventType,
        *,
        turn_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        model_id: Optional[str] = None,
        **payload: Any,
    ) -> "AgentEvent":
        """Convenience constructor that pulls out routing keys from payload."""
        return cls(
            type=type,
            payload=dict(payload),
            turn_id=turn_id,
            tool_call_id=tool_call_id,
            model_id=model_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "ts": self.ts,
            "turn_id": self.turn_id,
            "tool_call_id": self.tool_call_id,
            "model_id": self.model_id,
            "schema_version": self.schema_version,
            "payload": self.payload,
        }

    # ── Legacy compatibility ─────────────────────────────────

    def to_legacy_event_sink_args(
        self,
    ) -> Optional[tuple[str, str, str, str, dict[str, Any]]]:
        """Map to plan_executor.py:149 ``event_sink`` 5-tuple if applicable.

        Plan_executor _emit signature:
            event_sink(phase, actor, status, message, metadata)

        Returns None for events that don't bridge naturally (e.g. raw token
        deltas should not flood plan progress UI).
        """
        if self.type == AgentEventType.PLAN_PHASE:
            p = self.payload
            return (
                str(p.get("phase", "Execution")),
                str(p.get("actor", "biobank")),
                str(p.get("status", "running")),
                str(p.get("message", "")),
                dict(p.get("metadata") or {}),
            )

        if self.type == AgentEventType.TOOL_STARTED:
            return (
                "Execution",
                str(self.payload.get("skill", "tool")),
                "running",
                str(self.payload.get("description", self.payload.get("skill", ""))),
                {"tool_call_id": self.tool_call_id, **(self.payload.get("metadata") or {})},
            )
        if self.type == AgentEventType.TOOL_RESULT:
            return (
                "Execution",
                str(self.payload.get("skill", "tool")),
                "success",
                str(self.payload.get("description", self.payload.get("skill", ""))),
                {"tool_call_id": self.tool_call_id, **(self.payload.get("metadata") or {})},
            )
        if self.type == AgentEventType.TOOL_ERROR:
            return (
                "Execution",
                str(self.payload.get("skill", "tool")),
                "failed",
                str(self.payload.get("error", "tool failed")),
                {"tool_call_id": self.tool_call_id, **(self.payload.get("metadata") or {})},
            )
        if self.type == AgentEventType.SCHEMA_VIOLATION:
            return (
                "Validation",
                "study_spec",
                "failed",
                str(self.payload.get("reason", "schema violation")),
                dict(self.payload.get("metadata") or {}),
            )
        return None


# ── PII scrubbing ────────────────────────────────────────────

# Direct identifiers and quasi-identifiers from UKB that must never
# appear in rollout logs or telemetry. References (UKB Field IDs):
#   - 20074/20075 home coordinates
#   - 22001 genetic sex
#   - 21000 ethnic background (sensitive aggregate)
#   - 53/54 assessment-centre date / location
#   - 189 Townsend deprivation index (geo-derived)
#   - 31 sex (administrative)
#   - 34 year of birth, 52 month of birth (DOB triangulation)
#   - 40000 date of death, 40001 cause of death
#   - 21001 BMI is allowed (not PHI, but kept here as nothing for now)
SENSITIVE_FIELD_IDS: frozenset[str] = frozenset({
    "20074", "20075",
    "22001",
    "21000",
    "53", "54",
    "189",
    "31",
    "34", "52",
    "40000", "40001", "40002",
})

# CKB / HPP biobank-specific direct identifiers. Conservative: any
# field whose semantic includes residence/home/address.
SENSITIVE_FIELD_PATTERNS: tuple[str, ...] = (
    "_eid", "_id", "_uid", "_address", "_home_", "_postcode",
    "_dob", "_birth_", "_death", "_phone", "_email",
)

_PII_KEY_PATTERNS: frozenset[str] = frozenset({
    "eid", "participant_id", "subject_id", "patient_id", "ssn",
    "name", "first_name", "last_name", "fullname",
    "email", "phone", "phone_number",
    "dob", "date_of_birth", "birthdate",
    "address", "home_address", "residence",
    "postcode", "post_code", "postal_code", "zip_code", "zip",
    "lat", "long", "latitude", "longitude",
    "home_x", "home_y", "coord_x", "coord_y",
    "ip_address", "ip", "session_token", "auth_token", "api_key",
    "nhs_id", "nhs_number",
})


def _scrub_value(value: Any, depth: int = 0) -> Any:
    """Recursively redact known PII shapes."""
    if depth > 8:
        return "<TRUNCATED>"
    if isinstance(value, dict):
        return {k: _scrub_dict_entry(k, v, depth) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_value(v, depth + 1) for v in value]
    if isinstance(value, tuple):
        return tuple(_scrub_value(v, depth + 1) for v in value)
    if isinstance(value, str):
        # Heuristic: strings that look like UK postcodes get redacted.
        if len(value) <= 10 and _looks_like_postcode(value):
            return "<REDACTED:postcode>"
    return value


def _scrub_dict_entry(key: str, value: Any, depth: int) -> Any:
    norm = str(key).lower().strip()
    if norm in _PII_KEY_PATTERNS:
        return "<REDACTED>"
    # Field IDs are sometimes used as dict keys directly (e.g., {"22001": ...}).
    # Also strip a trailing instance/array suffix like "20074-0.0".
    norm_field = norm.split("-", 1)[0]
    if norm_field in SENSITIVE_FIELD_IDS:
        return "<REDACTED:phi_field>"
    # Fuzzy match on substring patterns (e.g. participant_eid, ckb_address).
    for pattern in SENSITIVE_FIELD_PATTERNS:
        if pattern in norm:
            return "<REDACTED:pii_pattern>"
    # Catch keys that *contain* a known sensitive token (e.g. "user_email",
    # "ckb_postcode") which exact-match wouldn't hit.
    for token in _PII_KEY_PATTERNS:
        if len(token) >= 4 and token in norm and not norm.startswith("p_"):
            return "<REDACTED:pii_token>"
    return _scrub_value(value, depth + 1)


def _looks_like_postcode(s: str) -> bool:
    """Cheap heuristic for UK postcodes (e.g., 'SW1A 1AA', 'E1 6AN')."""
    s = s.strip().upper()
    if not 5 <= len(s) <= 8:
        return False
    if " " not in s:
        return False
    head, tail = s.split(" ", 1)
    if not (2 <= len(head) <= 4 and len(tail) == 3):
        return False
    return head[0].isalpha() and tail[0].isdigit()


def scrub_pii(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a redacted copy of ``payload`` safe for persistent logs.

    The runtime keeps the raw payload in-memory for renderer use; only the
    rollout writer / telemetry collector apply scrub_pii() before disk.
    """
    if not isinstance(payload, dict):
        return payload  # type: ignore[return-value]
    return {k: _scrub_dict_entry(k, v, 0) for k, v in payload.items()}


# ── Event bus ────────────────────────────────────────────────


EventListener = Callable[[AgentEvent], Awaitable[None]]
SyncEventListener = Callable[[AgentEvent], None]


class AgentEventBus:
    """Asyncio fan-out for AgentEvent.

    Each ``subscribe()`` call returns a fresh AsyncIterator backed by an
    independent queue, so renderer / rollout / telemetry consumers do not
    starve each other. Sync listeners (e.g. plan_executor's event_sink) are
    invoked inline before fan-out so they observe the same ordering as the
    runtime.
    """

    def __init__(self, queue_maxsize: int = 1024) -> None:
        self._queues: list[asyncio.Queue[Optional[AgentEvent]]] = []
        self._sync_listeners: list[SyncEventListener] = []
        self._closed = False
        self._queue_maxsize = queue_maxsize
        self._lock = asyncio.Lock()

    # ── Subscriber API ───────────────────────────────────────

    async def subscribe(self) -> AsyncIterator[AgentEvent]:
        """Async iterator over events. Stops when ``close()`` is called.

        Usage::

            async for event in bus.subscribe():
                handle(event)
        """
        queue: asyncio.Queue[Optional[AgentEvent]] = asyncio.Queue(self._queue_maxsize)
        async with self._lock:
            if self._closed:
                return
            self._queues.append(queue)
        try:
            while True:
                event = await queue.get()
                if event is None:  # close sentinel
                    return
                yield event
        finally:
            async with self._lock:
                if queue in self._queues:
                    self._queues.remove(queue)

    def add_sync_listener(self, listener: SyncEventListener) -> None:
        """Register a sync callback (runs inline; must not block)."""
        self._sync_listeners.append(listener)

    def remove_sync_listener(self, listener: SyncEventListener) -> None:
        try:
            self._sync_listeners.remove(listener)
        except ValueError:
            pass

    # ── Publisher API ────────────────────────────────────────

    async def publish(self, event: AgentEvent) -> None:
        if self._closed:
            return
        # Inline sync listeners first to keep ordering deterministic.
        for listener in list(self._sync_listeners):
            try:
                listener(event)
            except Exception as e:  # listeners must never break the bus
                logger.debug("event sync listener raised: %s", e)

        async with self._lock:
            queues = list(self._queues)
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest to keep up. Telemetry can re-derive from rollout.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning("event bus queue full, dropping event %s", event.type)

    def publish_nowait(self, event: AgentEvent) -> None:
        """Sync entrypoint for callers that aren't in a coroutine.

        Schedules ``publish`` on the running loop if any; otherwise drops to
        sync-listener fan-out only. This is the bridge legacy code needs.
        """
        if self._closed:
            return
        for listener in list(self._sync_listeners):
            try:
                listener(event)
            except Exception as e:
                logger.debug("event sync listener raised: %s", e)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._publish_async_only(event))

    async def _publish_async_only(self, event: AgentEvent) -> None:
        async with self._lock:
            queues = list(self._queues)
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            queues = list(self._queues)
            self._queues.clear()
        for q in queues:
            await q.put(None)


# ── Bridge to legacy plan_executor event_sink ───────────────


def make_event_sink_bridge(bus: AgentEventBus) -> Callable[..., None]:
    """Adapt plan_executor.py:149 event_sink protocol onto the AgentEventBus.

    Plan_executor._emit signature:
        event_sink(phase, actor, status, message, metadata)
    """

    def _sink(
        phase: str,
        actor: str,
        status: str,
        message: str,
        metadata: dict | None = None,
    ) -> None:
        ev = AgentEvent.make(
            AgentEventType.PLAN_PHASE,
            phase=phase,
            actor=actor,
            status=status,
            message=message,
            metadata=dict(metadata or {}),
        )
        bus.publish_nowait(ev)

    return _sink


__all__ = [
    "AgentEventType",
    "AgentEvent",
    "AgentEventBus",
    "EventListener",
    "SyncEventListener",
    "SENSITIVE_FIELD_IDS",
    "scrub_pii",
    "make_event_sink_bridge",
]
