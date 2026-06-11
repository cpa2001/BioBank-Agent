"""Fold the flat ``session.events`` stream into a hierarchical run tree (LangSmith-style).

The runtime already emits everything a nested-span view needs — ``AgentEvent`` carries
``turn_id`` (groups a turn), ``tool_call_id`` (pairs TOOL_STARTED with its TOOL_RESULT/
TOOL_ERROR into one span), ``ts`` (start/end → latency) and ``model_id`` (LLM spans). So this
is a pure transform over the event list: no engine instrumentation, no new dependency, and it
never raises on missing / duplicate / out-of-order events (a live trace is always partial).

Use it for the ``/trace`` view and as the scoring substrate for online evaluation (run_eval.py)
and the council A/B harness (M16).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from biobank_agent.core.events import AgentEvent, AgentEventType

# Span lifecycle: an opener starts a span, a matching closer (keyed by tool_call_id) ends it.
# TWO taxonomies are folded so the tree is correct whether it is built from the live scheduler
# bus (TOOL_STARTED/TOOL_RESULT/TOOL_ERROR) or from a SAVED session — the engine persists
# TOOL_CALL_STARTED/TOOL_CALL_COMPLETED into session.events, not the bus events.
_TOOL_OPENERS = {AgentEventType.TOOL_STARTED, AgentEventType.TOOL_CALL_STARTED}
_TOOL_CLOSERS = {AgentEventType.TOOL_RESULT: "ok", AgentEventType.TOOL_ERROR: "error"}
_TOOL_CALL_DONE = AgentEventType.TOOL_CALL_COMPLETED         # status comes from payload["state"]
_TOOL_CLOSER_TYPES = set(_TOOL_CLOSERS) | {_TOOL_CALL_DONE}
_OK_TOOL_STATES = {"done", "ok", "completed", "success"}     # ToolState.DONE -> ok, else error
_LLM_OPEN = AgentEventType.MODEL_REQUEST_STARTED
_LLM_CLOSERS = {AgentEventType.MESSAGE_COMPLETE: "ok", AgentEventType.MODEL_DELTA: "ok"}
_TURN_DONE = {AgentEventType.USER_TURN_COMPLETED, AgentEventType.TURN_FINISHED}

SESSION = "session"
TURN = "turn"
PHASE = "phase"
TOOL = "tool"
LLM = "llm"

OK = "ok"
ERROR = "error"
RUNNING = "running"


@dataclass
class RunNode:
    """One span in the run tree. ``kind`` ∈ {session, turn, phase, tool, llm}."""

    id: str
    name: str
    kind: str
    start: float | None = None
    end: float | None = None
    status: str = RUNNING
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list["RunNode"] = field(default_factory=list)

    @property
    def duration(self) -> float | None:
        if self.start is None or self.end is None:
            return None
        return max(0.0, self.end - self.start)

    def add(self, child: "RunNode") -> "RunNode":
        self.children.append(child)
        return child

    def walk(self) -> Iterable["RunNode"]:
        """Pre-order traversal including self."""
        yield self
        for child in self.children:
            yield from child.walk()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "status": self.status,
            "duration": self.duration,
            "attributes": dict(self.attributes),
            "children": [c.to_dict() for c in self.children],
        }


def _event_type(ev: AgentEvent) -> AgentEventType:
    t = ev.type
    return t if isinstance(t, AgentEventType) else AgentEventType(str(t))


def _tool_name(ev: AgentEvent) -> str:
    p = ev.payload or {}
    return str(p.get("name") or p.get("tool") or p.get("skill") or ev.tool_call_id or "tool")


def _tool_close_status(etype: AgentEventType, payload: dict[str, Any] | None) -> str:
    """ok/error for a tool closer. The bus closers encode it in the event type; the persisted
    ``TOOL_CALL_COMPLETED`` encodes it in ``payload['state']`` (a ToolState value)."""
    if etype is _TOOL_CALL_DONE:
        state = str((payload or {}).get("state") or "").lower()
        return OK if state in _OK_TOOL_STATES else ERROR
    return _TOOL_CLOSERS.get(etype, OK)


def build_run_tree(events: Iterable[AgentEvent], *, session_id: str = "session") -> RunNode:
    """Group events into session → turn → (phase | tool | llm) spans.

    Tolerant by construction: an opener with no closer stays ``running``; a closer with no
    opener is ignored; events without a ``turn_id`` fall into a synthetic default turn. The
    span start/end derive from event ``ts`` so durations are wall-clock latencies."""
    root = RunNode(id=session_id, name=session_id, kind=SESSION)

    turns: dict[str | None, RunNode] = {}
    open_tools: dict[str, RunNode] = {}
    open_llms: dict[str, RunNode] = {}

    def turn_node(turn_id: str | None) -> RunNode:
        node = turns.get(turn_id)
        if node is None:
            node = RunNode(id=str(turn_id or "turn"), name=str(turn_id or "turn"), kind=TURN, status=OK)
            turns[turn_id] = node
            root.add(node)
        return node

    def touch(node: RunNode, ts: float) -> None:
        node.start = ts if node.start is None else min(node.start, ts)
        node.end = ts if node.end is None else max(node.end, ts)

    for ev in events or []:
        etype = _event_type(ev)
        ts = float(ev.ts)
        turn = turn_node(ev.turn_id)
        touch(turn, ts)
        touch(root, ts)

        if etype in _TOOL_OPENERS:
            key = ev.tool_call_id or f"{ev.turn_id}:{_tool_name(ev)}:{len(turn.children)}"
            span = RunNode(id=key, name=_tool_name(ev), kind=TOOL, start=ts, status=RUNNING,
                           attributes={"tool_call_id": ev.tool_call_id})
            turn.add(span)
            open_tools[key] = span
        elif etype in _TOOL_CLOSER_TYPES:
            key = ev.tool_call_id or ""
            span = open_tools.pop(key, None)
            if span is None and not key:
                # No id to match on — close the most recent still-running tool span in this turn.
                span = next((c for c in reversed(turn.children) if c.kind == TOOL and c.status == RUNNING), None)
            if span is not None:
                span.end = ts
                span.status = _tool_close_status(etype, ev.payload)
        elif etype is _LLM_OPEN:
            key = ev.tool_call_id or f"{ev.turn_id}:llm:{len(turn.children)}"
            span = RunNode(id=key, name=str(ev.model_id or "llm"), kind=LLM, start=ts, status=RUNNING,
                           attributes={"model_id": ev.model_id})
            turn.add(span)
            open_llms[key] = span
        elif etype in _LLM_CLOSERS and open_llms:
            # An LLM span closes on completion; streaming deltas just extend its end.
            span = next(reversed(open_llms.values())) if open_llms else None
            if span is not None:
                span.end = ts
                if etype is AgentEventType.MESSAGE_COMPLETE:
                    span.status = OK
                    open_llms.pop(span.id, None)
        elif etype in _TURN_DONE and open_llms:
            # Saved sessions omit MESSAGE_COMPLETE — close any LLM span still open at turn end
            # (keep a streamed end if MODEL_DELTA already set one; otherwise bound it here).
            for span in list(open_llms.values()):
                if span.end is None:
                    span.end = ts
                span.status = OK
            open_llms.clear()
        elif etype is AgentEventType.PLAN_PHASE:
            phase_name = str((ev.payload or {}).get("phase") or (ev.payload or {}).get("name") or "phase")
            status = str((ev.payload or {}).get("status") or OK)
            turn.add(RunNode(id=f"{turn.id}:{phase_name}", name=phase_name, kind=PHASE,
                             start=ts, end=ts, status=ERROR if status in ("error", "failed") else OK,
                             attributes={k: v for k, v in (ev.payload or {}).items() if k not in ("phase", "name")}))

    # Any span still open at end-of-stream stays ``running`` (a live trace is always partial).
    return root


def summarize_run_tree(root: RunNode) -> dict[str, Any]:
    """Aggregate counts, latency and status over the whole tree (deterministic)."""
    tool_spans = [n for n in root.walk() if n.kind == TOOL]
    llm_spans = [n for n in root.walk() if n.kind == LLM]
    errors = [n for n in root.walk() if n.status == ERROR]
    durations = [(n.name, n.duration) for n in root.walk() if n.duration is not None and n.kind in (TOOL, LLM)]
    durations.sort(key=lambda kv: kv[1], reverse=True)
    tool_ok = sum(1 for n in tool_spans if n.status == OK)
    return {
        "turns": sum(1 for n in root.children if n.kind == TURN),
        "tool_calls": len(tool_spans),
        "tool_errors": sum(1 for n in tool_spans if n.status == ERROR),
        "tool_success_rate": round(tool_ok / len(tool_spans), 4) if tool_spans else None,
        "llm_calls": len(llm_spans),
        "error_nodes": len(errors),
        "total_tool_latency": round(sum(d for _, d in durations if d is not None), 6),
        "slowest": durations[:5],
    }


__all__ = ["RunNode", "build_run_tree", "summarize_run_tree",
           "SESSION", "TURN", "PHASE", "TOOL", "LLM", "OK", "ERROR", "RUNNING"]
