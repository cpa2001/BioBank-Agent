"""Spawn a biobank child session for a subtask, with a hard structural depth guard.

"biobank-spawns-biobank": run a subtask in a fresh :class:`AgentSession` under the same
:class:`AgentRuntime`, one level deeper than its parent. The depth is stamped on the child's state, so a
subagent that itself spawns a subagent is bounded by ``max_depth`` — a STRUCTURAL guard, unlike the
legacy prompt-only "do not spawn" instruction. ``run_turn`` is injected so the orchestration is unit-
testable offline with no model. Gated OFF by default at the call site (``biobank_subagent_enabled``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

# Depth stamped on a child session's state.custom_data so the guard survives across nested spawns.
DEPTH_META_KEY = "subagent_depth"


@dataclass
class SubagentResult:
    ok: bool
    text: str = ""
    session_id: str = ""
    depth: int = 0
    error: str = ""


def current_depth(session: Any) -> int:
    """Depth of ``session`` (0 for a top-level session that was never spawned as a subagent)."""
    try:
        return int((session.state.custom_data or {}).get(DEPTH_META_KEY, 0) or 0)
    except Exception:
        return 0


def _final_text(session: Any) -> str:
    try:
        for turn in reversed(getattr(session, "turns", []) or []):
            for msg in reversed(getattr(turn, "assistant_messages", []) or []):
                if getattr(msg, "text", ""):
                    return str(msg.text)
    except Exception:
        pass
    return ""


def spawn_biobank_subagent(
    runtime: Any,
    parent_session: Any,
    task: str,
    *,
    title: str = "subtask",
    max_depth: int = 2,
    run_turn: Optional[Callable[[Any, str], Any]] = None,
) -> SubagentResult:
    """Run ``task`` in a fresh child :class:`AgentSession` under ``runtime``, one level deeper.

    Refuses to spawn beyond ``max_depth`` so nested subagents cannot recurse without bound. Never raises:
    a depth-limit hit or a child-turn error is returned as ``ok=False`` with a reason.
    """
    depth = current_depth(parent_session) + 1
    if depth > max_depth:
        return SubagentResult(ok=False, depth=depth, error=f"subagent depth {depth} exceeds max_depth {max_depth}")
    child = runtime.create_session(title=title, cwd=getattr(parent_session, "cwd", ""))
    try:
        child.state.custom_data[DEPTH_META_KEY] = depth
    except Exception:
        pass
    runner = run_turn or getattr(runtime, "run_turn", None)
    if runner is None:
        return SubagentResult(ok=False, depth=depth, error="runtime has no run_turn")
    try:
        out = runner(child, task)
    except Exception as exc:  # a child failure must not crash the parent orchestration
        return SubagentResult(ok=False, session_id=getattr(child, "session_id", ""), depth=depth, error=str(exc))
    session_out = out if out is not None else child
    return SubagentResult(
        ok=True,
        text=_final_text(session_out),
        session_id=getattr(session_out, "session_id", "") or getattr(child, "session_id", ""),
        depth=depth,
    )


__all__ = ["DEPTH_META_KEY", "SubagentResult", "current_depth", "spawn_biobank_subagent"]
