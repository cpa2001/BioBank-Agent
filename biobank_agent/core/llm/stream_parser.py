"""Incremental tool-arguments JSON parser.

The OpenAI ``chat.completions.create(..., stream=True)`` endpoint emits
``tool_calls`` deltas where each chunk carries a fragment of the function
arguments JSON in ``delta.tool_calls[i].function.arguments``. The legacy
``LLMClient.stream()`` already accumulates these into a string and only
parses the JSON at the end (``llm.py:294-300``), which means the runtime
cannot tell what the model is asking for until the entire call arrives.

This module accumulates deltas under a stable ``call_id`` and surfaces
two streams of information:

1. **Raw token deltas** for renderers that just want the new text.
2. **Best-effort partial-args dicts** parsed from a balanced-prefix view
   of the JSON, so the runtime can emit
   ``TOOL_ARG_DELTA(args={"icd10_code": "E11"})`` mid-stream.

The partial parse is conservative: it only parses prefixes that end at a
``,`` / ``}`` boundary outside of strings. If the buffer is mid-token the
parser returns the previously stable dict.

Kept dependency-free — no third-party streaming-JSON library required.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolArgFragment:
    """One streamed fragment for a single tool call."""

    call_id: str
    name: Optional[str] = None
    delta_text: str = ""
    full_text_so_far: str = ""
    partial_args: dict = field(default_factory=dict)
    finalized: bool = False


@dataclass
class _CallState:
    call_id: str
    name: str = ""
    buffer: str = ""
    last_partial: dict = field(default_factory=dict)
    finalized: bool = False


class ToolArgumentStreamParser:
    """Incrementally parse OpenAI tool_call argument deltas.

    Usage from inside a streaming loop::

        parser = ToolArgumentStreamParser()
        for delta in stream_chunks:
            for frag in parser.feed_openai_delta(delta):
                yield AgentEvent.make(AgentEventType.TOOL_ARG_DELTA, ...)
    """

    def __init__(self) -> None:
        self._calls: dict[int | str, _CallState] = {}
        self._index_to_id: dict[int, str] = {}

    # ── Public API ───────────────────────────────────────────

    def feed_openai_delta(self, tool_calls_delta: Iterable) -> list[ToolArgFragment]:
        """Consume one ``delta.tool_calls`` list from a streaming chunk.

        Returns a list of fragments (zero or more — multiple tool calls can
        update in the same chunk).
        """
        fragments: list[ToolArgFragment] = []
        for tc_delta in tool_calls_delta or []:
            idx = getattr(tc_delta, "index", None)
            new_id = getattr(tc_delta, "id", None) or ""
            new_name = ""
            new_args_chunk = ""
            fn = getattr(tc_delta, "function", None)
            if fn is not None:
                new_name = getattr(fn, "name", "") or ""
                new_args_chunk = getattr(fn, "arguments", "") or ""

            state = self._resolve_state(idx, new_id)
            if new_name and not state.name:
                state.name = new_name
            if new_id and not state.call_id:
                state.call_id = new_id
                if isinstance(idx, int):
                    self._index_to_id[idx] = new_id

            if new_args_chunk:
                state.buffer += new_args_chunk

            partial = _try_parse_partial(state.buffer, fallback=state.last_partial)
            if partial is not state.last_partial:
                state.last_partial = partial

            fragments.append(
                ToolArgFragment(
                    call_id=state.call_id or f"idx_{idx}",
                    name=state.name or None,
                    delta_text=new_args_chunk,
                    full_text_so_far=state.buffer,
                    partial_args=dict(state.last_partial),
                    finalized=False,
                )
            )
        return fragments

    def finalize(self) -> list[ToolArgFragment]:
        """Emit a finalized fragment per call once the stream ends."""
        out: list[ToolArgFragment] = []
        for state in self._calls.values():
            if state.finalized:
                continue
            args = _try_parse_strict(state.buffer)
            state.last_partial = args if args is not None else state.last_partial
            state.finalized = True
            out.append(
                ToolArgFragment(
                    call_id=state.call_id or "anon",
                    name=state.name or None,
                    delta_text="",
                    full_text_so_far=state.buffer,
                    partial_args=dict(state.last_partial),
                    finalized=True,
                )
            )
        return out

    def known_calls(self) -> list[tuple[str, str, str]]:
        """Return ``(call_id, name, raw_args_text)`` per known call."""
        return [
            (state.call_id or "anon", state.name, state.buffer)
            for state in self._calls.values()
        ]

    # ── Internals ────────────────────────────────────────────

    def _resolve_state(self, idx: Optional[int], call_id: str) -> _CallState:
        # Prefer index → id mapping if we've seen it before.
        if isinstance(idx, int) and idx in self._index_to_id:
            stable_id = self._index_to_id[idx]
            return self._calls.setdefault(
                stable_id, _CallState(call_id=stable_id)
            )
        if call_id:
            state = self._calls.get(call_id)
            if state is None:
                state = _CallState(call_id=call_id)
                self._calls[call_id] = state
                if isinstance(idx, int):
                    self._index_to_id[idx] = call_id
            return state
        # No id yet, key by index temporarily.
        if isinstance(idx, int):
            key = f"__idx_{idx}__"
            return self._calls.setdefault(key, _CallState(call_id=""))
        return self._calls.setdefault("__anon__", _CallState(call_id=""))


# ── JSON partial parsing helpers ─────────────────────────────


def _try_parse_strict(text: str) -> Optional[dict]:
    text = text.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _try_parse_partial(text: str, fallback: dict) -> dict:
    """Best-effort partial parse of a streaming JSON object.

    Strategy: walk the buffer balancing braces / brackets while skipping
    string contents, then attempt to close the structure at the last safe
    boundary and json.loads it. If anything goes wrong we return
    ``fallback`` so the caller never sees a regression.
    """
    text = text.lstrip()
    if not text:
        return fallback
    if not text.startswith("{"):
        # Some relays prefix with whitespace or a newline; tolerate.
        idx = text.find("{")
        if idx == -1:
            return fallback
        text = text[idx:]

    closed = _close_open_json(text)
    if closed is None:
        return fallback
    try:
        parsed = json.loads(closed)
    except json.JSONDecodeError:
        return fallback
    if not isinstance(parsed, dict):
        return fallback
    return parsed


def _close_open_json(text: str) -> Optional[str]:
    """Append closing braces / brackets so a streaming JSON prefix parses.

    Returns None if the buffer is in an unrecoverable state (e.g. ends
    inside an unterminated escape sequence).
    """
    stack: list[str] = []
    in_string = False
    escape = False
    last_safe_end = -1  # last index that ended a top-level value
    i = 0
    while i < len(text):
        ch = text[i]
        if escape:
            escape = False
            i += 1
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            stack.pop()
            if not stack:
                last_safe_end = i + 1
        elif ch == "," and len(stack) == 1 and stack[-1] == "{":
            last_safe_end = i  # safe to truncate just before this comma
        i += 1

    # Decide what to keep.
    if escape or in_string:
        # Unterminated string — back off to the last comma boundary.
        if last_safe_end <= 0:
            return None
        prefix = text[:last_safe_end].rstrip(", \t\n\r")
        return prefix + "}" * stack.count("{") + "]" * stack.count("[")

    if not stack:
        return text  # already balanced

    # Open structure: try to close it as-is. Trim trailing partial key/value.
    prefix = text.rstrip()
    # If the structure ends with ``"key":`` or ``"key": value,`` we can't
    # close cleanly; back off to last safe boundary.
    if prefix.endswith((":", ",")):
        if last_safe_end <= 0:
            return None
        prefix = text[:last_safe_end].rstrip(", \t\n\r")
    return prefix + "}" * stack.count("{") + "]" * stack.count("[")


__all__ = ["ToolArgFragment", "ToolArgumentStreamParser"]
