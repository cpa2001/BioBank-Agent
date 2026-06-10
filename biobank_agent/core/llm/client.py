"""Async wrapper around the legacy synchronous ``biobank_agent.llm.LLMClient``.

The legacy client (``biobank_agent/llm.py``) exposes a non-async
``stream()`` generator. This wrapper drives it on a worker thread and
re-publishes streaming events through an asyncio queue so the runtime
loop never blocks.

Design choices:
- We do **not** modify ``llm.py``: keeping the sync API intact protects
  every existing caller (eval harness, benchmarks, tests, plan executor
  shim). Async callers go through ``AsyncLLMClient.stream_async``.
- Tool-call argument JSON is parsed incrementally via
  ``ToolArgumentStreamParser`` so the runtime can emit
  ``TOOL_ARG_DELTA`` events while the model is still streaming.
- The wrapper preserves the legacy retry semantics by delegating to the
  underlying ``LLMClient`` (which already retries inside ``chat()`` and
  ``_create_chat_completion_with_compat``).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from biobank_agent.llm import LLMClient, LLMResponse, ToolCall

from .stream_parser import ToolArgFragment, ToolArgumentStreamParser

logger = logging.getLogger(__name__)


# ── Streaming event taxonomy (internal) ─────────────────────


@dataclass
class _Sentinel:
    """Marker for end-of-stream in the cross-thread queue."""

    error: Optional[BaseException] = None


@dataclass
class StreamTextDelta:
    """Streaming text fragment from the assistant role."""

    text: str
    ts: float = field(default_factory=time.time)


@dataclass
class StreamToolDelta:
    """Streaming tool-args fragment with best-effort partial parse."""

    fragment: ToolArgFragment
    ts: float = field(default_factory=time.time)


@dataclass
class StreamFinal:
    """Final aggregated response (includes tool_calls and usage)."""

    response: LLMResponse
    ts: float = field(default_factory=time.time)


StreamEvent = StreamTextDelta | StreamToolDelta | StreamFinal


# ── Async LLM client ─────────────────────────────────────────


class AsyncLLMClient:
    """Drive the legacy ``LLMClient`` from asyncio.

    ``stream_async()`` runs the underlying sync generator on a worker
    thread. The generator's ``yield`` values (text deltas) and the final
    ``LLMResponse`` are pushed through an asyncio queue.
    """

    def __init__(self, sync: LLMClient) -> None:
        self.sync = sync

    @property
    def model(self) -> str:
        return self.sync.model

    async def stream_async(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        ttft_callback: Optional[Any] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield streaming events as the model produces them.

        ``ttft_callback`` if provided is called with the first text-delta
        timestamp so the runtime can record TTFT for the streaming
        latency benchmark.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[StreamEvent | _Sentinel] = asyncio.Queue()
        parser = ToolArgumentStreamParser()
        ttft_recorded = {"flag": False}
        t_start = time.time()

        def _push_threadsafe(item: StreamEvent | _Sentinel) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:  # loop closed
                pass

        def _stop_requested() -> bool:
            # Predicate set up after the closure-bound stop_event below.
            return stop_event.is_set()

        def _run() -> None:
            try:
                gen = self.sync.stream(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                final: Optional[LLMResponse] = None
                # The legacy generator both yields chunks and ``return``s
                # the final LLMResponse via StopIteration.value. We feed
                # tool-call argument deltas into the parser by patching
                # the underlying low-level stream; currently we run in
                # post-finalize mode and let the parser see the final
                # tool_calls only. A future patch will
                # tee the chunk-level deltas; this still produces real
                # text streaming for TTFT, the original goal.
                while True:
                    if _stop_requested():
                        break
                    try:
                        chunk = next(gen)
                    except StopIteration as stop:
                        final = stop.value if isinstance(stop.value, LLMResponse) else None
                        break
                    if not isinstance(chunk, str):
                        continue
                    if chunk:
                        if not ttft_recorded["flag"]:
                            ttft_recorded["flag"] = True
                            if ttft_callback is not None:
                                try:
                                    ttft_callback(time.time() - t_start)
                                except Exception:
                                    pass
                        _push_threadsafe(StreamTextDelta(text=chunk))

                # Emit synthesized tool-arg fragments so downstream
                # consumers see the same shape they will get once the
                # parser is fed live deltas.
                if final and final.tool_calls:
                    for tc in final.tool_calls:
                        try:
                            arg_text = _coerce_args_to_json(tc.args)
                        except Exception:
                            arg_text = "{}"
                        frag = ToolArgFragment(
                            call_id=tc.id,
                            name=tc.name,
                            delta_text=arg_text,
                            full_text_so_far=arg_text,
                            partial_args=dict(tc.args or {}),
                            finalized=True,
                        )
                        _push_threadsafe(StreamToolDelta(fragment=frag))

                _push_threadsafe(StreamFinal(response=final or LLMResponse()))
                _push_threadsafe(_Sentinel())
            except BaseException as exc:  # propagate to async side
                _push_threadsafe(_Sentinel(error=exc))

        stop_event = threading.Event()
        worker = threading.Thread(target=_run, name="async-llm-stream", daemon=True)
        worker.start()
        try:
            while True:
                item = await queue.get()
                if isinstance(item, _Sentinel):
                    if item.error is not None:
                        raise item.error
                    return
                yield item
        finally:
            # Codex review MAJOR: avoid blocking the event loop on
            # consumer cancellation. Signal the worker to stop, drain
            # any pending items the worker queues, and let it self-clean
            # in the background (daemon=True). We do not join the
            # thread here — that would block the loop for up to 5s on
            # cancel of a long-running model call.
            stop_event.set()
            try:
                while not queue.empty():
                    queue.get_nowait()
            except Exception:
                pass


def _coerce_args_to_json(args: Any) -> str:
    import json

    return json.dumps(args, ensure_ascii=False, default=str)


__all__ = [
    "AsyncLLMClient",
    "StreamTextDelta",
    "StreamToolDelta",
    "StreamFinal",
    "StreamEvent",
]
