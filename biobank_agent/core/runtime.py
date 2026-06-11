"""AsyncAgent — streaming-first runtime that wraps the legacy ``Agent``.

Design summary:

- The legacy synchronous ``biobank_agent.agent.Agent`` keeps owning the
  ReAct loop and skill execution (it stays the source of truth for the
  56 production skills, multi-model orchestration, reflexion, audit
  log, etc.). The AsyncAgent here is **composition over inheritance**:
  it delegates one tool round at a time to the legacy code via
  ``asyncio.to_thread`` so we never block the event loop, and surfaces
  AgentEvents during the LLM call itself (the longest part of a turn).

- The legacy ``Agent.run()`` synchronous facade is preserved, but now
  calls ``AsyncAgent.run_to_text()`` from inside an ``asyncio.run``
  wrapper when ``settings.async_runtime_enabled`` is set and the
  current state is safe for async wrapping. Unsafe states keep using
  the synchronous loop.

The core innovation here vs. the legacy loop is that the LLM streaming
response is produced incrementally via ``AsyncLLMClient`` and exposed
through the AgentEventBus. The first text token reaches the renderer
in O(network RTT) instead of O(full completion).

Tool execution still runs synchronously on a worker thread (CPU-bound
DuckDB / pandas / lifelines work doesn't benefit from asyncio), but
each invocation is wrapped in ``asyncio.to_thread`` so other
coroutines (renderer, telemetry, cancellation watcher) keep running.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from biobank_agent.complexity import Strategy
from biobank_agent.llm import LLMResponse, ToolCall

from .compaction import (
    CompactionDecision,
    before_last_user_message,
    estimate_messages_tokens,
    evaluate as evaluate_compaction,
    force_aggressive,
    structured_extract,
)
from .events import AgentEvent, AgentEventBus, AgentEventType
from .llm.client import (
    AsyncLLMClient,
    StreamFinal,
    StreamTextDelta,
    StreamToolDelta,
)
from .safety.data_fingerprint import fingerprint as compute_fingerprint
from .safety.evidence_hooks import link_tool_result_to_evidence
from .safety.nli_causal_check import maybe_inject_disclaimer
from .telemetry import (
    LocalTelemetryWriter,
    OpenTelemetryBridge,
    configure_opentelemetry_from_settings,
)
from .tools.approval import ApprovalPolicy, builtin_profile
from .tools.protocol import ToolContext, ToolSpec, _BaseHandler
from .tools.registry import infer_legacy_capabilities, infer_legacy_is_mutating
from .tools.scheduler import ToolRequest, ToolScheduler, ToolState

if TYPE_CHECKING:  # pragma: no cover
    from biobank_agent.agent import Agent

logger = logging.getLogger(__name__)


_DEFAULT_CONTEXT_WINDOW = 180_000  # matches biobank_agent.config.Settings


# ── Runtime ─────────────────────────────────────────────────


class AsyncAgent:
    """Async streaming wrapper around the legacy ``Agent``.

    Holds a reference to the legacy agent for skill execution, memory,
    orchestration etc. Owns the AgentEventBus that consumers subscribe
    to.
    """

    def __init__(self, legacy: "Agent", *, bus: Optional[AgentEventBus] = None) -> None:
        self.legacy = legacy
        self.bus = bus or AgentEventBus()
        self._async_llm = AsyncLLMClient(legacy.llm)
        self._cancel_event: asyncio.Event = asyncio.Event()
        self._active_turn_id: Optional[str] = None
        self.telemetry: LocalTelemetryWriter | None = None
        self.otel: OpenTelemetryBridge | None = None
        if getattr(legacy.settings, "telemetry_enabled", True):
            try:
                self.telemetry = LocalTelemetryWriter.from_settings(legacy.settings)
                self.bus.add_sync_listener(self.telemetry.append_event)
            except Exception as exc:
                logger.debug("telemetry disabled: %s", exc)
        if getattr(legacy.settings, "otel_enabled", False):
            try:
                configure_opentelemetry_from_settings(legacy.settings)
                service_name = str(getattr(legacy.settings, "otel_service_name", "biobank-agent") or "biobank-agent")
                self.otel = OpenTelemetryBridge(service_name=service_name)
                self.bus.add_sync_listener(self.otel.handle_event)
            except Exception as exc:
                logger.debug("opentelemetry disabled: %s", exc)

    # ── Public API ───────────────────────────────────────────

    async def run_to_text(self, query: str) -> str:
        """Run a turn end-to-end and return the final text answer.

        Sync callers go through this via ``asyncio.run(agent.run_to_text(q))``.
        """
        text = ""
        async for event in self.stream_events(query):
            if event.type == AgentEventType.MESSAGE_COMPLETE:
                text = event.payload.get("text", "") or text
            elif event.type == AgentEventType.TURN_FINISHED:
                final = event.payload.get("final_text")
                if isinstance(final, str) and final:
                    text = final
        return text

    @staticmethod
    def is_safe_for_async(legacy: "Agent") -> tuple[bool, str]:
        """Async-safety guard against a silent multi-model downgrade.

        The async runtime currently routes every LLM call through a
        single model via ``AsyncLLMClient`` — it does not invoke the
        DEBATE / ENSEMBLE / SUPERVISOR strategies in
        ``MultiModelOrchestrator``. To avoid a silent capability
        downgrade, callers should refuse to run async when multi-model
        orchestration is enabled and the model pool has more than one
        candidate.

        Returns ``(True, "")`` when safe; ``(False, reason)`` when the
        caller should fall back to the synchronous loop.
        """
        if not getattr(legacy.settings, "multi_model_enabled", False):
            return True, ""
        try:
            pool_size = len(legacy.orchestrator.model_pool)
        except Exception:
            pool_size = 0
        if pool_size > 1:
            return (
                False,
                "multi_model_enabled with >1 models in pool; async runtime "
                "does not yet drive DEBATE/ENSEMBLE/SUPERVISOR. Falling back "
                "to synchronous loop. Disable multi_model_enabled or set "
                "model_pool to a single id to re-enable async streaming.",
            )
        return True, ""

    async def stream_events(self, query: str) -> AsyncIterator[AgentEvent]:
        """Yield events for a single user-query turn.

        The events also flow through ``self.bus`` so renderers / rollout
        / telemetry attached via ``bus.subscribe()`` receive the same
        stream out-of-band.
        """
        legacy = self.legacy
        self._cancel_event.clear()
        turn_id = f"turn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{abs(hash(query)) % 1_000_000}"
        self._active_turn_id = turn_id

        # Bookkeeping mirrored from legacy.run() so audit trails stay
        # identical for downstream tooling.
        legacy.messages.append({"role": "user", "content": query})
        legacy._active_query_id = f"q_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{abs(hash(query)) % 1_000_000}"
        try:
            legacy.memory.upsert_node(
                node_type="query",
                node_id=legacy._active_query_id,
                payload={"text": query, "timestamp": datetime.now().isoformat()},
                score=1.0,
            )
        except Exception:
            pass

        report_dir = legacy.settings.reports_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir.mkdir(parents=True, exist_ok=True)

        async for ev in self._emit_async(
            AgentEvent.make(
                AgentEventType.TURN_STARTED,
                turn_id=turn_id,
                query=query,
                report_dir=str(report_dir),
            )
        ):
            yield ev

        # StudySpec compilation (replaces agent.py:401-410). Failure is
        # not fatal; we surface it so the runtime can record the gap.
        legacy._current_study_spec = None
        try:
            spec = legacy._study_spec_compiler.compile(query)
            legacy._current_study_spec = spec
            async for ev in self._emit_async(
                AgentEvent.make(
                    AgentEventType.STUDY_SPEC_COMPILED,
                    turn_id=turn_id,
                    design=spec.design.value,
                    modalities=[m.value for m in spec.modalities],
                    tool_budget=spec.tool_budget,
                )
            ):
                yield ev
        except Exception as e:
            logger.debug("StudySpec compilation skipped: %s", e)

        max_rounds = max(1, int(legacy.settings.max_tool_rounds))
        final_text = ""

        for round_n in range(max_rounds):
            if self._cancel_event.is_set() or legacy.state.interrupted:
                async for ev in self._emit_async(
                    AgentEvent.make(
                        AgentEventType.CANCELLED,
                        turn_id=turn_id,
                        round=round_n,
                    )
                ):
                    yield ev
                final_text = "[Interrupted by user]"
                break

            # Compaction check before constructing the prompt for this round.
            messages = [legacy._system_message()] + legacy.messages
            decision = evaluate_compaction(
                messages,
                context_window=getattr(
                    legacy.settings, "context_window", _DEFAULT_CONTEXT_WINDOW
                ),
                warn_pct=getattr(legacy.settings, "compaction_warn_pct", 0.6),
                compact_pct=getattr(legacy.settings, "compaction_compact_pct", 0.75),
                force_pct=getattr(legacy.settings, "compaction_force_pct", 0.9),
            )
            if decision.severity == "warn":
                async for ev in self._emit_async(
                    AgentEvent.make(
                        AgentEventType.CONTEXT_WINDOW_WARNING,
                        turn_id=turn_id,
                        estimated_tokens=decision.estimated_tokens,
                        context_window=decision.context_window,
                        reason=decision.reason,
                    )
                ):
                    yield ev
            elif decision.should_compact:
                if decision.severity == "force":
                    legacy.messages = force_aggressive(legacy.messages)
                else:
                    legacy.messages = before_last_user_message(legacy.messages)
                messages = [legacy._system_message()] + legacy.messages
                async for ev in self._emit_async(
                    AgentEvent.make(
                        AgentEventType.COMPACTED,
                        turn_id=turn_id,
                        severity=decision.severity,
                        before_tokens=decision.estimated_tokens,
                        after_tokens=estimate_messages_tokens(messages),
                        reason=decision.reason,
                    )
                ):
                    yield ev

            # Streaming LLM call. We let the AsyncLLMClient drive the
            # legacy LLMClient.stream() on a worker thread and surface
            # text + tool deltas as events.
            self._last_round_response = None
            async for ev in self._stream_one_round(
                turn_id=turn_id,
                round_n=round_n,
                messages=messages,
            ):
                yield ev
            response = self._last_round_response

            if not response:
                # Stream failed; emit error and abort.
                final_text = "[LLM stream produced no response]"
                async for ev in self._emit_async(
                    AgentEvent.make(
                        AgentEventType.ERROR,
                        turn_id=turn_id,
                        message=final_text,
                    )
                ):
                    yield ev
                break

            # Track tokens.
            if response.usage:
                legacy.state.token_usage.update(response.usage)

            # No tool calls -> final answer.
            if not response.has_tool_calls:
                safe_text = _safe_final_text(response.text)
                legacy.messages.append({"role": "assistant", "content": safe_text})
                async for ev in self._emit_async(
                    AgentEvent.make(
                        AgentEventType.MESSAGE_COMPLETE,
                        turn_id=turn_id,
                        text=safe_text,
                        usage=response.usage,
                    )
                ):
                    yield ev
                # Mirror legacy bookkeeping.
                try:
                    wrapped = legacy.orchestrator._wrap_llm_response(
                        raw=response,
                        strategy=Strategy.SINGLE,
                        model_id=legacy.settings.llm_model,
                    )
                    legacy._record_orchestration_graph(wrapped)
                    legacy._record_executive_findings(query, safe_text, wrapped)
                except Exception:
                    pass
                final_text = safe_text
                break

            # Append assistant message (with tool_calls) before invoking
            # tools, matching the legacy contract for relay
            # compatibility.
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response.text or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                    }
                    for tc in response.tool_calls
                ],
            }
            legacy.messages.append(assistant_msg)

            # Execute each tool call. We run them sequentially to
            # preserve legacy ordering; parallel scheduling for
            # independent calls may come later.
            for tc in response.tool_calls:
                if self._cancel_event.is_set():
                    break
                async for ev in self._dispatch_tool_call(
                    tc=tc,
                    turn_id=turn_id,
                    report_dir=report_dir,
                ):
                    yield ev

        else:
            # Loop exhausted without break — max rounds reached.
            final_text = final_text or "[Max tool rounds reached]"
            legacy.messages.append({"role": "assistant", "content": final_text})

        # Post-run housekeeping (delegates to legacy).
        try:
            await asyncio.to_thread(legacy._post_run, query)
        except Exception as e:
            logger.debug("legacy _post_run failed (non-critical): %s", e)

        async for ev in self._emit_async(
            AgentEvent.make(
                AgentEventType.TURN_FINISHED,
                turn_id=turn_id,
                final_text=final_text,
                rounds=round_n + 1 if "round_n" in locals() else 0,
            )
        ):
            yield ev

        self._active_turn_id = None

    def cancel(self) -> None:
        """Request cancellation of the active turn."""
        self._cancel_event.set()
        try:
            self.legacy.state.interrupted = True
        except Exception:
            pass

    # ── Internals ────────────────────────────────────────────

    async def _emit_async(self, event: AgentEvent) -> AsyncIterator[AgentEvent]:
        """Publish ``event`` to the bus and yield it to the per-turn iter."""
        await self.bus.publish(event)
        yield event

    async def _yield_proxy(self, event: AgentEvent) -> AsyncIterator[AgentEvent]:
        """Used by ``_stream_one_round`` to bubble events to the caller."""
        await self.bus.publish(event)
        yield event

    async def _stream_one_round(
        self,
        *,
        turn_id: str,
        round_n: int,
        messages: list[dict],
    ) -> AsyncIterator[AgentEvent]:
        """Async generator: yield streaming events for one LLM call.

        Stores the final ``LLMResponse`` on ``self._last_round_response``
        so the caller can pick it up after iteration completes.
        Emits MESSAGE_DELTA / TOOL_ARG_DELTA / ERROR through both the
        per-turn iterator and the bus.
        """
        legacy = self.legacy
        ttft_seen = False
        t_start = time.time()

        try:
            tools = legacy.registry.tool_schemas() or None
        except Exception:
            tools = None

        async def _ttft_noop(_elapsed: float) -> None:
            return None

        # The legacy stream_async accepts a sync callback; we set TTFT
        # via the first delta arrival inside this generator instead.

        try:
            async for stream_event in self._async_llm.stream_async(
                messages=messages,
                tools=tools,
                temperature=0.1,
                max_tokens=4096,
            ):
                if isinstance(stream_event, StreamTextDelta):
                    elapsed = time.time() - t_start
                    if not ttft_seen and stream_event.text:
                        ttft_seen = True
                        ttft_event = AgentEvent.make(
                            AgentEventType.MESSAGE_DELTA,
                            turn_id=turn_id,
                            text="",
                            round=round_n,
                            ttft_seconds=elapsed,
                            marker="first_token",
                        )
                        await self.bus.publish(ttft_event)
                        yield ttft_event
                    delta_event = AgentEvent.make(
                        AgentEventType.MESSAGE_DELTA,
                        turn_id=turn_id,
                        text=stream_event.text,
                        round=round_n,
                    )
                    await self.bus.publish(delta_event)
                    yield delta_event
                elif isinstance(stream_event, StreamToolDelta):
                    frag = stream_event.fragment
                    arg_event = AgentEvent.make(
                        AgentEventType.TOOL_ARG_DELTA,
                        turn_id=turn_id,
                        tool_call_id=frag.call_id,
                        skill=frag.name,
                        partial_args=frag.partial_args,
                        finalized=frag.finalized,
                    )
                    await self.bus.publish(arg_event)
                    yield arg_event
                elif isinstance(stream_event, StreamFinal):
                    self._last_round_response = stream_event.response
        except Exception as e:
            logger.exception("Streaming LLM call failed: %s", e)
            err_event = AgentEvent.make(
                AgentEventType.ERROR,
                turn_id=turn_id,
                message=str(e),
                round=round_n,
            )
            await self.bus.publish(err_event)
            yield err_event
            self._last_round_response = None

    async def _dispatch_tool_call(
        self,
        *,
        tc: ToolCall,
        turn_id: str,
        report_dir: Path,
    ) -> AsyncIterator[AgentEvent]:
        """Dispatch one tool call through ToolScheduler into the legacy executor."""
        legacy = self.legacy

        # Compute pre-execution data fingerprint so we can compare it
        # against the post-execution fingerprint stored alongside the
        # checkpoint.
        try:
            pre_fp = await asyncio.to_thread(
                compute_fingerprint,
                settings=legacy.settings,
                duckdb_conn=getattr(legacy.dm, "conn", None),
                state=legacy.state,
                inputs=dict(tc.args or {}),
                measure_row_counts=False,  # cheap before, full after
            )
        except Exception as e:
            logger.debug("pre-fingerprint failed: %s", e)
            pre_fp = None

        handler = _LegacyExecutionHandler(
            legacy=legacy,
            skill_name=tc.name,
            report_dir=report_dir,
            allow_retry=True,
        )
        scheduler = ToolScheduler(
            policy=ApprovalPolicy(builtin_profile("yolo")),
            bus=self.bus,
        )
        event_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

        def _capture_tool_event(event: AgentEvent) -> None:
            if event.tool_call_id == tc.id:
                try:
                    event_queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass

        self.bus.add_sync_listener(_capture_tool_event)
        try:
            task = asyncio.create_task(
                scheduler.run(
                    ToolRequest(
                        call_id=tc.id,
                        handler=handler,
                        args=dict(tc.args or {}),
                        turn_id=turn_id,
                        context_factory=lambda req: ToolContext(
                            name=tc.name,
                            args=dict(req.args or {}),
                            capabilities=handler.required_capabilities(),
                            settings=legacy.settings,
                            duckdb_conn=getattr(legacy.dm, "conn", None),
                            catalog=getattr(legacy, "catalog", None),
                            state=getattr(legacy, "state", None),
                            memory=getattr(legacy, "memory", None),
                            report_dir=report_dir,
                            emit_progress=_legacy_emit_progress(legacy, report_dir),
                            turn_id=turn_id,
                            tool_call_id=tc.id,
                        ),
                    )
                )
            )
            while not task.done() or not event_queue.empty():
                try:
                    ev = await asyncio.wait_for(event_queue.get(), timeout=0.05)
                    yield ev
                except asyncio.TimeoutError:
                    continue
            outcome = await task
        finally:
            self.bus.remove_sync_listener(_capture_tool_event)

        execution = outcome.result if isinstance(outcome.result, dict) else {}
        elapsed = outcome.elapsed_seconds
        is_error = outcome.state != ToolState.DONE or bool(execution.get("is_error"))
        result_payload = execution.get("result") if isinstance(execution.get("result"), dict) else {}
        if not execution:
            result_payload = {"error": outcome.error or "tool execution failed"}
            execution = {"result": result_payload, "result_str": json.dumps(result_payload), "is_error": True}
            is_error = True
        elif outcome.error and "error" not in result_payload:
            result_payload = {**result_payload, "error": outcome.error}

        # Smart compaction of the tool message we hand back to the LLM.
        compacted = structured_extract(result_payload, target_tokens=2000)
        legacy.messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": compacted.text,
        })

        # Reproducibility hook: full checkpoint with data
        # fingerprint. Audit log is already written inside the legacy
        # path; checkpoint() is the heavier disk write.
        try:
            ctx_hash = await asyncio.to_thread(
                legacy.reproducibility.checkpoint,
                tc.name,
                {"args": tc.args or {}, "fingerprint": (pre_fp.to_dict() if pre_fp else None)},
                {"summary": result_payload, "compacted_text": compacted.text[:2000]},
                elapsed * 1000.0,
                "failed" if is_error else "success",
                "" if not is_error else str(result_payload.get("error", "")),
            )
        except Exception as e:
            logger.debug("reproducibility.checkpoint failed: %s", e)
            ctx_hash = ""

        async for ev in self._emit_async(
            AgentEvent.make(
                AgentEventType.REPRODUCIBILITY_CHECKPOINT,
                turn_id=turn_id,
                tool_call_id=tc.id,
                skill=tc.name,
                context_hash=ctx_hash,
                fingerprint_hash=(pre_fp.fingerprint_hash if pre_fp else ""),
                bank_id=(pre_fp.bank_id if pre_fp else ""),
            )
        ):
            yield ev

        if is_error:
            return

        # Auto-link the result into the Action Graph when relevant.
        ev_id = None
        try:
            ev_id = link_tool_result_to_evidence(
                legacy,
                skill=tc.name,
                args=dict(tc.args or {}),
                result=result_payload,
                turn_id=turn_id,
            )
        except Exception as e:
            logger.debug("evidence_hooks.link failed: %s", e)
        if ev_id:
            async for ev in self._emit_async(
                AgentEvent.make(
                    AgentEventType.EVIDENCE_LINKED,
                    turn_id=turn_id,
                    tool_call_id=tc.id,
                    skill=tc.name,
                    evidence_id=ev_id,
                )
            ):
                yield ev

        # Surface DisclosureLayer output if the skill produced one.
        disclosed = result_payload.get("disclosed") if isinstance(result_payload, dict) else None
        if isinstance(disclosed, dict):
            async for ev in self._emit_async(
                AgentEvent.make(
                    AgentEventType.DISCLOSURE_LAYER,
                    turn_id=turn_id,
                    tool_call_id=tc.id,
                    skill=tc.name,
                    layers=list(disclosed.get("layers", [])),
                    current_level=disclosed.get("current_level"),
                )
            ):
                yield ev

        async for ev in self._emit_async(
            AgentEvent.make(
                AgentEventType.TOOL_RESULT,
                turn_id=turn_id,
                tool_call_id=tc.id,
                skill=tc.name,
                summary=_extract_priority_summary(result_payload),
                elapsed_seconds=elapsed,
                compaction_strategy=compacted.strategy,
                dropped_keys=compacted.dropped_keys,
            )
        ):
            yield ev


# ── Helpers ─────────────────────────────────────────────────


_PRIORITY_SUMMARY_KEYS = (
    "auc", "mean_auc", "n_cases", "n_controls", "n_total", "n_subjects",
    "p_value", "log_rank_p", "or", "hr",
    "summary", "warnings", "scientific_finding",
)


def _extract_priority_summary(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    out: dict[str, Any] = {}
    for k in _PRIORITY_SUMMARY_KEYS:
        if k in result:
            out[k] = result[k]
    return out


def _safe_final_text(text: str) -> str:
    try:
        return maybe_inject_disclaimer(text or "")
    except Exception:
        return text or ""


def _legacy_emit_progress(legacy: "Agent", report_dir: Path):
    builder = getattr(legacy, "_build_ctx", None)
    if not callable(builder):
        return None
    try:
        return getattr(builder(report_dir), "emit_progress", None)
    except Exception:
        return None


__all__ = ["AsyncAgent"]


class _LegacyExecutionHandler(_BaseHandler):
    """ToolHandler wrapper around ``Agent._execute_skill_and_record``."""

    def __init__(
        self,
        *,
        legacy: "Agent",
        skill_name: str,
        report_dir: Path,
        allow_retry: bool,
    ) -> None:
        super().__init__(
            _name=skill_name,
            _spec=ToolSpec(
                name=skill_name,
                description="Legacy biobank skill execution",
                parameters={},
                required=[],
            ),
            _caps=infer_legacy_capabilities(skill_name),
            _is_mutating=infer_legacy_is_mutating(skill_name),
        )
        self.legacy = legacy
        self.report_dir = report_dir
        self.allow_retry = allow_retry

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.legacy._execute_skill_and_record,
            self.name,
            dict(ctx.args or {}),
            self.report_dir,
            allow_retry=self.allow_retry,
        )
