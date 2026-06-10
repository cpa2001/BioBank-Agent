"""Multi-agent council engine — the shared core for plan and research modes.

This runs a staged pipeline (draft candidates -> critique -> merge -> validate)
with *real* parallel fan-out over the runtime :class:`ProviderRouter`, emitting
honest per-stage / per-subagent progress events. It is deliberately generic: the
plan and research adapters (``RuntimePlanner`` / ``RuntimeResearcher``) own their
own prompts, return types, and validation, and call into this core for the
concurrency + event plumbing.

Concurrency uses the same ``ThreadPoolExecutor`` + ``as_completed`` + timeout +
cancel pattern as :mod:`biobank_agent.orchestrator` (``_parallel_chat``), because
the working provider path (``provider.complete``) is synchronous. We do NOT
rewrite providers async here — council parallelism stays internal to this module.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from dataclasses import dataclass, field
from typing import Any, Callable

from biobank_agent.runtime.types import ProviderRequest, ProviderRole

logger = logging.getLogger(__name__)

# emit(stage, *, status="running", message="", metadata=None)
EmitFn = Callable[..., None]


class CouncilError(RuntimeError):
    """Raised when the council cannot produce a usable result.

    Surfacing this (instead of silently returning a generic template) is the
    whole point: a planning failure must be visible, not disguised.
    """


@dataclass
class CouncilJob:
    """One model call to run (possibly in parallel with others)."""

    role: ProviderRole
    messages: list[dict[str, Any]]
    label: str
    stage: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CouncilResult:
    label: str
    stage: str
    role: str
    model: str
    text: str = ""
    error: str = ""
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text.strip())


@dataclass
class CouncilContext:
    """Shared dependencies + progress sink for a council run."""

    provider_router: Any
    session_id: str = "council"
    turn_id: str = "council"
    emit: EmitFn | None = None
    max_workers: int = 6
    timeout_s: float = 240.0
    clock: Callable[[], float] = field(default=lambda: 0.0)
    # When True (and an emit sink is set), LLM jobs stream their output token by
    # token; each delta is forwarded via ``emit_delta`` so a live dashboard can
    # show what each model is writing. ``cancel_event`` lets late deltas from an
    # orphaned worker (post-timeout / Ctrl-C) be dropped at the source.
    stream: bool = False
    cancel_event: Any = None

    def emit_event(self, stage: str, *, status: str = "running", message: str = "", **metadata: Any) -> None:
        if self.emit is None:
            return
        try:
            self.emit(stage, status=status, message=message, metadata=metadata)
        except Exception:  # progress must never break the pipeline
            logger.debug("council emit callback raised", exc_info=True)

    def emit_delta(self, label: str, stage: str, delta: str) -> None:
        """Forward one streamed token/chunk for ``label``. Routed through the
        same emit sink with a ``delta`` marker so the CLI can update the live
        row WITHOUT recording an event (token-rate ``record`` would saturate
        rendering). Never raises."""
        if self.emit is None or not delta:
            return
        if self.cancel_event is not None and getattr(self.cancel_event, "is_set", lambda: False)():
            return
        try:
            self.emit(stage, status="stream", message="", metadata={"subagent": label, "delta": delta})
        except Exception:
            logger.debug("council delta emit raised", exc_info=True)


def _now(ctx: CouncilContext) -> float:
    try:
        return float(ctx.clock())
    except Exception:
        return 0.0


def run_one(ctx: CouncilContext, job: CouncilJob, stage_cancel: "threading.Event | None" = None) -> CouncilResult:
    """Execute a single model call, capturing failures instead of raising.

    ``stage_cancel`` (set by ``run_parallel`` on timeout) and ``ctx.cancel_event``
    (set globally, e.g. on Ctrl-C) both suppress further streamed deltas from an
    orphaned worker whose HTTP read cannot be interrupted."""
    model = ctx.provider_router.model_for_role(job.role)
    started = _now(ctx)
    try:
        provider = ctx.provider_router.resolve(job.role)
    except Exception:
        # Fall back to the primary executor's provider if a role is unmapped.
        provider = ctx.provider_router.resolve(ProviderRole.PRIMARY_EXECUTOR)
        model = ctx.provider_router.model_for_role(ProviderRole.PRIMARY_EXECUTOR)
    stream_cb = None
    if ctx.stream and ctx.emit is not None:
        def stream_cb(delta: str, _label: str = job.label, _stage: str = job.stage) -> None:
            if stage_cancel is not None and stage_cancel.is_set():
                return
            ctx.emit_delta(_label, _stage, delta)
    request = ProviderRequest(
        session_id=ctx.session_id,
        turn_id=ctx.turn_id,
        role=job.role if isinstance(job.role, ProviderRole) else ProviderRole(str(job.role)),
        messages=list(job.messages),
        tools=[],
        model=model,
        metadata=dict(job.metadata),
        stream_cb=stream_cb,
    )
    try:
        response = provider.complete(request)
        return CouncilResult(
            label=job.label,
            stage=job.stage,
            role=request.role.value,
            model=response.model or model,
            text=response.text or "",
            elapsed_s=_now(ctx) - started,
        )
    except Exception as exc:  # captured; the caller decides on thresholds
        logger.warning("council job %s failed: %s", job.label, exc)
        return CouncilResult(label=job.label, stage=job.stage, role=request.role.value, model=model, error=str(exc), elapsed_s=_now(ctx) - started)


def _prewarm(ctx: CouncilContext, jobs: list[CouncilJob]) -> None:
    """Force lazy provider/client creation once in the main thread so concurrent
    workers don't race the lazy-init in LazyLLMProvider._ensure()."""
    seen: set[int] = set()
    # Warm each job's role provider plus the primary-executor fallback that
    # run_one() uses when a role is unmapped.
    roles = [job.role for job in jobs] + [ProviderRole.PRIMARY_EXECUTOR]
    for role in roles:
        try:
            provider = ctx.provider_router.resolve(role)
        except Exception:
            continue
        if id(provider) in seen:
            continue
        seen.add(id(provider))
        ensure = getattr(provider, "_ensure", None)
        if callable(ensure):
            try:
                ensure()
            except Exception:
                logger.debug("provider prewarm failed for role %s", role, exc_info=True)


def run_parallel(ctx: CouncilContext, jobs: list[CouncilJob]) -> list[CouncilResult]:
    """Run jobs concurrently; emit a dispatch event per job and a completion
    event per result. Returns all results (successes and failures)."""
    if not jobs:
        return []
    _prewarm(ctx, jobs)

    def _role_of(job: CouncilJob) -> str:
        return str(getattr(job.role, "value", job.role))

    for job in jobs:
        ctx.emit_event(
            job.stage,
            status="running",
            message=f"dispatch {job.label}",
            subagent=job.label,
            model=ctx.provider_router.model_for_role(job.role),
            role=_role_of(job),
            persona=job.metadata.get("persona_tag"),
            activity=job.metadata.get("activity"),
        )

    results: list[CouncilResult] = []
    completed: set[str] = set()
    # Per-stage cancel: set on timeout so orphaned workers in THIS stage stop
    # forwarding streamed deltas, without suppressing streaming for later stages
    # (a shared/global flag would). Ctrl-C uses ctx.cancel_event instead.
    stage_cancel = threading.Event()
    pool = ThreadPoolExecutor(max_workers=max(1, min(ctx.max_workers, len(jobs))))
    future_map = {pool.submit(run_one, ctx, job, stage_cancel): job for job in jobs}
    try:
        for future in as_completed(future_map, timeout=ctx.timeout_s):
            job = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - run_one already guards
                result = CouncilResult(label=job.label, stage=job.stage, role=_role_of(job), model="", error=str(exc))
            results.append(result)
            completed.add(job.label)
            ctx.emit_event(
                job.stage,
                status="success" if result.ok else "error",
                message=(f"{result.label} ({result.model})" if result.ok else f"{result.label} failed: {result.error}"),
                subagent=result.label,
                model=result.model,
                role=result.role,
                persona=job.metadata.get("persona_tag"),
                activity=job.metadata.get("activity"),
                elapsed_s=round(result.elapsed_s, 2),
            )
    except FuturesTimeoutError:
        logger.warning("council parallel run timed out after %.1fs; using partial results", ctx.timeout_s)
        stage_cancel.set()  # stop orphaned workers in this stage from streaming further
        ctx.emit_event(jobs[0].stage, status="error", message=f"timeout after {ctx.timeout_s:.0f}s; using {len(results)} partial result(s)")
        # Resolve every job that never came through ``as_completed``. A future
        # that actually finished between the timeout firing and now is reported
        # with its real result (no false "timed out"); only genuinely unfinished
        # jobs get a timeout. Either way a per-subagent terminal event is emitted
        # so the dashboard clears that model's live row (its timer would tick
        # forever otherwise). The orphaned worker thread itself cannot be stopped
        # here (HTTP reads are not interruptible) — handled in Phase C.
        for future, job in future_map.items():
            if job.label in completed:
                continue
            if future.done() and not future.cancelled():
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - run_one guards
                    result = CouncilResult(label=job.label, stage=job.stage, role=_role_of(job), model="", error=str(exc))
                results.append(result)
                completed.add(job.label)
                ctx.emit_event(
                    job.stage,
                    status="success" if result.ok else "error",
                    message=(f"{result.label} ({result.model})" if result.ok else f"{result.label} failed: {result.error}"),
                    subagent=result.label,
                    model=result.model,
                    role=result.role,
                    persona=job.metadata.get("persona_tag"),
                    activity=job.metadata.get("activity"),
                    elapsed_s=round(result.elapsed_s, 2),
                )
            else:
                ctx.emit_event(
                    job.stage,
                    status="error",
                    message=f"{job.label} timed out",
                    subagent=job.label,
                    model=ctx.provider_router.model_for_role(job.role),
                    role=_role_of(job),
                    persona=job.metadata.get("persona_tag"),
                    activity=job.metadata.get("activity"),
                )
    finally:
        for future, job in future_map.items():
            if not future.done():
                future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
    return results


@dataclass
class MapResult:
    index: int
    label: str
    value: Any = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def map_parallel(
    ctx: CouncilContext,
    items: list[Any],
    fn: Callable[[Any], Any],
    *,
    stage: str,
    label_fn: Callable[[int, Any], str] | None = None,
) -> list[MapResult]:
    """Run ``fn(item)`` for each item concurrently (same pool/timeout/cancel
    pattern as ``run_parallel``), emitting per-item events. Used for non-LLM
    fan-out such as research retrieval. Failures are captured, not raised."""
    if not items:
        return []
    label_for = label_fn or (lambda i, _x: f"task-{i + 1}")
    for i, item in enumerate(items):
        ctx.emit_event(stage, status="running", message=f"dispatch {label_for(i, item)}", subagent=label_for(i, item))
    results: list[MapResult] = []
    pool = ThreadPoolExecutor(max_workers=max(1, min(ctx.max_workers, len(items))))
    future_map = {pool.submit(fn, item): (i, item) for i, item in enumerate(items)}
    completed: set[int] = set()
    try:
        for future in as_completed(future_map, timeout=ctx.timeout_s):
            i, item = future_map[future]
            label = label_for(i, item)
            try:
                value = future.result()
                results.append(MapResult(index=i, label=label, value=value))
                ctx.emit_event(stage, status="success", message=label, subagent=label)
            except Exception as exc:
                results.append(MapResult(index=i, label=label, error=str(exc)))
                ctx.emit_event(stage, status="error", message=f"{label} failed: {exc}", subagent=label)
            completed.add(i)
    except FuturesTimeoutError:
        logger.warning("council map_parallel timed out after %.1fs; using partial results", ctx.timeout_s)
        ctx.emit_event(stage, status="error", message=f"timeout after {ctx.timeout_s:.0f}s; using {len(results)} partial result(s)")
        # Emit a per-item terminal event for every item not yet resolved so the
        # dashboard clears its live row (mirrors run_parallel). A future that
        # finished during the timeout window is reported with its real result.
        for future, (i, item) in future_map.items():
            if i in completed:
                continue
            label = label_for(i, item)
            if future.done() and not future.cancelled():
                try:
                    value = future.result()
                    results.append(MapResult(index=i, label=label, value=value))
                    ctx.emit_event(stage, status="success", message=label, subagent=label)
                except Exception as exc:
                    results.append(MapResult(index=i, label=label, error=str(exc)))
                    ctx.emit_event(stage, status="error", message=f"{label} failed: {exc}", subagent=label)
            else:
                results.append(MapResult(index=i, label=label, error="timed out"))
                ctx.emit_event(stage, status="error", message=f"{label} timed out", subagent=label)
    finally:
        for future in future_map:
            if not future.done():
                future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
    results.sort(key=lambda r: r.index)
    return results


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM response.

    Handles ```json fences and trailing prose by scanning for the first balanced
    ``{...}`` or ``[...]`` block. Raises ``ValueError`` if nothing parses.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty response")
    # Prefer a fenced block if present.
    fence = _JSON_FENCE.search(raw)
    candidates = []
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(raw)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            block = _first_balanced_block(candidate)
            if block is not None:
                try:
                    return json.loads(block)
                except Exception:
                    continue
    raise ValueError("no JSON object/array found in response")


def _first_balanced_block(text: str) -> str | None:
    """Return the first balanced {...} or [...] substring, respecting strings."""
    start_idx = None
    opener = closer = ""
    for i, ch in enumerate(text):
        if ch in "{[":
            start_idx = i
            opener = ch
            closer = "}" if ch == "{" else "]"
            break
    if start_idx is None:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]
    return None
