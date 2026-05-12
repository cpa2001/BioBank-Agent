"""Tool execution state machine.

Loosely models gemini's ``packages/core/src/core/scheduler/scheduler.ts``
state machine but adapted to biobank's ``ApprovalPolicy``:

    Pending
       │
       ▼
    Validating         (StudySpec gate, schema check, args coercion)
       │
       ▼
    Awaiting           (ApprovalPolicy.ASK_USER → wait for user)
       │
       ▼
    Executing
       │
       ▼
    Done | Failed | Cancelled

Events at every transition flow through the AgentEventBus so renderers
and rollouts see the full lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from ..events import AgentEvent, AgentEventBus, AgentEventType
from .approval import ApprovalOutcome, ApprovalPolicy, Decision
from .protocol import ToolContext, ToolHandler

logger = logging.getLogger(__name__)


class ToolState(str, Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    AWAITING = "awaiting"
    EXECUTING = "executing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ToolRequest:
    """Input to ``ToolScheduler.run``."""

    call_id: str
    handler: ToolHandler
    args: dict[str, Any]
    turn_id: Optional[str] = None
    context_factory: Optional[Callable[..., ToolContext]] = None  # builds the ToolContext


@dataclass
class ToolOutcome:
    call_id: str
    state: ToolState
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    elapsed_seconds: float = 0.0
    approval: Optional[ApprovalOutcome] = None


# A user-prompt callback: receives the ApprovalOutcome and returns
# ``True`` to allow execution, ``False`` to deny. The runtime can use
# the same Textual confirm modal that the M4 auto-merger uses.
ConfirmFn = Callable[[ApprovalOutcome, ToolRequest], Awaitable[bool]]


class ToolScheduler:
    """Coordinates tool execution lifecycle for one or many requests."""

    def __init__(
        self,
        *,
        policy: ApprovalPolicy,
        bus: AgentEventBus,
        confirm_fn: Optional[ConfirmFn] = None,
        validators: Optional[list[Callable[[ToolRequest], Optional[str]]]] = None,
    ) -> None:
        self.policy = policy
        self.bus = bus
        self.confirm_fn = confirm_fn
        self.validators = list(validators or [])
        self._cancel: dict[str, asyncio.Event] = {}

    def cancel(self, call_id: str) -> None:
        ev = self._cancel.get(call_id)
        if ev is not None:
            ev.set()

    async def run(self, request: ToolRequest) -> ToolOutcome:
        bus = self.bus
        cancel_ev = asyncio.Event()
        self._cancel[request.call_id] = cancel_ev

        await bus.publish(
            AgentEvent.make(
                AgentEventType.TOOL_STARTED,
                turn_id=request.turn_id,
                tool_call_id=request.call_id,
                skill=request.handler.name,
                args=dict(request.args or {}),
                state=ToolState.PENDING.value,
            )
        )

        # Validating
        for validator in self.validators:
            try:
                err = validator(request)
            except Exception as e:
                err = f"validator raised: {e}"
            if err:
                await bus.publish(
                    AgentEvent.make(
                        AgentEventType.SCHEMA_VIOLATION,
                        turn_id=request.turn_id,
                        tool_call_id=request.call_id,
                        skill=request.handler.name,
                        reason=err,
                    )
                )
                outcome = ToolOutcome(
                    call_id=request.call_id,
                    state=ToolState.FAILED,
                    error=err,
                )
                await self._emit_terminal(request, outcome)
                return outcome

        # Approval
        approval = self.policy.decide(request.handler, request.args)
        if approval.decision == Decision.DENY:
            outcome = ToolOutcome(
                call_id=request.call_id,
                state=ToolState.FAILED,
                error=approval.rationale,
                approval=approval,
            )
            await self._emit_terminal(request, outcome)
            return outcome
        if approval.decision == Decision.ASK_USER:
            await bus.publish(
                AgentEvent.make(
                    AgentEventType.TOOL_CONFIRMATION_REQ,
                    turn_id=request.turn_id,
                    tool_call_id=request.call_id,
                    skill=request.handler.name,
                    capability=approval.capability.value if approval.capability else None,
                    rationale=approval.rationale,
                    state=ToolState.AWAITING.value,
                )
            )
            confirmed = True  # default-allow when no confirm fn (eval harness)
            if self.confirm_fn is not None:
                try:
                    confirmed = bool(await self.confirm_fn(approval, request))
                except Exception as e:
                    logger.warning("confirm_fn raised: %s", e)
                    confirmed = False
            await bus.publish(
                AgentEvent.make(
                    AgentEventType.TOOL_CONFIRMATION_DONE,
                    turn_id=request.turn_id,
                    tool_call_id=request.call_id,
                    confirmed=confirmed,
                )
            )
            if not confirmed:
                outcome = ToolOutcome(
                    call_id=request.call_id,
                    state=ToolState.CANCELLED,
                    error="user denied execution",
                    approval=approval,
                )
                await self._emit_terminal(request, outcome)
                return outcome

        # Build context
        try:
            ctx: ToolContext
            if request.context_factory is not None:
                ctx = request.context_factory(request)
            else:
                ctx = ToolContext(
                    name=request.handler.name,
                    args=dict(request.args or {}),
                    capabilities=request.handler.required_capabilities(),
                    turn_id=request.turn_id,
                    tool_call_id=request.call_id,
                )
        except Exception as e:
            outcome = ToolOutcome(
                call_id=request.call_id,
                state=ToolState.FAILED,
                error=f"context_factory failed: {e}",
                approval=approval,
            )
            await self._emit_terminal(request, outcome)
            return outcome

        # Execute
        await bus.publish(
            AgentEvent.make(
                AgentEventType.TOOL_PROGRESS,
                turn_id=request.turn_id,
                tool_call_id=request.call_id,
                skill=request.handler.name,
                state=ToolState.EXECUTING.value,
            )
        )
        t_start = time.time()
        try:
            handle_task = asyncio.create_task(request.handler.handle(ctx))
            cancel_task = asyncio.create_task(cancel_ev.wait())
            done, pending = await asyncio.wait(
                {handle_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for p in pending:
                p.cancel()
            if cancel_task in done and handle_task not in done:
                outcome = ToolOutcome(
                    call_id=request.call_id,
                    state=ToolState.CANCELLED,
                    error="cancelled",
                    elapsed_seconds=time.time() - t_start,
                    approval=approval,
                )
            else:
                result = await handle_task
                error_msg = _result_error_message(result)
                disclosure_error = ""
                if not error_msg:
                    disclosure_error, result = _apply_disclosure_policy(
                        result,
                        settings=getattr(ctx, "settings", None),
                    )
                terminal_error = error_msg or disclosure_error
                outcome = ToolOutcome(
                    call_id=request.call_id,
                    state=ToolState.FAILED if terminal_error else ToolState.DONE,
                    result=result if isinstance(result, dict) else {"output": result},
                    error=terminal_error,
                    elapsed_seconds=time.time() - t_start,
                    approval=approval,
                )
        except Exception as e:
            outcome = ToolOutcome(
                call_id=request.call_id,
                state=ToolState.FAILED,
                error=str(e),
                elapsed_seconds=time.time() - t_start,
                approval=approval,
            )
        finally:
            self._cancel.pop(request.call_id, None)

        await self._emit_terminal(request, outcome)
        return outcome

    async def _emit_terminal(self, request: ToolRequest, outcome: ToolOutcome) -> None:
        if outcome.state == ToolState.DONE:
            await self.bus.publish(
                AgentEvent.make(
                    AgentEventType.TOOL_RESULT,
                    turn_id=request.turn_id,
                    tool_call_id=request.call_id,
                    skill=request.handler.name,
                    summary=_summarise(outcome.result),
                    elapsed_seconds=outcome.elapsed_seconds,
                )
            )
        elif outcome.state in (ToolState.FAILED, ToolState.CANCELLED):
            await self.bus.publish(
                AgentEvent.make(
                    AgentEventType.TOOL_ERROR,
                    turn_id=request.turn_id,
                    tool_call_id=request.call_id,
                    skill=request.handler.name,
                    error=outcome.error,
                    state=outcome.state.value,
                    elapsed_seconds=outcome.elapsed_seconds,
                )
            )


def _summarise(result: dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, dict) and isinstance(result.get("result"), dict):
        result = result["result"]
    out: dict[str, Any] = {}
    for k in ("auc", "n_cases", "n_subjects", "p_value", "summary", "warnings"):
        if k in result:
            out[k] = result[k]
    return out


def _result_error_message(result: Any) -> str:
    """Return a terminal error message for structured tool failure payloads."""
    if not isinstance(result, dict):
        return ""
    if result.get("is_error") is True:
        nested = result.get("result")
        if isinstance(nested, dict) and nested.get("error"):
            return str(nested.get("error"))
        return str(result.get("error") or "tool returned is_error=True")
    if result.get("error"):
        return str(result.get("error"))
    nested = result.get("result")
    if isinstance(nested, dict) and nested.get("error"):
        return str(nested.get("error"))
    return ""


def _apply_disclosure_policy(result: Any, *, settings: Any = None) -> tuple[str, Any]:
    """Sanitise aggregate output and fail unsafe disclosures before surfacing."""
    if not isinstance(result, dict):
        return "", result

    mode = str(getattr(settings, "disclosure_control_mode", "internal") or "internal").lower()
    deidentified = bool(getattr(settings, "data_deidentified", True))
    if mode in {"off", "internal", "research"} and deidentified:
        return "", result

    payload = result.get("result") if isinstance(result.get("result"), dict) else result
    if not isinstance(payload, dict):
        return "", result

    try:
        from biobank_agent.domain.compliance.policy_engine import enforce
    except Exception:
        return "", result

    strict = mode in {"strict", "external", "export"}
    check = enforce(payload=payload, strict=strict)
    if isinstance(result.get("result"), dict):
        result = dict(result)
        result["result"] = check.sanitised
    else:
        result = check.sanitised

    if not check.passed:
        messages = "; ".join(v.message for v in check.violations)
        return messages or "disclosure policy rejected tool output", result
    return "", result


__all__ = ["ToolState", "ToolRequest", "ToolOutcome", "ToolScheduler", "ConfirmFn"]
