"""Tests for the M2 tool layer (protocol / registry / approval / scheduler)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from biobank_agent.core.events import AgentEventBus, AgentEventType
from biobank_agent.core.tools.approval import (
    ApprovalPolicy,
    Decision,
    builtin_profile,
)
from biobank_agent.core.tools.protocol import (
    Capability,
    LegacySkillToolHandler,
    ToolContext,
    ToolHandler,
    ToolSpec,
)
from biobank_agent.core.tools.scheduler import (
    ToolRequest,
    ToolScheduler,
    ToolState,
)


# ── ToolSpec / ToolContext ─────────────────────────────────


def test_tool_spec_serialises_to_openai_schema():
    spec = ToolSpec(
        name="prevalence",
        description="compute disease prevalence",
        parameters={"icd10_code": {"type": "string"}},
        required=["icd10_code"],
    )
    out = spec.to_openai_schema()
    assert out["type"] == "function"
    assert out["function"]["name"] == "prevalence"
    assert out["function"]["parameters"]["required"] == ["icd10_code"]


def test_tool_context_has_and_require():
    ctx = ToolContext(
        name="prevalence",
        args={},
        capabilities=frozenset({Capability.READ_DATA}),
    )
    assert ctx.has(Capability.READ_DATA)
    assert not ctx.has(Capability.NETWORK)
    with pytest.raises(PermissionError):
        ctx.require(Capability.NETWORK)


# ── LegacySkillToolHandler ─────────────────────────────────


def test_legacy_skill_tool_handler_implements_protocol():
    def fake_skill(*, ctx, icd10_code):
        return {"icd10": icd10_code, "n": 100}

    spec = ToolSpec(
        name="prevalence",
        description="legacy",
        parameters={"icd10_code": {"type": "string"}},
        required=["icd10_code"],
    )
    handler = LegacySkillToolHandler(
        name="prevalence",
        spec=spec,
        callable_=fake_skill,
        capabilities={Capability.READ_DATA},
        is_mutating=False,
    )
    assert isinstance(handler, ToolHandler)
    assert handler.required_capabilities() == frozenset({Capability.READ_DATA})
    assert not handler.is_mutating


@pytest.mark.asyncio
async def test_legacy_skill_tool_handler_executes_via_to_thread():
    def fake_skill(*, ctx, icd10_code):
        return {"icd10": icd10_code, "n": 100}

    handler = LegacySkillToolHandler(
        name="prevalence",
        spec=ToolSpec(name="prevalence", description="", parameters={}),
        callable_=fake_skill,
    )
    ctx = ToolContext(
        name="prevalence",
        args={"icd10_code": "E11"},
        capabilities=frozenset({Capability.READ_DATA}),
    )
    result = await handler.handle(ctx)
    assert result == {"icd10": "E11", "n": 100}


# ── ApprovalPolicy ─────────────────────────────────────────


class _StubHandler:
    def __init__(self, name: str, caps: frozenset, mutating: bool = False):
        self._name = name
        self._caps = caps
        self._mutating = mutating

    @property
    def name(self):
        return self._name

    def spec(self):
        return ToolSpec(name=self._name, description="", parameters={})

    def required_capabilities(self):
        return self._caps

    @property
    def is_mutating(self):
        return self._mutating

    async def handle(self, ctx):
        return {"ok": True}


def test_default_profile_denies_pii_export():
    handler = _StubHandler("export", frozenset({Capability.EXPORT_PII}))
    policy = ApprovalPolicy()
    out = policy.decide(handler, {})
    assert out.decision == Decision.DENY


def test_default_profile_asks_for_network():
    handler = _StubHandler("web_fetch", frozenset({Capability.NETWORK}))
    policy = ApprovalPolicy()
    out = policy.decide(handler, {})
    assert out.decision == Decision.ASK_USER
    assert out.capability == Capability.NETWORK


def test_default_profile_allows_read_data_only():
    handler = _StubHandler("prevalence", frozenset({Capability.READ_DATA}))
    out = ApprovalPolicy().decide(handler, {})
    assert out.decision == Decision.ALLOW


def test_yolo_profile_allows_almost_everything():
    handler = _StubHandler(
        "external_agents",
        frozenset({Capability.CALL_REVIEWER, Capability.NETWORK}),
    )
    policy = ApprovalPolicy(builtin_profile("yolo"))
    out = policy.decide(handler, {})
    assert out.decision == Decision.ALLOW


def test_readonly_profile_denies_write_reports():
    handler = _StubHandler("generate_report", frozenset({Capability.WRITE_REPORTS}))
    policy = ApprovalPolicy(builtin_profile("readonly"))
    out = policy.decide(handler, {})
    assert out.decision == Decision.DENY


# ── ToolScheduler ─────────────────────────────────────────


class _CountingHandler:
    """Async handler that records args and returns canned output."""

    def __init__(
        self,
        *,
        name="counter",
        caps=frozenset({Capability.READ_DATA}),
        result=None,
        raises=None,
        delay=0.0,
    ):
        self._name = name
        self._caps = caps
        self.calls = []
        self._result = result or {"ok": True}
        self._raises = raises
        self._delay = delay

    @property
    def name(self):
        return self._name

    def spec(self):
        return ToolSpec(name=self._name, description="", parameters={})

    def required_capabilities(self):
        return self._caps

    @property
    def is_mutating(self):
        return False

    async def handle(self, ctx):
        if self._delay:
            await asyncio.sleep(self._delay)
        self.calls.append(dict(ctx.args))
        if self._raises:
            raise self._raises
        return self._result


@pytest.mark.asyncio
async def test_scheduler_runs_when_allowed():
    bus = AgentEventBus()
    handler = _CountingHandler()
    scheduler = ToolScheduler(policy=ApprovalPolicy(), bus=bus)
    outcome = await scheduler.run(
        ToolRequest(
            call_id="c1",
            handler=handler,
            args={"icd10_code": "E11"},
            turn_id="t1",
        )
    )
    assert outcome.state == ToolState.DONE
    assert handler.calls == [{"icd10_code": "E11"}]


@pytest.mark.asyncio
async def test_scheduler_denies_pii():
    handler = _CountingHandler(caps=frozenset({Capability.EXPORT_PII}))
    bus = AgentEventBus()
    scheduler = ToolScheduler(policy=ApprovalPolicy(), bus=bus)
    outcome = await scheduler.run(
        ToolRequest(call_id="c2", handler=handler, args={}, turn_id="t1")
    )
    assert outcome.state == ToolState.FAILED
    assert "denied" in outcome.error.lower() or "deny" in outcome.error.lower()
    assert handler.calls == []


@pytest.mark.asyncio
async def test_scheduler_prompts_then_executes_when_user_confirms():
    handler = _CountingHandler(caps=frozenset({Capability.NETWORK}))
    bus = AgentEventBus()

    async def confirm(_outcome, _request):
        return True

    scheduler = ToolScheduler(
        policy=ApprovalPolicy(),
        bus=bus,
        confirm_fn=confirm,
    )
    outcome = await scheduler.run(
        ToolRequest(call_id="c3", handler=handler, args={}, turn_id="t1")
    )
    assert outcome.state == ToolState.DONE
    assert handler.calls == [{}]


@pytest.mark.asyncio
async def test_scheduler_cancels_when_user_denies():
    handler = _CountingHandler(caps=frozenset({Capability.NETWORK}))
    bus = AgentEventBus()

    async def deny(_outcome, _request):
        return False

    scheduler = ToolScheduler(policy=ApprovalPolicy(), bus=bus, confirm_fn=deny)
    outcome = await scheduler.run(
        ToolRequest(call_id="c4", handler=handler, args={}, turn_id="t1")
    )
    assert outcome.state == ToolState.CANCELLED
    assert handler.calls == []


@pytest.mark.asyncio
async def test_scheduler_handles_handler_exception():
    handler = _CountingHandler(raises=ValueError("boom"))
    bus = AgentEventBus()
    scheduler = ToolScheduler(policy=ApprovalPolicy(), bus=bus)
    outcome = await scheduler.run(
        ToolRequest(call_id="c5", handler=handler, args={}, turn_id="t1")
    )
    assert outcome.state == ToolState.FAILED
    assert "boom" in outcome.error


@pytest.mark.asyncio
async def test_scheduler_treats_structured_error_result_as_failure():
    handler = _CountingHandler(result={"error": "bad args"})
    bus = AgentEventBus()
    scheduler = ToolScheduler(policy=ApprovalPolicy(), bus=bus)
    outcome = await scheduler.run(
        ToolRequest(call_id="c_error", handler=handler, args={}, turn_id="t1")
    )
    assert outcome.state == ToolState.FAILED
    assert "bad args" in outcome.error


@pytest.mark.asyncio
async def test_scheduler_preserves_deidentified_internal_small_counts():
    handler = _CountingHandler(result={"n_cases": 3, "summary": "small internal cohort"})
    bus = AgentEventBus()
    scheduler = ToolScheduler(policy=ApprovalPolicy(), bus=bus)
    settings = SimpleNamespace(data_deidentified=True, disclosure_control_mode="internal")
    outcome = await scheduler.run(
        ToolRequest(
            call_id="c_cell_internal",
            handler=handler,
            args={},
            turn_id="t1",
            context_factory=lambda _req: ToolContext(
                name=handler.name,
                args={},
                capabilities=handler.required_capabilities(),
                settings=settings,
            ),
        )
    )
    assert outcome.state == ToolState.DONE
    assert outcome.result["n_cases"] == 3


@pytest.mark.asyncio
async def test_scheduler_blocks_unsafe_small_cell_external_disclosure():
    handler = _CountingHandler(result={"n_cases": 3, "summary": "too small"})
    bus = AgentEventBus()
    scheduler = ToolScheduler(policy=ApprovalPolicy(), bus=bus)
    settings = SimpleNamespace(data_deidentified=False, disclosure_control_mode="external")
    outcome = await scheduler.run(
        ToolRequest(
            call_id="c_cell",
            handler=handler,
            args={},
            turn_id="t1",
            context_factory=lambda _req: ToolContext(
                name=handler.name,
                args={},
                capabilities=handler.required_capabilities(),
                settings=settings,
            ),
        )
    )
    assert outcome.state == ToolState.FAILED
    assert "minimum cell count" in outcome.error


@pytest.mark.asyncio
async def test_scheduler_validator_can_block_request():
    handler = _CountingHandler()
    bus = AgentEventBus()

    def reject_anything(_req):
        return "validator rejected the request"

    scheduler = ToolScheduler(
        policy=ApprovalPolicy(),
        bus=bus,
        validators=[reject_anything],
    )
    outcome = await scheduler.run(
        ToolRequest(call_id="c6", handler=handler, args={}, turn_id="t1")
    )
    assert outcome.state == ToolState.FAILED
    assert "validator" in outcome.error
    assert handler.calls == []


@pytest.mark.asyncio
async def test_scheduler_emits_lifecycle_events_to_bus():
    handler = _CountingHandler()
    bus = AgentEventBus()

    received_types = []

    async def consume():
        async for ev in bus.subscribe():
            received_types.append(ev.type)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    scheduler = ToolScheduler(policy=ApprovalPolicy(), bus=bus)
    await scheduler.run(
        ToolRequest(call_id="c7", handler=handler, args={}, turn_id="t1")
    )
    await asyncio.sleep(0.01)
    await bus.close()
    await consumer
    assert AgentEventType.TOOL_STARTED in received_types
    assert AgentEventType.TOOL_PROGRESS in received_types
    assert AgentEventType.TOOL_RESULT in received_types
