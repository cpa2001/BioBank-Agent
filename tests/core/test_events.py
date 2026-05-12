"""Unit tests for biobank_agent.core.events."""

from __future__ import annotations

import asyncio

import pytest

from biobank_agent.core.events import (
    AgentEvent,
    AgentEventBus,
    AgentEventType,
    make_event_sink_bridge,
    scrub_pii,
)


def test_event_make_and_to_dict():
    ev = AgentEvent.make(
        AgentEventType.TOOL_REQUEST,
        turn_id="t1",
        tool_call_id="call_1",
        skill="prevalence",
        args={"icd10_code": "E11"},
    )
    d = ev.to_dict()
    assert d["type"] == "tool_request"
    assert d["turn_id"] == "t1"
    assert d["tool_call_id"] == "call_1"
    assert d["payload"]["skill"] == "prevalence"


def test_legacy_event_sink_args_for_plan_phase():
    ev = AgentEvent.make(
        AgentEventType.PLAN_PHASE,
        phase="Execution",
        actor="biobank",
        status="running",
        message="step 1",
        metadata={"step_id": "s1"},
    )
    args = ev.to_legacy_event_sink_args()
    assert args == ("Execution", "biobank", "running", "step 1", {"step_id": "s1"})


def test_legacy_event_sink_args_for_tool_started():
    ev = AgentEvent.make(
        AgentEventType.TOOL_STARTED,
        tool_call_id="c1",
        skill="train_model",
        description="train E11 model",
    )
    args = ev.to_legacy_event_sink_args()
    assert args is not None
    phase, actor, status, message, metadata = args
    assert phase == "Execution"
    assert actor == "train_model"
    assert status == "running"
    assert metadata["tool_call_id"] == "c1"


def test_legacy_event_sink_returns_none_for_unmapped():
    ev = AgentEvent.make(AgentEventType.MESSAGE_DELTA, text="hello")
    assert ev.to_legacy_event_sink_args() is None


def test_scrub_pii_redacts_known_fields():
    p = {
        "eid": "1234567",
        "score": 0.91,
        "22001": "M",
        "inner": {
            "address": "12 Baker St",
            "ok_field": "value",
            "20074": 1234.5,
        },
        "list": [{"phone": "555-1234"}, {"safe": True}],
    }
    out = scrub_pii(p)
    assert out["eid"] == "<REDACTED>"
    assert out["22001"] == "<REDACTED:phi_field>"
    assert out["score"] == 0.91
    assert out["inner"]["address"] == "<REDACTED>"
    assert out["inner"]["20074"] == "<REDACTED:phi_field>"
    assert out["inner"]["ok_field"] == "value"
    assert out["list"][0]["phone"] == "<REDACTED>"
    assert out["list"][1]["safe"] is True


def test_scrub_pii_redacts_uk_postcode_string():
    p = {"home": "SW1A 1AA", "label": "control"}
    out = scrub_pii(p)
    assert out["home"] == "<REDACTED:postcode>"
    assert out["label"] == "control"


def test_scrub_pii_handles_non_dict():
    assert scrub_pii("x") == "x"  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_event_bus_publish_and_subscribe_one_consumer():
    bus = AgentEventBus()
    received: list[AgentEvent] = []

    async def consume():
        async for ev in bus.subscribe():
            received.append(ev)

    consumer = asyncio.create_task(consume())
    # Give the subscriber a tick to register before publishing.
    await asyncio.sleep(0)
    await bus.publish(AgentEvent.make(AgentEventType.TURN_STARTED))
    await bus.publish(AgentEvent.make(AgentEventType.TURN_FINISHED))
    await bus.close()
    await consumer

    types = [ev.type for ev in received]
    assert types == [AgentEventType.TURN_STARTED, AgentEventType.TURN_FINISHED]


@pytest.mark.asyncio
async def test_event_bus_fan_out_to_two_consumers():
    bus = AgentEventBus()
    a: list[AgentEvent] = []
    b: list[AgentEvent] = []

    async def consume(out: list[AgentEvent]) -> None:
        async for ev in bus.subscribe():
            out.append(ev)

    ta = asyncio.create_task(consume(a))
    tb = asyncio.create_task(consume(b))
    await asyncio.sleep(0)
    await bus.publish(AgentEvent.make(AgentEventType.MESSAGE_DELTA, text="x"))
    await bus.close()
    await asyncio.gather(ta, tb)
    assert len(a) == 1 and len(b) == 1


@pytest.mark.asyncio
async def test_event_bus_sync_listener_runs_inline():
    bus = AgentEventBus()
    seen: list[AgentEventType] = []
    bus.add_sync_listener(lambda ev: seen.append(ev.type))
    await bus.publish(AgentEvent.make(AgentEventType.RETRY))
    await bus.close()
    assert seen == [AgentEventType.RETRY]


@pytest.mark.asyncio
async def test_make_event_sink_bridge_publishes_plan_phase():
    bus = AgentEventBus()
    sink = make_event_sink_bridge(bus)
    received: list[AgentEvent] = []

    async def consume():
        async for ev in bus.subscribe():
            received.append(ev)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    sink("Validation", "biobank", "running", "checking", {"step_id": "s1"})
    # Give the loop a tick to drain the bridged publish_nowait task.
    await asyncio.sleep(0.01)
    await bus.close()
    await consumer

    assert len(received) == 1
    ev = received[0]
    assert ev.type == AgentEventType.PLAN_PHASE
    assert ev.payload["phase"] == "Validation"
    assert ev.payload["status"] == "running"
    assert ev.payload["metadata"]["step_id"] == "s1"
