"""Tests for local privacy-preserving telemetry."""

from __future__ import annotations

import json
from types import SimpleNamespace

from biobank_agent.core.events import AgentEvent, AgentEventType
from biobank_agent.core.telemetry import (
    LocalTelemetryWriter,
    OpenTelemetryBridge,
    configure_opentelemetry_from_settings,
    event_to_telemetry_row,
    opentelemetry_policy_from_settings,
)


class FakeSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attrs: dict[str, object] = {}
        self.ended = False

    def set_attribute(self, key: str, value: object) -> None:
        self.attrs[key] = value

    def end(self) -> None:
        self.ended = True


class FakeTracer:
    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []

    def start_span(self, name: str) -> FakeSpan:
        span = FakeSpan(name)
        self.spans.append(span)
        return span


def test_telemetry_row_drops_query_args_and_result_payload():
    event = AgentEvent.make(
        AgentEventType.TOOL_ERROR,
        turn_id="turn_1",
        tool_call_id="call_1",
        skill="train_model",
        args={"icd10_code": "E11"},
        summary={"auc": 0.8},
        error="/Users/example/raw participant 12345 failed",
        elapsed_seconds=1.2,
    )

    row = event_to_telemetry_row(event)

    assert row["type"] == "tool_error"
    assert row["skill"] == "train_model"
    assert row["elapsed_seconds"] == 1.2
    assert "args" not in row
    assert "summary" not in row
    assert "/Users/example" not in row["error_type"]
    assert "12345" not in row["error_type"]


def test_local_telemetry_writer_uses_memory_dir_default(tmp_path):
    settings = SimpleNamespace(memory_dir=tmp_path, telemetry_jsonl="")
    writer = LocalTelemetryWriter.from_settings(settings)

    writer.append_event(AgentEvent.make(AgentEventType.TOOL_RESULT, skill="missing_data", elapsed_seconds=0.4))

    path = tmp_path / "telemetry.jsonl"
    assert writer.path == path
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["type"] == "tool_result"
    assert row["skill"] == "missing_data"
    assert row["elapsed_seconds"] == 0.4


def test_opentelemetry_setup_none_uses_existing_provider_without_exporter_import():
    settings = SimpleNamespace(
        otel_enabled=True,
        otel_service_name="biobank-agent-test",
        otel_exporter="none",
        otel_endpoint="",
        otel_policy="local",
    )

    setup = configure_opentelemetry_from_settings(settings)

    assert setup.enabled is True
    assert setup.exporter == "none"
    assert setup.service_name == "biobank-agent-test"
    assert setup.policy == "local"
    assert "existing tracer provider" in setup.message


def test_opentelemetry_production_policy_requires_otlp_collector():
    console_settings = SimpleNamespace(
        otel_enabled=True,
        otel_service_name="biobank-agent-prod",
        otel_exporter="console",
        otel_endpoint="https://otel.example.org/v1/traces",
        otel_policy="production",
    )
    try:
        opentelemetry_policy_from_settings(console_settings)
    except ValueError as exc:
        assert "requires otel_exporter=otlp" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("production policy accepted console exporter")

    localhost_settings = SimpleNamespace(
        otel_enabled=True,
        otel_service_name="biobank-agent-prod",
        otel_exporter="otlp",
        otel_endpoint="http://127.0.0.1:4318/v1/traces",
        otel_policy="production",
        otel_allow_local_endpoint=False,
    )
    try:
        opentelemetry_policy_from_settings(localhost_settings)
    except ValueError as exc:
        assert "requires https" in str(exc) or "refuses loopback" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("production policy accepted loopback collector")

    valid_settings = SimpleNamespace(
        otel_enabled=True,
        otel_service_name="biobank-agent-prod",
        otel_exporter="otlp",
        otel_endpoint="https://otel.example.org/v1/traces",
        otel_policy="production",
        otel_allow_local_endpoint=False,
    )
    policy = opentelemetry_policy_from_settings(valid_settings)
    assert policy.exporter == "otlp"
    assert policy.endpoint == "https://otel.example.org/v1/traces"
    assert policy.policy == "production"


def test_opentelemetry_bridge_tracks_turn_and_tool_spans_without_raw_payloads():
    tracer = FakeTracer()
    bridge = OpenTelemetryBridge(tracer=tracer)

    bridge.handle_event(AgentEvent.make(AgentEventType.TURN_STARTED, turn_id="turn_1", query="raw research question"))
    bridge.handle_event(
        AgentEvent.make(
            AgentEventType.TOOL_STARTED,
            turn_id="turn_1",
            tool_call_id="call_1",
            skill="train_model",
            args={"icd10_code": "E11"},
        )
    )
    bridge.handle_event(
        AgentEvent.make(
            AgentEventType.TOOL_RESULT,
            turn_id="turn_1",
            tool_call_id="call_1",
            skill="train_model",
            summary={"auc": 0.8},
            elapsed_seconds=2.5,
        )
    )
    bridge.handle_event(AgentEvent.make(AgentEventType.TURN_FINISHED, turn_id="turn_1", final_text="raw answer"))

    assert [span.name for span in tracer.spans] == ["biobank.turn", "biobank.tool.train_model"]
    turn_span, tool_span = tracer.spans
    assert turn_span.ended is True
    assert tool_span.ended is True
    assert turn_span.attrs["biobank.type"] == "turn_finished"
    assert tool_span.attrs["biobank.type"] == "tool_result"
    assert tool_span.attrs["biobank.skill"] == "train_model"
    assert tool_span.attrs["biobank.elapsed_seconds"] == 2.5
    assert "biobank.ts" not in turn_span.attrs
    assert "biobank.turn_id" not in turn_span.attrs
    assert "biobank.tool_call_id" not in tool_span.attrs
    all_attr_text = json.dumps([span.attrs for span in tracer.spans], default=str)
    assert "raw research question" not in all_attr_text
    assert "raw answer" not in all_attr_text
    assert "icd10_code" not in all_attr_text
    assert "auc" not in all_attr_text


def test_opentelemetry_bridge_records_one_shot_tool_error_when_start_event_is_missing():
    tracer = FakeTracer()
    bridge = OpenTelemetryBridge(tracer=tracer)

    bridge.handle_event(
        AgentEvent.make(
            AgentEventType.TOOL_ERROR,
            tool_call_id="missing_start",
            skill="field_search",
            error="/Users/example failed on participant 12345",
        )
    )

    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.name == "biobank.tool.field_search"
    assert span.ended is True
    assert span.attrs["biobank.type"] == "tool_error"
    assert span.attrs["biobank.skill"] == "field_search"
    assert "/Users/example" not in span.attrs["biobank.error_type"]
    assert "12345" not in span.attrs["biobank.error_type"]
