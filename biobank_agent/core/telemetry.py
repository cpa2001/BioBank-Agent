"""Privacy-preserving local telemetry for AgentEvent streams."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .events import AgentEvent, AgentEventType


_PAYLOAD_KEYS = {
    "skill",
    "state",
    "status",
    "phase",
    "actor",
    "elapsed_seconds",
    "estimated_tokens",
    "context_window",
    "severity",
    "decision",
    "confirmed",
}

_OTEL_ATTRIBUTE_KEYS = {
    "type",
    "model_id",
    "skill",
    "state",
    "status",
    "phase",
    "actor",
    "elapsed_seconds",
    "estimated_tokens",
    "context_window",
    "severity",
    "decision",
    "confirmed",
    "error_type",
}

_LOCAL_HOSTS = {"", "localhost", "127.0.0.1", "::1", "[::1]"}
_SAFE_SERVICE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{1,127}$")


class LocalTelemetryWriter:
    """Append low-cardinality runtime metadata to local JSONL."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: Any) -> "LocalTelemetryWriter":
        configured = str(getattr(settings, "telemetry_jsonl", "") or "").strip()
        if configured:
            return cls(configured)
        root = Path(getattr(settings, "memory_dir", Path.home() / ".biobank_agent"))
        return cls(root / "telemetry.jsonl")

    def append_event(self, event: AgentEvent) -> None:
        row = event_to_telemetry_row(event)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


@dataclass(frozen=True)
class OpenTelemetrySetup:
    """Result of configuring an optional OpenTelemetry exporter."""

    enabled: bool
    exporter: str
    service_name: str
    endpoint: str = ""
    policy: str = "local"
    message: str = ""


@dataclass(frozen=True)
class OpenTelemetryPolicy:
    """Normalized OpenTelemetry collector policy.

    ``local`` is meant for developer traces and can use ``none`` or
    ``console``. ``production`` is intentionally stricter: production traces
    must go through OTLP/HTTP, point at an explicit collector endpoint, and
    avoid loopback collectors unless an operator opts in for sidecar setups.
    """

    enabled: bool
    exporter: str
    service_name: str
    endpoint: str = ""
    policy: str = "local"
    allow_local_endpoint: bool = False


def opentelemetry_policy_from_settings(settings: Any) -> OpenTelemetryPolicy:
    """Return and validate the normalized OpenTelemetry policy."""
    enabled = bool(getattr(settings, "otel_enabled", False))
    service_name = str(getattr(settings, "otel_service_name", "biobank-agent") or "biobank-agent").strip()
    exporter = str(getattr(settings, "otel_exporter", "none") or "none").strip().lower()
    endpoint = str(getattr(settings, "otel_endpoint", "") or "").strip()
    policy = str(getattr(settings, "otel_policy", "local") or "local").strip().lower()
    allow_local = bool(getattr(settings, "otel_allow_local_endpoint", False))

    if exporter == "default":
        exporter = "none"
    if exporter not in {"none", "console", "otlp"}:
        raise ValueError(f"Unsupported otel_exporter {exporter!r}; use none, console, or otlp")
    if policy not in {"local", "production"}:
        raise ValueError(f"Unsupported otel_policy {policy!r}; use local or production")
    if service_name and not _SAFE_SERVICE_NAME_RE.match(service_name):
        raise ValueError(
            "otel_service_name must be 2-128 characters and contain only letters, "
            "numbers, underscore, dot, or dash"
        )

    normalized = OpenTelemetryPolicy(
        enabled=enabled,
        exporter=exporter,
        service_name=service_name or "biobank-agent",
        endpoint=endpoint,
        policy=policy,
        allow_local_endpoint=allow_local,
    )
    if not enabled or policy == "local":
        return normalized

    if exporter != "otlp":
        raise ValueError("otel_policy=production requires otel_exporter=otlp")
    if not endpoint:
        raise ValueError("otel_policy=production requires OTEL_ENDPOINT")
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("OTEL_ENDPOINT must be an absolute http(s) URL")
    if parsed.path.rstrip("/") != "/v1/traces":
        raise ValueError("OTEL_ENDPOINT must target the OTLP/HTTP /v1/traces path")
    if parsed.scheme != "https" and not allow_local:
        raise ValueError(
            "otel_policy=production requires https unless otel_allow_local_endpoint=true "
            "for an explicitly managed sidecar collector"
        )
    if parsed.hostname in _LOCAL_HOSTS and not allow_local:
        raise ValueError(
            "otel_policy=production refuses loopback collectors unless "
            "otel_allow_local_endpoint=true"
        )
    return normalized


def configure_opentelemetry_from_settings(settings: Any) -> OpenTelemetrySetup:
    """Configure an OpenTelemetry tracer provider from settings.

    Supported exporters:

    - ``none``: use the process default provider, useful when an embedding app
      has already configured tracing.
    - ``console``: write spans to stdout for local debugging.
    - ``otlp``: send spans to an OTLP/HTTP endpoint such as an OpenTelemetry
      Collector or Jaeger all-in-one collector.
    """
    policy = opentelemetry_policy_from_settings(settings)
    service_name = policy.service_name
    exporter_name = policy.exporter
    endpoint = policy.endpoint
    if not policy.enabled:
        return OpenTelemetrySetup(False, exporter_name, service_name, endpoint, policy.policy, "otel disabled")
    if exporter_name in {"", "none"}:
        return OpenTelemetrySetup(
            True,
            "none",
            service_name,
            endpoint,
            policy.policy,
            "using existing tracer provider",
        )

    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency absent
        raise RuntimeError("OpenTelemetry SDK is not installed; install biobank-agent[tracing]") from exc

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )
    if exporter_name == "console":
        exporter = ConsoleSpanExporter()
    elif exporter_name == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency absent
            raise RuntimeError(
                "OTLP exporter is not installed; install biobank-agent[tracing]"
            ) from exc
        kwargs = {"endpoint": endpoint} if endpoint else {}
        exporter = OTLPSpanExporter(**kwargs)
    else:
        raise ValueError(f"Unsupported otel_exporter {exporter_name!r}; use none, console, or otlp")

    provider.add_span_processor(BatchSpanProcessor(exporter))
    try:
        trace.set_tracer_provider(provider)
    except Exception:
        # The OpenTelemetry API only allows setting the global provider once.
        # If a host app already set it, keep that provider and still create the
        # bridge; this avoids breaking embedded use.
        return OpenTelemetrySetup(
            True,
            exporter_name,
            service_name,
            endpoint,
            policy.policy,
            "tracer provider already configured by host process",
        )
    return OpenTelemetrySetup(True, exporter_name, service_name, endpoint, policy.policy, "configured")


class OpenTelemetryBridge:
    """Optional AgentEvent -> OpenTelemetry span bridge.

    The bridge is dependency-optional: production installs can enable it with
    ``otel_enabled=true`` and the ``tracing`` extra, while default local runs
    stay local-only. It reuses ``event_to_telemetry_row`` so spans never include
    raw prompts, tool args, result payloads, or participant identifiers.
    """

    def __init__(self, *, service_name: str = "biobank-agent", tracer: Any = None) -> None:
        if tracer is None:
            try:
                from opentelemetry import trace  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency absent
                raise RuntimeError("OpenTelemetry is not installed; install biobank-agent[tracing]") from exc
            tracer = trace.get_tracer(service_name)
        self.tracer = tracer
        self._turn_spans: dict[str, Any] = {}
        self._tool_spans: dict[str, Any] = {}

    def handle_event(self, event: AgentEvent) -> None:
        row = event_to_telemetry_row(event)
        if event.type == AgentEventType.TURN_STARTED:
            key = str(event.turn_id or row.get("turn_id") or f"turn:{event.ts}")
            span = self.tracer.start_span("biobank.turn")
            self._set_attributes(span, row)
            self._turn_spans[key] = span
            return

        if event.type in {AgentEventType.TURN_FINISHED, AgentEventType.CANCELLED, AgentEventType.ERROR}:
            key = str(event.turn_id or row.get("turn_id") or "")
            span = self._turn_spans.pop(key, None)
            if span is not None:
                self._set_attributes(span, row)
                self._end_span(span)
            return

        if event.type == AgentEventType.TOOL_STARTED:
            skill = str(row.get("skill") or "tool")
            key = self._tool_key(event, row)
            span = self.tracer.start_span(f"biobank.tool.{skill}")
            self._set_attributes(span, row)
            self._tool_spans[key] = span
            return

        if event.type in {AgentEventType.TOOL_RESULT, AgentEventType.TOOL_ERROR}:
            key = self._tool_key(event, row)
            span = self._tool_spans.pop(key, None)
            if span is None:
                skill = str(row.get("skill") or "tool")
                span = self.tracer.start_span(f"biobank.tool.{skill}")
            self._set_attributes(span, row)
            self._end_span(span)

    @staticmethod
    def _tool_key(event: AgentEvent, row: dict[str, Any]) -> str:
        return str(event.tool_call_id or row.get("tool_call_id") or row.get("skill") or f"tool:{event.ts}")

    @staticmethod
    def _set_attributes(span: Any, row: dict[str, Any]) -> None:
        setter = getattr(span, "set_attribute", None)
        if not callable(setter):
            return
        for key, value in row.items():
            if key not in _OTEL_ATTRIBUTE_KEYS:
                continue
            if isinstance(value, (str, int, float, bool)):
                setter(f"biobank.{key}", value)

    @staticmethod
    def _end_span(span: Any) -> None:
        end = getattr(span, "end", None)
        if callable(end):
            end()


def event_to_telemetry_row(event: AgentEvent) -> dict[str, Any]:
    """Convert an AgentEvent to a query/result-free telemetry row."""
    payload = event.payload or {}
    row: dict[str, Any] = {
        "type": event.type.value,
        "ts": event.ts,
        "turn_id": event.turn_id,
        "tool_call_id": event.tool_call_id,
        "model_id": event.model_id,
    }
    for key in _PAYLOAD_KEYS:
        if key in payload:
            row[key] = _scalar(payload[key])
    if event.type == AgentEventType.TOOL_ERROR:
        row["error_type"] = _normalise_error_type(payload.get("error", ""))
    return {k: v for k, v in row.items() if v not in (None, "")}


def _scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(type(value).__name__)


def _normalise_error_type(error: Any) -> str:
    text = " ".join(str(error or "").strip().split()).lower()
    if not text:
        return "unknown"
    text = re.sub(r"/users/[^ ]+", "<path>", text)
    text = re.sub(r"\b\d+\b", "<n>", text)
    return text[:80]


__all__ = [
    "LocalTelemetryWriter",
    "OpenTelemetryBridge",
    "OpenTelemetrySetup",
    "configure_opentelemetry_from_settings",
    "event_to_telemetry_row",
    "opentelemetry_policy_from_settings",
]
