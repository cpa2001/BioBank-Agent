# Observability

Biobank Agent exposes two privacy-preserving observability paths:

- local JSONL telemetry under `memory/telemetry.jsonl` by default;
- optional OpenTelemetry spans for turn/tool timing.

Neither path records raw user prompts, tool arguments, model answers, raw tool
results, participant identifiers, or local dataset rows.

OpenTelemetry attributes are stricter than local JSONL rows: spans only carry
low-cardinality metadata such as event type, skill name, model id, status,
timing, and normalized error type. Per-turn ids, tool-call ids, timestamps,
raw arguments, summaries, and result payloads are deliberately excluded from
span attributes.

## Local JSONL Telemetry

Enabled by default:

```bash
TELEMETRY_ENABLED=true
TELEMETRY_JSONL=
```

When `TELEMETRY_JSONL` is empty, rows are written to:

```text
<memory_dir>/telemetry.jsonl
```

Rows contain event type, skill name, status, elapsed time, and normalized error
type only.

## OpenTelemetry

Install tracing dependencies:

```bash
pip install -e ".[tracing]"
```

Use the process default tracer provider:

```bash
OTEL_ENABLED=true
OTEL_EXPORTER=none
OTEL_SERVICE_NAME=biobank-agent
```

Print spans to the console:

```bash
OTEL_ENABLED=true
OTEL_EXPORTER=console
OTEL_SERVICE_NAME=biobank-agent
```

Send spans to an OTLP/HTTP collector:

```bash
OTEL_ENABLED=true
OTEL_EXPORTER=otlp
OTEL_ENDPOINT=http://127.0.0.1:4318/v1/traces
OTEL_SERVICE_NAME=biobank-agent
OTEL_POLICY=local
```

## Production Collector Policy

Use production policy when spans leave the local developer machine:

```bash
OTEL_ENABLED=true
OTEL_POLICY=production
OTEL_EXPORTER=otlp
OTEL_ENDPOINT=https://otel.example.org/v1/traces
OTEL_SERVICE_NAME=biobank-agent-prod
```

Production policy is enforced at startup:

- exporter must be `otlp`;
- endpoint must be an absolute `http(s)` URL ending in `/v1/traces`;
- endpoint must use `https` and must not be loopback by default;
- loopback or cleartext endpoints are allowed only when
  `OTEL_ALLOW_LOCAL_ENDPOINT=true`, for explicitly managed sidecar collectors.

Keep collector-side processors aligned with the same privacy contract: do not
add raw prompts, SQL, tool arguments, participant ids, file paths, or result
payloads as resource attributes, span attributes, logs, or events.

## Jaeger Local Run

Start Jaeger all-in-one with OTLP enabled:

```bash
docker run --rm --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

Run Biobank Agent with OTLP:

```bash
OTEL_ENABLED=true \
OTEL_EXPORTER=otlp \
OTEL_ENDPOINT=http://127.0.0.1:4318/v1/traces \
OTEL_POLICY=local \
biobank
```

Open `http://127.0.0.1:16686` and select service `biobank-agent`.

Expected spans:

- `biobank.turn`
- `biobank.tool.<skill_name>`

If no spans appear, first verify the agent run emitted tool activity, then retry
with `OTEL_EXPORTER=console` to isolate collector connectivity from agent
instrumentation.
