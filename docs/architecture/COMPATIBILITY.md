# Biobank Agent v3.0 Compatibility Matrix

Last updated: 2026-05-12 (rc1 cut).

This matrix is the contract for what `biobank-agent` supports in the v3.0
release line. Items marked "Tier 1" are exercised in CI and any breakage is a
release blocker; Tier 2 is best-effort and may regress without blocking; "Not
supported" combinations are explicitly excluded.

> Source of truth: `pyproject.toml` (`requires-python = ">=3.10"`) and the
> reference implementations in `biobank_agent/extensions/mcp_manager.py`,
> `biobank_agent/cli/`, and `biobank_agent/core/telemetry.py`.

---

## Tier 1 — CI tested

| Dimension | Versions / values |
|-----------|-------------------|
| Python | 3.10, 3.11 |
| Operating system | Linux x86_64 (Ubuntu 22.04 LTS, Debian 12), macOS arm64 (14+) |
| `openai` SDK | `>=1.50,<2.0` |
| `textual` | `>=0.80` (TUI mode only; CLI mode does not require Textual) |
| MCP transport | `stdio` |
| OpenTelemetry exporter | `console`, `otlp` (HTTP/JSON to `/v1/traces`) |
| Terminal | iTerm2 ≥ 3.5, Apple Terminal (macOS Sonoma+), default Linux terminals + `tmux` ≥ 3.4 |

Tier 1 means the v3-completion audit, behavioral / scheduled eval, and the
manual benchmark live runner all execute successfully in CI for this
combination.

## Tier 2 — Best-effort

| Dimension | Versions / values | Notes |
|-----------|-------------------|-------|
| Python | 3.12 | Smoke-tested locally; not in CI |
| Operating system | macOS x86_64 (Intel) | Works but no longer the developer baseline |
| `textual` | `>=0.70,<0.80` | TUI panels render but newer-only key bindings may degrade gracefully |
| MCP transport | `http-sse` (local stub or self-hosted) | Protocol implemented and locally tested; production HTTP/SSE compatibility is a v3.1 target |
| Terminal | nested `tmux` panes | Textual may show occasional render artefacts; the CLI mode is fully functional |
| Data layouts | `UKB-RAP` (read-only mock) | Requires `dxpy` and DNAnexus credentials; only read paths covered |

Tier 2 is supported in the sense that we accept bug reports and aim to keep
functionality intact, but a regression is not a release blocker.

## Not supported

| Dimension | Why |
|-----------|-----|
| Python 3.9 | Below `pyproject.toml requires-python = ">=3.10"` |
| Python 3.13 | Some heavy optional dependencies (e.g. `cellxgene-census`, certain TUI back-ends) do not yet ship stable wheels |
| Windows native (cmd / PowerShell, no WSL) | The plan executor and `multiprocessing` paths assume POSIX-style file paths and signal handling |
| OpenTelemetry OTLP/gRPC | `core.telemetry` only ships the OTLP/HTTP exporter in v3.0; gRPC is a v3.2 candidate |
| MCP `websocket` transport | Not implemented; `stdio` and `http-sse` cover known biomedical MCP servers |
| Auto-published telemetry to external collectors | The default exporter is `none`; `production` policy forbids cleartext / loopback collectors. Users must opt-in by configuring `otel_exporter=otlp` and an explicit HTTPS `/v1/traces` endpoint. |

---

## Data-environment expectations

- **Local de-identified UKB parquet** is the canonical bank for the v3.0
  release line. The configured path (default `./milton_data`) must contain
  `ukb.parquet`, `hesin_diag.parquet`, `death_cause.parquet`, and the field
  catalogue (`field.txt`, `category.txt`). `biobank-agent skill
  bank_data_readiness --banks ukb` must return `READY` to satisfy
  v3-completion criterion 13a.
- **Full UKB raw CSV access** is supported through
  `UKB_RAW_DIR=/Users/chenpengan/Projects/CUHK/UKB`. The agent reads raw CSV
  headers for inventory and can materialize selected field IDs into the
  optional partitioned feature store at `UKB_FULL_PARQUET_DIR` (default:
  `/Users/chenpengan/Projects/CUHK/ukb_full_parquet`). This preserves the
  current Milton parquet subset for fast modelling while making non-Milton
  raw fields discoverable and usable on demand.
- **HPP / CKB / RAP** support is exercised against synthetic fixtures only.
  Real institutional data evidence is a v3.1 deliverable; rc1 ships with
  `BLOCKED_EXTERNAL` for `credentialed_hpp_ckb_rap_data` (criterion 13b).

## CLI surface

- Entry point: `biobank` (defined in `pyproject.toml [project.scripts]`).
- Default mode: `biobank` opens the interactive shell first; the shell uses
  `prompt_toolkit` when available and falls back to standard console input in
  lightweight environments.
- The single interactive `biobank` REPL is the only interaction mode. The former
  `--tui` (Textual) and `--legacy-repl` modes were removed during consolidation;
  `biobank [<task>]` always launches the REPL (seeding `<task>` as the first turn).
- Slash command surface ships with 75 built-in commands under
  `biobank_agent/cli/commands/`. Third-party command modules can be loaded
  through `BIOBANK_CLI_COMMAND_MODULES` (see `docs/guides/CLI_COMMAND_PLUGINS.md`).
