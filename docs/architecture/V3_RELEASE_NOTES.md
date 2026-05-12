# Biobank Agent v3.0-rc1 Release Notes

> Cut: 2026-05-12 from the v3-foundation tree.
> Audit: `biobank eval --suite v3_completion` → `status=BLOCKED_EXTERNAL`,
> `n_criteria=17`, `n_passed=12`, `n_blocked_external=5`, `n_failed=0`.

This is a release candidate. The final `v3.0` tag will be cut once the
remaining externally-credentialed criteria close per the
`v3.0-rc1 → v3.0-rc2 → v3.0` roadmap in
`/Users/chenpengan/.claude/plans/biobank-agent-biobank-agent-ukb-hpp-ckb-clever-robin.md`
§0.5.

---

## Highlights

- **Streaming async runtime**: `AsyncAgent.stream_events()` drives the CLI for
  every non-multi-model turn. Streaming time-to-first-token p50 is **52.4 ms**
  in the synthetic streaming benchmark. The multi-model orchestrator
  (`DEBATE/ENSEMBLE/SUPERVISOR`) still uses the synchronous loop; the runtime
  guards this via `AsyncAgent.is_safe_for_async()` and explicitly falls back
  to legacy mode rather than silently downgrading capability.
- **Plan + report hard gates**: `StudySpec` validation gates plan approval,
  the report contract guarantees `statistical_review → safety_check →
  world_model_audit → generate_report(format="dual")`, and `PlanExecutor`
  pauses on failure rather than printing false success.
- **Schema-first lazy tool registry**: 56+ skills register schemas at startup
  without importing heavy modules; benchmark
  (`benchmarks/skill_registration_latency.py`) protects the < 250 ms target.
- **MCP STDIO + HTTP/SSE locally**: `extensions/mcp_manager.py` discovers
  servers, normalises tool names, handles JSON-RPC over both transports,
  retries startup, reconnects on call failure, and exposes `/mcp-list`,
  `/mcp-start`, `/mcp-health`, `/mcp-call`, and `/mcp-stop`.
- **Bank-adapter main path**: `UKB / HPP / CKB / RAP` adapters route through
  `DataManager.code_prefix_filter`, `cohort.py`, and `train_model` for the
  configured synthetic fixtures. `bank_data_probe` and `bank_data_readiness`
  expose runtime + credential-gated readiness as agent-callable skills.
- **Paper-replication scaffolding**: `replicate_paper` produces spec / review /
  plan artifacts; `paper_replication_compare` lays five acceptance gates
  (`paper_access`, `cohort_count`, `model_auc`, `calibration_ece`,
  `feature_importance`, `figure_artifacts`) and surfaces the verdict in the
  dual report. MILTON (`docs/related_works/s41588-024-01898-1.pdf`) is the
  frozen fixture; `fetch_paper` prefers it over external network downloads.
- **Self-evolution safety gates**: LOW proposals require generated-skill
  allow-list + ALWAYS_PASSES eval; MEDIUM proposals demand explicit
  `confirm_fn` from the host or Textual TUI; HIGH proposals create local
  branches and return `pr_ready` until `gh` credentials are present.
- **OpenTelemetry production policy**: `otel_policy=production` enforces
  OTLP/HTTP, explicit `/v1/traces` path, default-HTTPS-non-loopback. Spans
  contain only low-cardinality metadata; queries, tool args, raw results,
  turn/tool ids, timestamps and participant ids are never recorded.

---

## Breaking changes

These changes ship in `3.0.0-rc1` and are part of the v3.0 final tag. Migrating
v2.x callers may require code or configuration updates.

- **`biobank_agent.cli` is now a package** (`cli/`) with `commands/`, `live/`,
  and `tui/` subpackages. The single-file `cli.py` from v2.x is replaced by
  `biobank_agent/cli_legacy.py` (compatibility entry — scheduled for removal
  in v3.2). External callers using `from biobank_agent.cli import main` keep
  working through the package `__init__`. External callers that imported
  internal symbols from the deleted `biobank_agent/cli.py` directly must
  re-import from `biobank_agent.cli_legacy` until v3.2.
- **Schema-first lazy skill registration**: `biobank_agent.registry`
  registers schemas at startup without importing heavy skill modules. Any
  external code that relied on `import biobank_agent` automatically loading
  every skill implementation needs to call the registry's resolution path or
  explicitly `import biobank_agent.skills.<name>`.
- **`default_analysis_sample_size = 0`** (and `max_train_rows_default = 0`):
  data-analysis skills default to **full** local de-identified data rather
  than fixed 50,000-row samples. Notebooks or scripts that relied on the
  v2.x implicit subsampling must either accept full-data execution or pass
  explicit `sample_size` / `n_samples` arguments.
- **`generate_report(format="dual")` artifact contract**: callers must accept
  the full set `report.md`, `report_technical.md`, `report_nature.md`,
  `_report_with_css.md`, `_report_nature_with_css.md`, `report.html`,
  `report_nature.html`. Missing dual artifacts now fail rather than silently
  succeed.
- **`v3_completion` criterion 13 split** into `credentialed_ukb_data` (13a,
  required for v3.0 final) + `credentialed_hpp_ckb_rap_data` (13b, v3.1
  milestone). Callers parsing the audit JSON need to expect 17 criteria, not
  16. The overall `status="COMPLETE"` rule is unchanged.

---

## Deprecations

- `biobank_agent/cli_legacy.py` — deprecation announced at v3.0; planned
  removal in v3.2 after the `cli/commands/` decomposition lands.
- `biobank_agent.cli.commands.cli_commands` — pure compatibility re-export
  for plug-ins still registering modules via the v2.x name. Use the new
  `biobank_agent.cli.commands.{session,plan,external,mcp,reproducibility,
  memory}` factories.

---

## Known `BLOCKED_EXTERNAL` (v3.0-rc1)

After the UKB full-data audit, criterion 13a can be closed locally by running
`biobank eval --suite external_evidence --collect bank-readiness --banks ukb`
against the configured `/Users/chenpengan/Projects/CUHK/UKB` raw-data path and
passing the resulting artifact to `v3_completion`. HPP / CKB / RAP remain
future ports for v3.1.

| ID | Criterion | Status | Why blocked | Closure target |
|----|-----------|--------|-------------|----------------|
| 13a | `credentialed_ukb_data` | PASS when local evidence is supplied | Requires a `bank_data_readiness` JSON with `ukb` row = `READY`; this is now obtainable from local settings-backed UKB data | Run `biobank eval --suite external_evidence --collect bank-readiness --banks ukb`, then `biobank eval --suite v3_completion --external-evidence-dir reports/eval/external_evidence` |
| 13b | `credentialed_hpp_ckb_rap_data` | BLOCKED_EXTERNAL | HPP / CKB / RAP institutional credentials are not available in CI; `bank_data_readiness` reports `SKIPPED_*` | **v3.1 milestone** — `v3.0` may ship with this BLOCKED_EXTERNAL by explicit user authorization |
| 14 | `real_mcp_compatibility_matrix` | BLOCKED_EXTERNAL | No external evidence artifact lists ≥ 2 real MCP servers with `health_checked=true` and `tool_called=true` | `v3.0-rc2`: run `biobank-agent-mcp-evidence collect --server filesystem --server github` (both STDIO) |
| 15 | `remote_ci_scheduled_eval` | BLOCKED_EXTERNAL | The remote `biobank-scheduled-eval.yml` workflow has not been dispatched + archived as JSON evidence | `v3.0-rc2`: `gh workflow run biobank-scheduled-eval.yml && gh run watch`, then `biobank eval --suite external_evidence --collect remote-ci` |
| 16 | `credentialed_high_risk_pr_path` | BLOCKED_EXTERNAL | No verified `gh pr create` evidence with `verified_by_gh=true` | `v3.0-rc2`: trigger one LOW `/evolve --apply-medium` patch on an owned repository, capture PR URL + state |

---

## v3.0 final tag entry conditions

Per the plan, either (A) or (B) must hold:

- (A) HPP / CKB / RAP credentials arrive **and** `bank_data_readiness` returns
  `READY` (non-`SKIPPED_*`) for at least one bank; criterion 13a + 13b both
  `PASS`; `v3_completion` reports `status=COMPLETE`, `n_passed=17`.
- (B) User provides written authorization to permanently defer 13b to v3.1;
  audit reports `n_passed=16`, `n_blocked_external=1` (only 13b); `v3.0`
  release notes explicitly list 13b as a v3.1 milestone.

---

## Compatibility matrix

See `docs/architecture/COMPATIBILITY.md` for the full Tier 1 / Tier 2 /
Not-supported grid across Python version, OS, Textual, openai SDK, and MCP
transport.

---

## Release gates passed

```text
pytest -q --ignore=tests/test_llm_e2e.py           # 1411 passed, 3 skipped
pytest -q benchmarks/streaming_latency.py          # TTFT p50 < 80 ms in stub
pytest -q benchmarks/skill_registration_latency.py # under threshold
biobank eval --suite behavioral --policy all       # 5/5 PASS
biobank eval --suite scheduled --policy all        # PASS
biobank-agent-live-test --run-all                  # 12/12 worker PASS (legacy markdown audit)
biobank eval --suite live_artifacts --run-dir <latest> --enforce-gate  # strict live audit PASS
biobank eval --suite v3_completion --run-dir <latest>                  # BLOCKED_EXTERNAL, 12/17 PASS, 5/17 BE, 0 FAIL
```

---

## Provenance

- Tree SHA (rc1): captured at `git tag -a v3.0-rc1 ...` time.
- Plan file: `/Users/chenpengan/.claude/plans/biobank-agent-biobank-agent-ukb-hpp-ckb-clever-robin.md`
  §0.5 v3 Closeout Roadmap.
- Round-by-round Codex consultation: §0.5 Decision Process.
