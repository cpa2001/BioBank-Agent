# V3 Remaining Implementation Plan

This document tracks the gap between the original Claude v3 roadmap and the
current verified foundation. It is intentionally evidence-based: a line should
only move to "done" after code is wired into the agent path and a real test or
live benchmark covers it.

## Current Verified Foundation

- Unit/regression suite: `1400 passed, 12 skipped` after adding the behavioral
  CLI/history runner and scheduled self-evolution pattern-mining artifacts.
  The earlier soft usually-pass behavioral gate and BankAdapter data-path probe
  slice continue to pass alongside the M3 domain tests.
- `bank_data_readiness` is agent-callable and live-tested. It writes aggregate
  JSON/Markdown readiness artifacts, uses configured HPP/CKB/RAP paths when
  available, and emits explicit `SKIPPED_*` statuses when credentials or
  datasets are absent.
- Full-data retry policy is enforced: retry/reflexion no longer mutates
  `sample_size` or `n_samples`, so recovery logic fixes compute knobs or asks
  for explicit operator sampling instead of silently discarding eligible rows.
- Performance checks: `benchmarks/streaming_latency.py` and
  `benchmarks/skill_registration_latency.py` pass; latest TTFT p50 was 52.4 ms.
- Human-style live benchmark:
  `reports/biobank_live_tests/full_after_scheduled_paper_gold/20260511_100909/LIVE_TEST_AUDIT.md`
  reports `PASS` with 12/12 workers passing.
- Trajectory behavior is covered by worker 06 in that run. The generated report
  contains `trajectory_tokenize`, 3,536,009 tokens across 501,611 participants,
  a tabular fallback model, `statistical_review`, `safety_check`, and
  `world_model_audit` limiting claims to `association_conditioned_forecast`.
  The audit now infers `available_tokens` from the latest session
  `trajectory_tokenize` result instead of reporting a misleading zero while
  keeping `training_distribution_coverage=0` until a real world-model
  validation exists.
- Paper replication behavior is covered by worker 02 in that run. The generated
  report renders `paper_replication_compare` acceptance gates with
  `PASS_WITH_LIMITATIONS`, including paper access, cohort count, AUC,
  calibration ECE, feature importance, and figure artifact checks.
- A strict post-run live artifact audit now passes on that run through
  `biobank eval --suite live_artifacts --run-dir reports/biobank_live_tests/full_after_scheduled_paper_gold/20260511_100909 --enforce-gate`.
  It verifies dual-report artifacts, the MILTON plan dependency graph and
  acceptance gates, and the trajectory worker's positive token count plus
  `association_conditioned_forecast` world-model boundary.
- A full objective completion audit now runs through
  `biobank eval --suite v3_completion --run-dir reports/biobank_live_tests/full_after_scheduled_paper_gold/20260511_100909`.
  Latest status is `BLOCKED_EXTERNAL`: 12 local/live criteria pass, 4 external
  credentialed criteria remain blocked, and 0 criteria fail.
- The MILTON paper fixture is now frozen at
  `docs/related_works/s41588-024-01898-1.pdf`; the hard behavioral suite checks
  local DOI cache lookup, PDF text extraction, planner inclusion of paper
  replication/model/guardrail/report steps, comparison prerequisites, and
  `generate_report(format="dual")`.
- Behavioral eval now has a project CLI entry:
  `biobank eval --suite behavioral --policy all|always|usually`, plus
  `python -m biobank_agent.eval.behavioral`. Both write `latest.json` and
  `behavioral_history.jsonl` under `reports/eval/behavioral`.
- Self-evolution now writes scheduled pattern-mining artifacts under
  `reports/eval/evolution` via `/evolve --write-history` or
  `/evolve --scheduled-run`, and MEDIUM patch confirmations include risk,
  eval summary, diff preview, and JSONL audit rows under
  `reports/generated_skills/evolution_decisions.jsonl`.
- Scheduled quality gates now compose behavioral eval and evolution pattern
  mining through `python -m biobank_agent.eval.scheduled` and
  `biobank eval --suite scheduled --policy all|always|usually`, writing a
  manifest under `reports/eval/scheduled` with recent run counts and
  behavioral pass-rate trend summaries. A weekly GitHub Actions wrapper exists
  at `.github/workflows/biobank-scheduled-eval.yml`.

## Remaining Gaps

| Area | Current state | Missing for full v3.0 | Next implementation slice |
|---|---|---|---|
| CLI/TUI productization | Textual scaffold, shared command registry, long-run dashboard, run-detail panel, worker/review queue panel, repair choices, report artifacts, confirmation callback bridge, and F5-F9 keyboard actions for approve/resume/repair choices | Production polish around live concurrent worker browsing, visual stress testing, and deeper mouse/keyboard affordances | Add screenshot-backed Textual QA and richer navigation for large worker sets and long repair histories |
| BankAdapter production data path | UKB/HPP/CKB/RAP adapters, synthetic fixture wiring, `bank_data_probe`, and `bank_data_readiness` exist. HPP/CKB fixture tests now run DataManager -> diagnosis filter -> cohort -> full-data train_model. The readiness skill records aggregate artifacts and cleanly skips absent HPP/CKB/RAP credentials. Trajectory default plans call `bank_data_probe` before cohort construction. | Real HPP/CKB/RAP data validation with private credentialed datasets, credentialed RAP read-only query execution, and a documented operator flow for those credentials | Run the existing `bank_data_readiness` artifact flow against real configured HPP/CKB/RAP datasets and add credentialed acceptance evidence |
| MCP production matrix | STDIO and HTTP/SSE-style JSON-RPC discovery/call/health/reconnect are tested | Long-lived SSE stream lifecycle, real external server compatibility, packaging | Add compatibility fixtures for GitHub/Postgres-style MCP servers and stream reconnect tests |
| Paper reproduction e2e | Review-to-execute chain, numeric/figure acceptance gates, frozen MILTON behavioral gold route, live worker 02 execution, and strict `live_artifacts` audit are implemented | Broader paper fixture set and true numeric/pixel diff where source tables/figures are extractable | Keep MILTON strict audit green; add a second published-paper fixture with extractable source tables/figures |
| Self-evolution closure | LOW allow-listed auto-merge, MEDIUM confirmation with diff/eval preview and decision audit, scheduled pattern-mining artifacts, scheduled quality-gate manifest with trend summary, HIGH local branch fallback | Screenshot-backed QA for long MEDIUM diff review, verified remote CI run history, credentialed `gh pr create` path | Add long-diff TUI stress tests; credential HIGH PR publishing only after credentials are configured |
| Behavioral eval CI | ALWAYS_PASSES and USUALLY_PASSES suites exist; `biobank eval --suite behavioral --policy ...` writes per-run JSON and historical JSONL; `biobank eval --suite scheduled` and weekly GitHub Actions wrapper compose behavioral/evolution gates with trend summaries | Verified remote CI run history | After the workflow runs in GitHub, archive its artifact URL as acceptance evidence |
| Release engineering | v3 foundation docs, examples, and verification contract exist | v3.0 release notes, SDK compatibility matrix, concrete collector packaging examples | Add release checklist, SDK matrix, and Jaeger/Tempo/vendor collector examples after the above implementation slices pass |

## Step-by-Step Order

1. Keep `pytest`, streaming/registration benchmarks, and
   `$biobank-agent-live-test --run-all` green after every slice.
2. Finish TUI productization first because it improves observability for all
   later long-running work.
3. Harden data paths next: fixture parity and credential-gated readiness are
   done, so the next slice is running the same readiness flow against real
   HPP/CKB/RAP credentials and storing acceptance evidence.
4. Keep the frozen paper reproduction strict live benchmark green, then expand
   to additional published-paper fixtures and exact diff checks where the
   source exposes comparable tables/figures.
5. Finish long-diff QA and remote CI evidence capture for self-evolution only
   after the benchmark suite can detect bad generated patches.
6. Add production packaging/release docs last, once behavior is no longer moving.

## External Completion Evidence

The local implementation is not allowed to mark v3 complete until the four
credentialed checks below are supplied to `biobank eval --suite v3_completion`.
These are intentionally external because synthetic fixtures cannot prove real
bank access, hosted MCP compatibility, remote CI execution, or GitHub PR
creation.

For a single command that first refreshes all external evidence and then audits
the full objective, run:

```bash
python -m biobank_agent.cli eval \
  --suite v3_completion \
  --run-dir reports/biobank_live_tests/full_after_scheduled_paper_gold/20260511_100909 \
  --collect-external \
  --mcp-config ~/.biobank_agent/mcp_servers.json \
  --pr-url https://github.com/<org>/<repo>/pull/<number> \
  --branch auto-improve/<skill>-<hash>
```

Use the external-evidence collector where possible instead of hand-writing
these files:

```bash
python -m biobank_agent.cli eval \
  --suite external_evidence \
  --collect bank-readiness \
  --banks hpp,ckb,ukb_rap \
  --icd10-code E11 \
  --probe-fields hba1c,bmi,glucose

python -m biobank_agent.cli eval \
  --suite external_evidence \
  --collect mcp \
  --mcp-config ~/.biobank_agent/mcp_servers.json \
  --mcp-call-args reports/<run>/mcp_call_args.json

python -m biobank_agent.cli eval \
  --suite external_evidence \
  --collect remote-ci \
  --workflow biobank-scheduled-eval.yml

python -m biobank_agent.cli eval \
  --suite external_evidence \
  --collect high-pr \
  --pr-url https://github.com/<org>/<repo>/pull/<number> \
  --branch auto-improve/<skill>-<hash>
```

Each collector also writes a `*_latest.json` alias under
`reports/eval/external_evidence`. After collecting all four evidence types, the
completion audit can auto-discover them:

```bash
python -m biobank_agent.cli eval \
  --suite v3_completion \
  --run-dir reports/biobank_live_tests/full_after_scheduled_paper_gold/20260511_100909
```

Use explicit evidence path flags only when you need to override the latest
collector outputs.

### 1. HPP / CKB / RAP readiness

Run `bank_data_readiness` against real configured HPP, CKB, and UKB-RAP data.
The accepted evidence is a JSON artifact with all required banks ready:

```json
{
  "artifact_type": "bank_data_readiness",
  "schema_version": 1,
  "generated_by": "bank_data_readiness",
  "banks": [
    {
      "bank_id": "hpp",
      "status": "READY",
      "n_subjects": 12345,
      "source_env": "BIOBANK_HPP_DATA_DIR",
      "diagnosis_probe": {"n_case_subjects": 123}
    },
    {
      "bank_id": "ckb",
      "status": "READY",
      "n_subjects": 12345,
      "source_env": "BIOBANK_CKB_DATA_DIR",
      "diagnosis_probe": {"n_case_subjects": 123}
    },
    {
      "bank_id": "ukb_rap",
      "status": "REMOTE_READY",
      "credential_envs_present": ["DX_PROJECT_CONTEXT_ID"]
    }
  ]
}
```

Then pass it to the completion audit:

```bash
python -m biobank_agent.cli eval \
  --suite v3_completion \
  --run-dir reports/biobank_live_tests/full_after_scheduled_paper_gold/20260511_100909 \
  --hpp-ckb-rap-readiness reports/<run>/bank_readiness_<stamp>.json
```

### 2. Real MCP compatibility

Start at least two real MCP servers, for example GitHub plus Postgres or another
project-local server, call one tool from each, and write a compatibility JSON:

```json
{
  "artifact_type": "mcp_compatibility_matrix",
  "servers": [
    {
      "name": "github",
      "transport": "stdio",
      "status": "PASS",
      "health_checked": true,
      "tool_called": true
    },
    {
      "name": "postgres",
      "transport": "http_sse",
      "status": "READY",
      "health_checked": true,
      "tool_called": true
    }
  ]
}
```

Then include it with `--mcp-compat-evidence`.

### 3. Remote scheduled CI

After `.github/workflows/biobank-scheduled-eval.yml` runs on GitHub Actions,
record the successful remote artifact URL:

```json
{
  "artifact_type": "remote_ci_scheduled_eval",
  "status": "SUCCESS",
  "url": "https://github.com/<org>/<repo>/actions/runs/<run-id>",
  "run_id": "<run-id>"
}
```

Then include it with `--remote-ci-evidence`.

### 4. HIGH-risk self-evolution PR path

Configure GitHub credentials and run a HIGH-risk self-evolution patch path until
it creates a remote PR instead of only a local fallback branch. The completion
audit accepts only a JSON artifact verified by `gh pr view`, with this shape:

```json
{
  "artifact_type": "high_risk_pr_evidence",
  "status": "PR_OPENED",
  "pr_url": "https://github.com/<org>/<repo>/pull/<number>",
  "branch": "auto-improve/<skill>-<hash>",
  "verified_by_gh": true
}
```

Do not hand-write this artifact. The completion gate also checks that it was
generated by `external_evidence.write_high_pr_evidence` and contains matching
`gh.url`, `gh.headRefName`, and `gh.state` fields from `gh pr view`.

```bash
python -m biobank_agent.cli eval \
  --suite v3_completion \
  --run-dir reports/biobank_live_tests/full_after_scheduled_paper_gold/20260511_100909 \
  --hpp-ckb-rap-readiness reports/<run>/bank_readiness_<stamp>.json \
  --mcp-compat-evidence reports/<run>/mcp_compat_<stamp>.json \
  --remote-ci-evidence reports/<run>/remote_ci_<stamp>.json \
  --high-pr-evidence reports/<run>/high_pr_<stamp>.json
```

Only when this command reports `COMPLETE` should the v3 roadmap be treated as
fully implemented.

## Behavioral Eval Commands

Run the hard gate:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q tests/eval/behavioral/test_behavioral_always.py --capture=no -p no:cacheprovider
```

Run the soft weekly gate:

```bash
BIOBANK_RUN_USUALLY_EVALS=1 PYTHONDONTWRITEBYTECODE=1 pytest -q tests/eval/behavioral/test_behavioral_always.py::test_behavioral_usually_passes_threshold --capture=no -p no:cacheprovider
```

Run the behavioral CLI/history gate:

```bash
python -m biobank_agent.cli eval --suite behavioral --policy all
```

Run the scheduled manifest gate:

```bash
python -m biobank_agent.eval.scheduled --behavioral-policy all
python -m biobank_agent.cli eval --suite scheduled --policy all
```

Run the strict live artifact audit after a human-style live benchmark:

```bash
python -m biobank_agent.cli eval \
  --suite live_artifacts \
  --run-dir reports/biobank_live_tests/full_after_scheduled_paper_gold/20260511_100909 \
  --enforce-gate
```

Run the full v3 objective completion audit:

```bash
python -m biobank_agent.cli eval \
  --suite v3_completion \
  --run-dir reports/biobank_live_tests/full_after_scheduled_paper_gold/20260511_100909
```

It should remain `BLOCKED_EXTERNAL` until credentialed HPP/CKB/RAP readiness,
real MCP compatibility evidence, remote scheduled CI evidence, and a HIGH-risk
GitHub PR evidence URL are supplied.

Run scheduled self-evolution pattern mining inside an interactive or scripted
agent session:

```text
/evolve --scheduled-run
```

Run the BankAdapter data-path gate:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q tests/core/test_bank_data_path.py tests/core/test_m3_domain.py --capture=no -p no:cacheprovider
```

Run the readiness live case using the local benchmark runner:

```bash
python scripts/run_live_test.py \
  --workers 1 --timeout-min 20 \
  --case-dir benchmarks/biobank_agent_manual_cases/cases/short_04_bank_readiness_probe.md \
  --output-root reports/biobank_live_tests/bank_readiness_probe \
  --review-hook skip
```
