# Examples

These examples are intentionally small and directly tied to the verified v3
foundation paths. They are meant for local smoke testing, documentation, and
new contributor onboarding.

## Files

- `manual_plan_benchmark.md` - human-style CLI runbook for plan/report testing.
- `sdk_streaming_client.py` - minimal embedded async client example.
- `mcp_http_stub_demo.py` - local JSON-RPC MCP stub plus `/mcp-*` commands.
- `cli_command_plugin_demo.py` - minimal third-party slash command module.
- `paper_replication_fixture.md` - fixture-level paper replication workflow.

## Strict Live Artifact Audit

After a full human-style live benchmark, run the stricter artifact gate against
the produced run directory:

```bash
python -m biobank_agent.cli eval \
  --suite live_artifacts \
  --run-dir reports/biobank_live_tests/full_after_scheduled_paper_gold/20260511_100909 \
  --enforce-gate
```

This verifies that the generic live runner PASS also contains the specific
trajectory and paper-replication evidence needed for v3 acceptance: dual report
artifacts, MILTON acceptance gates, positive trajectory tokenization, and
association-only world-model claim boundaries.

For the full objective-level stop/go check, run:

```bash
python -m biobank_agent.cli eval \
  --suite v3_completion \
  --run-dir reports/biobank_live_tests/full_after_scheduled_paper_gold/20260511_100909
```

This audit is expected to report `BLOCKED_EXTERNAL` on a local-only machine
until credentialed HPP/CKB/RAP readiness, real MCP compatibility, remote CI,
and HIGH-risk PR evidence are provided.

To collect external evidence after credentials are configured, use:

```bash
python -m biobank_agent.cli eval \
  --suite v3_completion \
  --run-dir reports/biobank_live_tests/full_after_scheduled_paper_gold/20260511_100909 \
  --collect-external

python -m biobank_agent.cli eval --suite external_evidence --collect bank-readiness
python -m biobank_agent.cli eval --suite external_evidence --collect mcp
python -m biobank_agent.cli eval --suite external_evidence --collect remote-ci
python -m biobank_agent.cli eval \
  --suite external_evidence \
  --collect high-pr \
  --pr-url https://github.com/<org>/<repo>/pull/<number> \
  --branch auto-improve/<skill>-<hash>
```

The completion audit auto-discovers `reports/eval/external_evidence/*_latest.json`,
so explicit evidence flags are only needed when reviewing older artifacts.

## Expected Environment

Run examples from the repository root after installing the package:

```bash
pip install -e ".[all,dev]"
```

The examples assume local UKB-style de-identified data are already configured
through `.env` or the usual `Settings` object.
