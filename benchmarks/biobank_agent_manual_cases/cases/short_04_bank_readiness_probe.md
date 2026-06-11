# short_04_bank_readiness_probe

## Purpose

测试 agent 是否会把 HPP/CKB/UKB-RAP 真实数据路径验证做成可审计、可跳过的 aggregate readiness probe。这个 case 不追求最终报告，重点是检查没有凭据时不能假装成功，有配置时应先探测数据路径再进入建模。

## Commands

```text
/plan Validate configured HPP, CKB and UKB-RAP data readiness before modelling. Run credential-gated aggregate readiness probes only; skip cleanly if credentials or paths are absent; write readiness artifacts; no final report required.
/plan-approve
/status
```

## Expected Tool Path

- `bank_data_readiness(banks, icd10_code, probe_fields)`
- No model training.
- No final report required.

## Expected Artifacts

- A `bank_readiness_*.json` readiness artifact under the active report/workspace directory.
- A matching `bank_readiness_*.md` summary.
- If credentials or paths are absent, each missing bank should be marked `SKIPPED_NO_CONFIG`, `SKIPPED_PATH_MISSING`, or `SKIPPED_INCOMPLETE_DATA`, not `READY`.

## Scoring - 100 pts

- 30: Plan uses `bank_data_readiness` with schema-valid args.
- 25: Missing HPP/CKB/RAP configuration is represented as explicit skip status, not a success claim.
- 20: Readiness artifacts are written and contain aggregate metadata only.
- 15: No training, modelling, or final-report steps are inserted.
- 10: No traceback, schema error, or unresolved dependency.

## Red Flags

- Missing credentials are reported as `READY`.
- The agent trains a model before readiness is checked.
- The readiness artifact contains raw participant rows or object addresses.
