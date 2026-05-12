# short_02_e11_dual_report_smoke

## Purpose

最短报告冒烟测试：从字段搜索和 E11 cohort summary 到 guardrails，再生成完整 dual report。这个 case 用来快速判断是否还有“显示成功但无 report”的问题。

## Main Prompt

```text
Run a concise UKB-only E11 Type 2 Diabetes workflow. Search available fields first, summarize cohort counts, run statistical_review, safety_check, and world_model_audit, then generate_report with format="dual". Keep it short but produce final report paths.
```

## Commands

```text
/plan Run a concise UKB-only E11 Type 2 Diabetes workflow. Search available fields first, summarize cohort counts, run statistical_review, safety_check, and world_model_audit, then generate_report with format="dual". Keep it short but produce final report paths.
/plan-approve
/status
/plans
```

## Expected Tool Path

- `field_search`
- `cohort_summary(icd10_code="E11")`
- `statistical_review`
- `safety_check`
- `world_model_audit`
- `generate_report(format="dual")`

## Expected Artifacts

The latest report directory must contain all files listed in `../scoring/REPORT_CHECKLIST.md`.

## Scoring - 100 pts

- 20: Plan is schema-valid before approval.
- 20: Cohort summary includes E11 case/control counts.
- 20: Guardrails run before report.
- 25: All dual report artifacts exist and are non-empty.
- 15: Report has clean executive summary, Nature abstract, limitations, and no AI/tool branding.

## Red Flags

- Plan DONE but no `report_technical.md` or `report_nature.md`.
- `generate_report` receives invalid args.
- Report lacks limitations or implies causal proof.

