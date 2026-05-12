# short_03_deep_research_brief

## Purpose

测试研究现状检索、文献摘要、UKB 字段交叉引用和报告摘要能力。

## Main Prompt

```text
Do a short deep research brief on UK Biobank Type 2 Diabetes biomarker prediction using BMI, HbA1c, glucose, blood pressure, lipids, and routine clinical biomarkers. Cross-reference available UKB fields, state what evidence is from literature versus local data, and generate a concise dual report.
```

## Commands

```text
/plan Do a short deep research brief on UK Biobank Type 2 Diabetes biomarker prediction using BMI, HbA1c, glucose, blood pressure, lipids, and routine clinical biomarkers. Cross-reference available UKB fields, state what evidence is from literature versus local data, and generate a concise dual report.
/plan-approve
/codex-check literature coverage, UKB field relevance, and report quality
```

## Expected Tool Path

- `deep_research(topic, max_sources)`
- `field_search`
- optional `cohort_summary`
- `statistical_review`
- `safety_check`
- `world_model_audit`
- `generate_report(format="dual")`

## Scoring - 100 pts

- 25: Deep research uses several search attempts and returns nontrivial sources or clear offline limitation.
- 20: UKB field cross-reference includes BMI/HbA1c/glucose/blood pressure/lipids when available.
- 15: Distinguishes literature evidence from local UKB analysis.
- 20: Report has references or clear source notes, plus limitations.
- 20: No fabricated DOI/source claims and no unsupported causal claims.

## Red Flags

- Only one shallow web query for a broad research-status task.
- No field discovery.
- Report treats literature statements as local UKB results.

