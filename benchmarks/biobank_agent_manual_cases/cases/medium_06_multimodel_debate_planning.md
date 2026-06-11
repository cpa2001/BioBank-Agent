# medium_06_multimodel_debate_planning

## Purpose

测试内置 multi-model planning council、debate、计划合并和查缺补漏能力。

## Main Prompt

```text
Plan a UKB-only E11 biomarker prediction and report workflow. Use the built-in multi-model debate planner to compare candidate plans, merge them, remove invalid args, add missing guardrails and model diagnostics, execute the merged plan, and generate a dual report. Make the planning-council appendix visible in the saved plan.
```

## Commands

```text
/plan Plan a UKB-only E11 biomarker prediction and report workflow. Use the built-in multi-model debate planner to compare candidate plans, merge them, remove invalid args, add missing guardrails and model diagnostics, execute the merged plan, and generate a dual report. Make the planning-council appendix visible in the saved plan.
/plan-approve
/routing-status
```

## Expected Tool Path

- Built-in multi-model planner with at least two candidate drafts.
- Merged executable plan with normal data/model/report skills.
- `generate_report(format="dual")`.

## Scoring - 100 pts

- 20: Multi-model planning council produces multiple candidate plans.
- 20: Candidate plans influence the final executable plan, not just appended as text.
- 20: Final plan remains schema-valid.
- 20: Missing diagnostics/guardrails are added before execution.
- 20: Saved plan contains a planning council appendix and a clear merged executable plan.

## Red Flags

- Candidate plans copied into invalid executable steps.
- A council slot failing causes the whole workflow to fail.
- Final report runs before model diagnostics or guardrails.
