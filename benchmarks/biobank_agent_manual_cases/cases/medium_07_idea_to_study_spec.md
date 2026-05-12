# medium_07_idea_to_study_spec

## Purpose

测试从模糊 idea 解析成可执行研究设计的能力：endpoint、cohort、field discovery、design rationale、guardrails 和报告。

## Main Prompt

```text
I have a vague research idea: maybe metabolic health, obesity, blood biomarkers, and diabetes progression in UKB are connected. Turn this into the safest feasible UKB-only study. You choose the endpoint, cohort design, fields, analysis path, and report structure. Push back on anything that cannot be supported by available data. Finish with a dual report that explains the chosen study design.
```

## Commands

```text
/plan I have a vague research idea: maybe metabolic health, obesity, blood biomarkers, and diabetes progression in UKB are connected. Turn this into the safest feasible UKB-only study. You choose the endpoint, cohort design, fields, analysis path, and report structure. Push back on anything that cannot be supported by available data. Finish with a dual report that explains the chosen study design.
/plan-approve
/codex-check idea interpretation, endpoint choice, scientific validity, and report clarity
```

## Expected Tool Path

- `project_doc` or memory/docs lookup when useful.
- `deep_research` or field/literature context.
- `field_search`
- `cohort_card`
- `cohort_summary`
- optional model branch if the agent chooses prediction.
- guardrails and dual report.

## Scoring - 100 pts

- 20: Converts vague idea into explicit endpoint and study design.
- 20: Justifies why E11 or another endpoint was chosen.
- 15: Searches fields rather than assuming data availability.
- 15: Includes safety/statistical/world-model constraints.
- 20: Report is readable to a researcher and separates feasible analysis from future work.
- 10: Does not overcomplicate the plan with irrelevant tools.

## Red Flags

- Agent asks the user for many details that it could reasonably choose.
- Agent claims causality or progression without temporal support.
- No rationale for endpoint or analysis path.

