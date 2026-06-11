# long_09_research_landscape_gap

## Purpose

测试研究现状分析：不仅跑本地数据，还要形成“这个领域现在做到了什么、UKB 能补什么 gap、下一步研究怎么设计”的报告。

## Main Prompt

```text
Create a research landscape and gap-analysis report for UKB cardiometabolic progression and Type 2 Diabetes prediction. Review current literature, identify common biomarkers and modelling approaches, cross-reference what UKB fields are available locally, run a small E11 cohort feasibility analysis, generate at least one useful figure if possible, and finish with a dual report that proposes the strongest next study design without overstating causality.
```

## Commands

```text
/plan Create a research landscape and gap-analysis report for UKB cardiometabolic progression and Type 2 Diabetes prediction. Review current literature, identify common biomarkers and modelling approaches, cross-reference what UKB fields are available locally, run a small E11 cohort feasibility analysis, generate at least one useful figure if possible, and finish with a dual report that proposes the strongest next study design without overstating causality.
/plan-approve
```

## Expected Tool Path

- `deep_research`
- `field_search`
- `cohort_summary`
- optional `smart_plot`
- optional `cohort_card`
- guardrails and dual report.

## Scoring - 100 pts

- 25: Literature review is broad enough and not one-query shallow.
- 20: UKB field availability is grounded in field search.
- 15: Local cohort feasibility is present.
- 15: Figure or table helps a reader inspect feasibility.
- 15: Report proposes a credible next study with limitations.
- 10: Clear distinction between current evidence, local analysis, and future work.

## Red Flags

- No literature sources.
- No local UKB feasibility check.
- Report is just a tool transcript rather than a landscape synthesis.

