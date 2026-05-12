# long_10_adaptive_repair_custom_skill

## Purpose

测试 agent 是否会在执行阶段修自己，而不是被动失败退出。这个 case 故意要求一个跨步骤 summary，如果现有 skill 不够，应创建低风险 custom skill 或重组计划。

## Main Prompt

```text
Run an adaptive UKB-only E11 and obesity cardiometabolic analysis. I want a reusable final "analysis readiness card" that summarizes fields found, cohort counts, missingness, model readiness, guardrail status, and report paths. If no existing skill can create this readiness card cleanly, create a small custom skill in custom_skills, hot-load it, use it in the plan, and then generate a dual report. If a step fails because data or arguments are missing, repair the plan by adding prerequisite discovery/cohort/model steps rather than stopping.
```

## Commands

```text
/plan Run an adaptive UKB-only E11 and obesity cardiometabolic analysis. I want a reusable final "analysis readiness card" that summarizes fields found, cohort counts, missingness, model readiness, guardrail status, and report paths. If no existing skill can create this readiness card cleanly, create a small custom skill in custom_skills, hot-load it, use it in the plan, and then generate a dual report. If a step fails because data or arguments are missing, repair the plan by adding prerequisite discovery/cohort/model steps rather than stopping.
/plan-approve
/status
/codex-check repair log, custom skill quality, hot-load evidence, and report quality
```

## Expected Tool Path

- Data discovery and cohort/model/report skills as relevant.
- `create_skill` or equivalent custom skill path if the planner decides it is needed.
- Repair actions visible in plan/replay/logs if failures occur.
- Dual report.

## Scoring - 100 pts

- 20: Agent decides whether a custom skill is actually needed; it does not create one gratuitously.
- 20: If created, custom skill is small, safe, schema-valid, hot-loaded, and used.
- 20: Failed prerequisites are repaired by adding concrete steps.
- 20: Readiness card contains fields/cohort/missingness/model/guardrail/report status.
- 20: Final report explains the repair or skill-extension path without leaking raw code chatter into human-facing sections.

## Red Flags

- Stops after the first error with no repair attempt.
- Creates broad unsafe code that accesses arbitrary files or secrets.
- Custom skill is created but never used.
- Report is missing or dominated by implementation logs.

