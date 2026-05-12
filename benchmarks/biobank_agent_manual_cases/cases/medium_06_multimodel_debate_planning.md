# medium_06_multimodel_debate_planning

## Purpose

测试外部 planning council、多模型 debate/检查、计划合并和查缺补漏能力。

## Main Prompt

```text
Use external planning if available: ask Biobank Agent, Codex plan mode, and Claude Code plan mode to independently plan a UKB-only E11 biomarker prediction and report workflow. Then merge the plans, remove invalid args, add missing guardrails and model diagnostics, execute the merged plan, and generate a dual report. Make the planning-council appendix visible in the saved plan.
```

## Commands

```text
/external-agents
/codex-plan UKB-only E11 biomarker prediction and dual report workflow with schema-valid tools
/claude-plan UKB-only E11 biomarker prediction and dual report workflow with schema-valid tools
/plan Use external planning if available: ask Biobank Agent, Codex plan mode, and Claude Code plan mode to independently plan a UKB-only E11 biomarker prediction and report workflow. Then merge the plans, remove invalid args, add missing guardrails and model diagnostics, execute the merged plan, and generate a dual report. Make the planning-council appendix visible in the saved plan.
/plan-approve
/routing-status
/codex-check compare Biobank, Codex, and Claude planning coverage and identify missed steps
```

## Expected Tool Path

- `external_agent_status`
- `codex_plan` if available
- `claude_plan` if available
- merged executable plan with normal data/model/report skills
- `generate_report(format="dual")`

## Scoring - 100 pts

- 20: External agent availability is checked and unavailable agents are handled gracefully.
- 20: External plans influence the final executable plan, not just appended as text.
- 20: Final plan remains schema-valid.
- 20: Missing diagnostics/guardrails are added before execution.
- 20: Saved plan contains a planning council appendix and a clear merged executable plan.

## Red Flags

- Codex/Claude plan output copied into invalid executable steps.
- External agents unavailable causes the whole workflow to fail.
- Final report runs before model diagnostics or guardrails.

