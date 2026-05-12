# grand_challenge_99_general_agent_showcase

## Purpose

这是最通用、最能展示 Biobank Agent 综合能力的 benchmark。目标是从一个很泛的问题出发，让 agent 自己调动尽量多的 tool/plugin/skills，形成一个复杂但可执行的研究 workflow，并输出能直观看出性能的技术报告和 Nature-style report。

## Main Prompt

```text
I only have a broad research question: can UKB support a compelling study of metabolic health trajectories, Type 2 Diabetes risk prediction, and potentially actionable cardiometabolic biomarkers?
```

This case intentionally uses only the broad question a human researcher would normally type. The benchmark is testing whether Biobank Agent can infer the scientific workflow, not whether the user can paste a prewritten plan.

## Recommended Commands

```text
/plan I only have a broad research question: can UKB support a compelling study of metabolic health trajectories, Type 2 Diabetes risk prediction, and potentially actionable cardiometabolic biomarkers?
/plan-approve
```

Notes:

- `/plan` should show live phases: clarification, three-step Research Setup, Biobank plan, Codex/Claude/Gemini planning council when selected and available, merge, validation, and review.
- Research Setup should ask one question per step: active data source, external planning council, then analysis strategy. It should confirm UKB as the active dataset, show HPP/CKB/RAP as future ports, and let the user accept or select available external planners.
- The user should not need to type the detailed pipeline. The agent should infer documentation/data inventory inspection, field discovery, literature review, E11 cohort design, trajectory feasibility, model training/evaluation, guardrails, and dual-report generation.
- If the agent asks clarification questions, answer them naturally or accept the recommended default.
- If execution stalls, use the displayed choices (`/plan-option A`, `/plan-option B`, `/plan-resume`, `/plan-exit`) instead of guessing.
- After completion, choose review hook `A` if prompted to run the full Codex + Claude + Gemini review council. `/status`, `/plans`, `/routing-status`, `/codex-check`, `/claude-check`, and `/gemini-check` remain optional diagnostics.

## Expected High-Coverage Tool Path

The exact plan can vary, but a strong run should include most of:

- `project_doc`
- `ukb_data_inventory`
- `ukb_field_resolve`
- optional `ukb_materialize_fields` for selected raw-only fields when it genuinely improves the analysis and will not block the demo.
- `external_agent_status`, `codex_plan`, `claude_plan`, `gemini_plan` when available
- `deep_research`
- `field_search`
- `cohort_summary`
- `cohort_card`
- `trajectory_tokenize`
- `missing_data`
- `train_model(model_type="auto")`
- `evaluate_model`
- `calibration`
- `feature_importance`
- `smart_plot`
- `statistical_review`
- `safety_check`
- `world_model_audit`
- `generate_report(format="dual")`
- optional `create_skill` only if it creates a genuinely reusable helper.

## Showcase Scoring - 100 pts

- 10: Starts from the short human prompt and opens a Research Setup confirmation instead of requiring a pasted plan.
- 10: Starts with project/schema awareness and external planning when selected and available.
- 10: Produces a merged plan that is long-horizon but executable.
- 10: Literature review and UKB field discovery are both meaningful.
- 10: Cohort design is explicit and auditable.
- 10: Trajectory feasibility is tested with correct limitations.
- 10: Auto model training includes candidate comparison, evaluation, calibration, and feature importance.
- 10: Figures/tables make the results inspectable.
- 10: Self-repair behavior is visible when a step is blocked or partial.
- 10: Guardrails run before final conclusions and final reports are polished enough to demo to another researcher.

## What "Impressive" Looks Like

- The user typed only the broad research question; the saved plan still contains the full scientific trajectory.
- Research Setup is split into data scope, external planning council, and analysis strategy steps; each step asks one question and supports Enter for the default.
- The saved plan has a planning council appendix and a merged executable plan.
- The technical report states the UKB data inventory: 502K-scale subject coverage, Milton parquet availability, full raw CSV availability, and which important fields were materialized or deferred.
- The report opens with clear executive findings, not a raw log.
- The Nature-style report reads like a cautious scientific manuscript.
- The technical report includes model selection, calibration, feature importance, trajectory feasibility, guardrail status, and execution appendix.
- The report explicitly says what the agent could not do and why, without making the output feel like a failure.

## Hard Fail Conditions

- Any false success.
- Passing only when the user provides a long prewritten pipeline prompt.
- Missing dual report artifacts.
- Unsupported claim of validated longitudinal forecast or causal effect.
- Unsupported claim that the current prevalent/ever-diagnosed E11 classifier is a prospective incident T2D risk model.
- No model branch.
- No trajectory branch.
- No guardrail branch.
- Human-facing report contains tool chatter, author/institution placeholders, or AI branding.
