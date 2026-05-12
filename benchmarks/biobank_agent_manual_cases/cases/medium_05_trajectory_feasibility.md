# medium_05_trajectory_feasibility

## Purpose

测试 HealthFormer-style longitudinal trajectory 任务的正确边界：能不能发现数据限制、做 feasibility audit、必要时 fallback 到 tabular prediction，而不是假装完成 forecast。

## Main Prompt

```text
Build a longitudinal HealthFormer-style trajectory forecast over time for UKB diabetes progression. Use UKB only, choose feasible endpoint and fields, search repeated BMI/HbA1c/glucose/blood pressure measurements, tokenize trajectory data if available, audit whether any forecast claim is supported, fall back to a tabular prediction model if trajectory rows are insufficient, and finish with a technical plus Nature-style dual report.
```

## Commands

```text
/plan Build a longitudinal HealthFormer-style trajectory forecast over time for UKB diabetes progression. Use UKB only, choose feasible endpoint and fields, search repeated BMI/HbA1c/glucose/blood pressure measurements, tokenize trajectory data if available, audit whether any forecast claim is supported, fall back to a tabular prediction model if trajectory rows are insufficient, and finish with a technical plus Nature-style dual report.
/plan-approve
/codex-check trajectory feasibility, fallback model branch, unsupported forecast claims, and report quality
```

## Ground Truth

Use `../ground_truth/TRAJECTORY_GROUND_TRUTH.md`.

## Expected Tool Path

- `deep_research`
- `field_search`
- `cohort_summary`
- `cohort_card`
- `trajectory_tokenize`
- fallback `missing_data -> train_model(model_type="auto") -> evaluate_model -> calibration -> feature_importance`
- `world_model_audit`
- `statistical_review`
- `safety_check`
- `generate_report(format="dual")`

## Scoring - 100 pts

- 20: Searches for repeated longitudinal fields and does not accept zero field results passively.
- 20: Runs trajectory-specific design/tokenization/audit steps.
- 20: Correctly says whether trajectory forecast is feasible, partial, or blocked.
- 20: Runs a fallback model branch if no usable trajectory tokens exist.
- 20: Report clearly separates trajectory feasibility from model performance and avoids unsupported forecast/causal claims.

## Red Flags

- Report claims a HealthFormer model was trained without trajectory tokens.
- No `world_model_audit`.
- No fallback model despite predictive prompt.

