# medium_04_auto_model_training

## Purpose

测试真正的模型训练能力：自动模型选择、评估、校准、特征重要性、模型选择理由和报告可解释性。

## Main Prompt

```text
Train the best feasible UKB-only predictive model for E11 Type 2 Diabetes using routine biomarkers. Start with field discovery and cohort_summary, assess missingness, use train_model with model_type="auto", compare reasonable candidate models, run evaluate_model, calibration, feature_importance, and smart_plot if useful. If performance is weak, explain what you tried and what should be improved. Finish with a technical plus Nature-style dual report.
```

## Commands

```text
/plan Train the best feasible UKB-only predictive model for E11 Type 2 Diabetes using routine biomarkers. Start with field discovery and cohort_summary, assess missingness, use train_model with model_type="auto", compare reasonable candidate models, run evaluate_model, calibration, feature_importance, and smart_plot if useful. If performance is weak, explain what you tried and what should be improved. Finish with a technical plus Nature-style dual report.
/plan-approve
/status
```

## Expected Tool Path

- `field_search`
- `cohort_summary`
- `missing_data`
- `train_model(icd10_code="E11", model_type="auto")`
- `evaluate_model`
- `calibration`
- `feature_importance`
- optional `smart_plot`
- `statistical_review`
- `safety_check`
- `world_model_audit`
- `generate_report(format="dual")`

## Scoring - 100 pts

- 15: Cohort count checked before training.
- 15: Missingness assessed before training.
- 20: Auto model selection compares multiple candidates and records rationale.
- 15: Evaluation and calibration outputs are present.
- 15: Feature importance is present and interpreted cautiously.
- 20: Report includes Model Selection, limitations, leakage/calibration caveats, and no deployment claim.

## Red Flags

- `train_model` uses a single hard-coded model when prompt asks for best feasible model.
- No calibration.
- AUC is reported without sample size or confidence/uncertainty.
- Report claims clinical readiness.

