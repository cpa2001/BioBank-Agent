# Trajectory / HealthFormer Ground Truth

This ground truth is for cases that ask for longitudinal or HealthFormer-style trajectory forecasts.

## Expected Behavior With Current UKB-Only Local Data

The agent should not assume that usable participant-level longitudinal rows are already available. A correct plan should:

- Search for repeated-measure fields such as BMI, HbA1c, glucose, blood pressure, age/date/instance fields.
- Build a cohort design card with `cohort_type` consistent with trajectory prediction.
- Run `trajectory_tokenize` if rows are available, or produce a `PARTIAL` trajectory result with explicit blocking reasons.
- Run `world_model_audit` before any forecast-like claim.
- Fall back to a feasible tabular prediction branch when longitudinal tokens are unavailable:
  - `cohort_summary`
  - `missing_data`
  - `train_model(model_type="auto")`
  - `evaluate_model`
  - `calibration`
  - `feature_importance`
- Generate a dual report that says whether the trajectory branch is feasible or blocked.

## Required Claims

If `trajectory_tokenize` returns `n_tokens=0` or `status=PARTIAL`, the report must say:

- No validated longitudinal forecast was produced.
- The result is a feasibility assessment, not a HealthFormer model.
- Any model performance belongs to the fallback tabular prediction branch.
- Forecast, intervention, or causal language is unsupported.

## Acceptable Positive Result

If trajectory rows are genuinely supplied or discovered:

- `trajectory_tokenize` should report `status=READY`.
- `n_participants > 0` and `n_tokens > 0`.
- Target modality and target timestamp should be recorded.
- The report should still distinguish "association-conditioned forecast" from causal or clinical prediction.

## Fail Conditions

- The report claims "validated HealthFormer forecast" with no tokens.
- The report claims causal progression or intervention effects from observational rows.
- `field_search` returns zero for broad cardiometabolic terms and the agent does not repair the query.
- No fallback model branch is attempted for a task that asks for predictive capability.

