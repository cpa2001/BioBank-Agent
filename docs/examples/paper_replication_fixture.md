# Paper Replication Fixture Workflow

This is a fixture-level workflow for testing the current paper replication
scaffold. It is not yet a claim of fully automated reproduction of a published
paper.

## REPL Input

```text
/replicate A UK Biobank study reports an E11 prediction model using baseline age, sex, BMI, HbA1c, glucose, blood pressure and lipid markers. The paper reports ROC AUC, calibration and feature importance figures. Build a guarded UKB-only replication plan that checks cohort definition, missingness, model performance, calibration and report limitations.
```

Expected behavior:

- The command writes a review-only StudySpec artifact under `reports/`.
- The proposed replication plan is loaded into Plan Review.
- The plan requires user review before execution; use `/plan-approve` only
  after checking the steps and arguments.
- The final report includes a replication comparison section if
  `paper_replication_compare` is executed after model and plot generation.

## Current Passing Standard

For v3 foundation, the scaffold passes when generated comparison artifacts link
local ROC/calibration/feature-importance figures into the report and label any
unmatched paper targets as limitations.

## Not Yet Complete

Full v3.0 paper reproduction still requires one fixed published UKB paper to run
through:

1. paper extraction,
2. StudySpec validation,
3. plan review,
4. execution,
5. numeric table diff,
6. figure diff,
7. final report with explicit reproducibility verdict.
