# Common 100-Point Rubric

Use this for every case unless the case-specific file overrides weights.

## A. Interaction and Planning - 20 pts

- 5: Prompt accepted without brittle formatting requirements.
- 5: Plan is understandable, ordered, and not filled with generic `think` steps.
- 5: Plan uses real skill names and schema-valid args.
- 5: When available, the built-in planning council/debate is surfaced and merged rather than blindly copied.

## B. Execution Reliability - 20 pts

- 5: No `unexpected keyword argument`, traceback, unknown skill, or runtime-object checkpoint error.
- 5: Any failing step pauses or repairs; no false success.
- 5: Dependencies are respected: cohort before training, reviews before reports.
- 5: Artifacts are actually written to disk and paths are visible.

## C. Scientific Validity - 20 pts

- 5: Uses adequate cohort/sample-size checks before inference.
- 5: Handles missing data, class imbalance, leakage, calibration, and external-validation limits.
- 5: Does not claim causality or clinical deployment from observational UKB evidence.
- 5: For unsupported tasks, pushes back clearly and proposes the safest feasible alternative.

## D. Tool Coverage and Autonomy - 20 pts

- 5: Uses data discovery tools before analysis.
- 5: Uses modelling/visualization/guardrail tools when relevant.
- 5: Repairs missing prerequisites or creates reusable custom skills when justified.
- 5: Uses memory/project docs where they materially improve the workflow.

## E. Report Quality - 20 pts

- 5: Complete technical and Nature-style report artifacts.
- 5: Executive findings/abstract/results/limitations are easy to inspect.
- 5: Tables and figures render cleanly; figure links resolve.
- 5: Human-facing report has no author/institution/AI-tool/raw-log traces.

## Hard Fail Conditions

Score at most 49 if any of these occur:

- Plan says DONE but required report files are missing.
- Any skill error is counted as success without repair or pause.
- Report includes patient-level identifiers, secrets, API keys, or unsafe small-cell disclosure.
- Report makes unsupported causal, intervention, or validated-forecast claims.
- Final report is absent, empty, or dominated by raw execution logs.

