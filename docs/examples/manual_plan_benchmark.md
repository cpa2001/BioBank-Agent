# Manual Plan Benchmark

Use this when you want to test the Biobank Agent as a human operator rather
than through pytest.

## Start

```bash
cd /path/to/biobank-agent
biobank
```

## Input

Paste this into the REPL:

```text
/plan I only have a broad research question: can UKB support a compelling study of metabolic health trajectories, Type 2 Diabetes risk prediction, and actionable cardiometabolic biomarkers? Act as an autonomous biobank research agent. Inspect available skills and project documentation. Discover relevant fields; review related literature; define an auditable E11-centered cohort; check case counts and missingness; search repeated BMI, HbA1c, glucose, blood pressure, lipid and diagnosis fields; attempt a HealthFormer-style trajectory feasibility branch; train the best feasible tabular prediction model with model_type="auto"; evaluate discrimination, calibration and feature importance; create useful figures; run statistical_review, safety_check and world_model_audit before conclusions; and produce final technical plus Nature-style reports with exact output paths. If some part is impossible, repair the plan, fall back to the safest valid analysis, and clearly distinguish achieved results, feasibility gaps, and next-step recommendations.
```

Then review the generated plan. If it is schema-valid and scientifically
reasonable:

```text
/plan-approve
```

If it pauses, use the displayed repair choices. Prefer auto-repair for
mechanical dependency issues and external review for ambiguous scientific
failures.

## Passing Artifacts

The final report directory must contain:

- `report.md`
- `report_technical.md`
- `report_nature.md`
- `_report_with_css.md`
- `_report_nature_with_css.md`
- `report.html`
- `report_nature.html`
- at least one figure referenced from the markdown report when figures were
  generated during execution

## Failure Signals

Treat these as failures:

- Plan approval allowed an invalid skill argument.
- A required skill error counted as success.
- `generate_report(format="dual")` claimed success without all artifacts.
- The trajectory section presents causal or temporal forecasts from unsupported
  association-only evidence.
- The report hides missingness, cohort definition, model choice, calibration, or
  safety/statistical review limitations.
