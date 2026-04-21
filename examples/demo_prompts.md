# Biobank Agent — Demo Prompts

These 15 prompts demonstrate the full capabilities of `bb`. Each prompt maps to 1–8 skills chained by the agent.

---

## 1. Disease Prevalence
```
What are the top 20 most common diseases in UK Biobank?
```
**Expected:** Horizontal bar chart + table of ICD10 codes with patient counts and percentages.

## 2. Cohort Summary
```
Build a Type 2 Diabetes (E11) cohort and show me the demographics.
```
**Expected:** Age histogram (cases vs controls), sex ratio, cohort sizes.

## 3. Biomarker Distributions
```
Compare glucose and HbA1c levels between Type 2 Diabetes cases and controls.
```
**Expected:** Violin plots with Mann-Whitney U test results and significance stars.

## 4. Correlation Heatmap
```
Show me the correlation structure of all blood biochemistry markers.
```
**Expected:** Clustered heatmap with hierarchical dendrogram, top correlated pairs.

## 5. Train Predictive Model
```
Train an XGBoost model to predict Type 2 Diabetes from biomarkers.
```
**Expected:** 5-fold CV AUC with 95% CI, metrics table, model stored in session.

## 6. ROC + PR Curves
```
Show the ROC and precision-recall curves for the diabetes model.
```
**Expected:** Side-by-side ROC (with per-fold curves) and PR curve, mean AUC/AP.

## 7. Feature Importance
```
What are the most important features for predicting diabetes?
```
**Expected:** Horizontal bar chart of top 20 features by tree-based importance.

## 8. Calibration Analysis
```
How well-calibrated is the diabetes model?
```
**Expected:** Reliability diagram + ECE, MCE, Brier score.

## 9. Missing Data
```
Analyse the missing data patterns across all biomarkers.
```
**Expected:** Bar chart of missing percentages, categorisation (complete/moderate/high).

## 10. Survival Analysis
```
Show Kaplan-Meier survival curves comparing T2D patients vs controls.
```
**Expected:** KM curves with 95% CI, log-rank p-value, mortality rates.

## 11. PheWAS
```
Run a PheWAS: which diseases are associated with high CRP levels?
```
**Expected:** Manhattan plot coloured by ICD10 chapter, FDR-corrected top associations.

## 12. Comorbidity Network
```
What diseases commonly co-occur with Type 2 Diabetes?
```
**Expected:** Odds ratio bar chart, top comorbidities with prevalence comparisons.

## 13. Patient Embedding
```
Create a UMAP embedding of patients coloured by diabetes status.
```
**Expected:** 2D scatter plot (Nature style, no ticks, rasterised points, legend).

## 14. Model Comparison
```
Compare XGBoost, LightGBM, and CatBoost for predicting hypertension (I10).
```
**Expected:** Three sequential train_model calls, comparison table of AUC/F1/precision.

## 15. Full Analysis Report
```
Run a complete analysis of Chronic Kidney Disease (N18) and generate a report.
```
**Expected:** Agent chains ~8 skills: prevalence → cohort_summary → biomarker_dist → train_model → evaluate_model → feature_importance → calibration → generate_report.
