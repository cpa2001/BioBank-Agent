"""Minimum sample size sensitivity study."""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from biobank_agent.data.cohort import build_cohort
from biobank_agent.registry import skill
from biobank_agent.utils.icd10 import icd10_name
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE


@skill(
    name="min_sample",
    description="Study how model performance changes with different training sample sizes. "
                "Trains models at increasing case counts and plots the learning curve.",
    parameters={
        "icd10_code": {
            "type": "string",
            "description": "ICD10 code prefix (e.g. 'E11')",
        },
        "case_counts": {
            "type": "string",
            "description": "Comma-separated case counts to test (default: '50,100,200,500,1000,2000,5000')",
            "default": "50,100,200,500,1000,2000,5000",
        },
        "n_repeats": {
            "type": "integer",
            "description": "Repetitions per sample size (default 3)",
            "default": 3,
        },
    },
    required=["icd10_code"],
)
def min_sample(icd10_code: str, case_counts: str = "50,100,200,500,1000,2000,5000",
               n_repeats: int = 3, *, ctx=None) -> dict:
    dm = ctx.dm
    id_col = ctx.settings.subject_id_col

    counts = [int(c.strip()) for c in case_counts.split(",")]

    # Build full cohort first
    cohort_key = f"{icd10_code}_1:all"
    if cohort_key in ctx.state.cohorts:
        full_df = ctx.state.cohorts[cohort_key]
    else:
        full_df = build_cohort(dm, icd10_code, controls_ratio=0)
        ctx.state.cohorts[cohort_key] = full_df

    feature_cols = [c for c in full_df.columns
                    if c not in (id_col, "label")
                    and full_df[c].dtype in ("float64", "float32", "int64", "int32")]
    X_full = full_df[feature_cols].fillna(full_df[feature_cols].median())
    y_full = full_df["label"]

    total_cases = int(y_full.sum())
    cases_idx = np.where(y_full == 1)[0]
    controls_idx = np.where(y_full == 0)[0]

    from xgboost import XGBClassifier

    results = []
    for n_cases in counts:
        if n_cases > total_cases:
            continue
        for rep in range(n_repeats):
            rng = np.random.RandomState(42 + rep)
            sel_cases = rng.choice(cases_idx, n_cases, replace=False)
            sel_controls = rng.choice(controls_idx, min(n_cases * 4, len(controls_idx)), replace=False)
            sel = np.concatenate([sel_cases, sel_controls])

            X_sub = X_full.iloc[sel]
            y_sub = y_full.iloc[sel]

            clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                eval_metric="logloss", random_state=42)
            cv = StratifiedKFold(n_splits=min(5, n_cases // 10 + 1), shuffle=True, random_state=42)
            try:
                scores = cross_val_score(clf, X_sub, y_sub, cv=cv, scoring="roc_auc")
                results.append({
                    "n_cases": n_cases, "repeat": rep,
                    "auc_mean": float(np.mean(scores)),
                    "auc_std": float(np.std(scores)),
                })
            except Exception:
                continue

    # Aggregate
    summary = {}
    for n in counts:
        reps = [r for r in results if r["n_cases"] == n]
        if reps:
            aucs = [r["auc_mean"] for r in reps]
            summary[n] = {
                "mean": float(np.mean(aucs)),
                "std": float(np.std(aucs)),
                "n_reps": len(reps),
            }

    # Plot learning curve
    fig, ax = nature_figure(width="single")
    ns = sorted(summary.keys())
    means = [summary[n]["mean"] for n in ns]
    stds = [summary[n]["std"] for n in ns]
    ax.errorbar(ns, means, yerr=stds, marker="o", color=PALETTE[0],
                markersize=4, linewidth=1, capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("Number of cases")
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"Sample Size Sensitivity: {icd10_code} {icd10_name(icd10_code)}")
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    paths = save_figure(fig, f"min_sample_{icd10_code}", ctx.report_dir)
    ctx.state.figures.extend(paths)

    return {
        "icd10_code": icd10_code,
        "disease": icd10_name(icd10_code),
        "total_available_cases": total_cases,
        "sample_sizes_tested": ns,
        "results": {str(k): v for k, v in summary.items()},
        "figure": str(paths[0]),
    }
