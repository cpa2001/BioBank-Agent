"""Biomarker distribution comparison — cases vs controls."""

import numpy as np
import seaborn as sns

from biobank_agent.data.cohort import build_cohort
from biobank_agent.data.features import ALL_BIOMARKERS
from biobank_agent.registry import skill
from biobank_agent.utils.icd10 import icd10_name
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE
from biobank_agent.utils.stats import mann_whitney


@skill(
    name="biomarker_dist",
    description="Compare biomarker distributions between disease cases and controls. "
                "Generates violin plots and performs Mann-Whitney U tests for each biomarker.",
    parameters={
        "icd10_code": {
            "type": "string",
            "description": "ICD10 code prefix (e.g. 'E11')",
        },
        "biomarkers": {
            "type": "string",
            "description": "Comma-separated biomarker field IDs or names. "
                           "Default: top 6 blood biochemistry markers. "
                           "Example: '30740,30750,30690' or 'glucose,cholesterol'",
            "default": "",
        },
        "controls_ratio": {
            "type": "integer",
            "description": "Controls per case when building the cohort. Use 0 or omit to include all eligible controls.",
            "default": 0,
        },
    },
    required=["icd10_code"],
)
def biomarker_dist(icd10_code: str, biomarkers: str = "", controls_ratio: int = 0, *, ctx=None) -> dict:
    dm = ctx.dm

    # Build or reuse cohort
    cohort_key = f"{icd10_code}_1:{controls_ratio or 'all'}"
    if cohort_key in ctx.state.cohorts:
        df = ctx.state.cohorts[cohort_key]
    else:
        df = build_cohort(dm, icd10_code, controls_ratio=controls_ratio)
        ctx.state.cohorts[cohort_key] = df

    # Select biomarkers
    if biomarkers:
        field_ids = [b.strip() for b in biomarkers.split(",")]
    else:
        # Default: top 6 blood biochemistry by importance
        field_ids = ["30740", "30750", "30690", "30870", "30710", "30760"]

    # Find matching columns
    results = []
    plot_data = []
    for fid in field_ids:
        col = f"{fid}-0.0"
        if col not in df.columns:
            continue
        name = ALL_BIOMARKERS.get(fid, fid)
        cases = df.loc[df["label"] == 1, col].dropna().values
        controls = df.loc[df["label"] == 0, col].dropna().values
        if len(cases) < 10 or len(controls) < 10:
            continue

        test = mann_whitney(cases, controls)
        results.append({
            "field_id": fid,
            "name": name,
            "cases_median": round(float(np.median(cases)), 3),
            "controls_median": round(float(np.median(controls)), 3),
            "p_value": test["p_value"],
            "effect_size": round(test["effect_size_r"], 3),
        })
        plot_data.append((fid, name, col))

    # Generate violin plots
    n_plots = min(len(plot_data), 6)
    if n_plots > 0:
        ncols = min(3, n_plots)
        nrows = (n_plots + ncols - 1) // ncols
        fig, axes = nature_figure(nrows=nrows, ncols=ncols, width="double")
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = np.atleast_1d(axes).flatten()

        for i, (fid, name, col) in enumerate(plot_data[:n_plots]):
            ax = axes[i]
            data_cases = df.loc[df["label"] == 1, col].dropna()
            data_controls = df.loc[df["label"] == 0, col].dropna()

            parts = ax.violinplot(
                [data_controls.values, data_cases.values],
                positions=[0, 1], showmedians=True, showextrema=False,
            )
            for j, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(PALETTE[j])
                pc.set_alpha(0.7)
            parts["cmedians"].set_color("black")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Control", "Case"])
            ax.set_ylabel(name)

            # Add p-value
            p = results[i]["p_value"]
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            ax.set_title(f"{name}\n{stars}", fontsize=7)

        # Hide unused axes
        for j in range(n_plots, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"{icd10_code} {icd10_name(icd10_code)}: Biomarker Distributions",
                     fontsize=8, y=1.02)
        fig.tight_layout()
        paths = save_figure(fig, f"biomarker_dist_{icd10_code}", ctx.report_dir)
        ctx.state.figures.extend(paths)

    return {
        "icd10_code": icd10_code,
        "disease": icd10_name(icd10_code),
        "n_cases": int((df["label"] == 1).sum()),
        "n_controls": int((df["label"] == 0).sum()),
        "controls_ratio": controls_ratio,
        "controls_sampling_applied": bool(controls_ratio and controls_ratio > 0),
        "comparisons": results,
        "figure": str(paths[0]) if n_plots > 0 else None,
    }
