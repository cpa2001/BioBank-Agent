"""Cohort demographics summary — age/sex distribution."""

from biobank_agent.data.cohort import build_cohort
from biobank_agent.registry import skill
from biobank_agent.utils.icd10 import icd10_name
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE
from biobank_agent.utils.stats import descriptive_stats

import numpy as np


@skill(
    name="cohort_summary",
    description="Build a case/control cohort for an ICD10 code and show demographics: "
                "age distribution, sex ratio, cohort sizes. Generates age histogram "
                "stratified by case/control status.",
    parameters={
        "icd10_code": {
            "type": "string",
            "description": "ICD10 code prefix (e.g. 'E11' for Type 2 Diabetes)",
        },
        "controls_ratio": {
            "type": "integer",
            "description": "Controls per case (default 4)",
            "default": 4,
        },
    },
    required=["icd10_code"],
)
def cohort_summary(icd10_code: str, controls_ratio: int = 4, *, ctx=None) -> dict:
    dm = ctx.dm

    # Build or reuse cohort
    cohort_key = f"{icd10_code}_1:{controls_ratio}"
    if cohort_key in ctx.state.cohorts:
        df = ctx.state.cohorts[cohort_key]
    else:
        df = build_cohort(dm, icd10_code, controls_ratio)
        ctx.state.cohorts[cohort_key] = df

    n_cases = int(df["label"].sum())
    n_controls = int((df["label"] == 0).sum())
    disease_name = icd10_name(icd10_code)

    # Sex distribution (field 31: 0=female, 1=male)
    sex_col = None
    for c in df.columns:
        if c.startswith("31-"):
            sex_col = c
            break

    sex_info = {}
    if sex_col:
        cases = df[df["label"] == 1]
        controls = df[df["label"] == 0]
        sex_info = {
            "cases_male_pct": float((cases[sex_col] == 1).mean() * 100),
            "cases_female_pct": float((cases[sex_col] == 0).mean() * 100),
            "controls_male_pct": float((controls[sex_col] == 1).mean() * 100),
            "controls_female_pct": float((controls[sex_col] == 0).mean() * 100),
        }

    # Age distribution (field 21022: age at recruitment)
    age_col = None
    for c in df.columns:
        if c.startswith("21022-") or c.startswith("21003-"):
            age_col = c
            break

    age_stats = {}
    if age_col:
        cases_age = df.loc[df["label"] == 1, age_col].dropna()
        controls_age = df.loc[df["label"] == 0, age_col].dropna()
        age_stats = {
            "cases_age": descriptive_stats(cases_age),
            "controls_age": descriptive_stats(controls_age),
        }

        # Plot age histogram
        fig, ax = nature_figure(width="single")
        bins = np.arange(35, 75, 2)
        ax.hist(controls_age, bins=bins, alpha=0.6, label=f"Controls (n={n_controls:,})",
                color=PALETTE[0], density=True)
        ax.hist(cases_age, bins=bins, alpha=0.6, label=f"Cases (n={n_cases:,})",
                color=PALETTE[1], density=True)
        ax.set_xlabel("Age at recruitment (years)")
        ax.set_ylabel("Density")
        ax.set_title(f"{icd10_code} {disease_name}")
        ax.legend(frameon=False)
        fig.tight_layout()
        paths = save_figure(fig, f"cohort_{icd10_code}_age", ctx.report_dir)
        ctx.state.figures.extend(paths)

    return {
        "icd10_code": icd10_code,
        "disease_name": disease_name,
        "n_cases": n_cases,
        "n_controls": n_controls,
        "n_features": df.shape[1] - 2,
        "sex": sex_info,
        "age": age_stats,
        "figure": str(paths[0]) if age_col else None,
    }
