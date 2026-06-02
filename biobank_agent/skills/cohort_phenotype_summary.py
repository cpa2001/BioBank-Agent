"""Cohort phenotype summary — demographics by phenotype group."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biobank_agent.registry import skill
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE
from biobank_agent.utils.stats import descriptive_stats
from biobank_agent.utils.wgs import wgs_results_dir


@skill(
    name="cohort_phenotype_summary",
    description=(
        "Summarize the cohort by phenotype group: age/sex distributions, "
        "sampling sites, and generate demographic plots. Works with any "
        "biobank that has a phenotype or categorical grouping column."
    ),
    parameters={
        "group_by": {
            "type": "string",
            "description": "Column to group by (default: phenotype_group)",
            "default": "phenotype_group",
        },
    },
    required=[],
)
def cohort_phenotype_summary(group_by: str = "phenotype_group", *, ctx=None) -> dict:
    dm = ctx.dm
    id_col = getattr(dm, "subject_id_col", ctx.settings.subject_id_col)

    df = dm.query("SELECT * FROM biomarkers")
    if df.empty:
        return {"error": "No data in biomarkers view."}

    if group_by not in df.columns:
        available = [c for c in df.columns if c != id_col]
        return {
            "error": f"Column '{group_by}' not found.",
            "available_columns": available,
        }

    groups = df[group_by].unique().tolist()
    n_total = len(df)

    age_col = None
    for c in ("age", "age_at_sampling"):
        if c in df.columns:
            age_col = c
            break
    if age_col is None:
        for c in df.select_dtypes(include=[np.number]).columns:
            if "age" in c.lower():
                age_col = c
                break

    sex_col = None
    for c in ("sex", "is_male"):
        if c in df.columns:
            sex_col = c
            break

    part_col = None
    for c in ("part", "site", "sampling_site"):
        if c in df.columns:
            part_col = c
            break

    group_stats = []
    for g in sorted(groups):
        sub = df[df[group_by] == g]
        info: dict = {"group": g, "n": len(sub), "pct": round(len(sub) / n_total * 100, 1)}
        if age_col and age_col in sub.columns:
            ages = sub[age_col].dropna()
            if len(ages) > 0:
                info["age"] = descriptive_stats(ages)
        if sex_col:
            if sex_col == "sex":
                info["male_n"] = int((sub[sex_col] == "M").sum())
                info["female_n"] = int((sub[sex_col] == "F").sum())
            elif sex_col == "is_male":
                info["male_n"] = int(sub[sex_col].sum())
                info["female_n"] = int((sub[sex_col] == 0).sum())
        if part_col:
            info["sites"] = sub[part_col].value_counts().to_dict()
        group_stats.append(info)

    figures = []
    report_dir = wgs_results_dir(ctx, "cohort")

    if age_col:
        fig, ax = nature_figure(width="single")
        positions = []
        data_arrays = []
        labels = []
        for i, gs in enumerate(group_stats):
            sub = df[df[group_by] == gs["group"]]
            vals = sub[age_col].dropna().values
            if len(vals) > 0:
                data_arrays.append(vals)
                positions.append(i)
                labels.append(f"{gs['group']}\n(n={gs['n']})")
        if data_arrays:
            bp = ax.boxplot(data_arrays, positions=positions, patch_artist=True, widths=0.6)
            for patch, color in zip(bp["boxes"], PALETTE[: len(data_arrays)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=6)
            ax.set_ylabel("Age (years)")
            ax.set_title(f"Age distribution by {group_by}")
            fig.tight_layout()
            paths = save_figure(fig, f"phenotype_age_by_{group_by}", report_dir)
            figures.extend(paths)
            if hasattr(ctx, "state") and hasattr(ctx.state, "figures"):
                ctx.state.figures.extend(paths)

    fig2, ax2 = nature_figure(width="single")
    grp_labels = [gs["group"] for gs in group_stats]
    grp_counts = [gs["n"] for gs in group_stats]
    bars = ax2.bar(range(len(grp_labels)), grp_counts,
                   color=PALETTE[: len(grp_labels)], edgecolor="none", width=0.6)
    ax2.set_xticks(range(len(grp_labels)))
    ax2.set_xticklabels(grp_labels, fontsize=7)
    ax2.set_ylabel("Count")
    ax2.set_title(f"Sample counts by {group_by}")
    for bar, count in zip(bars, grp_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(count), ha="center", va="bottom", fontsize=6)
    fig2.tight_layout()
    paths2 = save_figure(fig2, f"phenotype_counts_{group_by}", report_dir)
    figures.extend(paths2)
    if hasattr(ctx, "state") and hasattr(ctx.state, "figures"):
        ctx.state.figures.extend(paths2)

    cohort_key = f"phenotype_{group_by}"
    if hasattr(ctx, "state") and hasattr(ctx.state, "cohorts"):
        ctx.state.cohorts[cohort_key] = df

    return {
        "n_total": n_total,
        "group_by": group_by,
        "n_groups": len(groups),
        "groups": group_stats,
        "age_column": age_col,
        "sex_column": sex_col,
        "result_dir": str(report_dir),
        "figures": [str(p) for p in figures],
    }
