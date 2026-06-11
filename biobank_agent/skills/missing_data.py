"""Missing data analysis — heatmap and statistics."""

import numpy as np
import seaborn as sns

from biobank_agent.data.features import ALL_BIOMARKERS
from biobank_agent.registry import skill
from biobank_agent.utils.plotting import nature_figure, save_figure, apply_nature_style, PALETTE
from ._column_select import biomarker_select_expressions


@skill(
    name="missing_data",
    description="Analyse missing data patterns across biomarkers. Generates a heatmap "
                "showing percentage of missing values per feature, and a bar chart of "
                "the most/least complete features.",
    parameters={
        "sample_size": {
            "type": "integer",
            "description": "Optional subject sample size. Use 0 or omit to analyse the full biomarker table.",
            "default": 0,
        },
    },
    required=[],
)
def missing_data(sample_size: int = 0, *, ctx=None) -> dict:
    dm = ctx.dm
    if sample_size in (None, 0):
        sample_size = int(getattr(ctx.settings, "default_analysis_sample_size", 0) or 0)

    cols, col_names, column_source = biomarker_select_expressions(dm, ALL_BIOMARKERS)
    if not cols:
        return {"error": "No numeric biomarker columns available for missing-data analysis."}

    sample_clause = f" USING SAMPLE {int(sample_size)}" if sample_size and int(sample_size) > 0 else ""
    sql = f"SELECT {', '.join(cols)} FROM biomarkers{sample_clause}"
    df = dm.query(sql)
    n_subjects = int(len(df))
    sampling_applied = bool(sample_size and int(sample_size) > 0)

    # Compute missing percentage per column
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

    # Overall stats
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isnull().sum().sum())
    overall_pct = round(total_missing / total_cells * 100, 2)

    # Bar chart of missing percentages
    fig, ax = nature_figure(width="single", height_ratio=0.04 * min(len(missing_pct), 30))
    top_missing = missing_pct.head(30)
    y_pos = range(len(top_missing) - 1, -1, -1)
    colors = [PALETTE[1] if v > 20 else PALETTE[0] for v in top_missing.values]
    ax.barh(list(y_pos), top_missing.values, color=colors, height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top_missing.index)
    ax.set_xlabel("Missing (%)")
    ax.set_title(f"Missing Data (N={n_subjects:,}, overall={overall_pct}%)")
    ax.axvline(x=20, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    paths = save_figure(fig, "missing_data", ctx.report_dir)
    ctx.state.figures.extend(paths)

    # Categorise features
    complete = [n for n, v in missing_pct.items() if v < 1]
    moderate = [n for n, v in missing_pct.items() if 1 <= v < 20]
    high = [n for n, v in missing_pct.items() if v >= 20]

    return {
        "sample_size": n_subjects,
        "requested_sample_size": int(sample_size or 0),
        "sampling_applied": sampling_applied,
        "column_source": column_source,
        "n_subjects_analyzed": n_subjects,
        "n_features": len(missing_pct),
        "overall_missing_pct": overall_pct,
        "complete_features": len(complete),
        "moderate_missing": len(moderate),
        "high_missing": len(high),
        "top_missing": [
            {"feature": n, "missing_pct": round(float(v), 2)}
            for n, v in missing_pct.head(10).items()
        ],
        "most_complete": [
            {"feature": n, "missing_pct": round(float(v), 2)}
            for n, v in missing_pct.tail(5).items()
        ],
        "figure": str(paths[0]),
    }
