"""Correlation heatmap of biomarkers."""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from biobank_agent.data.features import BLOOD_BIOCHEMISTRY, BLOOD_COUNT, ALL_BIOMARKERS
from biobank_agent.registry import skill
from biobank_agent.utils.plotting import apply_nature_style, save_figure


@skill(
    name="correlation",
    description="Generate a clustered correlation heatmap for a set of biomarkers. "
                "Uses hierarchical clustering to order features by similarity.",
    parameters={
        "group": {
            "type": "string",
            "description": "Feature group: 'biochemistry', 'blood_count', or 'all' (default: biochemistry)",
            "default": "biochemistry",
            "enum": ["biochemistry", "blood_count", "all"],
        },
    },
    required=[],
)
def correlation(group: str = "biochemistry", *, ctx=None) -> dict:
    dm = ctx.dm

    groups = {
        "biochemistry": BLOOD_BIOCHEMISTRY,
        "blood_count": BLOOD_COUNT,
        "all": ALL_BIOMARKERS,
    }
    fields = groups.get(group, BLOOD_BIOCHEMISTRY)
    field_ids = list(fields.keys())

    # Build column select
    cols = []
    col_names = []
    for fid in field_ids:
        col = f'"{fid}-0.0"'
        name = fields[fid]
        cols.append(f'{col} AS "{name}"')
        col_names.append(name)

    sql = f"SELECT {', '.join(cols)} FROM biomarkers USING SAMPLE 50000"
    df = dm.query(sql)

    # Compute correlation
    corr = df.corr()

    # Plot clustered heatmap
    apply_nature_style()

    g = sns.clustermap(
        corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        figsize=(7, 7), linewidths=0.1,
        dendrogram_ratio=0.1, cbar_pos=(0.02, 0.8, 0.03, 0.15),
    )
    g.ax_heatmap.tick_params(axis="both", which="major", labelsize=5)
    g.fig.suptitle(f"Biomarker Correlation ({group})", y=1.01, fontsize=8)

    paths = save_figure(
        g.fig,
        f"correlation_{group}",
        ctx.report_dir,
        formats=("svg", "pdf"),
    )
    ctx.state.figures.extend(paths)

    # Top correlated pairs
    pairs = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                pairs.append({
                    "feature_1": corr.index[i],
                    "feature_2": corr.columns[j],
                    "r": round(float(r), 3),
                })
    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)

    return {
        "group": group,
        "n_features": len(field_ids),
        "n_subjects_sampled": 50000,
        "top_correlations": pairs[:15],
        "figures": [str(p) for p in paths],
    }
