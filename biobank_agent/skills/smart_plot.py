"""Smart plotting skill — publication-quality figures with style selection.

Generates figures with automatic style detection or explicit style preset.
Can search the web for reference figure styles.
"""

import logging

import numpy as np
import pandas as pd

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)

# Style presets with descriptions
STYLE_PRESETS = {
    "nature": {
        "description": "Nature family journals (Nature, Nat Methods, Nat Med)",
        "figure_func": "nature_figure",
        "font": "Arial",
        "single_col_mm": 89,
        "double_col_mm": 183,
    },
    "icml": {
        "description": "ICML / NeurIPS machine learning conferences",
        "figure_func": "icml_figure",
        "font": "Times",
        "textwidth_in": 6.75,
    },
    "nejm": {
        "description": "New England Journal of Medicine",
        "figure_func": "nature_figure",
        "font": "Arial",
        "single_col_mm": 86,
    },
    "lancet": {
        "description": "The Lancet family",
        "figure_func": "nature_figure",
        "font": "Arial",
        "single_col_mm": 89,
    },
}


@skill(
    name="smart_plot",
    description="Generate a publication-quality figure with automatic style selection. "
                "Supports multiple plot types and can search web for reference figure styles.",
    parameters={
        "plot_type": {
            "type": "string",
            "description": "Plot type: 'bar', 'violin', 'heatmap', 'forest', 'km', "
                           "'roc', 'manhattan', 'waterfall', 'network', 'scatter'",
        },
        "data_source": {
            "type": "string",
            "description": "What data to plot (e.g., 'cohort:E11', 'model:E11:xgboost', "
                           "'field:30750' for a specific field)",
        },
        "style": {
            "type": "string",
            "description": "Target style: 'nature', 'icml', 'nejm', 'lancet', 'auto'",
            "default": "auto",
        },
        "title": {
            "type": "string",
            "description": "Figure title (optional, auto-generated if empty)",
            "default": "",
        },
        "reference_search": {
            "type": "string",
            "description": "Optional: search query to find reference figures for style hints",
            "default": "",
        },
    },
    required=["plot_type", "data_source"],
)
def smart_plot(
    plot_type: str,
    data_source: str,
    style: str = "auto",
    title: str = "",
    reference_search: str = "",
    *,
    ctx=None,
) -> dict:
    """Generate publication-quality figure with smart style selection."""
    from biobank_agent.utils.plotting import (
        nature_figure, icml_figure, save_figure, PALETTE, add_panel_labels,
    )

    # Auto-detect style
    if style == "auto":
        style = "nature"  # Default to Nature for biomedical

    # Search for reference if requested
    ref_hints = {}
    if reference_search:
        try:
            from biobank_agent.skills.web_search import web_search
            search_results = web_search(
                query=f"{reference_search} figure style",
                max_results=3,
                ctx=ctx,
            )
            ref_hints["search_results"] = search_results.get("results", [])[:3]
        except Exception as e:
            ref_hints["search_error"] = str(e)

    # Create figure with selected style
    if style == "icml":
        fig, ax = icml_figure(nrows=1, ncols=1)
    else:
        width = "double" if plot_type in ("heatmap", "network", "manhattan") else "single"
        fig, ax = nature_figure(nrows=1, ncols=1, width=width)

    if hasattr(ax, "__len__"):
        ax = ax.flat[0]

    # Parse data source
    source_type, source_key = _parse_data_source(data_source)

    # Generate plot based on type
    try:
        if plot_type == "bar":
            _plot_bar(ax, source_type, source_key, ctx)
        elif plot_type == "violin":
            _plot_violin(ax, source_type, source_key, ctx)
        elif plot_type == "heatmap":
            _plot_heatmap(fig, ax, source_type, source_key, ctx)
        elif plot_type == "scatter":
            _plot_scatter(ax, source_type, source_key, ctx)
        elif plot_type == "roc":
            _plot_roc(ax, source_type, source_key, ctx)
        elif plot_type == "km":
            _plot_km(ax, source_type, source_key, ctx)
        else:
            ax.text(0.5, 0.5, f"Plot type '{plot_type}' — use dedicated skill",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7)
    except Exception as e:
        ax.text(0.5, 0.5, f"Plot error: {e}",
                ha="center", va="center", transform=ax.transAxes, fontsize=6, color="red")
        logger.warning("smart_plot error: %s", e)

    # Set title
    if title:
        ax.set_title(title)
    elif not ax.get_title():
        ax.set_title(f"{plot_type.title()} — {data_source}")

    fig.tight_layout()

    # Save
    fig_name = f"smart_{plot_type}_{source_key.replace(':', '_')}"
    fig_paths = save_figure(fig, fig_name, ctx.report_dir)
    for p in fig_paths:
        ctx.state.figures.append(p)

    return {
        "plot_type": plot_type,
        "data_source": data_source,
        "style": style,
        "figures": [str(p) for p in fig_paths],
        "reference_hints": ref_hints if ref_hints else None,
    }


def _parse_data_source(data_source: str):
    """Parse 'cohort:E11' or 'model:E11:xgboost' or 'field:30750'."""
    parts = data_source.split(":", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "raw", data_source


def _plot_bar(ax, source_type, source_key, ctx):
    """Bar chart from cohort or field data."""
    from biobank_agent.utils.plotting import PALETTE

    if source_type == "cohort" and source_key in ctx.state.cohorts:
        df = ctx.state.cohorts[source_key]
        if "label" in df.columns:
            counts = df["label"].value_counts().sort_index()
            labels = ["Controls", "Cases"]
            values = [counts.get(0, 0), counts.get(1, 0)]
            colors = [PALETTE[0], PALETTE[1]]
            ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_ylabel("Count")
            ax.set_title(f"Cohort {source_key}: Case/Control Distribution")
    else:
        ax.text(0.5, 0.5, "Bar: provide cohort data source", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)


def _plot_violin(ax, source_type, source_key, ctx):
    """Violin plot comparing cases vs controls."""
    from biobank_agent.utils.plotting import PALETTE

    if source_type == "cohort" and source_key in ctx.state.cohorts:
        df = ctx.state.cohorts[source_key]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "label"][:5]
        if numeric_cols and "label" in df.columns:
            data = [df[df["label"] == 0][numeric_cols[0]].dropna().values,
                    df[df["label"] == 1][numeric_cols[0]].dropna().values]
            parts = ax.violinplot(data, positions=[0, 1], showmedians=True)
            for i, pc in enumerate(parts.get("bodies", [])):
                pc.set_facecolor(PALETTE[i])
                pc.set_alpha(0.7)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Controls", "Cases"])
            ax.set_ylabel(numeric_cols[0])
            ax.set_title(f"{numeric_cols[0]} Distribution")
    else:
        ax.text(0.5, 0.5, "Violin: provide cohort data source", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)


def _plot_heatmap(fig, ax, source_type, source_key, ctx):
    """Correlation heatmap."""
    from biobank_agent.utils.plotting import PALETTE_DIV

    if source_type == "cohort" and source_key in ctx.state.cohorts:
        df = ctx.state.cohorts[source_key]
        numeric = df.select_dtypes(include=[np.number])
        numeric = numeric.drop(columns=["label"], errors="ignore")
        if len(numeric.columns) > 2:
            corr = numeric.corr()
            n = min(20, len(corr))
            corr = corr.iloc[:n, :n]
            im = ax.imshow(corr.values, cmap=PALETTE_DIV, vmin=-1, vmax=1, aspect="auto")
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks(range(n))
            ax.set_xticklabels([c[:10] for c in corr.columns], rotation=45, ha="right", fontsize=4)
            ax.set_yticks(range(n))
            ax.set_yticklabels([c[:10] for c in corr.index], fontsize=4)
            ax.set_title(f"Correlation Heatmap: {source_key}")
    else:
        ax.text(0.5, 0.5, "Heatmap: provide cohort data source", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)


def _plot_scatter(ax, source_type, source_key, ctx):
    """Scatter plot of two features."""
    from biobank_agent.utils.plotting import PALETTE

    if source_type == "cohort" and source_key in ctx.state.cohorts:
        df = ctx.state.cohorts[source_key]
        numeric = df.select_dtypes(include=[np.number])
        numeric = numeric.drop(columns=["label"], errors="ignore")
        cols = numeric.columns.tolist()[:2]
        if len(cols) >= 2 and "label" in df.columns:
            for label, color in [(0, PALETTE[0]), (1, PALETTE[1])]:
                mask = df["label"] == label
                ax.scatter(df.loc[mask, cols[0]], df.loc[mask, cols[1]],
                           c=color, s=1, alpha=0.3, label=f"Label {label}")
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            ax.legend(fontsize=5, markerscale=3)
            ax.set_title(f"Scatter: {cols[0]} vs {cols[1]}")
    else:
        ax.text(0.5, 0.5, "Scatter: provide cohort data source", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)


def _plot_roc(ax, source_type, source_key, ctx):
    """ROC curve from model results."""
    from biobank_agent.utils.plotting import PALETTE

    model_key = source_key if source_type == "model" else source_key
    meta = ctx.state.model_metadata.get(model_key, {})
    auc = meta.get("auc")

    if auc is not None:
        # Plot diagonal
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.5)
        # Placeholder ROC (we'd need the actual curve data)
        ax.text(0.5, 0.5, f"AUC = {float(auc):.4f}\n(Use evaluate_model for full ROC)",
                ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC: {model_key}")
    else:
        ax.text(0.5, 0.5, "ROC: train model first", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)


def _plot_km(ax, source_type, source_key, ctx):
    """Kaplan-Meier placeholder — delegates to survival skill."""
    ax.text(0.5, 0.5, f"KM for {source_key}\n(Use survival skill for full KM analysis)",
            ha="center", va="center", transform=ax.transAxes, fontsize=7)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival Probability")
    ax.set_title(f"Survival: {source_key}")
