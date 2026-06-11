"""Smart plotting skill — publication-quality figures with style selection.

Generates figures with automatic style detection or explicit style preset.
Can search the web for reference figure styles.
"""

import logging

import matplotlib.pyplot as plt
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

WIDE_PLOT_TYPES = {"heatmap", "network", "manhattan", "summary"}
SUPPORTED_INLINE_PLOTS = {"bar", "violin", "heatmap", "scatter", "roc", "km", "summary"}


def _palette_roles_for_plot(plot_type: str, source_type: str) -> list[str]:
    """Return semantic colour roles that should be used for plot metadata."""
    if plot_type in ("bar", "violin", "scatter"):
        return ["control", "case"]
    if plot_type == "roc":
        return ["observed", "reference"]
    if plot_type == "summary":
        return ["control", "case", "observed", "high_risk", "neutral"]
    if plot_type == "heatmap":
        return ["negative", "neutral", "positive"]
    if plot_type == "km":
        return ["low_risk", "high_risk"]
    if source_type == "model":
        return ["observed", "expected"]
    return ["primary", "secondary"]


def _select_plot_metadata(
    plot_type: str,
    source_type: str,
    source_key: str,
    requested_style: str,
) -> dict:
    """Choose style, figure width, and semantic palette for a smart plot."""
    from biobank_agent.utils.plotting import semantic_palette

    requested = (requested_style or "auto").strip().lower()
    if requested == "auto":
        selected_style = "nature"
        style_reason = "auto defaults to Nature style for biomedical cohort figures"
    elif requested in STYLE_PRESETS:
        selected_style = requested
        style_reason = f"explicit style '{requested}' requested"
    else:
        selected_style = "nature"
        style_reason = f"unknown style '{requested}' fell back to Nature style"

    figure_width = "double" if plot_type in WIDE_PLOT_TYPES else "single"
    palette_roles = _palette_roles_for_plot(plot_type, source_type)
    render_path = "inline" if plot_type in SUPPORTED_INLINE_PLOTS else "placeholder"

    return {
        "plot_type": plot_type,
        "data_source": {"type": source_type, "key": source_key},
        "requested_style": requested_style or "auto",
        "selected_style": selected_style,
        "style_reason": style_reason,
        "figure_width": figure_width,
        "palette_roles": palette_roles,
        "palette": semantic_palette(palette_roles),
        "render_path": render_path,
    }


@skill(
    name="smart_plot",
    description="Generate a publication-quality figure with automatic style selection. "
                "Supports multiple plot types and can search web for reference figure styles.",
    parameters={
        "plot_type": {
            "type": "string",
            "description": "Plot type: 'bar', 'violin', 'heatmap', 'forest', 'km', "
                           "'roc', 'summary', 'manhattan', 'waterfall', 'network', 'scatter'",
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
        nature_figure, icml_figure, save_figure,
    )

    plot_type = (plot_type or "").strip().lower()
    source_type, source_key = _parse_data_source(data_source)
    selection_metadata = _select_plot_metadata(plot_type, source_type, source_key, style)
    style = selection_metadata["selected_style"]

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
    if plot_type == "summary":
        fig, ax = nature_figure(nrows=2, ncols=2, width="double", height_ratio=0.62)
    elif style == "icml":
        fig, ax = icml_figure(nrows=1, ncols=1)
    else:
        width = selection_metadata["figure_width"]
        fig, ax = nature_figure(nrows=1, ncols=1, width=width)

    if plot_type != "summary" and hasattr(ax, "__len__"):
        ax = ax.flat[0]

    # Generate plot based on type
    try:
        if plot_type == "summary":
            _plot_summary(fig, ax, ctx)
        elif plot_type == "bar":
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
        target_ax = ax.flat[0] if hasattr(ax, "flat") else ax
        target_ax.text(0.5, 0.5, f"Plot error: {e}",
                       ha="center", va="center", transform=target_ax.transAxes, fontsize=6, color="red")
        logger.warning("smart_plot error: %s", e)

    # Set title
    if title and plot_type == "summary":
        fig.suptitle(title, y=1.01, fontsize=9)
    elif title:
        ax.set_title(title)
    elif plot_type != "summary" and not ax.get_title():
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
        "selection_metadata": selection_metadata,
        "plot_selection": selection_metadata,
    }


def _parse_data_source(data_source: str):
    """Parse 'cohort:E11' or 'model:E11:xgboost' or 'field:30750'."""
    parts = data_source.split(":", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "raw", data_source


def _plot_bar(ax, source_type, source_key, ctx):
    """Bar chart from cohort or field data."""
    from biobank_agent.utils.plotting import case_control_palette

    if source_type == "cohort" and source_key in ctx.state.cohorts:
        df = ctx.state.cohorts[source_key]
        if "label" in df.columns:
            counts = df["label"].value_counts().sort_index()
            labels = ["Controls", "Cases"]
            values = [counts.get(0, 0), counts.get(1, 0)]
            palette = case_control_palette()
            colors = [palette["control"], palette["case"]]
            ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_ylabel("Count")
            ax.set_title(f"Cohort {source_key}: Case/Control Distribution")
    else:
        ax.text(0.5, 0.5, "Bar: provide cohort data source", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)


def _plot_violin(ax, source_type, source_key, ctx):
    """Violin plot comparing cases vs controls."""
    from biobank_agent.utils.plotting import case_control_palette

    if source_type == "cohort" and source_key in ctx.state.cohorts:
        df = ctx.state.cohorts[source_key]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "label"][:5]
        if numeric_cols and "label" in df.columns:
            data = [df[df["label"] == 0][numeric_cols[0]].dropna().values,
                    df[df["label"] == 1][numeric_cols[0]].dropna().values]
            parts = ax.violinplot(data, positions=[0, 1], showmedians=True)
            palette = case_control_palette()
            colors = [palette["control"], palette["case"]]
            for i, pc in enumerate(parts.get("bodies", [])):
                pc.set_facecolor(colors[i])
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
    from biobank_agent.utils.plotting import case_control_palette

    if source_type == "cohort" and source_key in ctx.state.cohorts:
        df = ctx.state.cohorts[source_key]
        numeric = df.select_dtypes(include=[np.number])
        numeric = numeric.drop(columns=["label"], errors="ignore")
        cols = numeric.columns.tolist()[:2]
        if len(cols) >= 2 and "label" in df.columns:
            palette = case_control_palette()
            for label, color, text in [
                (0, palette["control"], "Controls"),
                (1, palette["case"], "Cases"),
            ]:
                mask = df["label"] == label
                ax.scatter(df.loc[mask, cols[0]], df.loc[mask, cols[1]],
                           c=color, s=1, alpha=0.3, label=text)
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            ax.legend(fontsize=5, markerscale=3)
            ax.set_title(f"Scatter: {cols[0]} vs {cols[1]}")
    else:
        ax.text(0.5, 0.5, "Scatter: provide cohort data source", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)


def _plot_roc(ax, source_type, source_key, ctx):
    """ROC curve from model results."""
    from biobank_agent.utils.plotting import semantic_color

    model_key = source_key if source_type == "model" else source_key
    meta = ctx.state.model_metadata.get(model_key, {})
    auc = meta.get("auc")

    if auc is not None:
        # Plot diagonal
        ax.plot([0, 1], [0, 1], "--", color=semantic_color("reference"), linewidth=0.5)
        # Placeholder ROC (we'd need the actual curve data)
        ax.text(0.5, 0.5, f"AUC = {float(auc):.4f}\n(Use evaluate_model for full ROC)",
                ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC: {model_key}")
    else:
        ax.text(0.5, 0.5, "ROC: train model first", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)


def _session_records(ctx):
    """Return analysis records from a live session-like context."""
    state = getattr(ctx, "state", None)
    records = getattr(state, "records", []) if state is not None else []
    return list(records or [])


def _latest_record(records, skill_name: str):
    for rec in reversed(records):
        if getattr(rec, "skill", "") == skill_name:
            return rec
    return None


def _record_results(rec) -> dict:
    value = getattr(rec, "key_results", {}) if rec is not None else {}
    return value if isinstance(value, dict) else {}


def _metric(results: dict, *keys, default=None):
    for key in keys:
        value = results.get(key)
        if value is not None and value != "":
            return value
    return default


def _as_float(value, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compact_label(value, max_len: int = 28) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _plot_no_session(ax, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=7)


def _plot_summary(fig, axes, ctx):
    """Render a session-level scientific diagnostic summary.

    This is intentionally not a decorative placeholder: it combines cohort size,
    trajectory feasibility, model diagnostics, feature importance, and guardrail
    status into one compact panel for demos and final reports.
    """
    from biobank_agent.utils.plotting import case_control_palette, semantic_color

    ax_arr = np.asarray(axes).reshape(-1)
    if len(ax_arr) < 4:
        _plot_no_session(ax_arr[0], "Summary plot requires a 2x2 axes layout")
        return
    ax_cohort, ax_metrics, ax_features, ax_guardrails = ax_arr[:4]

    records = _session_records(ctx)
    if not records:
        _plot_no_session(ax_cohort, "No session records available")
        for ax in (ax_metrics, ax_features, ax_guardrails):
            ax.axis("off")
        return

    cohort = _record_results(_latest_record(records, "cohort_summary"))
    trajectory = _record_results(_latest_record(records, "trajectory_tokenize"))
    train = _record_results(_latest_record(records, "train_model"))
    evaluation = _record_results(_latest_record(records, "evaluate_model"))
    calibration = _record_results(_latest_record(records, "calibration"))
    importance = _record_results(_latest_record(records, "feature_importance"))
    statistical = _record_results(_latest_record(records, "statistical_review"))
    safety = _record_results(_latest_record(records, "safety_check"))
    world = _record_results(_latest_record(records, "world_model_audit"))

    # Panel A: cohort composition.
    n_cases = _metric(cohort, "n_cases", default=_metric(train, "n_cases", default=0)) or 0
    n_controls = _metric(cohort, "n_controls", default=_metric(train, "n_controls", default=0)) or 0
    palette = case_control_palette()
    raw_values = [n_controls, n_cases]
    values = [_as_float(value, 0.0) or 0.0 for value in raw_values]
    ax_cohort.barh(
        ["Controls", "E11 cases"],
        values,
        color=[palette["control"], palette["case"]],
        edgecolor="white",
        linewidth=0.5,
    )
    ax_cohort.set_title("Cohort composition")
    ax_cohort.set_xlabel("Participants")
    ax_cohort.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    for idx, (raw_value, value) in enumerate(zip(raw_values, values)):
        try:
            label = f"{int(float(raw_value)):,}"
        except (TypeError, ValueError):
            label = str(raw_value)
        ax_cohort.text(value, idx, f" {label}", va="center", fontsize=6)

    # Panel B: key diagnostics as a compact dashboard.
    ax_metrics.axis("off")
    auc = _as_float(_metric(train, "auc_mean", "mean_auc", "auc"))
    cv_auc = _as_float(_metric(evaluation, "mean_auc", "auc"))
    ece = _as_float(_metric(calibration, "ece"))
    n_tokens = _metric(trajectory, "n_tokens", default=0) or 0
    n_participants = _metric(trajectory, "n_participants", default=0) or 0
    model_type = _metric(train, "selected_model_type", "model_type", default="model")
    eval_strategy = str(_metric(train, "evaluation_strategy", default="recorded")).replace("_", " ")
    incident_supported = _metric(train, "incident_risk_supported", default=None)
    diagnostics = [
        ("Model", f"{model_type} / {eval_strategy}"),
        ("Holdout AUC", f"{auc:.3f}" if auc is not None else "not recorded"),
        ("CV diagnostic AUC", f"{cv_auc:.3f}" if cv_auc is not None else "not recorded"),
        ("Calibration ECE", f"{ece:.3f}" if ece is not None else "not recorded"),
        ("Trajectory tokens", f"{int(n_tokens):,}" if isinstance(n_tokens, (int, float)) else str(n_tokens)),
        ("Tokenized participants", f"{int(n_participants):,}" if isinstance(n_participants, (int, float)) else str(n_participants)),
        ("Incident-risk design", "not built" if incident_supported is False else "not recorded"),
    ]
    ax_metrics.set_title("Execution diagnostics")
    y = 0.94
    for label, value in diagnostics:
        ax_metrics.text(0.02, y, label, transform=ax_metrics.transAxes, fontsize=6, color="#555555")
        ax_metrics.text(0.50, y, value, transform=ax_metrics.transAxes, fontsize=6, fontweight="bold")
        y -= 0.12

    # Panel C: top feature importances.
    top_features = importance.get("top_features") or []
    feature_rows: list[tuple[str, float]] = []
    if isinstance(top_features, list):
        for item in top_features[:6]:
            if isinstance(item, dict):
                feature_rows.append((str(item.get("feature", "feature")), _as_float(item.get("importance"), 0.0) or 0.0))
            else:
                feature_rows.append((str(item), 1.0))
    if not feature_rows and importance.get("top_feature"):
        feature_rows = [(str(importance["top_feature"]), 1.0)]
    if feature_rows:
        labels = [_compact_label(name) for name, _ in reversed(feature_rows)]
        values = [value for _, value in reversed(feature_rows)]
        ax_features.barh(labels, values, color=semantic_color("observed"), alpha=0.9)
        ax_features.set_title("Top model features")
        ax_features.set_xlabel("Tree importance")
    else:
        _plot_no_session(ax_features, "Feature importance not recorded")

    # Panel D: guardrail and claim boundary.
    ax_guardrails.axis("off")
    ax_guardrails.set_title("Guardrails and claim boundary")
    stat_text = str(statistical.get("overall_assessment") or statistical.get("overall") or "not recorded")
    safety_text = str(safety.get("overall") or safety.get("status") or "not recorded")
    world_text = str(world.get("safety_status") or "not recorded")
    allowed = str(world.get("allowed_claim_type") or "association / prediction only").replace("_", " ")
    rows = [
        ("Statistical review", stat_text, "high_risk" if "CRITICAL" in stat_text.upper() else "medium_risk"),
        ("Safety check", safety_text, "medium_risk" if "REVIEW" in safety_text.upper() else "positive"),
        ("World-model audit", world_text, "medium_risk" if "PARTIAL" in world_text.upper() else "positive"),
        ("Allowed claim", allowed, "neutral"),
    ]
    y = 0.88
    for label, value, role in rows:
        ax_guardrails.add_patch(
            plt.Rectangle((0.02, y - 0.045), 0.035, 0.035, transform=ax_guardrails.transAxes,
                          color=semantic_color(role), clip_on=False)
        )
        ax_guardrails.text(0.08, y, label, transform=ax_guardrails.transAxes, fontsize=6, color="#555555")
        ax_guardrails.text(0.42, y, _compact_label(value, 44), transform=ax_guardrails.transAxes, fontsize=6)
        y -= 0.16
    ax_guardrails.text(
        0.02,
        0.04,
        "Interpret as UKB internal prevalent-E11 discrimination and trajectory feasibility; not causal or deployment-ready.",
        transform=ax_guardrails.transAxes,
        fontsize=5.5,
        color="#555555",
        wrap=True,
    )

    fig.subplots_adjust(hspace=0.45, wspace=0.35)


def _plot_km(ax, source_type, source_key, ctx):
    """Kaplan-Meier placeholder — delegates to survival skill."""
    ax.text(0.5, 0.5, f"KM for {source_key}\n(Use survival skill for full KM analysis)",
            ha="center", va="center", transform=ax.transAxes, fontsize=7)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival Probability")
    ax.set_title(f"Survival: {source_key}")
