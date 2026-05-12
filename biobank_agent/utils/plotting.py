"""Publication-quality matplotlib plotting utilities.

Supports two style presets:
  - **Nature**: Arial 7 pt, 89/183 mm columns, 300 dpi, no top/right spines
  - **ICML**: Times 10 pt, two-column (6.75 in text-width), single-spaced

Colour-blind safe palettes (Okabe & Ito for categorical, viridis/RdBu_r for
sequential/diverging).

All figures default to SVG + PDF output for publication workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ── Palettes ────────────────────────────────────────────────────

# Colour-blind safe categorical palette (Okabe & Ito)
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#CC79A7",  # pink
    "#F0E442",  # yellow
    "#000000",  # black
]

PALETTE_SEQ = "viridis"    # sequential colourmap
PALETTE_DIV = "RdBu_r"     # diverging colourmap

# Stable semantic colours for common biomedical plot roles. These helpers keep
# case/control, risk, and significance colours consistent across skills.
SEMANTIC_PALETTE = {
    "primary": PALETTE[0],
    "secondary": PALETTE[4],
    "control": PALETTE[0],
    "controls": PALETTE[0],
    "case": PALETTE[1],
    "cases": PALETTE[1],
    "positive": PALETTE[2],
    "negative": PALETTE[1],
    "neutral": "#999999",
    "low_risk": PALETTE[0],
    "medium_risk": PALETTE[3],
    "high_risk": PALETTE[1],
    "significant": PALETTE[1],
    "non_significant": "#999999",
    "observed": PALETTE[0],
    "expected": "#666666",
    "reference": "#666666",
    "missing": "#BDBDBD",
}

SEMANTIC_ALIASES = {
    "label_0": "control",
    "label_1": "case",
    "non-significant": "non_significant",
    "nonsignificant": "non_significant",
    "risk_low": "low_risk",
    "risk_medium": "medium_risk",
    "risk_high": "high_risk",
}


def _normalise_semantic_role(role: str) -> str:
    key = str(role).strip().lower().replace(" ", "_").replace("-", "_")
    return SEMANTIC_ALIASES.get(key, key)


def semantic_color(role: str, default: str | None = None) -> str:
    """Return the stable colour assigned to a semantic plot role."""
    key = _normalise_semantic_role(role)
    if key in SEMANTIC_PALETTE:
        return SEMANTIC_PALETTE[key]
    return default if default is not None else PALETTE[0]


def semantic_palette(
    roles: Sequence[str],
    default_cycle: Sequence[str] | None = None,
) -> dict[str, str]:
    """Map semantic roles to stable colours, cycling for unknown roles."""
    cycle = list(default_cycle or PALETTE)
    assigned: dict[str, str] = {}
    unknown_idx = 0
    for role in roles:
        key = _normalise_semantic_role(role)
        if key in SEMANTIC_PALETTE:
            assigned[str(role)] = SEMANTIC_PALETTE[key]
        else:
            assigned[str(role)] = cycle[unknown_idx % len(cycle)]
            unknown_idx += 1
    return assigned


def case_control_palette() -> dict[str, str]:
    """Return the canonical control/case colour mapping."""
    return semantic_palette(["control", "case"])

# ── Nature style ────────────────────────────────────────────────

NATURE_RC: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# mm -> inches
SINGLE_COL = 89 / 25.4    # 3.50 in
DOUBLE_COL = 183 / 25.4   # 7.20 in
QUARTER_PAGE = 89 / 25.4   # width = single col, height ~ same

# ── ICML style ──────────────────────────────────────────────────

ICML_RC: dict = {
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

ICML_TEXTWIDTH = 6.75      # inches (two-column)
ICML_ROW_HEIGHT = 2.5      # inches per subplot row


# ── Style application ───────────────────────────────────────────

def apply_nature_style() -> None:
    """Apply Nature rcParams globally."""
    plt.rcParams.update(NATURE_RC)


def apply_icml_style() -> None:
    """Apply ICML rcParams globally."""
    plt.rcParams.update(ICML_RC)


# ── Figure constructors ─────────────────────────────────────────

def nature_figure(
    nrows: int = 1,
    ncols: int = 1,
    width: str = "single",
    height_ratio: float = 0.75,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray | plt.Axes]:
    """Create a figure with Nature dimensions.

    Parameters
    ----------
    width : "single" (89 mm) or "double" (183 mm)
    height_ratio : height = width * height_ratio per subplot row
    """
    apply_nature_style()
    w = SINGLE_COL if width == "single" else DOUBLE_COL
    h = w * height_ratio * nrows / max(ncols, 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)
    return fig, axes


def icml_figure(
    nrows: int = 1,
    ncols: int = 1,
    width: float | None = None,
    row_height: float | None = None,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray | plt.Axes]:
    """Create a figure with ICML two-column dimensions.

    Parameters
    ----------
    width : figure width in inches (default: ``ICML_TEXTWIDTH``, 6.75 in)
    row_height : height per subplot row in inches (default: 2.5 in)
    """
    apply_icml_style()
    w = width if width is not None else ICML_TEXTWIDTH
    rh = row_height if row_height is not None else ICML_ROW_HEIGHT
    h = rh * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)
    return fig, axes


# ── Save utility ────────────────────────────────────────────────

def save_figure(
    fig: plt.Figure,
    name: str,
    report_dir: Path,
    formats: tuple[str, ...] = ("svg", "pdf"),
) -> list[Path]:
    """Save figure in multiple formats, return paths."""
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = report_dir / f"{name}.{fmt}"
        fig.savefig(p, format=fmt)
        paths.append(p)
    plt.close(fig)
    return paths


# ── Panel labels ────────────────────────────────────────────────

def add_panel_labels(
    fig: plt.Figure,
    axes: np.ndarray | Sequence[plt.Axes] | plt.Axes,
    labels: Sequence[str] | None = None,
) -> None:
    """Add bold **(a)**, **(b)**, **(c)** labels to multi-panel figures.

    Labels are placed at the top-left corner of each axes panel.

    Parameters
    ----------
    fig : matplotlib Figure
    axes : single Axes, flat array / list of Axes
    labels : custom label strings; defaults to ``(a), (b), (c), ...``
    """
    ax_list: list[plt.Axes]
    if isinstance(axes, plt.Axes):
        ax_list = [axes]
    elif isinstance(axes, np.ndarray):
        ax_list = list(axes.flatten())
    else:
        ax_list = list(axes)

    if labels is None:
        labels = [f"({chr(ord('a') + i)})" for i in range(len(ax_list))]

    for ax, label in zip(ax_list, labels):
        if not ax.get_visible():
            continue
        ax.text(
            -0.1, 1.1, label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )


# ── Smart legend ────────────────────────────────────────────────

def smart_legend(ax: plt.Axes, ncol: int | None = None, **kwargs) -> plt.Legend:
    """Place legend inside or outside axes depending on item count.

    * <= 5 items  -> inside axes (upper right, frameon=False)
    * > 5 items   -> outside axes (to the right)

    Parameters
    ----------
    ncol : explicit column count; auto-chosen if *None*.
    **kwargs : forwarded to ``ax.legend()``.

    Returns
    -------
    matplotlib.legend.Legend
    """
    handles, labels_text = ax.get_legend_handles_labels()
    n_items = len(handles)

    if n_items == 0:
        # Nothing to draw; return a no-op legend to keep API consistent
        return ax.legend([], [])

    if n_items <= 5:
        # Inside axes
        _ncol = ncol if ncol is not None else 1
        kw = dict(frameon=False, loc="upper right", ncol=_ncol)
        kw.update(kwargs)
        return ax.legend(**kw)
    else:
        # Outside axes, to the right
        _ncol = ncol if ncol is not None else 1
        kw = dict(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=_ncol,
        )
        kw.update(kwargs)
        return ax.legend(**kw)


# ── Publication heatmap ─────────────────────────────────────────

def publication_heatmap(
    data: np.ndarray,
    ax: plt.Axes,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    cbar: bool = True,
    cbar_label: str = "",
    xticklabels: Sequence[str] | None = None,
    yticklabels: Sequence[str] | None = None,
    annot: bool = False,
    fmt: str = ".2f",
    **kwargs,
) -> matplotlib.image.AxesImage:
    """Draw a publication-ready heatmap with properly sized colourbar.

    Parameters
    ----------
    data : 2-D array-like
    ax : target Axes
    cmap, vmin, vmax, center : colour-map controls
    cbar : whether to draw a colourbar
    cbar_label : label for the colourbar
    xticklabels, yticklabels : tick-label sequences
    annot : annotate cells with numeric values
    fmt : format string for annotations
    **kwargs : forwarded to ``ax.imshow()``.

    Returns
    -------
    matplotlib.image.AxesImage
    """
    arr = np.asarray(data, dtype=float)

    # Resolve centre-based normalisation
    if center is not None:
        abs_max = max(
            abs(np.nanmin(arr) - center) if vmin is None else abs(vmin - center),
            abs(np.nanmax(arr) - center) if vmax is None else abs(vmax - center),
        )
        vmin = center - abs_max if vmin is None else vmin
        vmax = center + abs_max if vmax is None else vmax
    else:
        if vmin is None:
            vmin = float(np.nanmin(arr))
        if vmax is None:
            vmax = float(np.nanmax(arr))

    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    # Colourbar with correct proportioning (5 % of axes width)
    if cbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cb = ax.figure.colorbar(im, cax=cax)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=plt.rcParams.get("axes.labelsize", 7))

    # Tick labels
    n_rows, n_cols = arr.shape
    if xticklabels is not None:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    if yticklabels is not None:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(yticklabels)

    # Cell annotations
    if annot:
        for i in range(n_rows):
            for j in range(n_cols):
                val = arr[i, j]
                if np.isnan(val):
                    continue
                text_color = "white" if abs(val - center if center else val) > (vmax - vmin) * 0.4 else "black"
                ax.text(j, i, format(val, fmt), ha="center", va="center",
                        fontsize=max(plt.rcParams.get("font.size", 7) - 2, 4),
                        color=text_color)

    return im


# ── Significance bracket (kept for backward compat) ────────────

def add_significance_bracket(
    ax: plt.Axes,
    x1: float, x2: float,
    y: float,
    p_value: float,
    height: float = 0.02,
) -> None:
    """Add a significance bracket with stars."""
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "n.s."

    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=0.5, c="k")
    ax.text((x1 + x2) / 2, y + height, stars,
            ha="center", va="bottom", fontsize=6)
