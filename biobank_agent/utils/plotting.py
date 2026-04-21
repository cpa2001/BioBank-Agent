"""Nature-style matplotlib plotting utilities.

All figures follow Nature guidelines:
  - Arial font, 7pt body text, 300 dpi
  - Single column: 89 mm, double column: 183 mm
  - No top/right spines
  - Colour-blind friendly palettes
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ── Nature style ─────────────────────────────────────────────

NATURE_RC = {
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

# mm → inches
SINGLE_COL = 89 / 25.4    # 3.50 in
DOUBLE_COL = 183 / 25.4   # 7.20 in
QUARTER_PAGE = 89 / 25.4   # width = single col, height ≈ same

# Colour-blind safe palette (Okabe & Ito)
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


def apply_nature_style() -> None:
    """Apply Nature rcParams globally."""
    plt.rcParams.update(NATURE_RC)


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
    width : "single" (89mm) or "double" (183mm)
    height_ratio : height = width * height_ratio per subplot row
    """
    apply_nature_style()
    w = SINGLE_COL if width == "single" else DOUBLE_COL
    h = w * height_ratio * nrows / max(ncols, 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)
    return fig, axes


def save_figure(
    fig: plt.Figure,
    name: str,
    report_dir: Path,
    formats: tuple[str, ...] = ("png", "pdf"),
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
