"""Statistical tests commonly used in biomedical analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def mann_whitney(group1: np.ndarray, group2: np.ndarray) -> dict:
    """Two-sided Mann-Whitney U test."""
    g1 = group1[~np.isnan(group1)]
    g2 = group2[~np.isnan(group2)]
    stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    # Effect size: rank-biserial correlation
    n1, n2 = len(g1), len(g2)
    r = 1 - (2 * stat) / (n1 * n2)
    return {"U": stat, "p_value": p, "effect_size_r": r, "n1": n1, "n2": n2}


def chi2_test(table: np.ndarray) -> dict:
    """Chi-squared test of independence."""
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {"chi2": chi2, "p_value": p, "dof": dof}


def fisher_exact(table: np.ndarray) -> dict:
    """Fisher's exact test (2×2 only)."""
    odds_ratio, p = stats.fisher_exact(table)
    return {"odds_ratio": odds_ratio, "p_value": p}


def bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Bonferroni correction."""
    return np.minimum(p_values * len(p_values), 1.0)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    ranked = np.argsort(p_values)
    adjusted = np.empty(n)
    for i, idx in enumerate(ranked):
        adjusted[idx] = p_values[idx] * n / (i + 1)
    # Enforce monotonicity
    adjusted[ranked] = np.minimum.accumulate(adjusted[ranked][::-1])[::-1]
    return np.minimum(adjusted, 1.0)


def descriptive_stats(series: pd.Series) -> dict:
    """Compute standard descriptive statistics."""
    clean = series.dropna()
    return {
        "n": len(clean),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "median": float(clean.median()),
        "q25": float(clean.quantile(0.25)),
        "q75": float(clean.quantile(0.75)),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "missing": int(series.isna().sum()),
        "missing_pct": float(series.isna().mean() * 100),
    }


def log_rank_test(durations_a, event_a, durations_b, event_b) -> dict:
    """Log-rank test for survival analysis."""
    try:
        from lifelines.statistics import logrank_test
        result = logrank_test(durations_a, durations_b, event_a, event_b)
        return {"test_statistic": result.test_statistic, "p_value": result.p_value}
    except ImportError:
        return {"error": "lifelines not installed"}
