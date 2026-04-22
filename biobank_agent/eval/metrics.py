"""Evaluation metrics for Biobank Agent assessment.

Provides scoring functions for:
  - Answer accuracy (fuzzy text matching)
  - Skill call accuracy (correct tools called)
  - Report quality (structure, statistics, figures)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def answer_contains(response: str, expected: list[str]) -> float:
    """Score how many expected terms appear in the response.

    Returns 0.0-1.0 ratio of found terms.
    """
    if not expected:
        return 1.0
    response_lower = response.lower()
    found = sum(1 for term in expected if term.lower() in response_lower)
    return found / len(expected)


def skill_call_accuracy(
    actual_skills: list[str],
    expected_skills: list[str],
) -> float:
    """Score whether the expected skills were called.

    Returns 1.0 if all expected skills are in actual, 0.0 if none.
    """
    if not expected_skills:
        return 1.0
    actual_set = set(actual_skills)
    expected_set = set(expected_skills)
    return len(actual_set & expected_set) / len(expected_set)


def report_quality_score(report_text: str) -> dict[str, float]:
    """Score a generated report on structural quality.

    Returns a dict of component scores (0.0-1.0):
    - has_title: Report has a title heading
    - has_abstract: Has abstract or executive summary
    - has_methods: Has methodology section
    - has_results: Has results with metrics
    - has_figures: References figures
    - has_statistics: Contains statistical values (P, CI, AUC)
    - has_references: Has references section
    - overall: Mean of all components
    """
    text_lower = report_text.lower()

    scores = {}
    scores["has_title"] = 1.0 if report_text.startswith("#") else 0.0
    scores["has_abstract"] = 1.0 if any(k in text_lower for k in
        ("abstract", "executive summary", "key findings")) else 0.0
    scores["has_methods"] = 1.0 if any(k in text_lower for k in
        ("method", "methodology", "statistical analysis")) else 0.0
    scores["has_results"] = 1.0 if "result" in text_lower else 0.0
    scores["has_figures"] = 1.0 if any(k in text_lower for k in
        ("figure", "fig.", "![")) else 0.0
    scores["has_statistics"] = 1.0 if any(k in text_lower for k in
        ("p <", "p =", "auc", "95% ci", "confidence interval")) else 0.0
    scores["has_references"] = 1.0 if "reference" in text_lower else 0.0

    scores["overall"] = sum(scores.values()) / len(scores)
    return scores


def figure_quality_score(fig_path: Path) -> dict[str, bool]:
    """Check a figure file for quality standards.

    Returns dict of quality checks.
    """
    checks = {
        "exists": fig_path.exists(),
        "non_empty": fig_path.stat().st_size > 100 if fig_path.exists() else False,
        "correct_format": fig_path.suffix in (".svg", ".pdf", ".png"),
    }

    if fig_path.suffix == ".png" and fig_path.exists():
        try:
            from PIL import Image
            img = Image.open(fig_path)
            dpi = img.info.get("dpi", (72, 72))
            checks["dpi_300"] = dpi[0] >= 299
            checks["min_width_300px"] = img.width >= 300
        except ImportError:
            checks["dpi_300"] = None  # Can't check without Pillow

    return checks
