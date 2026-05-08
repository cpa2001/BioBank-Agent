"""Tests for evaluation metric helpers."""

import builtins
from pathlib import Path

from biobank_agent.eval.metrics import (
    answer_contains,
    figure_quality_score,
    report_quality_score,
    skill_call_accuracy,
)


def test_answer_contains_handles_empty_and_partial_matches():
    assert answer_contains("anything", []) == 1.0
    assert answer_contains("HbA1c and BMI predict E11", ["hba1c", "ldl"]) == 0.5
    assert answer_contains("", ["missing"]) == 0.0


def test_skill_call_accuracy_handles_empty_and_overlap():
    assert skill_call_accuracy(["prevalence"], []) == 1.0
    assert skill_call_accuracy(["prevalence", "train_model"], ["train_model", "report"]) == 0.5
    assert skill_call_accuracy([], ["report"]) == 0.0


def test_report_quality_score_detects_scientific_sections():
    report = """# Title

## Abstract
Results include AUC = 0.81 and 95% CI: 0.78-0.84.

## Methods
Statistical analysis used cross-validation.

## Results
![Figure 1](fig.png)

## References
1. Example.
"""
    scores = report_quality_score(report)

    assert scores["has_title"] == 1.0
    assert scores["has_statistics"] == 1.0
    assert scores["has_figures"] == 1.0
    assert scores["overall"] == 1.0


def test_report_quality_score_penalizes_sparse_text():
    scores = report_quality_score("no sections")

    assert scores["has_title"] == 0.0
    assert scores["has_references"] == 0.0
    assert 0.0 <= scores["overall"] < 0.5


def test_figure_quality_score_for_missing_and_existing_files(tmp_path):
    missing = figure_quality_score(tmp_path / "missing.svg")
    assert missing["exists"] is False
    assert missing["non_empty"] is False

    svg = tmp_path / "figure.svg"
    svg.write_text("<svg>" + ("x" * 150) + "</svg>")
    existing = figure_quality_score(svg)
    assert existing == {
        "exists": True,
        "non_empty": True,
        "correct_format": True,
    }

    txt = tmp_path / "figure.txt"
    txt.write_text("x" * 150)
    assert figure_quality_score(txt)["correct_format"] is False


def test_figure_quality_score_png_and_missing_pillow_paths(tmp_path, monkeypatch):
    from PIL import Image

    png = tmp_path / "figure.png"
    Image.new("RGB", (320, 240), "white").save(png, dpi=(300, 300))

    png_checks = figure_quality_score(png)
    assert png_checks["dpi_300"] is True
    assert png_checks["min_width_300px"] is True

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("Pillow unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    no_pillow = figure_quality_score(png)
    assert no_pillow["dpi_300"] is None
