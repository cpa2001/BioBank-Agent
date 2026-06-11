"""Focused coverage for statistical, ICD-10, and multimodal utilities."""

from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from biobank_agent.interfaces.multimodal import SimpleMultimodalGrounder
from biobank_agent.utils import icd10, stats


def test_mann_whitney_ignores_nan_and_reports_effect_size():
    result = stats.mann_whitney(
        np.array([1.0, np.nan, 2.0]),
        np.array([3.0, 4.0, np.nan]),
    )

    assert result["n1"] == 2
    assert result["n2"] == 2
    assert result["U"] == 0
    assert result["effect_size_r"] == pytest.approx(1.0)
    assert 0 <= result["p_value"] <= 1


def test_chi2_and_fisher_exact_return_expected_shapes():
    chi2 = stats.chi2_test(np.array([[8, 2], [1, 9]]))
    fisher = stats.fisher_exact(np.array([[8, 2], [1, 9]]))

    assert set(chi2) == {"chi2", "p_value", "dof"}
    assert chi2["dof"] == 1
    assert 0 <= chi2["p_value"] <= 1
    assert fisher["odds_ratio"] > 1
    assert 0 <= fisher["p_value"] <= 1


def test_multiple_testing_corrections_are_bounded_and_monotone():
    p_values = np.array([0.04, 0.001, 0.2, 0.8])

    bonf = stats.bonferroni(p_values)
    bh = stats.benjamini_hochberg(p_values)

    assert np.all((bonf >= 0) & (bonf <= 1))
    assert np.all((bh >= 0) & (bh <= 1))
    assert bonf.tolist() == pytest.approx([0.16, 0.004, 0.8, 1.0])
    order = np.argsort(p_values)
    assert np.all(np.diff(bh[order]) >= -1e-12)


def test_descriptive_stats_handles_missing_values():
    result = stats.descriptive_stats(pd.Series([1.0, 2.0, np.nan, 5.0]))

    assert result["n"] == 3
    assert result["missing"] == 1
    assert result["missing_pct"] == pytest.approx(25.0)
    assert result["median"] == 2.0
    assert result["min"] == 1.0
    assert result["max"] == 5.0


def test_log_rank_reports_missing_lifelines(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lifelines.statistics":
            raise ImportError("blocked for deterministic test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = stats.log_rank_test([1, 2], [1, 0], [1, 3], [1, 1])

    assert result == {"error": "lifelines not installed"}


def test_log_rank_success_with_fake_lifelines(monkeypatch):
    fake_statistics = ModuleType("lifelines.statistics")
    fake_statistics.logrank_test = lambda *args: SimpleNamespace(test_statistic=3.2, p_value=0.04)
    monkeypatch.setitem(sys.modules, "lifelines", ModuleType("lifelines"))
    monkeypatch.setitem(sys.modules, "lifelines.statistics", fake_statistics)

    result = stats.log_rank_test([1, 2], [1, 0], [1, 3], [1, 1])

    assert result == {"test_statistic": 3.2, "p_value": 0.04}


def test_icd10_exact_prefix_chapter_and_unknown_cases():
    assert icd10.icd10_name("E11") == "Type 2 diabetes mellitus"
    assert icd10.icd10_name("E11.9") == "Type 2 diabetes mellitus"
    assert icd10.icd10_name("U07") == "U07"

    assert icd10.icd10_chapter("i25") == (
        "IX",
        "Diseases of the circulatory system",
    )
    assert icd10.icd10_chapter("") == ("?", "Unknown")
    assert icd10.icd10_chapter("1BAD") == ("?", "Unknown")

    assert icd10.is_same_chapter("I10", "I25")
    assert not icd10.is_same_chapter("I10", "E11")


def test_multimodal_fusion_empty_concat_zscore_and_error_paths():
    grounder = SimpleMultimodalGrounder()

    assert grounder.supported_methods() == ["concat", "zscore_concat"]
    assert grounder.fuse({}).shape == (0, 0)
    assert grounder.fuse({"empty": np.array([])}).shape == (0, 0)

    tabular = np.array([[1.0, 2.0], [3.0, 2.0]])
    imaging = np.array([[10.0], [14.0]])

    concat = grounder.fuse({"tabular": tabular, "imaging": imaging})
    assert concat.shape == (2, 3)
    assert concat.tolist() == [[1.0, 2.0, 10.0], [3.0, 2.0, 14.0]]

    zscored = grounder.fuse({"tabular": tabular, "imaging": imaging}, method="zscore_concat")
    assert zscored.shape == (2, 3)
    assert zscored[:, 1].tolist() == [0.0, 0.0]
    assert zscored[:, 0].tolist() == pytest.approx([-1.0, 1.0])

    with pytest.raises(ValueError, match="Unsupported fusion method"):
        grounder.fuse({"tabular": tabular}, method="attention")


def test_multimodal_structural_signals_cover_table_and_tensor_paths():
    grounder = SimpleMultimodalGrounder()
    table = pd.DataFrame(
        {
            "stable": [1.0, 1.0, 1.0, 1.0],
            "marker": [0.0, 0.0, 0.0, 100.0],
            "category": ["a", "b", None, "d"],
        }
    )

    signals = grounder.extract_structural_signals(
        table=table,
        modality_tensors={"scan": np.array([[0.0, 2.0], [4.0, 6.0]]), "empty": np.array([])},
    )

    by_type = {(s.modality, s.signal_type): s for s in signals}
    assert by_type[("table", "missingness_ratio")].value == pytest.approx(1 / 12)
    assert by_type[("table", "outlier_ratio")].detail == "fraction of z-score > 3"
    assert by_type[("scan", "variance_proxy")].value == pytest.approx(np.std([[0.0, 2.0], [4.0, 6.0]]))
    assert ("empty", "variance_proxy") not in by_type

    no_table_signals = grounder.extract_structural_signals()
    categorical_only = grounder.extract_structural_signals(table=pd.DataFrame({"category": ["a", None]}))
    assert no_table_signals == []
    assert [(s.modality, s.signal_type) for s in categorical_only] == [("table", "missingness_ratio")]


def test_figure_to_tensor_missing_file_and_pillow_fallback(tmp_path, monkeypatch):
    grounder = SimpleMultimodalGrounder()

    assert grounder.figure_to_tensor(str(tmp_path / "missing.png")).shape == (0, 0)

    fake_image_module = ModuleType("PIL.Image")
    fake_image_module.open = lambda path: (_ for _ in ()).throw(RuntimeError("decode failed"))
    fake_pil = SimpleNamespace(Image=fake_image_module)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_module)

    figure_path = tmp_path / "figure.bin"
    figure_path.write_bytes(b"not a real image")

    tensor = grounder.figure_to_tensor(str(figure_path))

    assert tensor.shape == (1, 2)
    assert tensor[0, 0] == figure_path.stat().st_size


def test_figure_to_tensor_pillow_success_path(tmp_path, monkeypatch):
    grounder = SimpleMultimodalGrounder()

    class FakeImage:
        def convert(self, mode):
            assert mode == "L"
            return self

        def resize(self, size):
            assert size == (128, 128)
            return np.ones(size, dtype=np.float32) * 255

    fake_image_module = ModuleType("PIL.Image")
    fake_image_module.open = lambda path: FakeImage()
    fake_pil = SimpleNamespace(Image=fake_image_module)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_module)

    figure_path = tmp_path / "figure.png"
    figure_path.write_bytes(b"fake")

    tensor = grounder.figure_to_tensor(str(figure_path))

    assert tensor.shape == (1, 128 * 128)
    assert tensor.max() == pytest.approx(1.0)
