"""Edge coverage for patient-level prediction internals."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from biobank_agent.skills.predict import (
    _fetch_patient_features,
    _plot_risk_distribution,
    predict,
)


def _ctx(feature_names, X=None, y=None, tmp_path=None):
    model = MagicMock()
    ctx = SimpleNamespace(
        settings=SimpleNamespace(subject_id_col="eid"),
        state=SimpleNamespace(
            models={"E11:xgboost": model},
            model_metadata={"E11:xgboost": {"feature_names": feature_names}},
            feature_matrix=X,
            labels=y,
            figures=[],
        ),
        report_dir=tmp_path,
    )
    return ctx, model


@patch("biobank_agent.skills.predict._plot_risk_distribution", return_value=[])
@patch("biobank_agent.skills.predict._fetch_patient_features")
def test_specific_eids_repair_missing_columns_and_impute_median(mock_fetch, _mock_plot):
    feature_names = ["f1", "f2", "f3"]
    mock_fetch.return_value = pd.DataFrame(
        {"f2": [3.0, 4.0], "f1": [1.0, np.nan]},
        index=["100", "200"],
    )
    ctx, model = _ctx(feature_names)

    captured = {}

    def _predict_proba(X_in):
        captured["X"] = X_in.copy()
        return np.array([[0.8, 0.2], [0.1, 0.9]])

    model.predict_proba.side_effect = _predict_proba

    result = predict("E11:xgboost", patient_eids="100, 200", ctx=ctx)

    assert "error" not in result
    assert list(captured["X"].columns) == feature_names
    assert captured["X"].loc["200", "f1"] == 1.0
    assert np.isnan(captured["X"].loc["100", "f3"])
    assert [p["eid"] for p in result["predictions"]] == ["100", "200"]


def test_prediction_exception_returns_error():
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=list("abcde"))
    y = pd.Series([0, 1, 0, 1, 0], index=X.index)
    ctx, model = _ctx(["f1"], X=X, y=y)
    model.predict_proba.side_effect = RuntimeError("scoring failed")

    result = predict("E11:xgboost", ctx=ctx)

    assert result == {"error": "Prediction failed: scoring failed"}


@patch("biobank_agent.skills.predict._plot_risk_distribution", return_value=[])
def test_prediction_output_is_capped_but_patient_count_is_complete(_mock_plot):
    n = 125
    X = pd.DataFrame({"f1": np.linspace(0, 1, n)}, index=[str(i) for i in range(n)])
    y = pd.Series([0, 1] * 62 + [1], index=X.index)
    ctx, model = _ctx(["f1"], X=X, y=y)
    probs = np.linspace(0.05, 0.95, 25)
    model.predict_proba.return_value = np.column_stack([1 - probs, probs])

    result = predict("E11:xgboost", ctx=ctx)

    assert result["n_patients"] == 25
    assert len(result["predictions"]) == 20
    assert all("actual_label" in pred for pred in result["predictions"])


@patch("biobank_agent.skills.predict._plot_risk_distribution", return_value=[])
def test_auc_failure_is_ignored(_mock_plot, monkeypatch):
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=list("abcde"))
    y = pd.Series([0, 1, 0, 1, 0], index=X.index)
    ctx, model = _ctx(["f1"], X=X, y=y)
    model.predict_proba.return_value = np.array([[0.4, 0.6]])

    def _raise_auc(*_args, **_kwargs):
        raise ValueError("auc unavailable")

    monkeypatch.setattr("sklearn.metrics.roc_auc_score", _raise_auc)

    result = predict("E11:xgboost", ctx=ctx)

    assert "error" not in result
    assert "test_auc" not in result


def test_fetch_patient_features_uses_biomarker_group_ids():
    ctx = SimpleNamespace(settings=SimpleNamespace(subject_id_col="eid"))
    ctx.dm = MagicMock()
    ctx.dm.get_field.return_value = pd.DataFrame(
        {"eid": [100, 200], "30600-0.0": [5.1, 6.2]}
    )

    result = _fetch_patient_features(["100"], ["30600-0.0"], ctx)

    ctx.dm.get_field.assert_called_once_with("30600")
    assert result.index.tolist() == [100]
    assert result["30600-0.0"].tolist() == [5.1]


def test_fetch_patient_features_fallback_parses_numeric_prefix_and_skips_failures():
    ctx = SimpleNamespace(settings=SimpleNamespace(subject_id_col="eid"))

    def _get_field(fid):
        if fid == "88888":
            raise RuntimeError("field offline")
        return pd.DataFrame({"eid": ["100", "200"], f"{fid}-0.0": [1.0, 2.0]})

    ctx.dm = SimpleNamespace(get_field=_get_field)

    result = _fetch_patient_features(["100"], ["99999-0.0", "88888-0.0"], ctx)

    assert result.index.tolist() == ["100"]
    assert "99999-0.0" in result.columns


def test_fetch_patient_features_raises_when_no_fields_found():
    ctx = SimpleNamespace(settings=SimpleNamespace(subject_id_col="eid"))
    ctx.dm = SimpleNamespace(get_field=lambda _fid: None)

    try:
        _fetch_patient_features(["100"], ["not_a_field"], ctx)
    except ValueError as exc:
        assert "No features found" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_fetch_patient_features_skips_nonnumeric_prefixes_and_none_fields():
    ctx = SimpleNamespace(settings=SimpleNamespace(subject_id_col="eid"))
    ctx.dm = SimpleNamespace(get_field=lambda _fid: None)

    try:
        _fetch_patient_features(["100"], ["abc-0.0", "99999-0.0"], ctx)
    except ValueError as exc:
        assert "No features found" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_plot_risk_distribution_writes_svg_and_pdf(tmp_path):
    paths = _plot_risk_distribution(
        np.array([0.1, 0.35, 0.8]),
        ["low", "medium", "high"],
        "E11:xgboost",
        5,
        SimpleNamespace(report_dir=tmp_path),
    )

    assert {p.suffix for p in paths} == {".svg", ".pdf"}
    assert all(p.exists() for p in paths)
    assert all("E11_xgboost" in p.name for p in paths)
