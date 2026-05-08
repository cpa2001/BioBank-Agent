"""Tests for model training, evaluation, calibration, and importance skills."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from biobank_agent.skills import calibration as calibration_mod
from biobank_agent.skills import evaluate_model as evaluate_mod
from biobank_agent.skills import feature_importance as importance_mod
from biobank_agent.skills import train_model as train_mod


class ModelState:
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.feature_matrix = None
        self.labels = None
        self.figures = []
        self.cohorts = {}


def model_ctx(tmp_path):
    return SimpleNamespace(
        report_dir=tmp_path,
        state=ModelState(),
        dm=SimpleNamespace(),
        settings=SimpleNamespace(subject_id_col="eid"),
    )


def fake_save_figure(fig, name, report_dir, formats=("png", "svg")):
    return [report_dir / f"{name}.{fmt}" for fmt in formats]


def fitted_logistic_model():
    X = pd.DataFrame(
        {
            "30740-0.0": np.linspace(0, 1, 60),
            "30750-0.0": np.r_[np.linspace(0, 0.4, 30), np.linspace(0.6, 1, 30)],
        }
    )
    y = pd.Series([0, 1] * 30)
    model = LogisticRegression(solver="liblinear").fit(X, y)
    return model, X, y


def test_evaluate_model_errors_and_success(tmp_path, monkeypatch):
    ctx = model_ctx(tmp_path)
    assert evaluate_mod.evaluate_model(ctx=ctx)["error"] == "No trained model. Run train_model first."

    ctx.state.models["missing_data"] = LogisticRegression()
    assert "No feature matrix" in evaluate_mod.evaluate_model("missing_data", ctx=ctx)["error"]
    assert "not found" in evaluate_mod.evaluate_model("bad-key", ctx=ctx)["error"]

    model, X, y = fitted_logistic_model()
    ctx.state.models["E11_lr"] = model
    ctx.state.feature_matrix = X
    ctx.state.labels = y
    monkeypatch.setattr(evaluate_mod, "save_figure", fake_save_figure)

    result = evaluate_mod.evaluate_model(ctx=ctx)

    assert result["model_key"] == "E11_lr"
    assert len(result["fold_aucs"]) == 5
    assert 0 <= result["mean_auc"] <= 1
    assert 0 <= result["mean_ap"] <= 1
    assert result["figure"].endswith("roc_pr_E11_lr.png")
    assert len(ctx.state.figures) == 2
    plt.close("all")


class FakeImportanceModel:
    feature_importances_ = np.array([0.1, 0.6, 0.3])


def test_feature_importance_errors_tree_and_shap_paths(tmp_path, monkeypatch):
    ctx = model_ctx(tmp_path)
    assert "No trained model" in importance_mod.feature_importance(ctx=ctx)["error"]

    ctx.state.models["E11_tree"] = FakeImportanceModel()
    ctx.state.model_metadata["E11_tree"] = {
        "feature_names": ["30740-0.0", "custom_feature", "30750-0.0"]
    }
    ctx.state.feature_matrix = pd.DataFrame(
        {
            "30740-0.0": [1, 2, 3],
            "custom_feature": [0, 1, 0],
            "30750-0.0": [4, 5, 6],
        }
    )
    monkeypatch.setattr(importance_mod, "save_figure", fake_save_figure)

    missing = importance_mod.feature_importance("bad-key", ctx=ctx)
    tree = importance_mod.feature_importance(top_n=2, ctx=ctx)

    assert "Available" in missing["error"]
    assert tree["importance_type"] == "Tree-based"
    assert tree["top_features"] == [
        {"feature": "custom_feature", "importance": 0.6},
        {"feature": "Glycated haemoglobin (HbA1c)", "importance": 0.3},
    ]

    class FakeTreeExplainer:
        def __init__(self, model):
            self.model = model

        def shap_values(self, X):
            return [
                np.zeros((len(X), 3)),
                np.array([[0.1, 0.4, 0.2], [0.2, 0.3, 0.6], [0.0, 0.5, 0.1]]),
            ]

    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(TreeExplainer=FakeTreeExplainer))
    shap_result = importance_mod.feature_importance("E11_tree", top_n=1, method="shap", ctx=ctx)

    assert shap_result["importance_type"] == "SHAP"
    assert shap_result["top_features"][0]["feature"] == "custom_feature"
    assert len(ctx.state.figures) == 4
    plt.close("all")


def test_feature_importance_handles_3d_shap_and_extra_feature_names(tmp_path, monkeypatch):
    ctx = model_ctx(tmp_path)
    ctx.state.models["E11_tree"] = FakeImportanceModel()
    ctx.state.model_metadata["E11_tree"] = {
        "feature_names": ["a", "b", "c", "extra"]
    }
    ctx.state.feature_matrix = pd.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4]})
    monkeypatch.setattr(importance_mod, "save_figure", fake_save_figure)

    class FakeTreeExplainer:
        def __init__(self, model):
            self.model = model

        def shap_values(self, X):
            return np.dstack([
                np.zeros((len(X), 3)),
                np.array([[0.1, 0.5, 0.2], [0.3, 0.4, 0.6]]),
            ])

    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(TreeExplainer=FakeTreeExplainer))

    result = importance_mod.feature_importance("E11_tree", top_n=4, method="shap", ctx=ctx)

    assert result["importance_type"] == "SHAP"
    assert [item["feature"] for item in result["top_features"]] == ["b", "c", "a"]
    plt.close("all")


def test_feature_importance_shap_fallback_to_tree(tmp_path, monkeypatch):
    ctx = model_ctx(tmp_path)
    ctx.state.models["E11_tree"] = FakeImportanceModel()
    ctx.state.model_metadata["E11_tree"] = {"feature_names": ["a", "b", "c"]}
    ctx.state.feature_matrix = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    monkeypatch.setattr(importance_mod, "save_figure", fake_save_figure)
    monkeypatch.setitem(
        sys.modules,
        "shap",
        SimpleNamespace(TreeExplainer=lambda model: (_ for _ in ()).throw(RuntimeError("bad shap"))),
    )

    result = importance_mod.feature_importance("E11_tree", method="shap", ctx=ctx)

    assert result["importance_type"] == "Tree-based"


def test_calibration_errors_and_success(tmp_path, monkeypatch):
    ctx = model_ctx(tmp_path)
    assert "No trained model" in calibration_mod.calibration(ctx=ctx)["error"]

    model, X, y = fitted_logistic_model()
    ctx.state.models["E11_lr"] = model
    assert "No model/data" in calibration_mod.calibration("missing", ctx=ctx)["error"]

    ctx.state.feature_matrix = X
    ctx.state.labels = y
    monkeypatch.setattr(calibration_mod, "save_figure", fake_save_figure)

    result = calibration_mod.calibration(n_bins=5, ctx=ctx)

    assert result["model_key"] == "E11_lr"
    assert 0 <= result["ece"] <= 1
    assert 0 <= result["mce"] <= 1
    assert 0 <= result["brier_score"] <= 1
    assert result["n_bins"] == 5
    assert result["figure"].endswith("calibration_E11_lr.png")
    assert ctx.state.figures
    plt.close("all")


def training_cohort(n_cases=120, n_controls=480):
    labels = np.array([1] * n_cases + [0] * n_controls)
    n = len(labels)
    return pd.DataFrame(
        {
            "eid": np.arange(n),
            "label": labels,
            "30740-0.0": np.r_[np.linspace(7, 9, n_cases), np.linspace(4, 6, n_controls)],
            "30750-0.0": np.r_[np.linspace(60, 80, n_cases), np.linspace(35, 48, n_controls)],
            "constant": 1.0,
            "notes": ["case" if label else "control" for label in labels],
        }
    )


def test_train_model_insufficient_cases_and_successful_storage(tmp_path, monkeypatch):
    ctx = model_ctx(tmp_path)
    monkeypatch.setattr(train_mod, "build_cohort", lambda *args, **kwargs: training_cohort(n_cases=5, n_controls=20))

    small = train_mod.train_model("E11", ctx=ctx)
    assert small["error"] == "Insufficient cases (n=5). Minimum 100 required."

    ctx = model_ctx(tmp_path)
    estimators = [SimpleNamespace(name="fold0"), SimpleNamespace(name="fold1"), SimpleNamespace(name="fold2")]
    monkeypatch.setattr(train_mod, "build_cohort", lambda *args, **kwargs: training_cohort())
    monkeypatch.setattr(train_mod, "_get_estimator", lambda model_type: SimpleNamespace(model_type=model_type))
    monkeypatch.setattr(
        train_mod,
        "cross_validate",
        lambda *args, **kwargs: {
            "test_auc": np.array([0.71, 0.82, 0.76]),
            "test_f1": np.array([0.5, 0.6, 0.7]),
            "test_precision": np.array([0.4, 0.5, 0.6]),
            "test_recall": np.array([0.7, 0.8, 0.9]),
            "estimator": estimators,
        },
    )

    result = train_mod.train_model("E11", model_type="xgb", n_folds=3, ctx=ctx)

    assert result["model_key"] == "E11_xgb"
    assert result["n_cases"] == 120
    assert result["n_controls"] == 480
    assert result["n_features"] == 2
    assert result["fold_aucs"] == [0.71, 0.82, 0.76]
    assert ctx.state.models["E11_xgb"] is estimators[1]
    assert ctx.state.model_metadata["E11_xgb"]["feature_names"] == ["30740-0.0", "30750-0.0"]
    assert ctx.state.feature_matrix.isna().sum().sum() == 0


def test_train_model_reuses_cached_cohort_and_rejects_unknown_estimator(tmp_path, monkeypatch):
    ctx = model_ctx(tmp_path)
    with pytest.raises(ValueError, match="Unknown model type"):
        train_mod._get_estimator("bad")

    ctx.state.cohorts["E11_1:4"] = training_cohort()
    monkeypatch.setattr(train_mod, "_get_estimator", lambda model_type: SimpleNamespace(model_type=model_type))
    monkeypatch.setattr(
        train_mod,
        "cross_validate",
        lambda *args, **kwargs: {
            "test_auc": np.array([0.8, 0.81]),
            "test_f1": np.array([0.7, 0.71]),
            "test_precision": np.array([0.6, 0.61]),
            "test_recall": np.array([0.5, 0.51]),
            "estimator": [SimpleNamespace(name="a"), SimpleNamespace(name="b")],
        },
    )

    result = train_mod.train_model("E11", model_type="lgbm", n_folds=2, ctx=ctx)

    assert result["model_key"] == "E11_lgbm"
    assert ctx.state.models["E11_lgbm"].name == "b"


def test_get_estimator_optional_model_backends(monkeypatch):
    class FakeXGBClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLGBMClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCatBoostClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBClassifier=FakeXGBClassifier))
    monkeypatch.setitem(sys.modules, "lightgbm", SimpleNamespace(LGBMClassifier=FakeLGBMClassifier))
    monkeypatch.setitem(sys.modules, "catboost", SimpleNamespace(CatBoostClassifier=FakeCatBoostClassifier))

    xgb = train_mod._get_estimator("xgb")
    lgbm = train_mod._get_estimator("lgbm")
    catboost = train_mod._get_estimator("catboost")

    assert isinstance(xgb, FakeXGBClassifier)
    assert xgb.kwargs["eval_metric"] == "logloss"
    assert isinstance(lgbm, FakeLGBMClassifier)
    assert lgbm.kwargs["is_unbalance"] is True
    assert isinstance(catboost, FakeCatBoostClassifier)
    assert catboost.kwargs["auto_class_weights"] == "Balanced"
