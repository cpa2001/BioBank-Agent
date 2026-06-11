"""Test patient-level prediction skill."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, PropertyMock


class TestPredictImport:
    """Test predict skill registration and import."""

    def test_predict_imports(self):
        """Test that predict skill can be imported."""
        from biobank_agent.skills.predict import predict
        assert callable(predict)
        assert predict._skill_name == "predict"

    def test_predict_schema_has_required_params(self):
        """Test predict schema declares model_key as required."""
        from biobank_agent.skills.predict import predict
        schema = predict._skill_schema
        func_def = schema["function"]
        assert "model_key" in func_def["parameters"]["properties"]
        assert "patient_eids" in func_def["parameters"]["properties"]
        assert "model_key" in func_def["parameters"]["required"]


class TestPredictMissingModel:
    """Test error paths when model or metadata is missing."""

    def test_missing_model_key_returns_error(self):
        """Model key not in session state → error with available list."""
        from biobank_agent.skills.predict import predict

        ctx = MagicMock()
        ctx.state.models = {"E11:xgboost": MagicMock()}

        result = predict(model_key="I25:lightgbm", ctx=ctx)

        assert "error" in result
        assert "I25:lightgbm" in result["error"]
        assert "E11:xgboost" in result["error"]

    def test_missing_feature_names_returns_error(self):
        """Model metadata lacks feature_names → error."""
        from biobank_agent.skills.predict import predict

        ctx = MagicMock()
        ctx.state.models = {"E11:xgboost": MagicMock()}
        ctx.state.model_metadata = {"E11:xgboost": {}}

        result = predict(model_key="E11:xgboost", ctx=ctx)

        assert "error" in result
        assert "feature_names" in result["error"]


class TestPredictUnseen:
    """Test prediction on unseen (held-out) test set."""

    def _make_ctx(self, n=100, n_features=5):
        """Build a mock context with feature_matrix, labels, and a model."""
        rng = np.random.RandomState(42)
        feature_names = [f"feat_{i}" for i in range(n_features)]

        X = pd.DataFrame(
            rng.randn(n, n_features), columns=feature_names,
            index=[str(i) for i in range(n)],
        )
        y = pd.Series(rng.randint(0, 2, size=n), index=X.index)

        model = MagicMock()
        # predict_proba must return rows matching input size
        def _fake_proba(X_in):
            r = rng.rand(len(X_in))
            return np.column_stack([1 - r, r])
        model.predict_proba.side_effect = _fake_proba

        ctx = MagicMock()
        ctx.state.models = {"E11:xgboost": model}
        ctx.state.model_metadata = {
            "E11:xgboost": {"feature_names": feature_names}
        }
        ctx.state.feature_matrix = X
        ctx.state.labels = y
        ctx.state.figures = []
        ctx.report_dir = MagicMock()
        return ctx, model

    @patch("biobank_agent.skills.predict._plot_risk_distribution", return_value=[])
    def test_unseen_returns_predictions(self, mock_plot):
        """Unseen mode uses last 20% of feature matrix."""
        from biobank_agent.skills.predict import predict

        ctx, _ = self._make_ctx(n=100)
        result = predict(model_key="E11:xgboost", patient_eids="unseen", ctx=ctx)

        assert "error" not in result
        assert result["n_patients"] == 20  # 100 // 5
        assert "predictions" in result
        assert result["mean_risk"] >= 0

    @patch("biobank_agent.skills.predict._plot_risk_distribution", return_value=[])
    def test_unseen_no_feature_matrix_error(self, mock_plot):
        """Unseen mode with no feature_matrix → error."""
        from biobank_agent.skills.predict import predict

        ctx, _ = self._make_ctx()
        ctx.state.feature_matrix = None
        ctx.state.labels = None

        result = predict(model_key="E11:xgboost", patient_eids="unseen", ctx=ctx)

        assert "error" in result
        assert "feature matrix" in result["error"].lower()

    @patch("biobank_agent.skills.predict._plot_risk_distribution", return_value=[])
    def test_unseen_includes_actual_labels(self, mock_plot):
        """Unseen predictions should include actual_label field."""
        from biobank_agent.skills.predict import predict

        ctx, _ = self._make_ctx(n=50)
        result = predict(model_key="E11:xgboost", patient_eids="unseen", ctx=ctx)

        assert "error" not in result
        for pred in result["predictions"]:
            assert "actual_label" in pred


class TestPredictSpecificEIDs:
    """Test prediction for specific patient EIDs."""

    @patch("biobank_agent.skills.predict._plot_risk_distribution", return_value=[])
    @patch("biobank_agent.skills.predict._fetch_patient_features")
    def test_specific_eids_calls_fetch(self, mock_fetch, mock_plot):
        """Providing comma-separated EIDs queries patient features."""
        from biobank_agent.skills.predict import predict

        feature_names = ["feat_0", "feat_1"]
        X_mock = pd.DataFrame(
            {"feat_0": [1.0, 2.0], "feat_1": [3.0, 4.0]},
            index=["100", "200"],
        )
        mock_fetch.return_value = X_mock

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])

        ctx = MagicMock()
        ctx.state.models = {"E11:xgboost": model}
        ctx.state.model_metadata = {
            "E11:xgboost": {"feature_names": feature_names}
        }
        ctx.state.figures = []

        result = predict(
            model_key="E11:xgboost",
            patient_eids="100,200",
            ctx=ctx,
        )

        assert "error" not in result
        assert result["n_patients"] == 2
        mock_fetch.assert_called_once()

    @patch("biobank_agent.skills.predict._plot_risk_distribution", return_value=[])
    @patch("biobank_agent.skills.predict._fetch_patient_features",
           side_effect=ValueError("No features found"))
    def test_specific_eids_fetch_failure(self, mock_fetch, mock_plot):
        """Fetch failure for specific EIDs → error."""
        from biobank_agent.skills.predict import predict

        ctx = MagicMock()
        ctx.state.models = {"E11:xgboost": MagicMock()}
        ctx.state.model_metadata = {
            "E11:xgboost": {"feature_names": ["f1"]}
        }

        result = predict(
            model_key="E11:xgboost",
            patient_eids="999",
            ctx=ctx,
        )

        assert "error" in result
        assert "Failed to fetch" in result["error"]


class TestPredictModelFallback:
    """Test predict_proba fallback and risk categories."""

    @patch("biobank_agent.skills.predict._plot_risk_distribution", return_value=[])
    def test_fallback_to_predict_when_no_proba(self, mock_plot):
        """Model without predict_proba falls back to predict()."""
        from biobank_agent.skills.predict import predict

        model = MagicMock(spec=[])  # empty spec — no predict_proba
        model.predict = MagicMock(return_value=np.array([0.1, 0.5, 0.9]))
        # Ensure hasattr check fails for predict_proba
        assert not hasattr(model, "predict_proba")

        feature_names = ["f1"]
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["a", "b", "c"])
        y = pd.Series([0, 1, 1], index=X.index)

        ctx = MagicMock()
        ctx.state.models = {"E11:xgboost": model}
        ctx.state.model_metadata = {
            "E11:xgboost": {"feature_names": feature_names}
        }
        ctx.state.feature_matrix = X
        ctx.state.labels = y
        ctx.state.figures = []

        result = predict(model_key="E11:xgboost", patient_eids="unseen", ctx=ctx)

        assert "error" not in result
        model.predict.assert_called_once()

    @patch("biobank_agent.skills.predict._plot_risk_distribution", return_value=[])
    def test_risk_categories_thresholds(self, mock_plot):
        """Risk buckets: <0.3 low, 0.3–0.7 medium, >=0.7 high."""
        from biobank_agent.skills.predict import predict

        # Need enough rows so test split (last 20%) produces >= 3 samples
        # 15 rows → n_test = max(1, 15//5) = 3
        n = 15
        expected_probs = np.array([0.1, 0.5, 0.8])
        # Tile to cover all 15 rows, but predict_proba only called on last 3
        model = MagicMock()
        model.predict_proba.return_value = np.column_stack(
            [1 - expected_probs, expected_probs]
        )

        feature_names = ["f1"]
        X = pd.DataFrame(
            {"f1": np.arange(n, dtype=float)},
            index=[str(i) for i in range(n)],
        )
        y = pd.Series([0, 1] * 7 + [0], index=X.index)

        ctx = MagicMock()
        ctx.state.models = {"E11:xgboost": model}
        ctx.state.model_metadata = {
            "E11:xgboost": {"feature_names": feature_names}
        }
        ctx.state.feature_matrix = X
        ctx.state.labels = y
        ctx.state.figures = []

        result = predict(model_key="E11:xgboost", patient_eids="unseen", ctx=ctx)

        cats = [p["risk_category"] for p in result["predictions"]]
        assert "low" in cats
        assert "medium" in cats
        assert "high" in cats
        assert result["low_risk_count"] >= 1
        assert result["medium_risk_count"] >= 1
        assert result["high_risk_count"] >= 1


class TestPredictFigure:
    """Test figure generation in predict."""

    @patch("biobank_agent.skills.predict._plot_risk_distribution")
    def test_figures_appended_to_state(self, mock_plot):
        """Generated figure paths are appended to ctx.state.figures."""
        from biobank_agent.skills.predict import predict

        mock_plot.return_value = ["/tmp/fig1.svg", "/tmp/fig1.pdf"]

        feature_names = ["f1"]
        X = pd.DataFrame({"f1": [1.0, 2.0]}, index=["a", "b"])
        y = pd.Series([0, 1], index=X.index)

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.6, 0.4], [0.2, 0.8]])

        ctx = MagicMock()
        ctx.state.models = {"E11:xgboost": model}
        ctx.state.model_metadata = {
            "E11:xgboost": {"feature_names": feature_names}
        }
        ctx.state.feature_matrix = X
        ctx.state.labels = y
        ctx.state.figures = []

        result = predict(model_key="E11:xgboost", patient_eids="unseen", ctx=ctx)

        assert len(ctx.state.figures) == 2
        assert result["figures"] == ["/tmp/fig1.svg", "/tmp/fig1.pdf"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
