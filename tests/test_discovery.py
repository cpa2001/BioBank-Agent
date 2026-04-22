"""Test automated scientific discovery pipeline."""

import pytest
from unittest.mock import MagicMock, patch


class TestDiscoverImport:
    """Test discover skill registration."""

    def test_discover_imports(self):
        """Test that discover skill can be imported."""
        from biobank_agent.skills.discovery import discover
        assert callable(discover)
        assert discover._skill_name == "discover"

    def test_discover_schema(self):
        """Test discover has correct parameters."""
        from biobank_agent.skills.discovery import discover
        schema = discover._skill_schema
        func_def = schema["function"]
        assert "icd10_code" in func_def["parameters"]["properties"]
        assert "discovery_depth" in func_def["parameters"]["properties"]
        assert "model_type" in func_def["parameters"]["properties"]
        assert "icd10_code" in func_def["parameters"]["required"]


class TestDiscoverQuickDepth:
    """Test 'quick' depth — cohort + model + features only."""

    @patch("biobank_agent.skills.train_model.train_model")
    @patch("biobank_agent.skills.feature_importance.feature_importance")
    def test_quick_skips_phewas_and_literature(self, mock_fi, mock_train):
        """Quick depth should NOT run PheWAS or literature steps."""
        from biobank_agent.skills.discovery import discover

        ctx = MagicMock()
        ctx.dm = MagicMock(spec=[])  # no registry_ref attribute

        mock_train.return_value = {
            "mean_auc": 0.85, "n_features": 30, "figures": []
        }
        mock_fi.return_value = {
            "top_features": ["HbA1c", "BMI", "glucose"],
            "figures": [],
        }

        # Mock _run_cohort
        with patch("biobank_agent.skills.discovery._run_cohort") as mock_cohort:
            mock_cohort.return_value = {"n_cases": 5000, "n_controls": 10000}
            result = discover(
                icd10_code="E11", discovery_depth="quick", ctx=ctx,
            )

        assert result["depth"] == "quick"
        assert "cohort_built" in result["steps_completed"]
        assert "model_trained" in result["steps_completed"]
        assert "features_ranked" in result["steps_completed"]
        assert "phewas_done" not in result["steps_completed"]
        assert "literature_searched" not in result["steps_completed"]


class TestDiscoverStandardDepth:
    """Test 'standard' depth — adds PheWAS."""

    @patch("biobank_agent.skills.gwas_proxy.gwas_proxy")
    @patch("biobank_agent.skills.feature_importance.feature_importance")
    @patch("biobank_agent.skills.train_model.train_model")
    def test_standard_includes_phewas(self, mock_train, mock_fi, mock_gwas):
        """Standard depth should include PheWAS but not literature."""
        from biobank_agent.skills.discovery import discover

        ctx = MagicMock()
        ctx.dm = MagicMock(spec=[])

        mock_train.return_value = {"mean_auc": 0.82, "n_features": 20, "figures": []}
        mock_fi.return_value = {"top_features": ["BMI"], "figures": []}
        mock_gwas.return_value = {
            "n_significant": 12,
            "top_associations": [{"pheno": "I10", "p": 1e-8}],
            "figures": [],
        }

        with patch("biobank_agent.skills.discovery._run_cohort") as mock_cohort:
            mock_cohort.return_value = {"n_cases": 3000, "n_controls": 9000}
            result = discover(
                icd10_code="E11", discovery_depth="standard", ctx=ctx,
            )

        assert "phewas_done" in result["steps_completed"]
        assert result["phewas"]["n_significant"] == 12
        assert "literature_searched" not in result["steps_completed"]


class TestDiscoverDeepDepth:
    """Test 'deep' depth — adds literature search."""

    @patch("biobank_agent.skills.web_search.web_search")
    @patch("biobank_agent.skills.gwas_proxy.gwas_proxy")
    @patch("biobank_agent.skills.feature_importance.feature_importance")
    @patch("biobank_agent.skills.train_model.train_model")
    def test_deep_includes_literature(self, mock_train, mock_fi, mock_gwas, mock_web):
        """Deep depth should run all steps including literature."""
        from biobank_agent.skills.discovery import discover

        ctx = MagicMock()
        ctx.dm = MagicMock(spec=[])

        mock_train.return_value = {"mean_auc": 0.9, "n_features": 25, "figures": []}
        mock_fi.return_value = {"top_features": ["HbA1c"], "figures": []}
        mock_gwas.return_value = {"n_significant": 5, "top_associations": [], "figures": []}
        mock_web.return_value = {
            "results": [
                {"title": "UK Biobank diabetes biomarkers", "url": "http://example.com"}
            ],
        }

        with patch("biobank_agent.skills.discovery._run_cohort") as mock_cohort:
            mock_cohort.return_value = {"n_cases": 4000, "n_controls": 8000}
            result = discover(
                icd10_code="E11", discovery_depth="deep", ctx=ctx,
            )

        assert "literature_searched" in result["steps_completed"]
        assert isinstance(result["literature"], list)
        assert len(result["literature"]) > 0


class TestDiscoverFailureHandling:
    """Test graceful degradation when sub-steps fail."""

    def test_cohort_failure_records_error(self):
        """Cohort building failure → error recorded, pipeline continues."""
        from biobank_agent.skills.discovery import discover

        ctx = MagicMock()
        ctx.dm = MagicMock(spec=[])

        with patch("biobank_agent.skills.discovery._run_cohort",
                    side_effect=RuntimeError("DuckDB connection lost")):
            with patch("biobank_agent.skills.train_model.train_model",
                       side_effect=RuntimeError("No cohort")):
                with patch("biobank_agent.skills.feature_importance.feature_importance",
                           side_effect=RuntimeError("No model")):
                    result = discover(
                        icd10_code="E11", discovery_depth="quick", ctx=ctx,
                    )

        assert "error" in result["cohort"]
        assert "DuckDB" in result["cohort"]["error"]
        assert "summary" in result  # pipeline still completes

    @patch("biobank_agent.skills.feature_importance.feature_importance")
    @patch("biobank_agent.skills.train_model.train_model",
           side_effect=RuntimeError("Training OOM"))
    def test_model_failure_continues_pipeline(self, mock_train, mock_fi):
        """Model training failure → error recorded, features still attempted."""
        from biobank_agent.skills.discovery import discover

        ctx = MagicMock()
        ctx.dm = MagicMock(spec=[])

        mock_fi.side_effect = RuntimeError("No model to explain")

        with patch("biobank_agent.skills.discovery._run_cohort") as mock_cohort:
            mock_cohort.return_value = {"n_cases": 100, "n_controls": 200}
            result = discover(
                icd10_code="E11", discovery_depth="quick", ctx=ctx,
            )

        assert "error" in result["model"]
        assert "Training OOM" in result["model"]["error"]
        assert "cohort_built" in result["steps_completed"]
        assert "model_trained" not in result["steps_completed"]


class TestSynthesizeFindings:
    """Test the internal synthesis function."""

    def test_synthesis_with_all_results(self):
        """Full results → multi-line summary."""
        from biobank_agent.skills.discovery import _synthesize_findings

        results = {
            "cohort": {"n_cases": 5000, "n_controls": 10000},
            "model": {"type": "xgboost", "auc": 0.88},
            "phewas": {"n_significant": 15},
        }
        top_features = ["HbA1c", "BMI", "glucose", "creatinine", "CRP"]

        summary = _synthesize_findings("E11", results, top_features)

        assert "E11" in summary
        assert "5000 cases" in summary
        assert "0.8800" in summary
        assert "good" in summary
        assert "HbA1c" in summary
        assert "15 significant" in summary

    def test_synthesis_with_errors(self):
        """Error-laden results → only non-error steps in summary."""
        from biobank_agent.skills.discovery import _synthesize_findings

        results = {
            "cohort": {"error": "DB down"},
            "model": {"error": "No data"},
        }

        summary = _synthesize_findings("E11", results, [])

        assert "cases" not in summary
        assert "AUC" not in summary
        assert "E11" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
