"""Tests for FM interfaces and platform portability.

Verifies:
- All interface ABCs import without error
- Platform detection works
- GPU guard patterns don't crash on CPU-only machines
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestInterfaceImports:
    """All interface ABCs should import cleanly."""

    def test_fm_embedding_import(self):
        from biobank_agent.interfaces.fm_embedding import FMEmbeddingInterface
        assert hasattr(FMEmbeddingInterface, "encode")
        assert hasattr(FMEmbeddingInterface, "available_models")

    def test_multimodal_import(self):
        from biobank_agent.interfaces.multimodal import MultimodalFusionInterface
        assert hasattr(MultimodalFusionInterface, "fuse")

    def test_evolution_import(self):
        from biobank_agent.interfaces.evolution import EvolutionInterface
        assert hasattr(EvolutionInterface, "save_pipeline")
        assert hasattr(EvolutionInterface, "replay_pipeline")
        assert hasattr(EvolutionInterface, "suggest_improvement")
        assert hasattr(EvolutionInterface, "create_skill")

    def test_remote_fm_client_import(self):
        from biobank_agent.interfaces.fm_embedding import RemoteFMClient
        client = RemoteFMClient(base_url="http://fake", api_key="fake")
        # Should list models without connecting
        models = client.available_models()
        assert isinstance(models, list)
        assert len(models) > 0


class TestPlatformDetection:
    """Platform utilities work on ARM Mac."""

    def test_platform_info(self):
        from biobank_agent.utils.platform import platform_info
        info = platform_info()
        assert "python_version" in info
        assert "arch" in info
        assert "system" in info
        assert "gpu" in info

    def test_platform_summary(self):
        from biobank_agent.utils.platform import platform_summary
        s = platform_summary()
        assert isinstance(s, str)
        assert "Python" in s

    def test_has_gpu(self):
        from biobank_agent.utils.platform import has_gpu
        result = has_gpu()
        assert isinstance(result, bool)

    def test_gpu_info(self):
        from biobank_agent.utils.platform import gpu_info
        info = gpu_info()
        assert "available" in info


class TestGPUGuardPatterns:
    """GPU-dependent code falls back gracefully on CPU."""

    def test_embedding_umap_fallback(self):
        """UMAP import works (cuml falls back to umap-learn)."""
        try:
            from umap import UMAP
            assert UMAP is not None
        except ImportError:
            pytest.skip("umap-learn not installed")
        except Exception as e:
            # On some Python/numba combinations, importing umap raises a runtime
            # cache locator error; this is an environment issue, not a code bug.
            pytest.skip(f"umap import unavailable in this environment: {e}")

    def test_shap_import(self):
        """SHAP imports without GPU."""
        try:
            import shap
            assert shap is not None
        except ImportError:
            pytest.skip("shap not installed")

    def test_xgboost_no_gpu_required(self):
        """XGBoost works without GPU."""
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=2, max_depth=2)
        assert model is not None

    def test_lightgbm_no_gpu_required(self):
        """LightGBM works without GPU."""
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=2, max_depth=2)
        assert model is not None
