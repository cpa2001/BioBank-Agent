"""Unit tests for relay model discovery in LLMClient."""

from types import SimpleNamespace
from unittest.mock import MagicMock


class TestModelDiscovery:
    """Model list fetching should be cached and resilient."""

    def test_list_models_uses_cache(self):
        from biobank_agent.llm import LLMClient

        llm = LLMClient(base_url="http://relay.test", api_key="test", model="claude-sonnet-4-6")

        models_api = MagicMock()
        models_api.list.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(id="gpt-5.4"),
                SimpleNamespace(id="gemini-3.1-pro-preview"),
            ]
        )
        llm.client = SimpleNamespace(models=models_api)

        first = llm.list_models()
        second = llm.list_models()

        assert first == ["gemini-3.1-pro-preview", "gpt-5.4"]
        assert second == first
        assert models_api.list.call_count == 1

    def test_refresh_bypasses_cache(self):
        from biobank_agent.llm import LLMClient

        llm = LLMClient(base_url="http://relay.test", api_key="test", model="claude-sonnet-4-6")

        models_api = MagicMock()
        models_api.list.side_effect = [
            SimpleNamespace(data=[SimpleNamespace(id="gpt-5.4")]),
            SimpleNamespace(data=[SimpleNamespace(id="gemini-3.1-pro-preview")]),
        ]
        llm.client = SimpleNamespace(models=models_api)

        first = llm.list_models()
        refreshed = llm.list_models(refresh=True)

        assert first == ["gpt-5.4"]
        assert refreshed == ["gemini-3.1-pro-preview"]
        assert models_api.list.call_count == 2

    def test_failure_falls_back_to_cached_models(self):
        from biobank_agent.llm import LLMClient

        llm = LLMClient(base_url="http://relay.test", api_key="test", model="claude-sonnet-4-6")
        llm._model_cache = ["claude-sonnet-4-6"]
        llm._model_cache_ts = 0

        models_api = MagicMock()
        models_api.list.side_effect = RuntimeError("relay unavailable")
        llm.client = SimpleNamespace(models=models_api)

        models = llm.list_models(refresh=True)

        assert models == ["claude-sonnet-4-6"]
