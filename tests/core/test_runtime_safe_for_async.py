"""Tests for AsyncAgent.is_safe_for_async() guard."""

from __future__ import annotations

from types import SimpleNamespace

from biobank_agent.core.runtime import AsyncAgent


def _legacy(multi_model: bool, pool_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(multi_model_enabled=multi_model),
        orchestrator=SimpleNamespace(model_pool=[object()] * pool_size),
    )


def test_safe_when_multi_model_disabled():
    legacy = _legacy(multi_model=False, pool_size=3)
    safe, reason = AsyncAgent.is_safe_for_async(legacy)
    assert safe is True
    assert reason == ""


def test_safe_with_single_model_pool_and_multi_model_enabled():
    legacy = _legacy(multi_model=True, pool_size=1)
    safe, reason = AsyncAgent.is_safe_for_async(legacy)
    assert safe is True


def test_unsafe_when_multi_model_enabled_with_pool_gt_one():
    legacy = _legacy(multi_model=True, pool_size=3)
    safe, reason = AsyncAgent.is_safe_for_async(legacy)
    assert safe is False
    assert "multi_model_enabled" in reason
    assert "DEBATE/ENSEMBLE/SUPERVISOR" in reason
