"""Tests for robust token usage aggregation."""

from biobank_agent.state import TokenUsage


def test_token_usage_update_handles_none_and_invalid_values():
    usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
    usage.update({"prompt_tokens": None, "completion_tokens": "x"})
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5


def test_token_usage_update_accepts_numeric_strings():
    usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
    usage.update({"prompt_tokens": "12", "completion_tokens": 7.9})
    assert usage.prompt_tokens == 12
    assert usage.completion_tokens == 7
