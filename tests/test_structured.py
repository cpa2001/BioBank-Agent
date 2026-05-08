"""Tests for structured output extraction (Phase 3).

Tests both the instructor path (mocked) and the fallback path.
"""

import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field
from biobank_agent.structured import (
    extract_structured,
    is_instructor_available,
    _extract_fallback,
)


# ── Test models ──────────────────────────────────────────────────────────────


class SimpleOutput(BaseModel):
    """Simple test output model."""
    answer: str
    confidence: float = Field(ge=0, le=1)


class ComplexOutput(BaseModel):
    """Complex test output model."""
    title: str
    items: list[str]
    score: float = Field(ge=0, le=100)


# ── Mock LLM ─────────────────────────────────────────────────────────────────


def make_mock_llm(response_text: str):
    """Create a mock LLM that returns specific text."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.text = response_text
    mock.chat.return_value = mock_response
    return mock


# ── Tests ────────────────────────────────────────────────────────────────────


class TestInstructorAvailability:
    """Test availability detection."""

    def test_availability_is_bool(self):
        """is_instructor_available should return bool."""
        result = is_instructor_available()
        assert isinstance(result, bool)

    def test_import_detects_available_instructor(self):
        """Reloading with a fake instructor module should mark it available."""
        import biobank_agent.structured as structured_mod

        sentinel = object()
        original = sys.modules.get("instructor", sentinel)
        sys.modules["instructor"] = SimpleNamespace(from_openai=lambda client: client)
        try:
            reloaded = importlib.reload(structured_mod)
            assert reloaded.is_instructor_available() is True
        finally:
            if original is sentinel:
                sys.modules.pop("instructor", None)
            else:
                sys.modules["instructor"] = original
            importlib.reload(structured_mod)


class TestFallbackExtraction:
    """Test the fallback path (no instructor)."""

    def test_valid_json_extraction(self):
        """Should parse valid JSON into Pydantic model."""
        mock_llm = make_mock_llm('{"answer": "42", "confidence": 0.95}')
        result = _extract_fallback(mock_llm, SimpleOutput, "What is the answer?", 2, "sys")
        assert result.answer == "42"
        assert result.confidence == 0.95

    def test_json_with_markdown_fences(self):
        """Should handle JSON wrapped in markdown code fences."""
        mock_llm = make_mock_llm('```json\n{"answer": "hello", "confidence": 0.7}\n```')
        result = _extract_fallback(mock_llm, SimpleOutput, "Test", 2, "sys")
        assert result.answer == "hello"

    def test_json_with_plain_markdown_fence(self):
        """Fenced JSON without a language tag should also be parsed."""
        mock_llm = make_mock_llm('```\n{"answer": "plain", "confidence": 0.4}\n```')
        result = _extract_fallback(mock_llm, SimpleOutput, "Test", 2, "sys")
        assert result.answer == "plain"

    def test_complex_model_extraction(self):
        """Should handle complex Pydantic models."""
        response = json.dumps({"title": "Test", "items": ["a", "b"], "score": 85.5})
        mock_llm = make_mock_llm(response)
        result = _extract_fallback(mock_llm, ComplexOutput, "Test", 2, "sys")
        assert result.title == "Test"
        assert len(result.items) == 2
        assert result.score == 85.5

    def test_retry_on_invalid_json(self):
        """Should retry when JSON is invalid."""
        # First call returns invalid, second returns valid
        mock_llm = MagicMock()
        responses = [
            MagicMock(text="not json at all"),
            MagicMock(text='{"answer": "retry worked", "confidence": 0.5}'),
        ]
        mock_llm.chat.side_effect = [responses[0], responses[1]]
        result = _extract_fallback(mock_llm, SimpleOutput, "Test", 2, "sys")
        assert result.answer == "retry worked"

    def test_retry_on_fenced_non_json_content(self):
        """Code fences without JSON should still flow through retry handling."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            MagicMock(text="```text\nnot json\n```"),
            MagicMock(text='{"answer": "second try", "confidence": 0.5}'),
        ]

        result = _extract_fallback(mock_llm, SimpleOutput, "Test", 2, "sys")

        assert result.answer == "second try"

    def test_validation_error_retries(self):
        """Should retry when Pydantic validation fails."""
        # confidence > 1 is invalid
        mock_llm = MagicMock()
        responses = [
            MagicMock(text='{"answer": "bad", "confidence": 5.0}'),  # Invalid
            MagicMock(text='{"answer": "good", "confidence": 0.8}'),  # Valid
        ]
        mock_llm.chat.side_effect = [responses[0], responses[1]]
        result = _extract_fallback(mock_llm, SimpleOutput, "Test", 2, "sys")
        assert result.confidence == 0.8

    def test_all_retries_exhausted_raises(self):
        """Should raise ValueError when all retries fail."""
        mock_llm = make_mock_llm("this will never parse as json {{{")
        with pytest.raises(ValueError, match="Failed to extract"):
            _extract_fallback(mock_llm, SimpleOutput, "Test", 1, "sys")


class TestExtractStructured:
    """Test the main extract_structured function."""

    def test_uses_fallback_when_no_instructor(self):
        """Without instructor, should use fallback path."""
        mock_llm = make_mock_llm('{"answer": "fallback", "confidence": 0.6}')
        with patch("biobank_agent.structured._INSTRUCTOR_AVAILABLE", False):
            result = extract_structured(mock_llm, SimpleOutput, "Test")
            assert result.answer == "fallback"

    def test_custom_system_prompt(self):
        """Should pass custom system prompt to LLM."""
        mock_llm = make_mock_llm('{"answer": "custom", "confidence": 0.5}')
        with patch("biobank_agent.structured._INSTRUCTOR_AVAILABLE", False):
            extract_structured(
                mock_llm, SimpleOutput, "Test",
                system_prompt="You are a special extractor."
            )
            # Check that system prompt was passed
            call_args = mock_llm.chat.call_args
            messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
            assert any("special extractor" in m.get("content", "") for m in messages)

    def test_uses_instructor_when_available(self):
        """The instructor branch should return the patched client's model."""
        import biobank_agent.structured as structured_mod

        class FakeCompletions:
            def create(self, **kwargs):
                assert kwargs["model"] == "gpt-test"
                assert kwargs["response_model"] is SimpleOutput
                assert kwargs["max_retries"] == 3
                return SimpleOutput(answer="typed", confidence=0.9)

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        fake_instructor = SimpleNamespace(from_openai=lambda client: fake_client)
        llm = SimpleNamespace(client=object(), model="gpt-test")

        with patch("biobank_agent.structured._INSTRUCTOR_AVAILABLE", True):
            with patch.object(structured_mod, "instructor", fake_instructor, create=True):
                result = extract_structured(llm, SimpleOutput, "Test", max_retries=3)

        assert result.answer == "typed"

    def test_instructor_failure_falls_back(self):
        """Instructor errors should fall back to manual JSON extraction."""
        import biobank_agent.structured as structured_mod

        def failing_from_openai(client):
            raise RuntimeError("patch failed")

        llm = make_mock_llm('{"answer": "fallback", "confidence": 0.6}')
        llm.client = object()
        fake_instructor = SimpleNamespace(from_openai=failing_from_openai)

        with patch.object(structured_mod, "instructor", fake_instructor, create=True):
            result = structured_mod._extract_with_instructor(llm, SimpleOutput, "Test", 1, "sys")

        assert result.answer == "fallback"
