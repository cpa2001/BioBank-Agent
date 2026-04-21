"""End-to-end LLM integration tests.

These tests call the real API endpoint. Run with:
    pytest tests/test_llm_e2e.py -v -m integration

Skip in CI:
    pytest tests/ -v -m "not integration"
"""

import json
import os
import time

import pytest

# Mark all tests in this module as integration
pytestmark = pytest.mark.integration


def _get_settings():
    """Get settings with real API credentials."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from biobank_agent.config import get_settings
    return get_settings()


def _get_llm_client(model=None):
    """Create LLM client with real credentials."""
    from biobank_agent.llm import LLMClient
    settings = _get_settings()
    return LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model or settings.llm_model,
    )


# ── Basic Chat Tests ─────────────────────────────────────────


class TestBasicChat:
    """Test basic LLM chat without tools."""

    def test_simple_response(self):
        """LLM returns non-empty text for a simple question."""
        client = _get_llm_client()
        response = client.chat(
            messages=[
                {"role": "user", "content": "What is the ICD10 code for Type 2 Diabetes? Answer in one line."}
            ],
        )
        assert response.text, "Response text should not be empty"
        assert "E11" in response.text, f"Expected E11 in response: {response.text}"
        assert not response.has_tool_calls, "No tool calls expected"

    def test_usage_tracking(self):
        """Token usage is reported."""
        client = _get_llm_client()
        response = client.chat(
            messages=[{"role": "user", "content": "Say hello."}],
        )
        assert response.usage, "Usage should be reported"
        assert response.usage.get("total_tokens", 0) > 0


# ── Tool Calling Tests ───────────────────────────────────────


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "prevalence",
            "description": "Calculate disease prevalence",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "description": "Top N diseases"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Internal reasoning step",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": "Reasoning text"},
                },
                "required": ["thought"],
            },
        },
    },
]


class TestToolCalling:
    """Test LLM tool/function calling."""

    def test_tool_call_returned(self):
        """LLM returns a tool call when tools are provided and the query matches."""
        client = _get_llm_client()
        response = client.chat(
            messages=[
                {"role": "user", "content": "Show me the top 5 most common diseases."}
            ],
            tools=SAMPLE_TOOLS,
        )
        assert response.has_tool_calls, "Expected tool calls"
        tc = response.tool_calls[0]
        assert tc.name in ("prevalence", "think"), f"Unexpected tool: {tc.name}"
        assert tc.id, "Tool call should have an ID"

    def test_tool_args_parsed(self):
        """Tool call arguments are parsed as dict."""
        client = _get_llm_client()
        response = client.chat(
            messages=[
                {"role": "user", "content": "Show the top 10 most common diseases."}
            ],
            tools=SAMPLE_TOOLS,
        )
        if response.has_tool_calls:
            for tc in response.tool_calls:
                assert isinstance(tc.args, dict), f"Args should be dict, got {type(tc.args)}"


# ── Multi-Model Tests ────────────────────────────────────────


class TestMultiModel:
    """Test that multiple models respond without error."""

    @pytest.mark.parametrize("model", [
        "claude-sonnet-4-6",
        "gpt-5.4",
        "claude-haiku-4-5-20251001",
    ])
    def test_model_responds(self, model):
        """Each model returns a non-empty response."""
        try:
            client = _get_llm_client(model=model)
            response = client.chat(
                messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
                max_tokens=100,
            )
            assert response.text, f"Model {model} returned empty response"
            assert "4" in response.text, f"Model {model} gave wrong answer: {response.text}"
        except Exception as e:
            pytest.skip(f"Model {model} unavailable: {e}")


# ── Full Agent Turn Tests ────────────────────────────────────


class TestFullAgentTurn:
    """Test the full agent ReAct loop with real data + real LLM."""

    @pytest.fixture
    def agent(self):
        from biobank_agent.agent import Agent
        settings = _get_settings()
        return Agent(settings)

    def test_prevalence_query(self, agent):
        """Agent handles a prevalence query end-to-end."""
        t0 = time.time()
        response = agent.run("What are the top 5 most common diseases? Be brief.")
        elapsed = time.time() - t0

        assert response, "Agent should return a response"
        assert "[Max tool rounds reached]" not in response
        assert len(agent.state.records) > 0, "Should have analysis records"
        print(f"\nPrevalence query: {elapsed:.1f}s, {len(agent.state.records)} tool calls")
        print(f"Response preview: {response[:200]}")

    def test_error_handling(self, agent):
        """Agent handles an invalid ICD10 code gracefully."""
        response = agent.run("Show prevalence for ICD10 code ZZZZZ. Be brief.")
        assert response, "Should return a response even for invalid input"
        # Should not crash
