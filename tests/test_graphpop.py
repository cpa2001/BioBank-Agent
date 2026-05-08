"""Tests for GraphPop MCP client (Stream A).

Tests tool registry, parameter validation, and graceful error handling.
Does NOT require a running GraphPop server.
"""

import asyncio
import builtins
import importlib

import pytest
from biobank_agent.mcp import graphpop_client as gp
from biobank_agent.mcp.graphpop_client import (
    GraphPopMCPClient,
    GRAPHPOP_TOOLS,
    MCPToolSpec,
    MCPResult,
)


@pytest.fixture
def client():
    return GraphPopMCPClient(endpoint="http://localhost:9999/mcp")  # intentionally unreachable


class TestToolRegistry:
    """Test the 21-tool registry."""

    def test_21_tools_registered(self):
        """Should have exactly 21 tools."""
        assert len(GRAPHPOP_TOOLS) == 21

    def test_tool_categories(self):
        """Tools should be categorized correctly."""
        categories = {t.category for t in GRAPHPOP_TOOLS.values()}
        assert categories == {"fast_path", "full_path", "query", "admin"}

    def test_fast_path_tools(self):
        """Fast path should include diversity, fst, sfs, etc."""
        fast_tools = [t.name for t in GRAPHPOP_TOOLS.values() if t.category == "fast_path"]
        assert "diversity" in fast_tools
        assert "fst" in fast_tools
        assert "genome_scan" in fast_tools

    def test_full_path_tools(self):
        """Full path should include haplotype statistics."""
        full_tools = [t.name for t in GRAPHPOP_TOOLS.values() if t.category == "full_path"]
        assert "ihs" in full_tools
        assert "xpehh" in full_tools
        assert "roh" in full_tools

    def test_all_tools_have_descriptions(self):
        """Every tool should have a non-empty description."""
        for name, spec in GRAPHPOP_TOOLS.items():
            assert spec.description, f"Tool '{name}' has no description"


class TestParameterValidation:
    """Test pre-flight parameter validation."""

    def test_missing_required_params(self, client):
        """Should fail before network call if required params missing."""
        result = asyncio.run(client.call_tool("diversity", {}))
        assert not result.success
        assert "Missing required parameters" in result.error
        assert "chromosome" in result.error

    def test_partial_params_still_fail(self, client):
        """Partial params should still be caught."""
        result = asyncio.run(client.call_tool("diversity", {"chromosome": "chr1"}))
        assert not result.success
        assert "Missing required" in result.error

    def test_all_required_params_passes_validation(self, client):
        """With all required params, should attempt network (and fail gracefully)."""
        result = asyncio.run(client.call_tool("diversity", {
            "chromosome": "chr1",
            "start": 1,
            "end": 1000000,
            "population": "EUR",
        }))
        # Should fail due to unreachable server, NOT parameter validation
        assert not result.success
        assert "Missing required" not in result.error

    def test_unknown_tool_rejected(self, client):
        """Unknown tool name should be rejected immediately."""
        result = asyncio.run(client.call_tool("nonexistent_tool", {}))
        assert not result.success
        assert "Unknown tool" in result.error

    def test_admin_tools_no_required_params(self, client):
        """Admin tools with no required params should pass validation."""
        result = asyncio.run(client.call_tool("list_populations", {}))
        # Should fail at network level, not validation
        assert not result.success
        assert "Missing required" not in result.error


class TestToolSchemaExport:
    """Test OpenAI function-calling schema export."""

    def test_list_tools_format(self, client):
        """list_tools should return OpenAI-compatible schemas."""
        tools = client.list_tools()
        assert len(tools) == 21
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "parameters" in tool["function"]
            assert tool["function"]["name"].startswith("graphpop_")

    def test_tool_summary(self, client):
        """get_tool_summary should return readable text."""
        summary = client.get_tool_summary()
        assert "GraphPop" in summary
        assert "Fast Path" in summary
        assert "diversity" in summary

    def test_tool_summary_handles_empty_categories(self, client, monkeypatch):
        monkeypatch.setattr(gp, "GRAPHPOP_TOOLS", {
            "custom": MCPToolSpec(
                name="custom",
                description="Custom tool",
                category="other",
            )
        })

        summary = client.get_tool_summary()

        assert "GraphPop" in summary
        assert "Custom tool" not in summary

    def test_tools_property_returns_registry(self, client):
        assert client.tools is GRAPHPOP_TOOLS


class TestGracefulFailure:
    """Test graceful handling of server unavailability."""

    def test_connect_fails_gracefully(self, client):
        """Connection failure should not raise."""
        result = asyncio.run(client.connect())
        assert result is False
        assert not client.connected

    def test_call_without_connect(self, client):
        """Calling tool without connection should fail gracefully."""
        result = asyncio.run(client.call_tool("list_populations", {}))
        assert not result.success
        assert "unavailable" in result.error.lower() or "Connection" in result.error


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text="OK"):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeAsyncClient:
    responses = []
    calls = []

    def __init__(self, timeout=None):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, endpoint, json=None):
        self.__class__.calls.append((endpoint, json, self.timeout))
        item = self.__class__.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class TestMockedHTTP:
    def _install(self, monkeypatch, responses):
        _FakeAsyncClient.responses = list(responses)
        _FakeAsyncClient.calls = []
        monkeypatch.setattr(gp, "_HTTPX_AVAILABLE", True)
        monkeypatch.setattr(gp.httpx, "AsyncClient", _FakeAsyncClient)

    def test_no_httpx_disables_connect_and_call(self, monkeypatch):
        monkeypatch.setattr(gp, "_HTTPX_AVAILABLE", False)
        client = GraphPopMCPClient()

        connected = asyncio.run(client.connect())
        result = asyncio.run(client.call_tool("list_populations", {}))

        assert connected is False
        assert result.success is False
        assert "httpx not installed" in result.error

    def test_import_without_httpx_marks_client_unavailable(self, monkeypatch):
        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("no httpx")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        reloaded = importlib.reload(gp)
        assert reloaded._HTTPX_AVAILABLE is False

        monkeypatch.setattr(builtins, "__import__", original_import)
        restored = importlib.reload(gp)
        assert restored._HTTPX_AVAILABLE is True

    def test_connect_success_and_non_200(self, monkeypatch):
        self._install(monkeypatch, [_FakeResponse(200), _FakeResponse(503, text="down")])
        client = GraphPopMCPClient(endpoint="http://graphpop/mcp")

        assert asyncio.run(client.connect()) is True
        assert client.connected is True
        assert _FakeAsyncClient.calls[0][1]["method"] == "initialize"

        client._connected = False
        assert asyncio.run(client.connect()) is False
        assert client.connected is False

    def test_connect_timeout_and_unexpected_exception(self, monkeypatch):
        self._install(
            monkeypatch,
            [
                gp.httpx.TimeoutException("slow"),
                RuntimeError("boom"),
            ],
        )
        client = GraphPopMCPClient()

        assert asyncio.run(client.connect()) is False
        assert asyncio.run(client.connect()) is False

    def test_call_tool_success_json_text_and_resource_blocks(self, monkeypatch):
        payload_json = {"result": {"content": [{"type": "text", "text": "{\"pi\": 0.1}"}]}}
        payload_text = {"result": {"content": [{"type": "text", "text": "plain output"}]}}
        resource = {"type": "resource", "uri": "graphpop://result/1"}
        payload_resource = {"result": {"content": [resource]}}
        self._install(
            monkeypatch,
            [
                _FakeResponse(200, payload_json),
                _FakeResponse(200, payload_text),
                _FakeResponse(200, payload_resource),
            ],
        )
        client = GraphPopMCPClient(timeout=12)
        client._connected = True

        json_result = asyncio.run(client.call_tool("list_populations", None))
        text_result = asyncio.run(client.call_tool("list_populations", {}))
        resource_result = asyncio.run(client.call_tool("list_populations", {}))

        assert json_result.success is True
        assert json_result.data == {"pi": 0.1}
        assert text_result.data == "plain output"
        assert resource_result.data == resource
        assert _FakeAsyncClient.calls[0][1]["id"] == 1
        assert _FakeAsyncClient.calls[0][2] == 12

    def test_call_tool_connects_on_demand_and_ignores_unknown_content_blocks(self, monkeypatch):
        payload = {"result": {"content": [{"type": "unknown", "value": 1}]}}
        self._install(monkeypatch, [_FakeResponse(200, payload)])
        client = GraphPopMCPClient()

        async def fake_connect():
            client._connected = True
            return True

        client.connect = fake_connect
        result = asyncio.run(client.call_tool("list_populations", {}))

        assert result.success is True
        assert result.data is None

    def test_call_tool_jsonrpc_error_and_http_error(self, monkeypatch):
        self._install(
            monkeypatch,
            [
                _FakeResponse(200, {"error": {"message": "bad request"}}),
                _FakeResponse(500, text="server exploded" * 30),
            ],
        )
        client = GraphPopMCPClient()
        client._connected = True

        rpc = asyncio.run(client.call_tool("list_populations", {}))
        http = asyncio.run(client.call_tool("list_populations", {}))

        assert rpc.success is False
        assert rpc.error == "bad request"
        assert http.success is False
        assert http.error.startswith("HTTP 500")
        assert len(http.error) < 230

    def test_call_tool_timeout_connect_and_unexpected_errors(self, monkeypatch):
        self._install(
            monkeypatch,
            [
                gp.httpx.TimeoutException("slow"),
                gp.httpx.ConnectError("lost"),
                RuntimeError("bad"),
            ],
        )
        client = GraphPopMCPClient(timeout=1)
        client._connected = True

        timeout = asyncio.run(client.call_tool("list_populations", {}))
        lost = asyncio.run(client.call_tool("list_populations", {}))
        client._connected = True
        unexpected = asyncio.run(client.call_tool("list_populations", {}))

        assert timeout.success is False
        assert "Timeout after 1" in timeout.error
        assert lost.success is False
        assert "Connection lost" in lost.error
        assert client.connected is True  # reset before third call
        assert unexpected.success is False
        assert "Unexpected error" in unexpected.error
