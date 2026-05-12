"""M2 sub-agent and MCP manager tests."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from biobank_agent.core.tools.protocol import ToolContext
from biobank_agent.core.events import AgentEventType
from biobank_agent.core.llm.client import StreamFinal, StreamTextDelta
from biobank_agent.core.orchestration.subagent import (
    SubAgent,
    SubAgentMode,
    SubAgentSpec,
)
from biobank_agent.core.runtime import AsyncAgent
from biobank_agent.core.tools.registry import ToolRegistry
from biobank_agent.extensions.mcp_manager import McpManager, McpServerConfig
from biobank_agent.llm import LLMResponse


# Reuse the same stubs the runtime tests built, in stripped form.

class _StubAsyncLLM:
    def __init__(self, rounds):
        if rounds and isinstance(rounds[0], (list, tuple)):
            self.rounds = [list(r) for r in rounds]
        else:
            self.rounds = [list(rounds)]
        self._idx = 0

    async def stream_async(self, **_):
        if self._idx >= len(self.rounds):
            yield StreamFinal(response=LLMResponse(text="", tool_calls=[], usage={}))
            return
        script = self.rounds[self._idx]
        self._idx += 1
        for item in script:
            await asyncio.sleep(0)
            yield item


def _stub_legacy(tmp_path):
    from types import SimpleNamespace

    settings = SimpleNamespace(
        reports_dir=tmp_path / "reports",
        memory_dir=tmp_path / "mem",
        context_window=8192,
        max_tool_rounds=2,
        llm_model="m",
        bank_id="b",
        biobank_name="b",
        async_runtime_enabled=True,
    )
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.memory_dir.mkdir(parents=True, exist_ok=True)
    state = SimpleNamespace(
        interrupted=False,
        records=[],
        figures=[],
        cohorts={},
        token_usage=SimpleNamespace(update=lambda u: None),
        custom_data={},
        executive_findings=[],
        provenances=[],
        last_orchestration={},
    )
    return SimpleNamespace(
        settings=settings,
        state=state,
        memory=SimpleNamespace(upsert_node=lambda **_: None),
        registry=SimpleNamespace(tool_schemas=lambda: []),
        dm=SimpleNamespace(conn=None),
        catalog=SimpleNamespace(fields={}),
        reproducibility=SimpleNamespace(checkpoint=lambda *a, **kw: "ctx"),
        _study_spec_compiler=SimpleNamespace(
            compile=lambda q: (_ for _ in ()).throw(RuntimeError("no spec"))
        ),
        orchestrator=SimpleNamespace(
            _wrap_llm_response=lambda **kw: SimpleNamespace(
                claims=[],
                evidence_links=[],
                safety_status="UNKNOWN",
                debate_trace={},
                text="",
                tool_calls=[],
                usage={},
                has_tool_calls=False,
                to_dict=lambda: {},
            ),
        ),
        llm=SimpleNamespace(model="m"),
        messages=[
            {"role": "user", "content": "old turn"},
            {"role": "assistant", "content": "old answer"},
        ],
        _active_query_id=None,
        _system_message=lambda: {"role": "system", "content": "sys"},
        _execute_skill_and_record=lambda *a, **kw: {
            "result": {"summary": "sub_ok"},
            "result_str": "sub_ok",
            "elapsed_s": 0.01,
            "new_figures": [],
            "is_error": False,
            "args": {},
        },
        _record_orchestration_graph=lambda *a, **kw: None,
        _record_executive_findings=lambda *a, **kw: None,
        _post_run=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_subagent_streams_events_and_returns_final_text(tmp_path):
    legacy = _stub_legacy(tmp_path)
    parent_messages = legacy.messages
    parent = AsyncAgent(legacy)
    spec = SubAgentSpec(mode=SubAgentMode.WORKER, last_n_turns=1, label="reviewer")
    sub = SubAgent.from_parent(parent, spec)
    sub.runtime._async_llm = _StubAsyncLLM([
        StreamTextDelta(text="reviewed"),
        StreamFinal(response=LLMResponse(text="reviewed", tool_calls=[], usage={})),
    ])
    run = await sub.run("verify the cohort")
    assert run.final_text == "reviewed"
    types = [ev.type for ev in run.events]
    assert AgentEventType.TURN_STARTED in types
    assert AgentEventType.TURN_FINISHED in types
    # Sub-agent label propagates into payloads.
    started = next(ev for ev in run.events if ev.type == AgentEventType.TURN_STARTED)
    assert started.payload.get("subagent_label") == "reviewer"
    assert legacy.messages is parent_messages


def test_mcp_manager_returns_zero_when_no_config(tmp_path):
    registry = ToolRegistry()
    cfg_path = tmp_path / "mcp_servers.json"
    mgr = McpManager(registry, config_path=cfg_path)
    assert mgr.load_config() == []


def test_mcp_manager_uses_env_config_path(tmp_path, monkeypatch):
    registry = ToolRegistry()
    cfg_path = tmp_path / "custom_mcp.json"
    cfg_path.write_text(json.dumps({"servers": []}), encoding="utf-8")
    monkeypatch.setenv("BIOBANK_MCP_CONFIG", str(cfg_path))

    mgr = McpManager(registry)

    assert mgr.config_path == cfg_path


def test_mcp_manager_parses_config(tmp_path):
    registry = ToolRegistry()
    cfg_path = tmp_path / "mcp_servers.json"
    cfg_path.write_text(json.dumps({
        "servers": [
            {
                "name": "github",
                "transport": "stdio",
                "command": "echo",
                "args": ["hi"],
                "env": {},
            },
            {
                "name": "rap",
                "transport": "http-sse",
                "url": "https://example.com/mcp",
                "auth": {
                    "bearer_token_env": "RAP_MCP_TOKEN",
                    "api_key_env": "RAP_MCP_KEY",
                    "api_key_header": "x-rap-key",
                },
                "headers": {"x-client": "biobank-agent-test"},
            },
        ],
    }))
    mgr = McpManager(registry, config_path=cfg_path)
    cfgs = mgr.load_config()
    assert len(cfgs) == 2
    assert cfgs[0].name == "github"
    assert cfgs[1].transport == "http-sse"
    assert cfgs[1].bearer_token_env == "RAP_MCP_TOKEN"
    assert cfgs[1].api_key_env == "RAP_MCP_KEY"
    assert cfgs[1].api_key_header == "x-rap-key"
    assert cfgs[1].headers["x-client"] == "biobank-agent-test"


def test_mcp_http_auth_headers_are_structured_and_env_backed(monkeypatch):
    import biobank_agent.extensions.mcp_manager as mcp_mod

    monkeypatch.setenv("BIOBANK_TEST_MCP_TOKEN", "token-123")
    monkeypatch.setenv("BIOBANK_TEST_MCP_KEY", "key-456")
    client = mcp_mod._HttpMcpClient(
        name="secure",
        url="https://example.test/mcp",
        env={"HEADER_X_LEGACY": "legacy"},
        headers={"x-client": "biobank"},
        bearer_token_env="BIOBANK_TEST_MCP_TOKEN",
        api_key_env="BIOBANK_TEST_MCP_KEY",
        api_key_header="x-api-key",
    )

    headers = client._headers()

    assert headers["Authorization"] == "Bearer token-123"
    assert headers["x-api-key"] == "key-456"
    assert headers["X-LEGACY"] == "legacy"
    assert headers["x-client"] == "biobank"


def test_mcp_manager_skips_unsupported_transports(tmp_path, monkeypatch):
    registry = ToolRegistry()
    cfg_path = tmp_path / "mcp_servers.json"
    cfg_path.write_text(json.dumps({
        "servers": [
            {"name": "rap", "transport": "websocket", "url": "https://example.com"},
        ],
    }))
    mgr = McpManager(registry, config_path=cfg_path)

    async def go():
        return await mgr.start()

    loaded = asyncio.run(go())
    assert loaded == 0
    assert "unsupported transport" in mgr.errors["rap"]


def test_mcp_manager_registers_http_sse_tools(tmp_path):
    """HTTP-SSE config should register and call remote MCP tools, not skip."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802 - stdlib callback
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            method = payload.get("method")
            request_id = payload.get("id")
            if method == "notifications/initialized":
                body = b"{}"
            elif method == "initialize":
                body = json.dumps({"jsonrpc": "2.0", "id": request_id, "result": {"serverInfo": {"name": "stub"}}}).encode()
            elif method == "tools/list":
                body = json.dumps({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "lookup fields",
                                "description": "Lookup one item",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "required": ["query"],
                                },
                            }
                        ]
                    },
                }).encode()
            elif method == "tools/call":
                query = payload.get("params", {}).get("arguments", {}).get("query", "")
                body = json.dumps({"jsonrpc": "2.0", "id": request_id, "result": {"echo": query}}).encode()
            else:
                body = json.dumps({"jsonrpc": "2.0", "id": request_id, "error": {"message": "unknown"}}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        registry = ToolRegistry()
        cfg_path = tmp_path / "mcp_servers.json"
        cfg_path.write_text(json.dumps({
            "servers": [
                {
                    "name": "http stub",
                    "transport": "http-sse",
                    "url": f"http://127.0.0.1:{server.server_port}/mcp",
                }
            ],
        }))
        mgr = McpManager(registry, config_path=cfg_path)
        loaded = asyncio.run(mgr.start())
        assert loaded == 1
        status = mgr.status()[0]
        assert status["running"] is True
        assert status["loaded_tools"] == 1
        assert status["error"] == ""
        handler = registry.get("mcp_http_stub__lookup_fields")
        assert handler is not None

        async def call_remote():
            ctx = ToolContext(
                name=handler.name,
                args={"query": "E11"},
                capabilities=handler.required_capabilities(),
            )
            return await handler.handle(ctx)

        assert asyncio.run(call_remote()) == {"echo": "E11"}
        asyncio.run(mgr.stop())
        assert registry.get("mcp_http_stub__lookup_fields") is None
    finally:
        server.shutdown()
        server.server_close()


def test_mcp_manager_parses_text_event_stream_json_rpc_responses(tmp_path):
    """Compatibility fixture for servers returning JSON-RPC payloads as SSE data lines."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_sse(self, payload: dict) -> None:
            body = (
                ": keepalive\n"
                f"data: {json.dumps({'jsonrpc': '2.0', 'id': 'unrelated', 'result': {}})}\n\n"
                f"event: message\ndata: {json.dumps(payload)}\n\n"
                "data: [DONE]\n\n"
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802 - stdlib callback
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            method = payload.get("method")
            request_id = payload.get("id")
            if method == "initialize":
                self._send_sse({"jsonrpc": "2.0", "id": request_id, "result": {"serverInfo": {"name": "sse"}}})
                return
            if method == "tools/list":
                self._send_sse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "query cohort",
                                "description": "Query one aggregate cohort",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"icd10_code": {"type": "string"}},
                                    "required": ["icd10_code"],
                                },
                            }
                        ]
                    },
                })
                return
            if method == "tools/call":
                code = payload.get("params", {}).get("arguments", {}).get("icd10_code", "")
                self._send_sse({"jsonrpc": "2.0", "id": request_id, "result": {"icd10_code": code, "n_cases": 123}})
                return
            self._send_sse({"jsonrpc": "2.0", "id": request_id, "result": {}})

        def log_message(self, *_):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        registry = ToolRegistry()
        cfg_path = tmp_path / "mcp_servers.json"
        cfg_path.write_text(json.dumps({
            "servers": [
                {
                    "name": "sse stream",
                    "transport": "http-sse",
                    "url": f"http://127.0.0.1:{server.server_port}/mcp",
                }
            ],
        }))
        mgr = McpManager(registry, config_path=cfg_path)
        assert asyncio.run(mgr.start()) == 1
        handler = registry.get("mcp_sse_stream__query_cohort")
        assert handler is not None

        async def call_remote():
            ctx = ToolContext(
                name=handler.name,
                args={"icd10_code": "E11"},
                capabilities=handler.required_capabilities(),
            )
            return await handler.handle(ctx)

        assert asyncio.run(call_remote()) == {"icd10_code": "E11", "n_cases": 123}
    finally:
        asyncio.run(mgr.stop())
        server.shutdown()
        server.server_close()


def test_mcp_manager_retries_transient_http_start_failure(tmp_path):
    attempts = {"initialize": 0}

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802 - stdlib callback
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            method = payload.get("method")
            request_id = payload.get("id")
            if method == "initialize":
                attempts["initialize"] += 1
                if attempts["initialize"] == 1:
                    self._send_json(503, {"error": "warming up"})
                    return
                self._send_json(200, {"jsonrpc": "2.0", "id": request_id, "result": {}})
                return
            if method == "tools/list":
                self._send_json(200, {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]},
                })
                return
            self._send_json(200, {"jsonrpc": "2.0", "id": request_id, "result": {}})

        def log_message(self, *_):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        registry = ToolRegistry()
        cfg_path = tmp_path / "mcp_servers.json"
        cfg_path.write_text(json.dumps({
            "servers": [
                {
                    "name": "flaky",
                    "transport": "http-sse",
                    "url": f"http://127.0.0.1:{server.server_port}/mcp",
                    "max_retries": 1,
                    "retry_backoff_s": 0,
                }
            ],
        }))
        mgr = McpManager(registry, config_path=cfg_path)
        assert asyncio.run(mgr.start()) == 1
        assert attempts["initialize"] == 2
        assert mgr.status()[0]["loaded_tools"] == 1
        assert registry.get("mcp_flaky__echo") is not None
    finally:
        asyncio.run(mgr.stop())
        server.shutdown()
        server.server_close()


def test_mcp_tool_call_reconnects_once_after_http_failure(tmp_path):
    counts = {"call": 0, "initialize": 0}

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802 - stdlib callback
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            method = payload.get("method")
            request_id = payload.get("id")
            if method == "initialize":
                counts["initialize"] += 1
                self._send_json(200, {"jsonrpc": "2.0", "id": request_id, "result": {}})
                return
            if method == "tools/list":
                self._send_json(200, {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "required": ["query"],
                                },
                            }
                        ]
                    },
                })
                return
            if method == "tools/call":
                counts["call"] += 1
                if counts["call"] == 1:
                    self._send_json(503, {"error": "transient"})
                    return
                query = payload.get("params", {}).get("arguments", {}).get("query")
                self._send_json(200, {"jsonrpc": "2.0", "id": request_id, "result": {"echo": query}})
                return
            self._send_json(200, {"jsonrpc": "2.0", "id": request_id, "result": {}})

        def log_message(self, *_):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        registry = ToolRegistry()
        cfg_path = tmp_path / "mcp_servers.json"
        cfg_path.write_text(json.dumps({
            "servers": [
                {
                    "name": "flaky",
                    "transport": "http-sse",
                    "url": f"http://127.0.0.1:{server.server_port}/mcp",
                    "max_retries": 0,
                    "retry_backoff_s": 0,
                    "restart_on_call_failure": True,
                }
            ],
        }))
        mgr = McpManager(registry, config_path=cfg_path)
        assert asyncio.run(mgr.start()) == 1
        handler = registry.get("mcp_flaky__echo")
        assert handler is not None

        async def call_remote():
            ctx = ToolContext(
                name=handler.name,
                args={"query": "E11"},
                capabilities=handler.required_capabilities(),
            )
            return await handler.handle(ctx)

        assert asyncio.run(call_remote()) == {"echo": "E11"}
        assert counts["call"] == 2
        assert counts["initialize"] == 2
        assert mgr.restart_counts["flaky"] == 1
        assert mgr.status()[0]["restarts"] == 1
    finally:
        asyncio.run(mgr.stop())
        server.shutdown()
        server.server_close()


def test_tool_registry_hydrates_from_legacy_count():
    """Smoke test: hydrating from legacy registry exposes >0 tools."""
    from biobank_agent.registry import autodiscover_skills, get_registry

    autodiscover_skills()
    legacy = get_registry()
    if len(legacy) == 0:
        pytest.skip("no legacy skills loaded in this environment")
    registry = ToolRegistry(legacy=legacy)
    n = registry.hydrate_from_legacy()
    assert n == len(legacy)
    sample_name = next(iter(legacy._schemas))
    handler = registry.get(sample_name)
    assert handler is not None
    assert handler.name == sample_name
