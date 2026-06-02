"""Auto-spawning Model Context Protocol (MCP) client manager.

The manager speaks JSON-RPC over STDIO to child processes and over HTTP
for Streamable-HTTP / HTTP-SSE compatible servers, discovers each
server's tool list via the standard ``tools/list`` request, and registers
remote tools as ``ToolHandler`` instances so the LLM sees them alongside
native skills.

Configuration file: ``~/.biobank_agent/mcp_servers.json`` ::

    {
        "servers": [
            {
                "name": "github",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_TOKEN": "..."}
            },
            {
                "name": "panukb",
                "transport": "http-sse",
                "url": "https://mcp.panukb.example/mcp",
                "auth": {"bearer_token_env": "PANUKB_MCP_TOKEN"}
            }
        ]
    }

On ``manager.start()`` each server is spawned, ``initialize`` /
``tools/list`` are exchanged, and remote tools become callable via
``ToolRegistry``. The MCP wire format follows
https://modelcontextprotocol.io/specification/.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import select
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from biobank_agent.core.tools.protocol import (
    ActionClass,
    Capability,
    SafetyClass,
    ToolContext,
    ToolHandler,
    ToolSpec,
    TrajectorySerialization,
    WorkspaceScope,
    _BaseHandler,
)
from biobank_agent.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class _McpClient(Protocol):
    name: str

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def list_tools(self) -> list[dict[str, Any]]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class McpServerConfig:
    name: str
    transport: str = "stdio"  # stdio | http-sse
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    bearer_token: str = ""
    bearer_token_env: str = ""
    api_key_env: str = ""
    api_key_header: str = "x-api-key"
    url: str = ""  # for http-sse
    enabled: bool = True
    timeout_s: float = 20.0
    max_retries: int = 0
    retry_backoff_s: float = 0.25
    restart_on_call_failure: bool = True


# ── JSON-RPC over STDIO client ──────────────────────────────


class _StdioMcpClient:
    """Minimal JSON-RPC client for an MCP STDIO server.

    We deliberately don't pull in a third-party SDK so M2 stays self-
    contained. M4 may swap in ``mcp-sdk-python`` once the protocol
    stabilises further.
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str],
        timeout_s: float = 20.0,
    ) -> None:
        self.name = name
        self.command = command
        self.args = list(args)
        self.env = dict(env)
        self.timeout_s = float(timeout_s)
        self.proc: Optional[subprocess.Popen] = None
        self._next_id = 1
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if not self.command:
            raise RuntimeError(f"MCP STDIO server {self.name!r} has no command")
        merged_env = {**os.environ, **self.env}
        # Use sync Popen so this code path also works in offline tests
        # without depending on asyncio subprocess transport availability.
        self.proc = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
            text=True,
            bufsize=1,
        )
        # MCP servers typically expect an ``initialize`` request first.
        await self._call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "biobank-agent", "version": "v3"},
                "capabilities": {"tools": {}},
            },
        )
        await self._notify("notifications/initialized", {})

    async def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc = None

    async def list_tools(self) -> list[dict[str, Any]]:
        out = await self._call("tools/list", {})
        return list(out.get("tools", []))

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        out = await self._call("tools/call", {"name": name, "arguments": arguments})
        return out

    async def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_call, method, params)

    def _sync_call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.proc or not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError(f"MCP server {self.name!r} not running")
        request_id = self._next_id
        self._next_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        # Read until we see a matching id (skip notifications).
        deadline = time.monotonic() + max(self.timeout_s, 0.1)
        stdout_fd = self.proc.stdout.fileno()
        while True:
            if self.proc.poll() is not None:
                stderr = ""
                if self.proc.stderr:
                    try:
                        stderr = self.proc.stderr.read(1000)
                    except Exception:
                        stderr = ""
                suffix = f": {stderr.strip()}" if stderr.strip() else ""
                raise RuntimeError(f"MCP server {self.name!r} exited{suffix}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"MCP server {self.name!r} timed out waiting for {method}"
                )
            readable, _, _ = select.select([stdout_fd], [], [], remaining)
            if not readable:
                raise TimeoutError(
                    f"MCP server {self.name!r} timed out waiting for {method}"
                )
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP server {self.name!r} closed pipe")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("MCP %s: non-JSON line %r", self.name, line.strip())
                continue
            if msg.get("id") == request_id:
                if "error" in msg:
                    raise RuntimeError(f"MCP error: {msg['error']}")
                return msg.get("result") or {}

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        if not self.proc or not self.proc.stdin:
            return
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()


# ── JSON-RPC over HTTP / HTTP-SSE-compatible client ─────────


class _HttpMcpClient:
    """Minimal HTTP JSON-RPC client for streamable MCP servers.

    Real deployments commonly expose either the newer Streamable HTTP
    endpoint or a legacy HTTP-SSE message endpoint. Both accept JSON-RPC
    POSTs; responses may be JSON or ``text/event-stream`` containing
    ``data: {...}`` lines. This client supports those server shapes without
    introducing a mandatory MCP SDK dependency.
    """

    def __init__(
        self,
        name: str,
        url: str,
        env: dict[str, str],
        timeout_s: float = 20.0,
        headers: dict[str, str] | None = None,
        bearer_token: str = "",
        bearer_token_env: str = "",
        api_key_env: str = "",
        api_key_header: str = "x-api-key",
    ) -> None:
        self.name = name
        self.url = str(url or "").strip()
        self.env = dict(env)
        self.timeout_s = float(timeout_s)
        self.headers = dict(headers or {})
        self.bearer_token = str(bearer_token or "")
        self.bearer_token_env = str(bearer_token_env or "")
        self.api_key_env = str(api_key_env or "")
        self.api_key_header = str(api_key_header or "x-api-key")
        self._next_id = 1
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if not self.url:
            raise RuntimeError(f"MCP HTTP server {self.name!r} has no url")
        await self._call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "biobank-agent", "version": "v3"},
                "capabilities": {"tools": {}},
            },
        )
        await self._notify("notifications/initialized", {})

    async def stop(self) -> None:
        return None

    async def list_tools(self) -> list[dict[str, Any]]:
        out = await self._call("tools/list", {})
        return list(out.get("tools", []))

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._call("tools/call", {"name": name, "arguments": arguments})

    async def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_call, method, params)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        headers.update({str(k): str(v) for k, v in self.headers.items()})
        for key, value in self.env.items():
            if key.upper().startswith("HEADER_"):
                headers[key[7:].replace("_", "-")] = str(value)
        bearer = self.bearer_token or (
            os.environ.get(self.bearer_token_env, "") if self.bearer_token_env else ""
        )
        if bearer and not any(k.lower() == "authorization" for k in headers):
            headers["Authorization"] = f"Bearer {bearer}"
        api_key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if api_key and not any(k.lower() == self.api_key_header.lower() for k in headers):
            headers[self.api_key_header] = api_key
        return headers

    def _sync_call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        import httpx

        request_id = self._next_id
        self._next_id += 1
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        response = httpx.post(self.url, json=message, headers=self._headers(), timeout=self.timeout_s)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" in content_type:
            return self._parse_sse_response(response.text, request_id)
        payload = response.json()
        if payload.get("id") != request_id:
            raise RuntimeError(f"MCP HTTP response id mismatch for {self.name!r}")
        if "error" in payload:
            raise RuntimeError(f"MCP HTTP error: {payload['error']}")
        return payload.get("result") or {}

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await asyncio.to_thread(self._sync_notify, method, params)

    def _sync_notify(self, method: str, params: dict[str, Any]) -> None:
        import httpx

        message = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            httpx.post(self.url, json=message, headers=self._headers(), timeout=self.timeout_s)
        except Exception:
            logger.debug("MCP HTTP notify %s failed for %s", method, self.name, exc_info=True)

    @staticmethod
    def _parse_sse_response(text: str, request_id: int) -> dict[str, Any]:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if payload.get("id") != request_id:
                continue
            if "error" in payload:
                raise RuntimeError(f"MCP HTTP error: {payload['error']}")
            return payload.get("result") or {}
        raise RuntimeError("MCP HTTP-SSE response did not contain matching JSON-RPC result")


# ── ToolHandler that routes calls to the MCP client ─────────


class _McpToolHandler(_BaseHandler):
    """Wraps a remote MCP tool as a v3 ToolHandler."""

    def __init__(
        self,
        server_name: str,
        remote_name: str,
        spec: ToolSpec,
        capabilities: frozenset[Capability],
        manager: "McpManager",
    ) -> None:
        # Prefix remote tool with server name so namespaces don't collide
        # with native skills. Display name shown to the LLM is
        # ``mcp_<server>__<tool>``.
        local_name = f"mcp_{_sanitize_tool_name(server_name)}__{_sanitize_tool_name(remote_name)}"
        renamed_spec = ToolSpec(
            name=local_name,
            description=spec.description,
            parameters=spec.parameters,
            required=spec.required,
        )
        super().__init__(
            _name=local_name,
            _spec=renamed_spec,
            _caps=capabilities,
            _is_mutating=False,
        )
        self.server_name = server_name
        self.manager = manager
        self.remote_name = remote_name

    async def handle(self, ctx: ToolContext) -> dict[str, Any]:
        out = await self.manager.call_tool(self.server_name, self.remote_name, dict(ctx.args or {}))
        return out if isinstance(out, dict) else {"output": out}


# ── Manager ─────────────────────────────────────────────────


class McpManager:
    """Orchestrates auto-discovery + registration of MCP servers."""

    def __init__(self, registry: ToolRegistry, config_path: Optional[Path] = None) -> None:
        self.registry = registry
        env_config = os.environ.get("BIOBANK_MCP_CONFIG", "").strip()
        self.config_path = Path(config_path or env_config or (
            Path.home() / ".biobank_agent" / "mcp_servers.json"
        )).expanduser()
        self.clients: dict[str, _McpClient] = {}
        self.configs: dict[str, McpServerConfig] = {}
        self.handlers: list[_McpToolHandler] = []
        self.handlers_by_server: dict[str, list[_McpToolHandler]] = {}
        self.errors: dict[str, str] = {}
        self.loaded_counts: dict[str, int] = {}
        self.restart_counts: dict[str, int] = {}
        self.last_started_at: dict[str, float] = {}
        self.last_health_at: dict[str, float] = {}

    def load_config(self) -> list[McpServerConfig]:
        if not self.config_path.exists():
            return []
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("MCP config %s unreadable: %s", self.config_path, e)
            return []
        servers = data.get("servers") or []
        out: list[McpServerConfig] = []
        for entry in servers:
            try:
                row = dict(entry)
                auth = row.pop("auth", None)
                if isinstance(auth, dict):
                    for key in ("bearer_token", "bearer_token_env", "api_key_env", "api_key_header"):
                        if key in auth and key not in row:
                            row[key] = auth[key]
                    if isinstance(auth.get("headers"), dict):
                        row["headers"] = {**dict(auth["headers"]), **dict(row.get("headers") or {})}
                cfg = McpServerConfig(**row)
            except TypeError as e:
                logger.warning("Skipping malformed MCP server entry %s: %s", entry, e)
                continue
            out.append(cfg)
        return out

    async def start(self) -> int:
        await self.stop()
        self.errors.clear()
        self.loaded_counts.clear()
        self.configs = {cfg.name: cfg for cfg in self.load_config()}
        loaded = 0
        for cfg in self.configs.values():
            if not cfg.enabled:
                continue
            loaded += await self._start_server_with_retries(cfg)
        return loaded

    async def _start_server_with_retries(self, cfg: McpServerConfig) -> int:
        attempts = max(1, int(cfg.max_retries) + 1)
        last_error = ""
        for attempt in range(attempts):
            try:
                return await self._start_server_once(cfg)
            except Exception as e:
                last_error = str(e)
                self.errors[cfg.name] = last_error
                logger.warning(
                    "Could not start MCP server %s (attempt %s/%s): %s",
                    cfg.name,
                    attempt + 1,
                    attempts,
                    e,
                )
                await self._stop_server(cfg.name)
                if attempt + 1 < attempts:
                    await asyncio.sleep(max(float(cfg.retry_backoff_s), 0.0))
        self.loaded_counts[cfg.name] = 0
        if last_error:
            self.errors[cfg.name] = last_error
        return 0

    async def _start_server_once(self, cfg: McpServerConfig) -> int:
        await self._stop_server(cfg.name)
        transport = cfg.transport.strip().lower()
        if transport == "stdio":
            client: _McpClient = _StdioMcpClient(
                name=cfg.name,
                command=cfg.command,
                args=list(cfg.args),
                env=dict(cfg.env),
                timeout_s=cfg.timeout_s,
            )
        elif transport in {"http", "http-sse", "sse", "streamable-http"}:
            client = _HttpMcpClient(
                name=cfg.name,
                url=cfg.url,
                env=dict(cfg.env),
                timeout_s=cfg.timeout_s,
                headers=dict(cfg.headers),
                bearer_token=cfg.bearer_token,
                bearer_token_env=cfg.bearer_token_env,
                api_key_env=cfg.api_key_env,
                api_key_header=cfg.api_key_header,
            )
        else:
            self.errors[cfg.name] = f"unsupported transport: {cfg.transport}"
            logger.info("Transport %s not supported for server %s; skipping", cfg.transport, cfg.name)
            return 0

        try:
            await asyncio.wait_for(client.start(), timeout=max(cfg.timeout_s, 0.1) + 1)
            tools = await asyncio.wait_for(client.list_tools(), timeout=max(cfg.timeout_s, 0.1) + 1)
        except Exception:
            try:
                await client.stop()
            except Exception:
                logger.debug("Failed to stop MCP client after start error", exc_info=True)
            raise

        self.clients[cfg.name] = client
        self.last_started_at[cfg.name] = time.time()
        self.last_health_at[cfg.name] = time.time()
        self.errors.pop(cfg.name, None)
        server_loaded = self._register_tools(cfg.name, tools)
        self.loaded_counts[cfg.name] = server_loaded
        return server_loaded

    def _register_tools(self, server_name: str, tools: list[dict[str, Any]]) -> int:
        server_loaded = 0
        self.handlers_by_server[server_name] = []
        for tool in tools:
            remote_name = str(tool.get("name", "")).strip()
            if not remote_name:
                self.errors[server_name] = "tools/list returned a tool without a name"
                continue
            spec = ToolSpec(
                name=remote_name,
                description=str(tool.get("description", "")),
                parameters=dict(
                    (tool.get("inputSchema") or {}).get("properties", {})
                ),
                required=list(
                    (tool.get("inputSchema") or {}).get("required", [])
                ),
                safety_class=SafetyClass.NETWORK,
                approval_requirement="ask_before_network",
                workspace_scope=WorkspaceScope.EXTERNAL_READ,
                action_classes=(ActionClass.NETWORK,),
                trajectory_serialization=TrajectorySerialization.METADATA_ONLY,
            )
            handler = _McpToolHandler(
                server_name=server_name,
                remote_name=spec.name,
                spec=spec,
                capabilities=frozenset({Capability.NETWORK, Capability.READ_DATA}),
                manager=self,
            )
            try:
                self.registry.register(handler)
            except Exception as e:
                self.errors[handler.name] = str(e)
                logger.warning("Registering MCP tool %s failed: %s", handler.name, e)
                continue
            self.handlers.append(handler)
            self.handlers_by_server.setdefault(server_name, []).append(handler)
            server_loaded += 1
        return server_loaded

    async def stop(self) -> None:
        for name in list(self.clients):
            await self._stop_server(name)
        self.clients.clear()
        self.handlers.clear()
        self.handlers_by_server.clear()
        self.loaded_counts.clear()

    async def _stop_server(self, name: str) -> None:
        handlers = list(self.handlers_by_server.get(name, []))
        for handler in handlers:
            try:
                self.registry.unregister(handler.name)
            except Exception:
                logger.debug("Failed to unregister MCP tool %s", handler.name, exc_info=True)
            try:
                self.handlers.remove(handler)
            except ValueError:
                pass
        self.handlers_by_server.pop(name, None)
        self.loaded_counts.pop(name, None)
        client = self.clients.pop(name, None)
        if client is not None:
            await client.stop()

    async def reconnect(self, name: str) -> bool:
        """Restart one configured server and refresh its registered tools."""
        cfg = self.configs.get(name)
        if cfg is None:
            self.configs = {row.name: row for row in self.load_config()}
            cfg = self.configs.get(name)
        if cfg is None or not cfg.enabled:
            self.errors[name] = "server is not configured or disabled"
            return False
        self.restart_counts[name] = self.restart_counts.get(name, 0) + 1
        loaded = await self._start_server_with_retries(cfg)
        return loaded > 0

    async def call_tool(self, server_name: str, remote_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a remote tool, reconnecting once when configured."""
        client = self.clients.get(server_name)
        if client is None:
            if not await self.reconnect(server_name):
                raise RuntimeError(f"MCP server {server_name!r} is not running")
            client = self.clients[server_name]
        try:
            return await client.call_tool(remote_name, arguments)
        except Exception as exc:
            cfg = self.configs.get(server_name)
            self.errors[server_name] = f"tools/call failed: {exc}"
            if cfg is None or not cfg.restart_on_call_failure:
                raise
            logger.warning("MCP call failed for %s; reconnecting once: %s", server_name, exc)
            if not await self.reconnect(server_name):
                raise
            return await self.clients[server_name].call_tool(remote_name, arguments)

    async def health_check(self, *, repair: bool = False) -> list[dict[str, Any]]:
        """Probe configured servers; optionally reconnect unhealthy ones."""
        if not self.configs:
            self.configs = {row.name: row for row in self.load_config()}
        rows: list[dict[str, Any]] = []
        for cfg in self.configs.values():
            if not cfg.enabled:
                rows.append({"name": cfg.name, "ok": False, "running": False, "error": "disabled"})
                continue
            client = self.clients.get(cfg.name)
            if client is None:
                ok = await self.reconnect(cfg.name) if repair else False
                rows.append({
                    "name": cfg.name,
                    "ok": ok,
                    "running": cfg.name in self.clients,
                    "error": "" if ok else "not running",
                })
                continue
            try:
                tools = await asyncio.wait_for(client.list_tools(), timeout=max(cfg.timeout_s, 0.1) + 1)
            except Exception as exc:
                self.errors[cfg.name] = f"health check failed: {exc}"
                ok = await self.reconnect(cfg.name) if repair else False
                rows.append({
                    "name": cfg.name,
                    "ok": ok,
                    "running": cfg.name in self.clients,
                    "error": "" if ok else str(exc),
                })
                continue
            self.last_health_at[cfg.name] = time.time()
            self.loaded_counts[cfg.name] = len(tools)
            self.errors.pop(cfg.name, None)
            rows.append({"name": cfg.name, "ok": True, "running": True, "error": "", "tools": len(tools)})
        return rows

    def status(self) -> list[dict[str, Any]]:
        configs = self.load_config()
        rows: list[dict[str, Any]] = []
        for cfg in configs:
            rows.append({
                "name": cfg.name,
                "transport": cfg.transport,
                "enabled": cfg.enabled,
                "running": cfg.name in self.clients,
                "loaded_tools": self.loaded_counts.get(cfg.name, 0),
                "error": self.errors.get(cfg.name, ""),
                "restarts": self.restart_counts.get(cfg.name, 0),
                "last_started_at": self.last_started_at.get(cfg.name, 0.0),
                "last_health_at": self.last_health_at.get(cfg.name, 0.0),
            })
        return rows


def _sanitize_tool_name(name: str) -> str:
    """Return a JSON-schema/OpenAI-safe tool name component."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned or "tool"


__all__ = ["McpServerConfig", "McpManager"]
