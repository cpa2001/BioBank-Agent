"""Local HTTP JSON-RPC MCP stub for Biobank Agent.

Run this in one terminal:

    python docs/examples/mcp_http_stub_demo.py

Then create an MCP config, for example `/tmp/biobank_mcp_servers.json`:

    {
      "servers": [
        {
          "name": "demo",
          "transport": "http-sse",
          "url": "http://127.0.0.1:8765/mcp",
          "auth": {"bearer_token_env": "OPTIONAL_MCP_TOKEN"},
          "timeout_s": 5,
          "max_retries": 1,
          "retry_backoff_s": 0.25,
          "restart_on_call_failure": true
        }
      ]
    }

In the Biobank Agent REPL:

    /mcp-start
    /mcp-list
    /mcp-health
    /mcp-call demo__echo {"query":"E11"}
    /mcp-health --repair
    /mcp-stop

Point the agent at that file with either:

    BIOBANK_MCP_CONFIG=/tmp/biobank_mcp_servers.json biobank

or `MCP_CONFIG_PATH=/tmp/biobank_mcp_servers.json` in `.env`.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        method = payload.get("method")
        request_id = payload.get("id")
        if method == "initialize":
            result = {"serverInfo": {"name": "demo"}}
        elif method == "notifications/initialized":
            self._send({})
            return
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo a query for MCP smoke testing",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    }
                ]
            }
        elif method == "tools/call":
            args = payload.get("params", {}).get("arguments", {})
            result = {"echo": args.get("query", ""), "source": "mcp_stub"}
        else:
            self._send({"jsonrpc": "2.0", "id": request_id, "error": {"message": "unknown method"}})
            return
        self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: object) -> None:
        return


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8765), Handler)
    print("MCP stub listening on http://127.0.0.1:8765/mcp")
    server.serve_forever()
