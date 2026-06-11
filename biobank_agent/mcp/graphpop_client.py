"""GraphPop MCP client — connects to GraphPop's 21 population genomics tools.

GraphPop (bioRxiv:2026.04.11.717929v1) is a graph-native population genomics
engine built on Neo4j with 12 stored procedures. It exposes 21 MCP tools via
the Model Context Protocol (JSON-RPC 2.0 over HTTP/SSE).

Key capabilities:
  - O(V×K) population statistics (diversity, Fst, SFS) via pre-aggregated allele counts
  - Annotation-conditioned queries (consequence/pathway/gene filtering)
  - Multi-statistic convergence (selection scans stored on graph nodes)
  - Bit-packed haplotype computation (iHS, XP-EHH, nSL, ROH)
  - Persistent analytical records (results become graph properties)

Architecture decision: Use MCP protocol over HTTP via httpx
(already in deps), not direct Neo4j driver. This preserves GraphPop's domain
abstraction layer (21 tools) rather than reimplementing from raw Cypher.

Usage
-----
    from biobank_agent.mcp.graphpop_client import GraphPopMCPClient

    client = GraphPopMCPClient(endpoint="http://localhost:8080/mcp")
    await client.connect()
    result = await client.call_tool("diversity", {
        "chromosome": "chr1", "start": 1, "end": 43270923,
        "population": "EUR", "consequence": "missense_variant"
    })
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# httpx is already in biobank_agent dependencies
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False


# ── GraphPop Tool Registry (from paper's 12 procedures + MCP tools) ────

@dataclass
class MCPToolSpec:
    """Specification of a single MCP tool."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    category: str = ""  # "fast_path" | "full_path" | "query" | "admin"


# 21 MCP tools documented in GraphPop paper (12 procedures + query/admin tools)
GRAPHPOP_TOOLS: dict[str, MCPToolSpec] = {
    # ── FAST PATH (O(V×K) pre-aggregated) ──────────────────────
    "diversity": MCPToolSpec(
        name="diversity",
        description="Compute nucleotide diversity (π, θW, Tajima's D) for a population in a genomic region. Supports consequence conditioning.",
        parameters={
            "chromosome": {"type": "string", "required": True},
            "start": {"type": "integer", "required": True},
            "end": {"type": "integer", "required": True},
            "population": {"type": "string", "required": True},
            "consequence": {"type": "string", "required": False, "description": "VEP consequence type filter"},
            "pathway": {"type": "string", "required": False},
        },
        category="fast_path",
    ),
    "fst": MCPToolSpec(
        name="fst",
        description="Compute Weir-Cockerham Fst between two populations. Supports annotation conditioning.",
        parameters={
            "chromosome": {"type": "string", "required": True},
            "start": {"type": "integer", "required": True},
            "end": {"type": "integer", "required": True},
            "population1": {"type": "string", "required": True},
            "population2": {"type": "string", "required": True},
            "consequence": {"type": "string", "required": False},
        },
        category="fast_path",
    ),
    "sfs": MCPToolSpec(
        name="sfs",
        description="Compute Site Frequency Spectrum for a population.",
        parameters={
            "chromosome": {"type": "string", "required": True},
            "start": {"type": "integer", "required": True},
            "end": {"type": "integer", "required": True},
            "population": {"type": "string", "required": True},
        },
        category="fast_path",
    ),
    "joint_sfs": MCPToolSpec(
        name="joint_sfs",
        description="Compute Joint Site Frequency Spectrum between two populations.",
        parameters={
            "chromosome": {"type": "string", "required": True},
            "start": {"type": "integer", "required": True},
            "end": {"type": "integer", "required": True},
            "population1": {"type": "string", "required": True},
            "population2": {"type": "string", "required": True},
        },
        category="fast_path",
    ),
    "genome_scan": MCPToolSpec(
        name="genome_scan",
        description="Sliding-window genome scan computing π, θW, Tajima's D, Fst per window. Results stored as GenomicWindow nodes.",
        parameters={
            "chromosome": {"type": "string", "required": True},
            "population": {"type": "string", "required": True},
            "window_size": {"type": "integer", "required": False, "default": 50000},
            "step_size": {"type": "integer", "required": False, "default": 25000},
        },
        category="fast_path",
    ),
    "population_summary": MCPToolSpec(
        name="population_summary",
        description="Compute summary statistics for a population (mean diversity, heterozygosity, etc.).",
        parameters={
            "population": {"type": "string", "required": True},
        },
        category="fast_path",
    ),

    # ── FULL PATH (bit-packed haplotype computation) ───────────
    "ihs": MCPToolSpec(
        name="ihs",
        description="Compute integrated Haplotype Score (iHS) for selection detection.",
        parameters={
            "chromosome": {"type": "string", "required": True},
            "population": {"type": "string", "required": True},
            "maf_threshold": {"type": "number", "required": False, "default": 0.05},
        },
        category="full_path",
    ),
    "xpehh": MCPToolSpec(
        name="xpehh",
        description="Compute Cross-Population Extended Haplotype Homozygosity (XP-EHH).",
        parameters={
            "chromosome": {"type": "string", "required": True},
            "population1": {"type": "string", "required": True},
            "population2": {"type": "string", "required": True},
        },
        category="full_path",
    ),
    "nsl": MCPToolSpec(
        name="nsl",
        description="Compute number of Segregating Sites by Length (nSL).",
        parameters={
            "chromosome": {"type": "string", "required": True},
            "population": {"type": "string", "required": True},
        },
        category="full_path",
    ),
    "roh": MCPToolSpec(
        name="roh",
        description="Detect Runs of Homozygosity (ROH).",
        parameters={
            "chromosome": {"type": "string", "required": True},
            "population": {"type": "string", "required": True},
            "min_length_kb": {"type": "integer", "required": False, "default": 1000},
        },
        category="full_path",
    ),
    "garud_h": MCPToolSpec(
        name="garud_h",
        description="Compute Garud's H12/H2H1 statistics for soft sweep detection.",
        parameters={
            "chromosome": {"type": "string", "required": True},
            "population": {"type": "string", "required": True},
            "window_size": {"type": "integer", "required": False},
        },
        category="full_path",
    ),
    "pbs": MCPToolSpec(
        name="pbs",
        description="Compute Population Branch Statistic for branch-specific selection.",
        parameters={
            "chromosome": {"type": "string", "required": True},
            "target_pop": {"type": "string", "required": True},
            "reference_pop1": {"type": "string", "required": True},
            "reference_pop2": {"type": "string", "required": True},
        },
        category="full_path",
    ),

    # ── QUERY tools (graph traversal) ─────────────────────────
    "convergent_selection": MCPToolSpec(
        name="convergent_selection",
        description="Find genes under convergent selection across multiple populations via multi-statistic graph pattern match.",
        parameters={
            "statistic": {"type": "string", "required": True, "enum": ["ihs", "xpehh", "garud_h12"]},
            "threshold": {"type": "number", "required": True},
            "min_populations": {"type": "integer", "required": False, "default": 2},
        },
        category="query",
    ),
    "pathway_query": MCPToolSpec(
        name="pathway_query",
        description="Query pathway-level statistics via gene→pathway edge traversal.",
        parameters={
            "pathway": {"type": "string", "required": True},
            "statistic": {"type": "string", "required": False},
        },
        category="query",
    ),
    "gene_query": MCPToolSpec(
        name="gene_query",
        description="Query all statistics and annotations for a gene.",
        parameters={
            "gene": {"type": "string", "required": True},
        },
        category="query",
    ),
    "variant_query": MCPToolSpec(
        name="variant_query",
        description="Query variant-level information including annotations and stored statistics.",
        parameters={
            "chromosome": {"type": "string", "required": True},
            "position": {"type": "integer", "required": True},
            "allele": {"type": "string", "required": False},
        },
        category="query",
    ),
    "annotation_filter": MCPToolSpec(
        name="annotation_filter",
        description="Filter variants by VEP consequence type, gene, or pathway before computing statistics.",
        parameters={
            "consequence": {"type": "string", "required": False},
            "gene": {"type": "string", "required": False},
            "pathway": {"type": "string", "required": False},
            "impact": {"type": "string", "required": False, "enum": ["HIGH", "MODERATE", "LOW", "MODIFIER"]},
        },
        category="query",
    ),

    # ── ADMIN tools ───────────────────────────────────────────
    "import_vcf": MCPToolSpec(
        name="import_vcf",
        description="Import VCF file into GraphPop Neo4j database with allele count pre-aggregation.",
        parameters={
            "vcf_path": {"type": "string", "required": True},
            "population_map": {"type": "string", "required": True, "description": "Path to sample→population mapping file"},
        },
        category="admin",
    ),
    "list_populations": MCPToolSpec(
        name="list_populations",
        description="List all populations in the database with sample counts.",
        parameters={},
        category="admin",
    ),
    "list_chromosomes": MCPToolSpec(
        name="list_chromosomes",
        description="List all chromosomes with variant counts.",
        parameters={},
        category="admin",
    ),
    "database_stats": MCPToolSpec(
        name="database_stats",
        description="Return database statistics (total variants, genes, pathways, populations).",
        parameters={},
        category="admin",
    ),
}


@dataclass
class MCPResult:
    """Result from an MCP tool call."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    tool_name: str = ""
    duration_ms: float = 0.0


class GraphPopMCPClient:
    """Async MCP client for GraphPop population genomics tools.

    Communicates via JSON-RPC 2.0 over HTTP with the GraphPop MCP server.
    Falls back gracefully when the server is unreachable.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8080/mcp",
        timeout: float = 60.0,
        neo4j_uri: Optional[str] = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.neo4j_uri = neo4j_uri
        self._connected = False
        self._request_id = 0

    @property
    def tools(self) -> dict[str, MCPToolSpec]:
        """Return the registry of available GraphPop tools."""
        return GRAPHPOP_TOOLS

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Test connection to GraphPop MCP server."""
        if not _HTTPX_AVAILABLE:
            logger.warning("httpx not available — GraphPop MCP client disabled")
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # MCP initialize handshake
                response = await client.post(
                    self.endpoint,
                    json={
                        "jsonrpc": "2.0",
                        "id": 0,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {
                                "name": "biobank-agent",
                                "version": "2.0.0",
                            },
                        },
                    },
                )
                if response.status_code == 200:
                    self._connected = True
                    logger.info("Connected to GraphPop MCP server at %s", self.endpoint)
                    return True
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.info("GraphPop MCP server unavailable at %s: %s", self.endpoint, e)
        except Exception as e:
            logger.warning("GraphPop MCP connection error: %s", e)

        self._connected = False
        return False

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] = None) -> MCPResult:
        """Call a GraphPop MCP tool.

        Parameters
        ----------
        tool_name : str
            One of the 21 registered tool names
        arguments : dict
            Tool-specific arguments

        Returns
        -------
        MCPResult with success status and data/error
        """
        if tool_name not in GRAPHPOP_TOOLS:
            return MCPResult(
                success=False,
                error=f"Unknown tool: '{tool_name}'. Available: {list(GRAPHPOP_TOOLS.keys())}",
                tool_name=tool_name,
            )

        # Pre-flight parameter validation (avoid unnecessary network round-trip)
        spec = GRAPHPOP_TOOLS[tool_name]
        missing = [
            k for k, v in spec.parameters.items()
            if v.get("required") and k not in (arguments or {})
        ]
        if missing:
            return MCPResult(
                success=False,
                error=f"Missing required parameters for '{tool_name}': {missing}",
                tool_name=tool_name,
            )

        if not _HTTPX_AVAILABLE:
            return MCPResult(
                success=False,
                error="httpx not installed — cannot communicate with GraphPop MCP server",
                tool_name=tool_name,
            )

        if not self._connected:
            # Try to connect
            if not await self.connect():
                return MCPResult(
                    success=False,
                    error=f"GraphPop MCP server unavailable at {self.endpoint}",
                    tool_name=tool_name,
                )

        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {},
            },
        }

        start = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.endpoint, json=payload)
                duration_ms = (time.perf_counter() - start) * 1000

                if response.status_code != 200:
                    return MCPResult(
                        success=False,
                        error=f"HTTP {response.status_code}: {response.text[:200]}",
                        tool_name=tool_name,
                        duration_ms=duration_ms,
                    )

                result = response.json()

                # Handle JSON-RPC error
                if "error" in result:
                    return MCPResult(
                        success=False,
                        error=result["error"].get("message", str(result["error"])),
                        tool_name=tool_name,
                        duration_ms=duration_ms,
                    )

                # Extract MCP tool result
                content = result.get("result", {}).get("content", [])
                # MCP returns list of content blocks; extract text
                data = None
                for block in content:
                    if block.get("type") == "text":
                        try:
                            data = json.loads(block["text"])
                        except json.JSONDecodeError:
                            data = block["text"]
                    elif block.get("type") == "resource":
                        data = block

                return MCPResult(
                    success=True,
                    data=data,
                    tool_name=tool_name,
                    duration_ms=duration_ms,
                )

        except httpx.TimeoutException:
            duration_ms = (time.perf_counter() - start) * 1000
            return MCPResult(
                success=False,
                error=f"Timeout after {self.timeout}s calling {tool_name}",
                tool_name=tool_name,
                duration_ms=duration_ms,
            )
        except httpx.ConnectError as e:
            self._connected = False
            return MCPResult(
                success=False,
                error=f"Connection lost to GraphPop: {e}",
                tool_name=tool_name,
            )
        except Exception as e:
            return MCPResult(
                success=False,
                error=f"Unexpected error calling {tool_name}: {e}",
                tool_name=tool_name,
            )

    def list_tools(self) -> list[dict[str, Any]]:
        """Return tool specifications in OpenAI function-calling format.

        This allows the agent to discover GraphPop capabilities dynamically.
        """
        tools = []
        for spec in GRAPHPOP_TOOLS.values():
            properties = {}
            required = []
            for param_name, param_spec in spec.parameters.items():
                properties[param_name] = {
                    "type": param_spec.get("type", "string"),
                    "description": param_spec.get("description", ""),
                }
                if param_spec.get("enum"):
                    properties[param_name]["enum"] = param_spec["enum"]
                if param_spec.get("required", False):
                    required.append(param_name)

            tools.append({
                "type": "function",
                "function": {
                    "name": f"graphpop_{spec.name}",
                    "description": spec.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return tools

    def get_tool_summary(self) -> str:
        """Return a concise summary of available tools for system prompt."""
        lines = ["## GraphPop Population Genomics Tools (21 MCP tools)\n"]
        for category in ("fast_path", "full_path", "query", "admin"):
            cat_tools = [t for t in GRAPHPOP_TOOLS.values() if t.category == category]
            if cat_tools:
                label = {
                    "fast_path": "Fast Path (O(V×K) pre-aggregated)",
                    "full_path": "Full Path (bit-packed haplotype)",
                    "query": "Graph Queries",
                    "admin": "Administration",
                }[category]
                lines.append(f"### {label}")
                for t in cat_tools:
                    lines.append(f"- **{t.name}**: {t.description}")
                lines.append("")
        return "\n".join(lines)
