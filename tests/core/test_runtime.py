"""Integration tests for AsyncAgent that mock the LLM stream.

The legacy Agent has many side-effects (DuckDB, parquet, registry
hot-load), so for unit testing we substitute a stub object that
exposes the minimal surface AsyncAgent uses.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from biobank_agent.core.events import AgentEvent, AgentEventType
import biobank_agent.core.runtime as runtime_mod
from biobank_agent.core.llm.client import StreamFinal, StreamTextDelta, StreamToolDelta
from biobank_agent.core.llm.stream_parser import ToolArgFragment
from biobank_agent.core.runtime import AsyncAgent, _LegacyExecutionHandler
from biobank_agent.core.tools.protocol import Capability
from biobank_agent.llm import LLMResponse, ToolCall


# ── Stub LLM ─────────────────────────────────────────────────


class _StubAsyncLLMClient:
    """Replays a canned StreamEvent script, one round per stream_async() call.

    ``rounds`` may be:
      - a flat list of StreamEvents (single-round scripts), or
      - a list of lists, one per round (multi-round).
    """

    def __init__(self, rounds: list) -> None:
        if rounds and isinstance(rounds[0], (list, tuple)):
            self.rounds: list[list] = [list(r) for r in rounds]
        else:
            self.rounds = [list(rounds)]
        self._idx = 0

    async def stream_async(self, **_: Any):
        if self._idx >= len(self.rounds):
            # No more script -> emit empty final response.
            from biobank_agent.llm import LLMResponse  # local import for stub

            yield StreamFinal(response=LLMResponse(text="", tool_calls=[], usage={}))
            return
        script = self.rounds[self._idx]
        self._idx += 1
        for item in script:
            await asyncio.sleep(0)
            yield item


# ── Stub legacy Agent ────────────────────────────────────────


class _StubReproducibility:
    def checkpoint(self, *args, **kwargs) -> str:
        return "ctx_stub"

    def create_audit_log(self, *args, **kwargs):
        return SimpleNamespace(entry_id="audit_stub")


class _StubRegistry:
    def tool_schemas(self):
        return []


class _StubMemory:
    def upsert_node(self, **_):
        pass


class _StubStudySpecCompiler:
    def compile(self, _query):
        raise RuntimeError("no spec")


class _StubDataManager:
    def __init__(self):
        self.conn = None

    def count_subjects(self):
        return 0


class _StubSettings:
    def __init__(self, tmp: Path):
        self.reports_dir = tmp / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir = tmp / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.context_window = 8192
        self.max_tool_rounds = 4
        self.llm_model = "stub-model"
        self.llm_base_url = "http://localhost"
        self.llm_api_key = "key"
        self.tool_call_content_mode = "null"
        self.bank_id = "test_bank"
        self.biobank_name = "test_bank"


class _StubState:
    def __init__(self):
        self.interrupted = False
        self.records: list = []
        self.figures: list = []
        self.cohorts: dict = {}
        self.token_usage = SimpleNamespace(update=lambda usage: None)
        self.custom_data: dict = {}
        self.executive_findings: list = []
        self.provenances: list = []
        self.last_orchestration: dict = {}


class _StubOrchestrator:
    def _wrap_llm_response(self, raw, strategy, model_id):
        return SimpleNamespace(
            claims=[],
            evidence_links=[],
            safety_status="UNKNOWN",
            debate_trace={},
            text=raw.text,
            tool_calls=raw.tool_calls,
            usage=raw.usage,
            has_tool_calls=bool(raw.tool_calls),
            to_dict=lambda: {},
        )


class _StubAgent:
    """Minimal legacy-Agent surface for AsyncAgent."""

    def __init__(self, tmp: Path, on_skill: Any = None):
        self.settings = _StubSettings(tmp)
        self.state = _StubState()
        self.memory = _StubMemory()
        self.registry = _StubRegistry()
        self.dm = _StubDataManager()
        self.catalog = SimpleNamespace(fields={})
        self.reproducibility = _StubReproducibility()
        self._study_spec_compiler = _StubStudySpecCompiler()
        self.orchestrator = _StubOrchestrator()
        self.llm = SimpleNamespace(model="stub-model")
        self.messages: list[dict] = []
        self._active_query_id = None
        self._on_skill = on_skill or (lambda name, args: {"summary": f"ok:{name}"})

    def _system_message(self) -> dict:
        return {"role": "system", "content": "you are biobank agent"}

    def _execute_skill_and_record(self, name, args, report_dir, allow_retry=True):
        try:
            result = self._on_skill(name, args)
            return {
                "result": result,
                "result_str": str(result),
                "elapsed_s": 0.01,
                "new_figures": [],
                "is_error": False,
                "args": dict(args or {}),
            }
        except Exception as e:
            return {
                "result": {"error": str(e)},
                "result_str": str({"error": str(e)}),
                "elapsed_s": 0.01,
                "new_figures": [],
                "is_error": True,
                "args": dict(args or {}),
            }

    def _record_orchestration_graph(self, *_):
        pass

    def _record_executive_findings(self, *_):
        pass

    def _post_run(self, *_):
        pass


# ── Tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_agent_streams_simple_response(tmp_path):
    legacy = _StubAgent(tmp_path)
    runtime = AsyncAgent(legacy)

    # Patch the AsyncLLMClient with our stub.
    runtime._async_llm = _StubAsyncLLMClient([
        StreamTextDelta(text="hello "),
        StreamTextDelta(text="world"),
        StreamFinal(response=LLMResponse(text="hello world", tool_calls=[], usage={})),
    ])

    events = []
    async for event in runtime.stream_events("greet me"):
        events.append(event)

    types = [e.type for e in events]
    assert AgentEventType.TURN_STARTED in types
    assert AgentEventType.MESSAGE_DELTA in types
    assert AgentEventType.MESSAGE_COMPLETE in types
    assert AgentEventType.TURN_FINISHED in types
    assert events[-1].payload.get("final_text") == "hello world"


@pytest.mark.asyncio
async def test_async_agent_dispatches_tool_call_through_legacy(tmp_path):
    skill_calls: list[tuple[str, dict]] = []

    def on_skill(name, args):
        skill_calls.append((name, dict(args)))
        return {"auc": 0.83, "n_cases": 1500, "summary": "ok"}

    legacy = _StubAgent(tmp_path, on_skill=on_skill)
    runtime = AsyncAgent(legacy)

    tc = ToolCall(id="call_1", name="prevalence", args={"icd10_code": "E11"})
    runtime._async_llm = _StubAsyncLLMClient([
        [StreamFinal(response=LLMResponse(text="", tool_calls=[tc], usage={}))],
        # Second round: model returns final answer after seeing tool result.
        [StreamFinal(response=LLMResponse(text="prevalence summary", tool_calls=[], usage={}))],
    ])

    events = []
    async for event in runtime.stream_events("compute prevalence for E11"):
        events.append(event)

    types = [e.type for e in events]
    assert AgentEventType.TOOL_STARTED in types
    assert AgentEventType.TOOL_RESULT in types
    assert AgentEventType.REPRODUCIBILITY_CHECKPOINT in types
    assert AgentEventType.MESSAGE_COMPLETE in types
    assert skill_calls == [("prevalence", {"icd10_code": "E11"})]
    final_event = next(e for e in events if e.type == AgentEventType.TURN_FINISHED)
    assert final_event.payload["final_text"] == "prevalence summary"


@pytest.mark.asyncio
async def test_async_agent_emits_tool_error_when_skill_fails(tmp_path):
    def on_skill(name, args):
        raise ValueError("bad arg")

    legacy = _StubAgent(tmp_path, on_skill=on_skill)
    runtime = AsyncAgent(legacy)
    tc = ToolCall(id="call_x", name="prevalence", args={"icd10_code": "E11"})
    runtime._async_llm = _StubAsyncLLMClient([
        [StreamFinal(response=LLMResponse(text="", tool_calls=[tc], usage={}))],
        [StreamFinal(response=LLMResponse(text="couldn't run", tool_calls=[], usage={}))],
    ])

    events = []
    async for event in runtime.stream_events("force a failure"):
        events.append(event)

    types = [e.type for e in events]
    assert AgentEventType.TOOL_STARTED in types
    assert AgentEventType.TOOL_ERROR in types


@pytest.mark.asyncio
async def test_async_agent_run_to_text_collects_final_answer(tmp_path):
    legacy = _StubAgent(tmp_path)
    runtime = AsyncAgent(legacy)
    runtime._async_llm = _StubAsyncLLMClient([
        StreamTextDelta(text="answer "),
        StreamTextDelta(text="text"),
        StreamFinal(response=LLMResponse(text="answer text", tool_calls=[], usage={})),
    ])
    text = await runtime.run_to_text("hello")
    assert text == "answer text"


def test_legacy_execution_handler_uses_inferred_capabilities(tmp_path):
    legacy = _StubAgent(tmp_path)

    train = _LegacyExecutionHandler(
        legacy=legacy,
        skill_name="train_model",
        report_dir=tmp_path,
        allow_retry=True,
    )
    review = _LegacyExecutionHandler(
        legacy=legacy,
        skill_name="web_search",
        report_dir=tmp_path,
        allow_retry=True,
    )

    assert train.required_capabilities() == frozenset({
        Capability.READ_DATA,
        Capability.WRITE_REPORTS,
        Capability.EXPORT_AGGREGATE,
    })
    assert review.required_capabilities() == frozenset({Capability.NETWORK})
    assert Capability.EXPORT_PII not in train.required_capabilities()
    assert Capability.SHELL_EXEC not in review.required_capabilities()


def test_async_agent_wires_opentelemetry_bridge_when_enabled(tmp_path, monkeypatch):
    created: list[Any] = []
    configured: list[Any] = []

    class _FakeOtelBridge:
        def __init__(self, *, service_name: str):
            self.service_name = service_name
            self.events: list[AgentEvent] = []
            created.append(self)

        def handle_event(self, event: AgentEvent) -> None:
            self.events.append(event)

    monkeypatch.setattr(runtime_mod, "OpenTelemetryBridge", _FakeOtelBridge)
    monkeypatch.setattr(
        runtime_mod,
        "configure_opentelemetry_from_settings",
        lambda settings: configured.append(settings),
    )

    legacy = _StubAgent(tmp_path)
    legacy.settings.otel_enabled = True
    legacy.settings.otel_service_name = "biobank-agent-test"

    runtime = runtime_mod.AsyncAgent(legacy)
    event = AgentEvent.make(AgentEventType.TOOL_RESULT, skill="missing_data")
    runtime.bus.publish_nowait(event)

    assert runtime.otel is created[0]
    assert configured == [legacy.settings]
    assert created[0].service_name == "biobank-agent-test"
    assert created[0].events == [event]
