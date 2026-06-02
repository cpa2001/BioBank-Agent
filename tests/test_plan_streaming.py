"""Phase C: live token streaming from the council into the dashboard.

Pins: the provider streams plain-text calls token-by-token via ``stream_cb``
(tool-call turns stay non-streaming); streamed usage is captured; the council
forwards each delta on a dedicated ``delta`` channel (never as a recorded
event); and streaming is opt-in via ``ctx.stream`` / ``build_plan(stream=...)``.
"""

from __future__ import annotations

import json

from biobank_agent.cli.interactive import LLMProvider
from biobank_agent.llm import LLMResponse
from biobank_agent.runtime.council import CouncilContext, CouncilJob, run_parallel
from biobank_agent.runtime.engine import ProviderRouter
from biobank_agent.runtime.planner import RuntimePlanner
from biobank_agent.runtime.types import ProviderRequest, ProviderResponse, ProviderRole, RuntimeConfig

from tests.test_council_planner import _WGS_PLAN, _plan_handler


class _FakeStreamLLM:
    model = "fake-model"

    def __init__(self) -> None:
        self.chat_calls = 0

    def stream(self, messages, tools=None, **kw):
        for tok in ["先做", " QC，", "再跑 SAIGE"]:
            yield tok
        return LLMResponse(text="先做 QC，再跑 SAIGE", usage={"total_tokens": 12})

    def chat(self, messages, tools=None, **kw):
        self.chat_calls += 1
        return LLMResponse(text="non-streamed", usage={"total_tokens": 3})


def test_llmprovider_streams_text_and_captures_usage():
    llm = _FakeStreamLLM()
    provider = LLMProvider(llm)
    seen: list[str] = []
    req = ProviderRequest(session_id="s", turn_id="t", messages=[{"role": "user", "content": "x"}], tools=[])
    req.stream_cb = seen.append
    resp = provider.complete(req)
    assert seen == ["先做", " QC，", "再跑 SAIGE"]          # each delta delivered live
    assert resp.text == "先做 QC，再跑 SAIGE"               # final assembled text
    assert resp.usage == {"total_tokens": 12}             # streamed usage captured
    assert resp.deltas == ["先做", " QC，", "再跑 SAIGE"]
    assert llm.chat_calls == 0                            # took the streaming path


def test_llmprovider_no_stream_cb_uses_plain_chat():
    llm = _FakeStreamLLM()
    provider = LLMProvider(llm)
    req = ProviderRequest(session_id="s", turn_id="t", messages=[{"role": "user", "content": "x"}], tools=[])
    resp = provider.complete(req)  # no stream_cb
    assert resp.text == "non-streamed"
    assert llm.chat_calls == 1


def test_llmprovider_with_tools_does_not_stream():
    """Tool-call turns must keep the non-streaming path even if a stream_cb is
    somehow present (streaming tool-call accumulation is out of scope)."""
    llm = _FakeStreamLLM()
    provider = LLMProvider(llm)
    req = ProviderRequest(session_id="s", turn_id="t", messages=[{"role": "user", "content": "x"}],
                          tools=[{"type": "function", "function": {"name": "f"}}])
    req.stream_cb = lambda d: None
    resp = provider.complete(req)
    assert resp.text == "non-streamed"
    assert llm.chat_calls == 1


def test_stream_falls_back_when_usage_option_rejected_at_iteration():
    """If a provider rejects ``stream_options.include_usage`` only when the
    stream is iterated (not at creation), and nothing has been yielded yet,
    LLMClient.stream() must retry a plain stream rather than fail."""
    import types

    from biobank_agent.llm import LLMClient

    client = LLMClient.__new__(LLMClient)
    client.model = "fake"
    client.tool_call_content_mode = "null"
    client._deprecated_params = set()
    client.sanitize_messages = lambda m, _mode: m

    def mkdelta(c):
        d = types.SimpleNamespace(content=c, tool_calls=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=d)], usage=None)

    def raising_iter():
        raise RuntimeError("include_usage not supported")
        yield  # pragma: no cover

    def good_iter():
        yield mkdelta("hello")
        yield mkdelta(" world")

    calls = {"n": 0}

    def fake_create(kwargs):
        calls["n"] += 1
        return raising_iter() if "stream_options" in kwargs else good_iter()

    client._create_chat_completion_with_compat = fake_create
    gen = client.stream(messages=[{"role": "user", "content": "hi"}])
    out: list[str] = []
    final = None
    while True:
        try:
            out.append(next(gen))
        except StopIteration as stop:
            final = stop.value
            break
    assert out == ["hello", " world"]
    assert final.text == "hello world"
    assert calls["n"] == 2  # tried with stream_options, then fell back


class _StreamProvider:
    """Provider that honours ``request.stream_cb`` by emitting the response text
    in two chunks before returning it."""

    def __init__(self, handler) -> None:
        self.handler = handler

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        text = self.handler(request)
        cb = getattr(request, "stream_cb", None)
        if cb is not None:
            mid = max(1, len(text) // 2)
            cb(text[:mid])
            cb(text[mid:])
        return ProviderResponse(text=text, provider="fake", model=request.model or "m", usage={"total_tokens": 5})


def _stream_router():
    config = RuntimeConfig(primary_model="m", planner_model="m", critic_model="m", summarizer_model="m", safety_reviewer_model="m")
    return ProviderRouter({"m": _StreamProvider(_plan_handler)}, config)


def test_run_parallel_streams_deltas_on_dedicated_channel():
    events: list[tuple] = []

    def emit(stage, *, status="running", message="", metadata=None):
        events.append((status, metadata or {}))

    ctx = CouncilContext(provider_router=_stream_router(), emit=emit, stream=True)
    job = CouncilJob(role=ProviderRole.PLANNER, messages=[{"role": "user", "content": "x"}],
                     label="candidate-1", stage="Planning", metadata={"persona_tag": "biostatistician"})
    run_parallel(ctx, [job])
    deltas = [m for (st, m) in events if st == "stream" and m.get("delta")]
    assert deltas, "expected streamed delta events"
    assert all(m.get("subagent") == "candidate-1" for m in deltas)
    # deltas are NOT recorded as running/terminal phase events
    assert all("delta" in m for (_st, m) in events if _st == "stream")


def test_stream_disabled_emits_no_deltas():
    events: list[tuple] = []

    def emit(stage, *, status="running", message="", metadata=None):
        events.append((status, metadata or {}))

    ctx = CouncilContext(provider_router=_stream_router(), emit=emit, stream=False)  # default
    job = CouncilJob(role=ProviderRole.PLANNER, messages=[{"role": "user", "content": "x"}],
                     label="candidate-1", stage="Planning")
    run_parallel(ctx, [job])
    assert not [m for (st, m) in events if st == "stream"]


def test_cancel_event_drops_late_deltas():
    import threading

    events: list[tuple] = []

    def emit(stage, *, status="running", message="", metadata=None):
        events.append((status, metadata or {}))

    cancel = threading.Event()
    cancel.set()
    ctx = CouncilContext(provider_router=_stream_router(), emit=emit, stream=True, cancel_event=cancel)
    job = CouncilJob(role=ProviderRole.PLANNER, messages=[{"role": "user", "content": "x"}],
                     label="candidate-1", stage="Planning")
    run_parallel(ctx, [job])
    assert not [m for (st, m) in events if st == "stream"], "cancelled run must drop deltas at the source"


def test_build_plan_stream_routes_deltas_through_emit():
    """End-to-end: RuntimePlanner.build_plan(stream=True) drives the streaming
    provider and surfaces deltas through the emit sink while still producing a
    valid objective-specific plan."""
    planner = RuntimePlanner(_stream_router(), RuntimeConfig(primary_model="m", planner_model="m", critic_model="m", summarizer_model="m", safety_reviewer_model="m"), num_candidates=2)
    events: list[tuple] = []

    def emit(stage, *, status="running", message="", metadata=None):
        events.append((status, metadata or {}))

    plan = planner.build_plan("vitiligo WGS case/control", tool_names=["vcf_qc"], emit=emit, stream=True)
    assert [s.id for s in plan.steps]  # produced a real plan
    assert [m for (st, m) in events if st == "stream" and m.get("delta")], "expected streamed deltas end-to-end"
