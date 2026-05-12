"""TTFT (time-to-first-token) benchmark for the v3 AsyncAgent.

The legacy synchronous loop returned the entire LLM response in one
shot, so TTFT is undefined for it. The v3 runtime, by contrast, emits
``MESSAGE_DELTA`` events as soon as the first chunk arrives.

This benchmark wires up an in-memory ``AsyncLLMClient`` substitute that
generates artificial network jitter, then measures how long it takes
the runtime to emit the first ``MESSAGE_DELTA`` event with
``marker == "first_token"``.

Run with::

    python -m pytest benchmarks/streaming_latency.py -v
    # or, for trend tracking:
    python benchmarks/streaming_latency.py
"""

from __future__ import annotations

import asyncio
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from biobank_agent.core.events import AgentEventType
from biobank_agent.core.llm.client import StreamFinal, StreamTextDelta
from biobank_agent.core.runtime import AsyncAgent
from biobank_agent.llm import LLMResponse


class _JitterAsyncLLMClient:
    """Stub stream client with deterministic per-chunk delay."""

    def __init__(self, *, first_chunk_delay_s: float, between_chunks_s: float, n_chunks: int = 6):
        self.first_chunk_delay_s = first_chunk_delay_s
        self.between_chunks_s = between_chunks_s
        self.n_chunks = n_chunks

    async def stream_async(self, **_: Any):
        await asyncio.sleep(self.first_chunk_delay_s)
        for i in range(self.n_chunks):
            yield StreamTextDelta(text=f"tok{i} ")
            if i < self.n_chunks - 1:
                await asyncio.sleep(self.between_chunks_s)
        yield StreamFinal(response=LLMResponse(text="tok0 tok1 tok2 ", tool_calls=[], usage={}))


def _legacy_stub(tmp_path: Path) -> SimpleNamespace:
    settings = SimpleNamespace(
        reports_dir=tmp_path / "reports",
        memory_dir=tmp_path / "mem",
        context_window=8192,
        max_tool_rounds=4,
        llm_model="stub",
        bank_id="stub",
        biobank_name="stub",
        async_runtime_enabled=True,
        compaction_warn_pct=0.6,
        compaction_compact_pct=0.75,
        compaction_force_pct=0.9,
    )
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.memory_dir.mkdir(parents=True, exist_ok=True)

    state = SimpleNamespace(
        interrupted=False,
        records=[],
        figures=[],
        cohorts={},
        token_usage=SimpleNamespace(update=lambda usage: None),
        custom_data={},
        executive_findings=[],
        provenances=[],
        last_orchestration={},
    )
    legacy = SimpleNamespace(
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
        llm=SimpleNamespace(model="stub"),
        messages=[],
        _active_query_id=None,
        _system_message=lambda: {"role": "system", "content": "sys"},
        _execute_skill_and_record=lambda *a, **kw: {
            "result": {"summary": "ok"},
            "result_str": "ok",
            "elapsed_s": 0.01,
            "new_figures": [],
            "is_error": False,
            "args": {},
        },
        _record_orchestration_graph=lambda *a, **kw: None,
        _record_executive_findings=lambda *a, **kw: None,
        _post_run=lambda *a, **kw: None,
    )
    return legacy


async def _measure_ttft(legacy: SimpleNamespace, *, first_chunk_delay_s: float) -> float:
    runtime = AsyncAgent(legacy)
    runtime._async_llm = _JitterAsyncLLMClient(
        first_chunk_delay_s=first_chunk_delay_s,
        between_chunks_s=0.01,
    )
    t_start = time.time()
    ttft = None
    async for event in runtime.stream_events("benchmark query"):
        if (
            event.type == AgentEventType.MESSAGE_DELTA
            and event.payload.get("marker") == "first_token"
        ):
            ttft = time.time() - t_start
            # Continue iteration to keep the runtime alive but we have what we need.
            break
    if ttft is None:
        # Fall back to the first non-marker delta.
        async for event in runtime.stream_events("benchmark query 2"):
            if event.type == AgentEventType.MESSAGE_DELTA:
                ttft = time.time() - t_start
                break
    assert ttft is not None
    return ttft


@pytest.mark.asyncio
async def test_ttft_under_threshold_with_50ms_network(tmp_path):
    """Synthetic baseline: with a 50ms first-byte delay the runtime
    should still report TTFT ≤ 200ms — i.e. the runtime overhead on top
    of the LLM is < 150ms.
    """
    samples = []
    for _ in range(5):
        legacy = _legacy_stub(tmp_path)
        ttft = await _measure_ttft(legacy, first_chunk_delay_s=0.05)
        samples.append(ttft)
    p50 = statistics.median(samples)
    print(f"\n[stream-bench] TTFT samples (s): {samples}")
    print(f"[stream-bench] p50: {p50*1000:.1f}ms")
    assert p50 < 0.20, f"runtime overhead too high (p50={p50:.3f}s)"


def main() -> None:
    """Run a richer benchmark sweep and emit a one-line summary."""
    import json
    from tempfile import TemporaryDirectory

    async def _run() -> dict:
        results = {}
        for label, delay in [("instant", 0.0), ("50ms", 0.05), ("200ms", 0.2)]:
            samples = []
            with TemporaryDirectory() as tmp:
                for _ in range(10):
                    legacy = _legacy_stub(Path(tmp))
                    samples.append(await _measure_ttft(legacy, first_chunk_delay_s=delay))
            results[label] = {
                "min": min(samples),
                "p50": statistics.median(samples),
                "p95": sorted(samples)[int(len(samples) * 0.95) - 1],
                "max": max(samples),
            }
        return results

    out = asyncio.run(_run())
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
