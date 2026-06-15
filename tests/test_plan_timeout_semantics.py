"""Plan-timeout semantics: the SIGALRM-raised deadlines must propagate through the runtime's broad
``except Exception`` guards, so they inherit BaseException (like KeyboardInterrupt), not Exception."""

from __future__ import annotations

import time

import pytest

from biobank_agent.cli.interactive import (
    LLMProvider,
    PlanBuildTimeoutError,
    PlanStepTimeoutError,
    _wall_clock_timeout,
    mark_activity,
)


@pytest.mark.parametrize("exc_type", [PlanStepTimeoutError, PlanBuildTimeoutError])
def test_plan_timeout_bypasses_broad_except(exc_type):
    # Inherits BaseException, NOT Exception — so the real runtime's tool/agent-loop `except Exception`
    # guards cannot swallow the deadline before the explicit `except PlanStepTimeoutError` handler runs.
    assert issubclass(exc_type, BaseException)
    assert not issubclass(exc_type, Exception)

    swallowed_by_broad = False
    with pytest.raises(exc_type):
        try:
            raise exc_type("deadline")
        except Exception:  # noqa: BLE001 - intentionally broad, mirrors the runtime guards
            swallowed_by_broad = True
    assert not swallowed_by_broad, f"{exc_type.__name__} was swallowed by except Exception"


def test_mark_activity_is_safe_noop_when_idle():
    # No active watchdog -> heartbeat is a harmless no-op (must never raise).
    mark_activity()


# --------------------------------------------------------------------------- inactivity watchdog


def test_activity_based_timeout_does_not_fire_while_producing():
    """The per-step watchdog measures SILENCE: while activity is marked far more often than the window,
    it must NOT fire — a model that is actively producing is never killed."""
    fired = False
    try:
        with _wall_clock_timeout(0.4, PlanStepTimeoutError, "test", activity_based=True, hard_ceiling_s=10.0):
            for _ in range(16):
                time.sleep(0.05)
                mark_activity()  # refresh every ~0.05s, well under the 0.4s window
    except PlanStepTimeoutError:
        fired = True
    assert not fired, "activity-based watchdog fired despite continuous activity"


def test_activity_based_timeout_fires_on_real_silence():
    """With no activity for longer than the window, the watchdog fires — catching a genuine hang."""
    fired = False
    try:
        with _wall_clock_timeout(0.3, PlanStepTimeoutError, "test", activity_based=True, hard_ceiling_s=10.0):
            time.sleep(1.0)  # silent past the 0.3s window
    except PlanStepTimeoutError:
        fired = True
    assert fired, "watchdog did not fire after silence beyond the inactivity window"


def test_hard_ceiling_fires_even_under_continuous_activity():
    """The absolute hard ceiling stops a step that keeps 'making progress' forever (anti-runaway /
    endless-keepalive guard), independent of the inactivity window. The window must be <= ceiling so
    the handler re-arms frequently enough to observe the ceiling."""
    fired = False
    message = ""
    try:
        with _wall_clock_timeout(0.2, PlanStepTimeoutError, "test", activity_based=True, hard_ceiling_s=0.6):
            for _ in range(40):
                time.sleep(0.05)
                mark_activity()  # always active, but the 0.6s ceiling must still fire
    except PlanStepTimeoutError as exc:
        fired = True
        message = str(exc)
    assert fired, "hard ceiling did not fire under continuous activity"
    assert "hard ceiling" in message


def test_hard_ceiling_observed_even_when_window_exceeds_it():
    """Even if the inactivity window is (mis)configured LARGER than the hard ceiling, the ceiling is
    still enforced — the initial timer arms no later than the ceiling. Without that, the first fire
    (and the ceiling check) would be delayed all the way to the window."""
    fired = False
    message = ""
    try:
        with _wall_clock_timeout(5.0, PlanStepTimeoutError, "test", activity_based=True, hard_ceiling_s=0.4):
            for _ in range(40):
                time.sleep(0.05)
                mark_activity()  # window is 5s so inactivity never trips; the 0.4s ceiling must fire
    except PlanStepTimeoutError as exc:
        fired = True
        message = str(exc)
    assert fired, "hard ceiling must fire even when timeout_s > hard_ceiling_s"
    assert "hard ceiling" in message


# --------------------------------------------------------------------------- streaming with tools


class _FakeStreamLLM:
    """Minimal LLM client whose stream() yields text deltas, fires on_chunk per raw chunk (including a
    tool-call-only chunk that yields nothing), and returns a final response carrying a tool call."""

    model = "fake-model"

    def __init__(self) -> None:
        self.on_chunk_count = 0
        self.streamed_with_tools = None
        self.chat_called = False

    def stream(self, messages, tools=None, temperature=0.1, max_tokens=4096, on_chunk=None):
        from biobank_agent.llm import LLMResponse
        from biobank_agent.llm import ToolCall as _LLMToolCall

        self.streamed_with_tools = tools
        for piece in ["partial ", "answer"]:
            if on_chunk is not None:
                on_chunk()
                self.on_chunk_count += 1
            yield piece
        if on_chunk is not None:  # a tool-call-only chunk: liveness with no yielded text
            on_chunk()
            self.on_chunk_count += 1
        return LLMResponse(
            text="partial answer",
            tool_calls=[_LLMToolCall(id="t1", name="search", args={"q": "x"})],
            usage={"completion_tokens": 7},
        )

    def chat(self, messages, tools=None, **kwargs):
        self.chat_called = True
        raise AssertionError("chat() must not be used when streaming succeeds")


def test_llm_provider_streams_with_tools_and_fires_on_chunk():
    """LLMProvider.complete streams WITH tools (the client assembles tool_calls from deltas) and wires
    on_chunk so every raw chunk — including the tool-call-only chunk — refreshes the watchdog."""
    from biobank_agent.runtime.types import ProviderRequest, ProviderRole

    fake = _FakeStreamLLM()
    provider = LLMProvider(fake)
    seen: list[str] = []
    req = ProviderRequest(
        session_id="s",
        turn_id="t",
        role=ProviderRole.PRIMARY_EXECUTOR,
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
        model="fake-model",
        stream_cb=seen.append,
    )
    resp = provider.complete(req)
    assert fake.streamed_with_tools is not None, "must stream WITH tools, not drop them"
    assert not fake.chat_called, "must not fall back to chat() when streaming works"
    assert resp.tool_calls and resp.tool_calls[0].name == "search", "tool call assembled from the stream"
    assert "".join(seen) == "partial answer", "stream_cb receives the yielded text deltas"
    assert fake.on_chunk_count >= 3, "on_chunk fired per raw chunk, including the tool-only chunk"
