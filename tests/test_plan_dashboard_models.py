"""Phase B: PlanRunDashboard per-model live tracking + council role/persona emit.

These pin the Claude-Code-style multi-model view: each parallel model call
becomes a live "active" row (model · activity · persona · ticking timer); on
completion it moves to history with its duration; a timeout clears the row so
its timer cannot tick forever; and model-derived text can never break the
Rich render.
"""

from __future__ import annotations

import time
from io import StringIO

from rich.console import Console

from biobank_agent.progress import PlanRunDashboard, _elapsed_str, _short_model
from biobank_agent.runtime.council import CouncilContext, CouncilJob, run_parallel
from biobank_agent.runtime.types import ProviderRole

from tests.test_council_planner import _router


def _dash() -> PlanRunDashboard:
    return PlanRunDashboard(Console(file=StringIO(), force_terminal=True, width=100), title="Planning")


def _render(d: PlanRunDashboard) -> str:
    buf = StringIO()
    Console(file=buf, force_terminal=True, width=100).print(d._build_panel())
    return buf.getvalue()


def test_dispatch_opens_active_row_then_completion_moves_to_history():
    d = _dash()
    d.record(
        "Planning", status="running", message="dispatch candidate-1",
        metadata={"subagent": "candidate-1", "model": "moonshotai/kimi-k2.6", "role": "planner", "persona": "biostatistician"},
    )
    assert "candidate-1" in d._active
    assert d._active["candidate-1"]["model"] == "moonshotai/kimi-k2.6"
    assert d._history == []

    d.record(
        "Planning", status="success", message="candidate-1 (kimi)",
        metadata={"subagent": "candidate-1", "model": "moonshotai/kimi-k2.6", "elapsed_s": 12.5, "persona": "biostatistician"},
    )
    assert "candidate-1" not in d._active
    assert len(d._history) == 1
    assert d._history[0]["elapsed_s"] == 12.5
    assert d._history[0]["status"] == "success"
    # the live short model name is rendered
    d.record("Planning", status="running", metadata={"subagent": "candidate-2", "model": "moonshotai/kimi-k2.6"})
    assert "kimi" in _render(d)


def test_active_row_timer_advances_live():
    d = _dash()
    d.record("Planning", status="running", metadata={"subagent": "candidate-1", "model": "kimi"})
    d._active["candidate-1"]["start_ts"] = time.time() - 5
    assert "5s" in _render(d)


def test_timeout_error_clears_active_row():
    d = _dash()
    d.record("Planning", status="running", metadata={"subagent": "candidate-3", "model": "kimi"})
    assert "candidate-3" in d._active
    d.record("Planning", status="error", message="candidate-3 timed out", metadata={"subagent": "candidate-3", "model": "kimi"})
    assert "candidate-3" not in d._active
    assert d._history[-1]["status"] == "error"


def test_note_partial_appends_and_drops_after_completion():
    d = _dash()
    d.record("Planning", status="running", metadata={"subagent": "candidate-2", "model": "kimi"})
    d.note_partial("candidate-2", "先做 QC")
    d.note_partial("candidate-2", " 再跑 SAIGE")
    assert d._active["candidate-2"]["partial"].endswith("再跑 SAIGE")
    assert "QC" in _render(d)
    d.record("Planning", status="success", metadata={"subagent": "candidate-2", "model": "kimi", "elapsed_s": 3})
    d.note_partial("candidate-2", " LATE")  # late delta after completion: dropped, no crash
    assert "candidate-2" not in d._active


def test_note_partial_dropped_after_stop():
    """After the dashboard stops, an orphaned worker's late delta must be dropped
    (stop() clears active rows under the lock, so note_partial finds no row)."""
    d = _dash()
    d.record("Planning", status="running", metadata={"subagent": "c1", "model": "kimi"})
    d.note_partial("c1", "before")
    assert d._active["c1"]["partial"] == "before"
    d.stop()
    d.note_partial("c1", " after-stop")  # orphaned late delta
    assert "c1" not in d._active


def test_markup_in_streamed_text_does_not_break_render():
    d = _dash()
    d.record("Planning", status="running", metadata={"subagent": "c1", "model": "kimi"})
    d.note_partial("c1", "weird [bold]not markup[/bold] [unclosed and \x1b[31m ansi")
    _render(d)  # must not raise MarkupError


def test_error_status_event_renders_without_markup_crash():
    """A council job that fails emits status='error', which is NOT in
    DASHBOARD_STATUS_STYLES. The old code did `style = ....get(status, "")`,
    so the recent line rendered `[]<icon>[/]` and Rich raised
    MarkupError("closing tag '[/]' has nothing to close"). This is the live
    `/plan` crash; rendering an error event must never raise."""
    d = _dash()
    d.record("Planning", status="running", metadata={"subagent": "candidate-1", "model": "kimi"})
    d.record(
        "Planning", status="error", message="candidate-1 failed: boom",
        metadata={"subagent": "candidate-1", "model": "kimi"},
    )
    _render(d)  # _build_panel -> _recent_renderable must not raise on empty style
    # the history path (render_activity_summary -> _history_renderable) too
    d.render_activity_summary()


def test_unknown_status_event_renders_without_markup_crash():
    """Defense-in-depth: even a status nobody mapped (e.g. 'mystery') must not
    crash the render — _styled() emits no empty tag when the style is unknown."""
    d = _dash()
    d.record("Planning", status="mystery", message="some [weird] state", metadata={})
    _render(d)


def test_format_choices_escapes_markup_in_title_and_detail():
    """Failure-panel choices are built from plan-derived title/detail (e.g. a step
    id in the title). A stray '[/]' must be escaped, not raise MarkupError."""
    from rich.console import Console
    from io import StringIO
    from biobank_agent.progress import _format_choices, ChoiceOption

    text = _format_choices([ChoiceOption(key="A", title="Skip step [/bold] x", detail="weird [/] detail")])
    # Rendering the formatted markup must not raise.
    Console(file=StringIO(), force_terminal=True, width=80).print(text)


def test_short_model_and_elapsed_helpers():
    assert _short_model("moonshotai/kimi-k2.6") == "kimi"
    assert _short_model("deepseek/deepseek-v4-pro") == "deepseek"
    assert _short_model("z-ai/glm-5.1") == "glm"
    assert _short_model("") == "model"
    assert _elapsed_str(5) == "5s"
    assert _elapsed_str(75) == "1m15s"


def test_active_rows_capped_for_wide_fanout():
    d = _dash()
    for i in range(20):
        d.record("Planning", status="running", metadata={"subagent": f"c{i}", "model": "kimi"})
    table = d._active_renderable()
    assert table is not None and table.row_count <= 8


def test_run_parallel_timeout_emits_per_subagent_terminal():
    """On a council timeout, every unfinished job must still get a per-subagent
    terminal event so the dashboard can clear its live row (no forever-ticking
    timer)."""
    events: list[tuple] = []

    def emit(stage, *, status="running", message="", metadata=None):
        events.append((status, metadata or {}))

    def slow(_req):
        time.sleep(0.3)
        return "late"

    router, _ = _router(slow)
    ctx = CouncilContext(provider_router=router, emit=emit, clock=time.time, timeout_s=0.1)
    jobs = [
        CouncilJob(role=ProviderRole.PLANNER, messages=[{"role": "user", "content": "x"}],
                   label="candidate-1", stage="Planning", metadata={"persona_tag": "biostatistician"})
    ]
    run_parallel(ctx, jobs)
    terminal = [m for (st, m) in events if st in ("success", "error") and m.get("subagent") == "candidate-1"]
    assert terminal, "expected a per-subagent terminal event on timeout"


def test_map_parallel_timeout_emits_per_item_terminal():
    """Research retrieval fan-out (map_parallel) must also emit per-item terminal
    events on timeout (mirrors run_parallel) so the /research dashboard clears
    stale rows."""
    from biobank_agent.runtime.council import map_parallel

    events: list[tuple] = []

    def emit(stage, *, status="running", message="", metadata=None):
        events.append((status, metadata or {}))

    def slow(item):
        time.sleep(0.3)
        return item

    router, _ = _router(lambda req: "x")
    ctx = CouncilContext(provider_router=router, emit=emit, clock=time.time, timeout_s=0.1)
    res = map_parallel(ctx, ["a"], slow, stage="Planning", label_fn=lambda i, _x: f"retrieve-{i + 1}")
    terminal = [m for (st, m) in events if st == "error" and m.get("subagent") == "retrieve-1"]
    assert terminal, "expected a per-item terminal event on timeout"
    assert res and res[0].error


def test_dashboard_renders_three_distinct_models():
    """Multi-model planning shows three DIFFERENT models, not 3×kimi."""
    d = _dash()
    for label, model in (("candidate-1", "moonshotai/kimi-k2.6"),
                         ("candidate-2", "deepseek/deepseek-v4-pro"),
                         ("candidate-3", "z-ai/glm-5.1")):
        d.record("Planning", status="running", metadata={"subagent": label, "model": model, "activity": "drafting"})
    text = _render(d)
    assert "kimi" in text and "deepseek" in text and "glm" in text


def test_dashboard_shows_per_round_debate_activity_verb():
    """Each debating model shows its OWN per-round activity ('revising R2'),
    so the frontend reflects what each model is doing rather than a generic verb."""
    d = _dash()
    d.record("Debate", status="running",
             metadata={"subagent": "debate-r2-s1", "model": "deepseek/deepseek-v4-pro",
                       "persona": "bioinformatics", "activity": "revising R2"})
    text = _render(d)
    assert "revising R2" in text
    assert "deepseek" in text


def test_dashboard_debate_warning_status_renders_safely():
    """A pruned-node Debate event uses status='warning' (mapped) and must render
    without the empty-style MarkupError, in both the live panel and history."""
    d = _dash()
    d.record("Debate", status="running", metadata={"subagent": "debate-r1-s1", "model": "glm"})
    d.record("Debate", status="warning", message="R1: pruned candidate-2 (deepseek) — homogeneous",
             metadata={"subagent": "debate-r1-s1", "model": "glm"})
    _render(d)
    d.render_activity_summary()


def test_council_run_parallel_emits_activity_metadata():
    events: list[tuple] = []

    def emit(stage, *, status="running", message="", metadata=None):
        events.append((status, metadata or {}))

    router, _ = _router(lambda req: "ok")
    ctx = CouncilContext(provider_router=router, emit=emit, clock=time.time)
    jobs = [CouncilJob(role=ProviderRole.PLANNER, messages=[{"role": "user", "content": "x"}],
                       label="debate-r1-s0", stage="Debate",
                       metadata={"persona_tag": "biostatistician", "activity": "revising R1"})]
    run_parallel(ctx, jobs)
    dispatch = [m for (st, m) in events if st == "running" and m.get("subagent") == "debate-r1-s0"]
    assert dispatch and dispatch[0].get("activity") == "revising R1"


def test_council_run_parallel_emits_role_and_persona():
    events: list[tuple] = []

    def emit(stage, *, status="running", message="", metadata=None):
        events.append((stage, status, metadata or {}))

    router, _ = _router(lambda req: "ok")
    ctx = CouncilContext(provider_router=router, emit=emit, clock=time.time)
    jobs = [
        CouncilJob(
            role=ProviderRole.PLANNER,
            messages=[{"role": "user", "content": "x"}],
            label="candidate-1",
            stage="Planning",
            metadata={"persona_tag": "biostatistician"},
        )
    ]
    run_parallel(ctx, jobs)
    dispatch = [m for (_s, st, m) in events if st == "running" and m.get("subagent") == "candidate-1"]
    done = [m for (_s, st, m) in events if st in ("success", "error") and m.get("subagent") == "candidate-1"]
    assert dispatch and dispatch[0].get("role") == "planner" and dispatch[0].get("persona") == "biostatistician"
    assert done and done[0].get("role") == "planner" and "elapsed_s" in done[0]
