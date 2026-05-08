"""Coverage for long-horizon planning structures and decomposition paths."""

import logging
from types import SimpleNamespace

from biobank_agent.planner import LongHorizonPlan, LongHorizonPlanner, PlanMode, PlanStep


class _FakeLLM:
    def __init__(self, text=None, exc=None):
        self.text = text
        self.exc = exc
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.exc:
            raise self.exc
        return SimpleNamespace(text=self.text)


def test_plan_step_and_empty_plan_properties():
    step = PlanStep(id="s1", skill="think", depends_on=["s0"])
    skipped = PlanStep(id="s2", skill="report", status="skipped")
    empty = LongHorizonPlan(goal="empty")

    assert step.is_done is False
    assert step.is_blocked is True
    assert skipped.is_done is True
    assert empty.total_steps == 0
    assert empty.done_steps == 0
    assert empty.failed_steps == 0
    assert empty.progress() == 0.0
    assert empty.next_runnable() == []


def test_long_horizon_plan_progress_mark_done_failed_and_markdown():
    plan = LongHorizonPlan(
        goal="biomarker workflow",
        steps=[
            PlanStep(id="s1", skill="prevalence", description="Find endpoints"),
            PlanStep(id="s2", skill="train_model", depends_on=["s1"]),
            PlanStep(id="s3", skill="report", depends_on=["s2"]),
            PlanStep(id="s4", skill="safety_check", depends_on=["s3"]),
        ],
    )

    assert [s.id for s in plan.next_runnable()] == ["s1"]
    plan.mark_done("s1", {"n": 10})
    assert plan.steps[0].result == {"n": 10}
    assert [s.id for s in plan.next_runnable()] == ["s2"]
    plan.mark_failed("s2", "model error")

    assert plan.failed_steps == 1
    assert plan.steps[1].status == "failed"
    assert plan.steps[2].status == "skipped"
    assert plan.steps[3].status == "skipped"
    assert "3/4 done" in plan.summary()
    markdown = plan.to_markdown()
    assert "# Plan: biomarker workflow" in markdown
    assert "`prevalence({})`" in markdown
    assert "**Error:** model error" in markdown
    assert "Progress:" in markdown


def test_long_horizon_plan_noop_markers_for_unknown_steps():
    plan = LongHorizonPlan(
        goal="noop",
        steps=[PlanStep(id="s1", skill="think")],
    )

    plan.mark_done("missing", {"ignored": True})
    assert plan.steps[0].status == "pending"
    assert plan.steps[0].result == {}

    plan.mark_failed("missing", "not found")
    assert plan.steps[0].status == "pending"
    assert plan.steps[0].error == ""


def test_long_horizon_planner_default_without_llm():
    planner = LongHorizonPlanner()

    plan = planner.decompose("Discover T2DM biomarkers", ["think", "prevalence"])

    assert [s.skill for s in plan.steps] == ["think", "prevalence"]
    assert plan.steps[1].depends_on == ["s1"]


def test_decompose_parses_fenced_json_applies_spec_and_truncates(caplog):
    llm = _FakeLLM(
        "```json\n"
        "["
        "{\"id\":\"a\",\"skill\":\"prevalence\",\"args\":{\"top_n\":5},\"description\":\"Prevalence\",\"depends_on\":[],\"can_parallelize\":true},"
        "{\"id\":\"b\",\"skill\":\"train_model\",\"args\":{\"icd10_code\":\"E11\"},\"description\":\"Train\",\"depends_on\":[\"a\"],\"can_parallelize\":false}"
        "]\n```"
    )
    spec = SimpleNamespace(
        tool_budget=1,
        constrain_skills=lambda skills: [s for s in skills if s != "report"],
    )
    planner = LongHorizonPlanner(llm)

    with caplog.at_level(logging.WARNING):
        plan = planner.decompose(
            "Build a diabetes model",
            ["prevalence", "train_model", "report"],
            context="x" * 800,
            spec=spec,
        )

    assert len(plan.steps) == 1
    assert plan.steps[0].id == "a"
    assert plan.steps[0].can_parallelize is True
    assert llm.calls
    assert "report" not in llm.calls[0][0][1]["content"]
    assert "truncating" in caplog.text


def test_decompose_falls_back_for_non_list_or_llm_exception():
    non_list = LongHorizonPlanner(_FakeLLM('{"not":"a list"}')).decompose("goal", ["think"])
    fenced_non_json = LongHorizonPlanner(_FakeLLM("```text\nnot json\n```")).decompose("goal", ["think"])
    failed = LongHorizonPlanner(_FakeLLM(exc=RuntimeError("bad json"))).decompose("goal", ["think"])

    assert [s.skill for s in non_list.steps] == ["think", "prevalence"]
    assert [s.skill for s in fenced_non_json.steps] == ["think", "prevalence"]
    assert [s.skill for s in failed.steps] == ["think", "prevalence"]


def test_decompose_tolerates_bad_spec_and_temporal_safety_exception(monkeypatch):
    class BadSpec:
        def constrain_skills(self, _skills):
            raise TypeError("bad spec")

    class BrokenChecker:
        def check_plan(self, _steps):
            raise RuntimeError("checker unavailable")

    monkeypatch.setattr("biobank_agent.guardrails.TemporalSafetyChecker", lambda: BrokenChecker())
    llm = _FakeLLM('[{"id":"s1","skill":"think"}]')

    plan = LongHorizonPlanner(llm).decompose("goal", ["think"], spec=BadSpec())

    assert plan.steps[0].skill == "think"


def test_temporal_safety_logs_violations(monkeypatch, caplog):
    class FakeViolation:
        description = "report must follow safety_check"

    class FakeChecker:
        def check_plan(self, _steps):
            return [FakeViolation()]

    monkeypatch.setattr("biobank_agent.guardrails.TemporalSafetyChecker", lambda: FakeChecker())
    planner = LongHorizonPlanner()

    with caplog.at_level(logging.WARNING):
        planner._check_temporal_safety(LongHorizonPlan(goal="g", steps=[PlanStep(id="s1", skill="report")]))

    assert "report must follow safety_check" in caplog.text


def test_plan_mode_remaining_edges(tmp_path):
    pm = PlanMode(plans_dir=tmp_path)
    assert pm.update_plan("content") == "No active plan."
    assert pm._has_open_questions() is False

    pm.enter("task")
    assert pm.approve().startswith("Cannot approve")
    content = pm.current_plan.read_text().replace("## Open Questions", "## Questions")
    pm.current_plan.write_text(content)
    assert pm._has_open_questions() is False

    path = pm.current_plan
    path.unlink()
    assert pm.get_plan_content() == "(no active plan)"
    assert pm.set_status("ALIGNMENT").startswith("Plan status changed")

    (tmp_path / "unknown.md").write_text("# No status here\n", encoding="utf-8")
    statuses = {p["file"]: p["status"] for p in pm.list_plans()}
    assert statuses["unknown.md"] == "UNKNOWN"

    pm2 = PlanMode(plans_dir=tmp_path / "fresh")
    pm2.enter("fresh task")
    no_timestamp = pm2.get_plan_content().replace("- Updated:", "- Last touched:")
    assert pm2.update_plan(no_timestamp).startswith("Plan updated at")
    assert "- Last touched:" in pm2.current_plan.read_text()
