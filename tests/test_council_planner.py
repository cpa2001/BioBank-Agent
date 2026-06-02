"""Tests for the multi-agent council planner (Phase 2).

These pin the behaviours that distinguish the council planner from the old
static template: objective-specific plans, *fail-loud* on total failure (never a
silent generic fallback), the clarification gate, and real parallel fan-out.
"""

from __future__ import annotations

import json

import pytest

from biobank_agent.runtime.council import CouncilError, extract_json
from biobank_agent.runtime.engine import ProviderRouter
from biobank_agent.runtime.planner import RuntimePlanner
from biobank_agent.runtime.types import PlanStatus, ProviderResponse, RuntimeConfig


class _FakeProvider:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list = []

    def complete(self, request):
        self.calls.append(request)
        return ProviderResponse(text=self.handler(request), provider="fake", model=request.model or "m")


def _router(handler) -> tuple[ProviderRouter, _FakeProvider]:
    fake = _FakeProvider(handler)
    config = RuntimeConfig(primary_model="m", planner_model="m", critic_model="m", summarizer_model="m", safety_reviewer_model="m")
    return ProviderRouter({"m": fake}, config), fake


_WGS_PLAN = {
    "title": "Vitiligo WGS case/control analysis",
    "summary": "QC, annotate, and compare phenotype groups.",
    "steps": [
        {"id": "qc", "title": "QC the VCF", "purpose": "Filter low-quality variants", "dependencies": [], "tool_scope": ["vcf_qc"], "file_scope": ["data/cohort.vcf"], "verification": ["Ti/Tv in range"], "risks": []},
        {"id": "annotate", "title": "Annotate variants", "purpose": "Add functional annotation", "dependencies": ["qc"], "tool_scope": ["vcf_annotation"], "file_scope": [], "verification": ["annotations present"], "risks": []},
        {"id": "assoc", "title": "Case/control association", "purpose": "Compare the two phenotype groups", "dependencies": ["annotate"], "tool_scope": ["vcf_association"], "file_scope": [], "verification": ["QQ plot, lambda"], "risks": []},
    ],
    "risks": ["population stratification"],
    "verification_plan": ["replicate top hits"],
    "required_approvals": ["approve before writing outputs"],
    "proposed_tool_scope": ["vcf_qc", "vcf_annotation", "vcf_association"],
    "proposed_file_scope": ["data/cohort.vcf"],
    "open_questions": [],
}


def _plan_handler(req):
    # Dispatch by PROMPT CONTENT, not provider role: draft slots now use distinct
    # roles (planner/primary/critic) so each resolves to a different model, so a
    # draft can legitimately arrive on the critic role. Real models see the prompt,
    # so the fake must too.
    content = req.messages[-1]["content"] if req.messages else ""
    if "Critique this plan" in content:
        return '{"score": 80, "strengths": ["clear"], "weaknesses": [], "missing": []}'
    if "clarifying questions" in content:
        return '{"questions": []}'
    return json.dumps(_WGS_PLAN)


def test_council_produces_objective_specific_plan():
    router, fake = _router(_plan_handler)
    planner = RuntimePlanner(router, num_candidates=3, enable_clarification=False)
    plan = planner.build_plan("vitiligo WGS case/control analysis", tool_names=["vcf_qc", "vcf_annotation", "vcf_association"])

    # Not the old static template.
    ids = [s.id for s in plan.steps]
    assert ids == ["qc", "annotate", "assoc"]
    assert ids != ["context", "design", "execute", "verify"]
    assert "VCF" in plan.steps[0].title
    assert plan.status == PlanStatus.DRAFT
    assert plan.revision == 1
    # Real fan-out: 3 candidate drafts + 3 critics + 1 merge.
    assert len(fake.calls) >= 7
    assert any(c.role.value == "critic" for c in fake.calls)


def test_council_fails_loudly_when_no_valid_json():
    router, _ = _router(lambda req: "Sorry, I can't produce JSON.")
    planner = RuntimePlanner(router, num_candidates=2, enable_clarification=False)
    with pytest.raises(CouncilError):
        planner.build_plan("analyze something", tool_names=["vcf_qc"])


def test_single_candidate_skips_merge():
    router, fake = _router(_plan_handler)
    planner = RuntimePlanner(router, num_candidates=1, enable_clarification=False)
    plan = planner.build_plan("one shot", tool_names=[])
    assert len(plan.steps) == 3
    # 1 candidate, no critics (single candidate), no merge.
    assert len(fake.calls) == 1


def test_clarification_non_interactive_defers_to_open_questions():
    def handler(req):
        content = req.messages[-1]["content"] if req.messages else ""
        if "clarifying questions" in content:
            return json.dumps({"questions": [{"id": "q1", "header": "Cohort", "question": "Which cohort?", "options": [{"label": "UKB", "description": "UK Biobank"}, {"label": "Custom", "description": "Bring your own"}]}]})
        if req.role.value == "critic":
            return '{"score": 50}'
        return json.dumps(_WGS_PLAN)

    router, _ = _router(handler)
    planner = RuntimePlanner(router, num_candidates=1, enable_clarification=True)
    plan = planner.build_plan("ambiguous goal", tool_names=[], interactive=False, clarifier=None)
    assert "Which cohort?" in plan.open_questions


def test_clarification_interactive_invokes_clarifier():
    asked: list[dict] = []

    def clarifier(question):
        asked.append(question)
        return "UKB"

    def handler(req):
        content = req.messages[-1]["content"] if req.messages else ""
        if "clarifying questions" in content:
            return json.dumps({"questions": [{"id": "q1", "header": "Cohort", "question": "Which cohort?", "options": [{"label": "UKB", "description": "UK Biobank"}]}]})
        if req.role.value == "critic":
            return '{"score": 50}'
        return json.dumps(_WGS_PLAN)

    router, _ = _router(handler)
    planner = RuntimePlanner(router, num_candidates=1, enable_clarification=True)
    plan = planner.build_plan("ambiguous goal", tool_names=[], interactive=True, clarifier=clarifier)
    assert asked and asked[0]["question"] == "Which cohort?"
    # Answered question is folded in, not left dangling.
    assert "Which cohort?" not in plan.open_questions


def test_clarification_tolerates_string_options_and_text_key():
    """Robustness: real models (e.g. kimi) emit options as plain STRINGS and use a
    'text' field instead of 'question'. The parser must normalize these into a
    usable menu rather than silently dropping the question (which made the live
    clarification step no-op). Reproduces the exact shape observed on OpenRouter."""
    asked: list[dict] = []

    def clarifier(question):
        asked.append(question)
        return question["options"][0]["label"]  # pick the first normalized option

    def handler(req):
        content = req.messages[-1]["content"] if req.messages else ""
        if "clarifying questions" in content:
            # kimi-style: 'text' not 'question', options as bare strings
            return json.dumps({"questions": [{
                "id": 1,
                "text": "这两组表型的具体定义是什么？",
                "options": ["A. 白癜风病例 vs 健康对照", "B. 节段型 vs 非节段型"],
            }]})
        if req.role.value == "critic":
            return '{"score": 50}'
        return json.dumps(_WGS_PLAN)

    router, _ = _router(handler)
    planner = RuntimePlanner(router, num_candidates=1, enable_clarification=True)
    plan = planner.build_plan("ambiguous goal", tool_names=[], interactive=True, clarifier=clarifier)
    assert asked, "string-option question was silently dropped (parser too strict)"
    q = asked[0]
    assert q["question"] == "这两组表型的具体定义是什么？"   # 'text' normalized to 'question'
    assert [o["label"] for o in q["options"]] == ["A. 白癜风病例 vs 健康对照", "B. 节段型 vs 非节段型"]
    assert all(isinstance(o, dict) and "label" in o and "description" in o for o in q["options"])


def test_emit_receives_real_stage_events():
    events: list[tuple] = []

    def emit(stage, *, status="running", message="", metadata=None):
        events.append((stage, status))

    router, _ = _router(_plan_handler)
    planner = RuntimePlanner(router, num_candidates=2, enable_clarification=False)
    planner.build_plan("obj", tool_names=[], emit=emit)
    stages = {stage for stage, _ in events}
    assert {"Planning", "External council", "Validation"}.issubset(stages)
    assert ("Validation", "success") in events


def test_malformed_merge_steps_does_not_crash():
    """A merge response with a non-list `steps` must fall back to the best
    candidate, not raise TypeError."""
    def handler(req):
        role = req.role.value
        content = req.messages[-1]["content"] if req.messages else ""
        if "clarifying questions" in content:
            return '{"questions": []}'
        if role == "critic":
            return '{"score": 60}'
        if role == "summarizer":
            return '{"steps": [1, "garbage"]}'  # non-empty list of non-dicts
        return json.dumps(_WGS_PLAN)

    router, _ = _router(handler)
    planner = RuntimePlanner(router, num_candidates=2, enable_clarification=False)
    plan = planner.build_plan("obj", tool_names=[])
    assert [s.id for s in plan.steps] == ["qc", "annotate", "assoc"]


def test_scalar_list_fields_are_not_split_into_characters():
    plan_json = {
        "title": "t",
        "summary": "s",
        "steps": [{"id": "qc", "title": "QC", "dependencies": "", "tool_scope": "vcf_qc"},
                  {"id": "assoc", "title": "Assoc", "dependencies": "qc", "tool_scope": ["vcf_association"]}],
        "risks": "population stratification",
    }

    def handler(req):
        if req.role.value == "critic":
            return '{"score": 50}'
        return json.dumps(plan_json)

    router, _ = _router(handler)
    planner = RuntimePlanner(router, num_candidates=1, enable_clarification=False)
    plan = planner.build_plan("obj", tool_names=[])
    assert plan.steps[1].dependencies == ["qc"]  # not ["q", "c"]
    assert plan.steps[0].tool_scope == ["vcf_qc"]
    assert plan.risks == ["population stratification"]


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here is the plan: {"a": [1, 2], "b": "x"} done.') == {"a": [1, 2], "b": "x"}
    assert extract_json('[{"id": "s1"}]') == [{"id": "s1"}]
    with pytest.raises(ValueError):
        extract_json("no json here")


# --- Part 3: multi-model multi-round debate ---------------------------------


def _router_distinct(handler):
    """Router with THREE distinct models (kimi/deepseek/glm) so debate engages.
    All map to the same fake instance, but request.model differs per role, so the
    fake echoes a distinct model id -> nodes carry distinct models."""
    fake = _FakeProvider(handler)
    config = RuntimeConfig(
        primary_model="deepseek", planner_model="kimi", critic_model="glm",
        summarizer_model="deepseek", safety_reviewer_model="glm",
    )
    return ProviderRouter({"kimi": fake, "deepseek": fake, "glm": fake}, config), fake


def _plan(ids, tools):
    return {
        "title": "t", "summary": "s",
        "steps": [{"id": i, "title": i, "purpose": "p", "dependencies": [], "tool_scope": [t], "file_scope": [], "verification": [], "risks": []}
                  for i, t in zip(ids, tools)],
        "risks": [], "verification_plan": [], "required_approvals": [], "proposed_tool_scope": tools, "proposed_file_scope": [], "open_questions": [],
    }


_PLAN_A = _plan(["qc", "annotate", "assoc"], ["vcf_qc", "vcf_annotation", "vcf_association"])
_PLAN_B = _plan(["load", "impute", "burden"], ["vcf_load", "vcf_impute", "burden_test"])
_PLAN_C = _plan(["fetch", "normalize", "glm"], ["fetch_data", "normalize", "run_glm"])


def test_three_draft_slots_use_distinct_roles_and_models():
    """The 3 draft jobs must use DISTINCT roles -> DISTINCT models (kimi/deepseek/
    glm), so the dashboard shows three different models, not 3×kimi."""
    router, fake = _router_distinct(_plan_handler)
    planner = RuntimePlanner(router, num_candidates=3, enable_clarification=False, enable_debate=False)
    planner.build_plan("obj", tool_names=["vcf_qc"])
    # First 3 provider calls are the parallel drafts (before any critic/merge).
    draft_calls = [c for c in fake.calls if "Critique this plan" not in c.messages[-1]["content"]
                   and "Synthesize the single strongest" not in c.messages[-1]["content"]][:3]
    draft_models = {c.model for c in draft_calls}
    draft_roles = {c.role.value for c in draft_calls}
    assert draft_models == {"kimi", "deepseek", "glm"}
    assert draft_roles == {"planner", "primary_executor", "critic"}


def test_debate_runs_and_revises_when_models_distinct():
    """With >=2 distinct models and non-converged drafts, a debate round runs:
    each model revises its plan ('Improve YOUR plan' prompt)."""
    def handler(req):
        content = req.messages[-1]["content"]
        if "Critique this plan" in content:
            return '{"score": 70}'
        if "clarifying questions" in content:
            return '{"questions": []}'
        if "Improve YOUR plan" in content:
            return json.dumps({**_PLAN_A, "confidence": 0.9, "debate_rationale": "merged peers"})
        # distinct drafts per model so round 1 is not short-circuited
        return json.dumps({"kimi": _PLAN_A, "deepseek": _PLAN_B, "glm": _PLAN_C}.get(req.model, _PLAN_A))

    router, fake = _router_distinct(handler)
    planner = RuntimePlanner(router, num_candidates=3, enable_clarification=False, debate_rounds=1)
    plan = planner.build_plan("obj", tool_names=[])
    debate_calls = [c for c in fake.calls if "Improve YOUR plan" in c.messages[-1]["content"]]
    assert debate_calls, "expected at least one debate revision call"
    assert plan.steps  # still produces a valid plan


def test_debate_short_circuits_on_consensus():
    """When all drafts are identical the models already agree; debate must NOT
    spend a round (the MAD 'don't debate when they agree' rule)."""
    def handler(req):
        content = req.messages[-1]["content"]
        if "Critique this plan" in content:
            return '{"score": 70}'
        if "clarifying questions" in content:
            return '{"questions": []}'
        return json.dumps(_PLAN_A)  # identical for every model

    router, fake = _router_distinct(handler)
    planner = RuntimePlanner(router, num_candidates=3, enable_clarification=False, debate_rounds=2)
    planner.build_plan("obj", tool_names=[])
    debate_calls = [c for c in fake.calls if "Improve YOUR plan" in c.messages[-1]["content"]]
    assert debate_calls == [], "identical drafts must short-circuit debate (0 rounds)"


def test_debate_prunes_homogeneous_low_confidence_node():
    """After a revision round, two nodes that converge to the same plan are
    homogeneous; the lower-confidence one is pruned and excluded from merge."""
    events: list[tuple] = []

    def emit(stage, *, status="running", message="", metadata=None):
        events.append((stage, status, message))

    def handler(req):
        content = req.messages[-1]["content"]
        model = req.model
        if "Critique this plan" in content:
            return '{"score": 70}'
        if "clarifying questions" in content:
            return '{"questions": []}'
        if "Synthesize the single strongest" in content:
            return json.dumps(_PLAN_A)
        if "Improve YOUR plan" in content:
            if model == "kimi":
                return json.dumps({**_PLAN_A, "confidence": 0.9, "debate_rationale": "x"})
            if model == "deepseek":
                return json.dumps({**_PLAN_A, "confidence": 0.2, "debate_rationale": "x"})  # homogeneous + low conf
            return json.dumps({**_PLAN_B, "confidence": 0.85, "debate_rationale": "x"})
        # distinct drafts so round 1 proceeds
        return json.dumps({"kimi": _PLAN_A, "deepseek": _PLAN_C, "glm": _PLAN_B}.get(model, _PLAN_A))

    router, fake = _router_distinct(handler)
    planner = RuntimePlanner(router, num_candidates=3, enable_clarification=False, debate_rounds=1)
    planner.build_plan("obj", tool_names=[], emit=emit)
    pruned = [m for (stage, st, m) in events if stage == "Debate" and st == "warning" and "pruned" in m]
    assert pruned, "expected a prune event for the homogeneous low-confidence node"
    # only 2 survivors reach the merge prompt (no 'Candidate 3')
    merge_calls = [c for c in fake.calls if "Synthesize the single strongest" in c.messages[-1]["content"]]
    assert merge_calls and "Candidate 3" not in merge_calls[0].messages[-1]["content"]


def test_debate_all_revisions_fail_falls_back_to_valid_plan():
    """If every debate revision returns garbage, nodes keep their prior plans
    (confidence decays, never deleted) -> a valid plan still ships, no crash."""
    def handler(req):
        content = req.messages[-1]["content"]
        if "Critique this plan" in content:
            return '{"score": 70}'
        if "clarifying questions" in content:
            return '{"questions": []}'
        if "Improve YOUR plan" in content:
            return "garbage not json"
        if "Synthesize the single strongest" in content:
            return json.dumps(_PLAN_A)
        return json.dumps({"kimi": _PLAN_A, "deepseek": _PLAN_B, "glm": _PLAN_C}.get(req.model, _PLAN_A))

    router, _ = _router_distinct(handler)
    planner = RuntimePlanner(router, num_candidates=3, enable_clarification=False, debate_rounds=2)
    plan = planner.build_plan("obj", tool_names=[])  # must NOT raise / hang
    assert plan.steps


def test_debate_disabled_reproduces_single_shot():
    """enable_debate=False -> no debate calls, classic draft/critique/merge."""
    router, fake = _router_distinct(_plan_handler)
    planner = RuntimePlanner(router, num_candidates=3, enable_clarification=False, enable_debate=False)
    planner.build_plan("obj", tool_names=[])
    assert [c for c in fake.calls if "Improve YOUR plan" in c.messages[-1]["content"]] == []


def test_debate_skipped_when_single_distinct_model():
    """Even with debate enabled, a single-model router (<2 distinct models) must
    auto-skip debate so single-provider setups behave as before."""
    router, fake = _router(_plan_handler)  # all roles -> "m"
    planner = RuntimePlanner(router, num_candidates=3, enable_clarification=False, enable_debate=True, debate_rounds=2)
    planner.build_plan("obj", tool_names=[])
    assert [c for c in fake.calls if "Improve YOUR plan" in c.messages[-1]["content"]] == []
