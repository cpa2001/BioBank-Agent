"""里程碑7: the M16 adversarial council wired into the live RuntimePlanner (flag-gated).

Deterministic + offline: a FakeProvider scripts proposer / red-team / referee responses by
prompt content (the same dispatch the symmetric council tests use). Proves the flag actually
routes planning, the symmetric path is untouched when off, failures degrade, and the
council_ab A/B gate runs against the real planner.
"""

from __future__ import annotations

import json

from biobank_agent.runtime.council_ab import ab_compare
from biobank_agent.runtime.engine import ProviderRouter
from biobank_agent.runtime.planner import RuntimePlanner
from biobank_agent.runtime.types import ProviderResponse, RuntimeConfig


class _FakeProvider:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list = []

    def complete(self, request):
        self.calls.append(request)
        return ProviderResponse(text=self.handler(request), provider="fake", model=request.model or "m")


_PLAN = {
    "title": "Vitiligo WGS case/control analysis",
    "summary": "QC, annotate, and compare phenotype groups.",
    "steps": [
        {"id": "qc", "title": "QC the VCF", "purpose": "Filter low-quality variants", "dependencies": [],
         "tool_scope": ["vcf_qc"], "file_scope": ["data/cohort.vcf"], "verification": ["Ti/Tv in range"], "risks": []},
        {"id": "assoc", "title": "Association", "purpose": "Compare phenotype groups", "dependencies": ["qc"],
         "tool_scope": ["vcf_association"], "file_scope": [], "verification": ["QQ plot"], "risks": []},
    ],
    "risks": ["population stratification"],
    "verification_plan": ["replicate top hits"],
    "required_approvals": ["approve before writing outputs"],
    "proposed_tool_scope": ["vcf_qc", "vcf_association"],
    "proposed_file_scope": ["data/cohort.vcf"],
    "open_questions": [],
}
_FLAW = {"severity": "major", "kind": "covariates", "claim": "no PCs", "fix": "adjust for 10 PCs"}


def _router(handler, *, adversarial: bool, rounds: int = 1) -> tuple[ProviderRouter, _FakeProvider]:
    fake = _FakeProvider(handler)
    config = RuntimeConfig(
        primary_model="m", planner_model="m", critic_model="m", summarizer_model="m",
        safety_reviewer_model="m", adversarial_council_enabled=adversarial, debate_rounds=rounds,
    )
    return ProviderRouter({"m": fake}, config), config


def _adv_handler(req):
    content = req.messages[-1]["content"] if req.messages else ""
    if "adversarial red-team" in content:
        return json.dumps({"flaws": [_FLAW]})
    if "You are the referee" in content:
        revised = {**_PLAN, "title": "Revised: " + _PLAN["title"]}
        return json.dumps({"plan": revised, "accepted": [_FLAW], "dismissed": [], "notes": "added PCs"})
    if "clarifying questions" in content:
        return json.dumps({"questions": []})
    return json.dumps(_PLAN)  # proposer (and symmetric drafts)


def test_adversarial_flag_routes_planning_and_folds_referee_revision():
    router, config = _router(_adv_handler, adversarial=True, rounds=1)
    planner = RuntimePlanner(router, config, num_candidates=1, enable_clarification=False)
    events: list = []
    plan = planner.build_plan("vitiligo WGS case/control", tool_names=["vcf_qc", "vcf_association"],
                              emit=lambda *a, **k: events.append((a, k)))
    assert plan.steps                                   # a valid PlanState came out
    assert plan.title.startswith("Revised:")            # the referee's revision flowed through
    fake = router.providers["m"]
    assert any("adversarial red-team" in c.messages[-1]["content"] for c in fake.calls)
    assert any("You are the referee" in c.messages[-1]["content"] for c in fake.calls)


def test_flag_off_keeps_symmetric_path():
    router, config = _router(_adv_handler, adversarial=False)
    planner = RuntimePlanner(router, config, num_candidates=2, enable_clarification=False)
    plan = planner.build_plan("vitiligo WGS case/control", tool_names=["vcf_qc", "vcf_association"])
    assert plan.steps
    fake = router.providers["m"]
    # No adversarial role calls were made — the symmetric debate handled it.
    assert not any("adversarial red-team" in c.messages[-1]["content"] for c in fake.calls)


def test_adversarial_degrades_to_scaffold_when_proposer_fails():
    # Proposer (and everything) returns junk -> CouncilError inside the pipeline -> the existing
    # build_plan fallback yields a scaffold plan rather than raising.
    router, config = _router(lambda req: "not json at all", adversarial=True, rounds=1)
    planner = RuntimePlanner(router, config, num_candidates=1, enable_clarification=False)
    plan = planner.build_plan("some analysis objective", tool_names=["vcf_qc"])
    assert plan is not None and plan.title


def test_ab_compare_runs_both_planner_modes_against_real_planner():
    def _plan_in(adversarial: bool):
        def _run(objective: str):
            router, config = _router(_adv_handler, adversarial=adversarial, rounds=1)
            planner = RuntimePlanner(router, config, num_candidates=1, enable_clarification=False)
            return planner.build_plan(objective, tool_names=["vcf_qc", "vcf_association"])
        return _run

    report = ab_compare(
        ["objective one", "objective two"],
        run_a=_plan_in(True), run_b=_plan_in(False),
        score_fn=lambda plan: float(len(plan.steps)),
        label_a="adversarial", label_b="symmetric",
    )
    assert report["n"] == 2
    assert report["winner"] in {"adversarial", "symmetric", "tie"}
    assert set(report["wins"]) == {"adversarial", "symmetric", "tie"}
