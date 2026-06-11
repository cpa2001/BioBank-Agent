"""External plan-review council: orchestration logic, dead-flag reachability, and planner wiring."""

from __future__ import annotations

import biobank_agent.runtime.external_council as ec
from biobank_agent.runtime.external_agents import ExternalAgentResult, ExternalAgentSpec
from biobank_agent.runtime.planner import RuntimePlanner
from biobank_agent.runtime.types import RuntimeConfig
from tests.test_council_planner import _plan_handler, _router


def _detector(names):
    return [ExternalAgentSpec(n, (n, "{prompt}")) for n in names]


def _ok(spec, prompt, **kw):
    return ExternalAgentResult(spec.name, ok=True, text="the plan is missing a QC step")


def _fail(spec, prompt, **kw):
    return ExternalAgentResult(spec.name, ok=False, error="boom")


# ── policy + request detection ───────────────────────────────────────────────

def test_policy_permits():
    assert ec.policy_permits("always", requested=False) is True
    assert ec.policy_permits("never", requested=True) is False
    assert ec.policy_permits("requested", requested=True) is True
    assert ec.policy_permits("requested", requested=False) is False


def test_review_requested_keywords():
    assert ec.review_requested("run gwas", "please get a second opinion") is True
    assert ec.review_requested("ask codex to double-check") is True
    assert ec.review_requested("just run it", "") is False


def test_build_review_prompt_includes_objective_and_steps():
    p = ec.build_review_prompt("vitiligo GWAS", ["QC the VCF", "association test"])
    assert "vitiligo GWAS" in p and "QC the VCF" in p and "association test" in p


# ── collect_plan_reviews (offline, injected detector/invoker) ────────────────

def test_collect_never_policy_returns_empty():
    assert ec.collect_plan_reviews("o", ["s"], agents="codex", policy="never", requested=True,
                                   detector=_detector, invoker=_ok) == []


def test_collect_always_collects_each_available_agent():
    notes = ec.collect_plan_reviews("o", ["s"], agents="codex,claude", policy="always", requested=False,
                                    detector=_detector, invoker=_ok)
    assert len(notes) == 2 and notes[0].startswith("[codex]") and "QC step" in notes[0]


def test_collect_requested_policy_gates_on_flag():
    kw = dict(agents="codex", policy="requested", detector=_detector, invoker=_ok)
    assert ec.collect_plan_reviews("o", ["s"], requested=False, **kw) == []
    assert len(ec.collect_plan_reviews("o", ["s"], requested=True, **kw)) == 1


def test_collect_skips_failed_agents():
    assert ec.collect_plan_reviews("o", ["s"], agents="codex", policy="always", requested=False,
                                   detector=_detector, invoker=_fail) == []


def test_collect_no_agents_available():
    assert ec.collect_plan_reviews("o", ["s"], agents="codex", policy="always", requested=False,
                                   detector=lambda names: [], invoker=_ok) == []


def test_collect_runs_each_agent_in_its_own_isolated_dir():
    import os

    seen = []

    def spy(spec, prompt, *, cwd, timeout_s):
        seen.append(cwd)
        assert os.path.isdir(cwd)  # a real throwaway dir, present during the call
        return ExternalAgentResult(spec.name, ok=True, text="ok")

    ec.collect_plan_reviews("o", ["s"], agents="codex,claude", policy="always", requested=False,
                            detector=_detector, invoker=spy)
    # Each agent runs in its OWN throwaway dir — never the live repo, never shared between agents.
    assert len(seen) == 2
    assert all(c != os.getcwd() and "biobank-review-" in c for c in seen)
    assert seen[0] != seen[1]


# ── dead-flag reachability: Settings -> RuntimeConfig (the load-bearing pattern) ──

def test_external_council_flags_reachable_via_from_settings():
    class S:
        plan_external_council_enabled = True
        plan_external_council_agents = "codex,claude"
        plan_external_council_policy = "always"
        plan_external_council_timeout_s = 42

    cfg = RuntimeConfig.from_settings(S())
    assert cfg.external_council_enabled is True
    assert cfg.external_council_agents == "codex,claude"
    assert cfg.external_council_policy == "always"
    assert cfg.external_council_timeout_s == 42


def test_external_council_defaults_off():
    assert RuntimeConfig.from_settings(object()).external_council_enabled is False


def test_external_council_default_off_with_real_settings():
    from biobank_agent.config import Settings

    # Shelling out to external coding-agent CLIs must be opt-in: the REAL Settings default (not just a
    # bare object) must keep the council off. This guards the Settings->RuntimeConfig default wiring.
    assert RuntimeConfig.from_settings(Settings()).external_council_enabled is False


# ── planner wiring: notes fold into the drafted plan's open_questions ────────

def _cfg(**over):
    base = dict(primary_model="m", planner_model="m", critic_model="m", summarizer_model="m",
                safety_reviewer_model="m")
    base.update(over)
    return RuntimeConfig(**base)


def test_planner_folds_external_review_into_open_questions(monkeypatch):
    monkeypatch.setattr(ec, "collect_plan_reviews", lambda *a, **k: ["[codex] add a kinship/PCA step"])
    router, _ = _router(_plan_handler)
    planner = RuntimePlanner(router, _cfg(external_council_enabled=True, external_council_policy="always"),
                             num_candidates=1, enable_clarification=False)
    plan = planner.build_plan("vitiligo WGS case/control analysis",
                              tool_names=["vcf_qc", "vcf_annotation", "vcf_association"])
    assert any("kinship/PCA" in q for q in plan.open_questions)


def test_planner_passes_plan_steps_to_external_review(monkeypatch):
    captured = {}

    def spy(objective, step_descriptions, **kwargs):
        captured["steps"] = list(step_descriptions)
        return []

    monkeypatch.setattr(ec, "collect_plan_reviews", spy)
    router, _ = _router(_plan_handler)
    planner = RuntimePlanner(router, _cfg(external_council_enabled=True, external_council_policy="always"),
                             num_candidates=1, enable_clarification=False)
    planner.build_plan("vitiligo WGS case/control analysis",
                       tool_names=["vcf_qc", "vcf_annotation", "vcf_association"])
    # The drafted plan's steps must reach the review (not be dropped to '(none)').
    joined = " ".join(captured.get("steps") or [])
    assert "QC the VCF" in joined and "association" in joined.lower()


def test_planner_skips_external_review_when_disabled(monkeypatch):
    called = {"n": 0}

    def spy(*a, **k):
        called["n"] += 1
        return ["[codex] note"]

    monkeypatch.setattr(ec, "collect_plan_reviews", spy)
    router, _ = _router(_plan_handler)
    planner = RuntimePlanner(router, _cfg(external_council_enabled=False),
                             num_candidates=1, enable_clarification=False)
    planner.build_plan("vitiligo WGS case/control analysis",
                       tool_names=["vcf_qc", "vcf_annotation", "vcf_association"])
    assert called["n"] == 0
