"""Tests for M16: adversarial-game council + A/B harness (pure, injected roles)."""

from __future__ import annotations

from biobank_agent.runtime.adversarial_council import (
    BLOCK, MAJOR, MINOR, AdjudicatedPlan, Flaw, adversarial_plan,
)
from biobank_agent.runtime.council_ab import ab_compare


def test_referee_accepts_blocking_flaw_and_revises_plan():
    redteam = lambda obj, draft: [{"severity": BLOCK, "kind": "no_covariates", "claim": "no PCs", "fix": "add 10 PCs"}]
    calls = {"n": 0}

    def referee(obj, draft, flaws):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"plan": draft + " + adjust for 10 PCs", "accepted": flaws, "dismissed": [], "notes": "fix PCs"}
        return {"plan": draft, "accepted": [], "dismissed": [], "notes": "clean"}

    out = adversarial_plan("GWAS", proposer_fn=lambda o: "run GWAS", redteam_fn=redteam,
                           referee_fn=referee, rounds=3)
    assert "10 PCs" in out.plan
    assert any(f.kind == "no_covariates" for f in out.addressed)
    assert out.game_score["redteam"] == 1          # one accepted blocking flaw
    assert out.rounds == 2                          # round 2 accepted nothing -> converged


def test_clean_draft_converges_in_one_round():
    out = adversarial_plan(
        "obj", proposer_fn=lambda o: "solid plan",
        redteam_fn=lambda o, d: [],
        referee_fn=lambda o, d, f: {"plan": d, "accepted": [], "dismissed": [], "notes": "no issues"},
        rounds=3,
    )
    assert out.rounds == 1 and out.plan == "solid plan"
    assert out.addressed == [] and out.survived == []


def test_dismissed_flaws_survive_and_score_proposer():
    out = adversarial_plan(
        "obj", proposer_fn=lambda o: "p",
        redteam_fn=lambda o, d: [{"severity": MINOR, "claim": "nitpick"}],
        referee_fn=lambda o, d, f: {"plan": d, "accepted": [], "dismissed": f, "notes": "dismissed"},
        rounds=2,
    )
    assert len(out.survived) == 1
    assert out.game_score["proposer"] == 1
    assert out.game_score["redteam"] == 0
    assert out.rounds == 1                          # no accepted load-bearing flaw -> stop


def test_rounds_cap_is_respected_when_referee_keeps_accepting():
    # Referee always accepts a major flaw -> never converges -> bounded by `rounds`.
    out = adversarial_plan(
        "obj", proposer_fn=lambda o: "p",
        redteam_fn=lambda o, d: [{"severity": MAJOR, "claim": "x", "fix": "y"}],
        referee_fn=lambda o, d, f: {"plan": d + ".", "accepted": f, "dismissed": [], "notes": ""},
        rounds=3,
    )
    assert out.rounds == 3
    assert len(out.addressed) == 3


def test_adjudicated_plan_to_dict_roundtrips_shape():
    plan = AdjudicatedPlan(plan="p", addressed=[Flaw(severity=BLOCK, claim="c")], survived=[Flaw(claim="d")])
    d = plan.to_dict()
    assert d["plan"] == "p" and d["game_score"]["redteam"] == 1 and d["game_score"]["proposer"] == 1


def test_ab_compare_ranks_better_mode_higher():
    # mode_a always scores 1.0, mode_b 0.0 -> a wins every objective.
    report = ab_compare(
        ["o1", "o2", "o3"],
        run_a=lambda o: {"q": 1.0}, run_b=lambda o: {"q": 0.0},
        score_fn=lambda r: r["q"], label_a="adversarial", label_b="symmetric",
    )
    assert report["winner"] == "adversarial"
    assert report["wins"]["adversarial"] == 3 and report["wins"]["symmetric"] == 0
    assert report["mean"]["adversarial"] == 1.0
