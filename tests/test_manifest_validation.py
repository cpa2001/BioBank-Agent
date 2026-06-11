"""Manifest validation, graceful provider fallback,
and edge coverage for the run_eval / council_ab harness primitives."""

from __future__ import annotations

from biobank_agent.runtime.council_ab import ab_compare
from biobank_agent.runtime.engine import ProviderRouter
from biobank_agent.runtime.run_eval import evaluate_run_tree, latency_budget, tool_success_rate
from biobank_agent.runtime.run_tree import build_run_tree
from biobank_agent.runtime.types import ProviderResponse, RuntimeConfig
from biobank_agent.skills import manifest


# ── manifest validation ────────────────────────────────────────────

def test_real_manifest_validates_clean():
    # Structural self-check (tier overlap / tree dup / dangling node refs) on the shipped manifest.
    assert manifest.validate_manifest() == []


def test_validate_manifest_detects_tier_overlap():
    bad = {"tiers": {"direct": ["a", "b"], "hidden": ["b", "c"]}}
    issues = manifest.validate_manifest(data=bad)
    assert any("both the direct and hidden" in i and "'b'" in i for i in issues)


def test_validate_manifest_detects_duplicate_tree_skill():
    bad = {"tree": {"nodes": {"n1": {"skills": ["s"]}, "n2": {"skills": ["s"]}}}}
    issues = manifest.validate_manifest(data=bad)
    assert any("filed into multiple tree nodes" in i for i in issues)


def test_validate_manifest_detects_missing_node_refs():
    bad = {"tree": {"root": ["ghost"], "nodes": {"n1": {"children": ["nope"]}}}}
    issues = manifest.validate_manifest(data=bad)
    assert any("root node 'ghost'" in i for i in issues)
    assert any("missing child node 'nope'" in i for i in issues)


def test_validate_manifest_flags_unregistered_skill():
    data = {"tiers": {"direct": ["real", "phantom"]}}
    issues = manifest.validate_manifest(["real"], data=data)
    assert any("unregistered skill 'phantom'" in i for i in issues)


# ── graceful provider fallback (robustness) ──────────────────────────────────

class _FP:
    def __init__(self, tag):
        self.tag = tag

    def complete(self, request):
        return ProviderResponse(text=self.tag, provider="fake", model=request.model or "m")


def test_resolve_falls_back_to_primary_when_role_model_unregistered():
    cfg = RuntimeConfig(primary_model="P", planner_model="MISSING")
    router = ProviderRouter({"P": _FP("primary")}, cfg)
    # PLANNER maps to "MISSING" (unregistered) -> degrade to the primary provider, not crash.
    assert router.resolve("planner").tag == "primary"


def test_resolve_raises_only_when_no_providers_at_all():
    router = ProviderRouter({}, RuntimeConfig(primary_model="P"))
    try:
        router.resolve("planner")
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected KeyError when no providers are registered")


# ── council_ab + run_eval edges (coverage) ───────────────────────────────────

def test_ab_compare_handles_empty_items_without_division_error():
    report = ab_compare([], run_a=lambda o: 1.0, run_b=lambda o: 0.0, score_fn=float)
    assert report["n"] == 0 and report["winner"] == "tie"


def test_ab_compare_reports_tie_on_equal_scores():
    report = ab_compare(["x", "y"], run_a=lambda o: 1.0, run_b=lambda o: 1.0, score_fn=float)
    assert report["winner"] == "tie" and report["wins"]["tie"] == 2


def test_run_eval_latency_budget_and_empty_tree():
    empty = build_run_tree([])
    # No tool spans -> vacuous passes.
    report = evaluate_run_tree(empty, [tool_success_rate(), latency_budget(1.0)])
    assert report.passed is True
