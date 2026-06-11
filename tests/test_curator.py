"""Tests for M9: skill curator (usage-driven promotion + auto-harness)."""

from __future__ import annotations

import json

from biobank_agent.runtime.curator import (
    SkillUsageStats, collect_usage, recommend_curation, apply_curation,
    propose_harness_tasks, curate,
)


def _records():
    # vcf_pca: 6 calls, all ok (deferred -> promote candidate)
    # smart_plot: 6 calls, mostly failing (direct -> demote candidate)
    # field_search: 2 calls (below min_calls -> ignored)
    recs = []
    recs += [{"skill": "vcf_pca", "status": "ok"} for _ in range(6)]
    recs += [{"skill": "smart_plot", "status": "failed"} for _ in range(5)] + [{"skill": "smart_plot", "status": "ok"}]
    recs += [{"skill": "field_search", "status": "ok"} for _ in range(2)]
    return recs


def _exposure(name):
    return {"smart_plot": "direct", "field_search": "direct"}.get(name, "deferred")


def test_collect_usage_counts_and_rates():
    usage = collect_usage(_records())
    assert usage["vcf_pca"].calls == 6 and usage["vcf_pca"].successes == 6
    assert usage["vcf_pca"].success_rate == 1.0
    assert usage["smart_plot"].calls == 6 and usage["smart_plot"].failures == 5
    assert round(usage["smart_plot"].success_rate, 2) == 0.17


def test_collect_usage_ignores_records_without_skill():
    usage = collect_usage([{"status": "ok"}, {"skill": "x", "status": "ok"}])
    assert set(usage) == {"x"}


def test_recommend_promote_demote_and_min_calls():
    usage = collect_usage(_records())
    recs = recommend_curation(usage, _exposure, min_calls=5)
    assert "vcf_pca" in recs["promote"]          # deferred + 6 calls + 100% -> promote
    assert "smart_plot" in recs["demote"]        # direct + 6 calls + 17% -> demote
    assert "field_search" not in recs["promote"] + recs["demote"]  # below min_calls


def test_pinned_skill_is_not_demoted():
    usage = collect_usage(_records())
    recs = recommend_curation(usage, _exposure, min_calls=5, pinned=lambda n: n == "smart_plot")
    assert "smart_plot" not in recs["demote"]


def test_external_skill_is_not_auto_promoted():
    usage = collect_usage(_records())
    # vcf_pca is deferred + 6/6 success -> normally a promote candidate.
    base = recommend_curation(usage, _exposure, min_calls=5, trust_of=lambda n: "internal")
    assert "vcf_pca" in base["promote"]
    # Tagging it external (ingested corpus) blocks the usage-only auto-promotion.
    gated = recommend_curation(usage, _exposure, min_calls=5,
                               trust_of=lambda n: "external" if n == "vcf_pca" else "internal")
    assert "vcf_pca" not in gated["promote"]


def test_apply_curation_mutates_manifest(tmp_path):
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps({"tiers": {"direct": ["smart_plot", "think"], "hidden": ["secret_skill"]}}), encoding="utf-8")
    applied = apply_curation({"promote": ["vcf_pca", "secret_skill"], "demote": ["smart_plot"]}, mpath)
    assert applied["promoted"] == ["vcf_pca"]    # secret_skill is hidden -> not promoted
    assert applied["demoted"] == ["smart_plot"]
    after = json.loads(mpath.read_text(encoding="utf-8"))["tiers"]["direct"]
    assert "vcf_pca" in after and "smart_plot" not in after and "think" in after


def test_propose_harness_tasks():
    tasks = propose_harness_tasks(["vcf_pca", "prevalence", "vcf_pca"])
    assert [t["skill"] for t in tasks] == ["prevalence", "vcf_pca"]   # deduped + sorted
    assert all(t["auto_generated"] and t["expectations"]["skill_invoked"] == t["skill"] for t in tasks)


def test_curate_dry_run_does_not_mutate(tmp_path):
    mpath = tmp_path / "manifest.json"
    original = {"tiers": {"direct": ["smart_plot"], "hidden": []}}
    mpath.write_text(json.dumps(original), encoding="utf-8")
    out = curate(_records(), _exposure, mpath, dry_run=True, min_calls=5)
    assert out["dry_run"] is True
    assert out["recommendations"]["promote"] == ["vcf_pca"]
    assert out["applied"] == {"promoted": [], "demoted": []}
    assert json.loads(mpath.read_text(encoding="utf-8")) == original   # untouched
    assert any(t["skill"] == "vcf_pca" for t in out["harness_tasks"])
