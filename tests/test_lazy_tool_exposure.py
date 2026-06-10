"""Tests for M0: skill-tree manifest + lazy tool exposure."""

from __future__ import annotations

import asyncio

from biobank_agent.registry import autodiscover_skills
from biobank_agent.core.tools.registry import ToolRegistry
from biobank_agent.core.tools.protocol import ToolContext, LegacySkillToolHandler
from biobank_agent.skills import manifest as m


def _registry() -> ToolRegistry:
    autodiscover_skills()
    reg = ToolRegistry()
    reg.hydrate_from_legacy()
    return reg


def test_manifest_tiers_and_domains():
    assert m.manifest_available()
    assert m.exposure_of("python_exec") == m.DIRECT
    assert m.exposure_of("vcf_association") == m.DEFERRED
    assert m.exposure_of("create_skill") == m.HIDDEN
    assert m.exposure_of("a_totally_unlisted_skill") == m.DEFERRED  # default
    assert m.domain_of("vcf_pca") == "genomics"
    assert m.domain_of("train_model") == "modeling"
    assert "genomics" in m.category_summaries()


def test_exposed_handlers_is_a_small_subset():
    reg = _registry()
    total = len(reg.list_handlers())
    exposed = len(reg.exposed_handlers())
    assert total > 100                 # native + ~106 legacy skills
    assert exposed < total // 2        # lazy exposure is a real cut
    for h in reg.exposed_handlers():
        if isinstance(h, LegacySkillToolHandler):
            assert m.exposure_of(h.name) == m.DIRECT


def test_activation_surfaces_deferred_not_hidden():
    reg = _registry()
    base = {h.name for h in reg.exposed_handlers()}
    assert "vcf_association" not in base          # deferred, hidden until activated
    after = {h.name for h in reg.exposed_handlers({"vcf_association"})}
    assert "vcf_association" in after
    hidden_after = {h.name for h in reg.exposed_handlers({"create_skill"})}
    assert "create_skill" not in hidden_after      # hidden never surfaced


def test_search_ranks_deferred_excludes_direct_and_hidden():
    reg = _registry()
    hits = reg.search("genome-wide association covariates PCA", k=6)
    names = [h["name"] for h in hits]
    assert "vcf_pca" in names or "vcf_association" in names
    for h in hits:
        assert h["exposure"] == m.DEFERRED
    assert "create_skill" not in names            # hidden excluded
    assert "skill_search" not in names            # direct excluded


def test_skill_search_tool_activates():
    reg = _registry()
    store: dict[str, list[str]] = {"active_skills": []}

    def activate(names):
        for n in names:
            if n not in store["active_skills"]:
                store["active_skills"].append(n)

    tool = reg.get("skill_search")
    ctx = ToolContext(name="skill_search", args={"query": "survival analysis time to event", "k": 3}, capabilities=set())
    ctx.tool_registry = reg
    ctx.activate_skills = activate
    res = asyncio.new_event_loop().run_until_complete(tool.handle(ctx))
    assert res["status"] == "ok"
    assert res["loaded"]
    assert store["active_skills"] == res["loaded"]
    exposed_after = {h.name for h in reg.exposed_handlers(store["active_skills"])}
    assert set(res["loaded"]).issubset(exposed_after)


def test_tool_schemas_unchanged_for_backcompat():
    reg = _registry()
    assert len(reg.tool_schemas()) == len(reg.list_handlers())
