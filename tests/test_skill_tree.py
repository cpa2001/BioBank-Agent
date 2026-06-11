"""Hierarchical skill-tree + classify_skill."""

from __future__ import annotations

from biobank_agent.skills import manifest as m
from biobank_agent.skills import skill_tree as st


def test_navigate_root_lists_top_nodes():
    out = st.navigate_tree()
    assert out["tree_available"] is True
    ids = {c["id"] for c in out["children"]}
    assert {"genomics_germline", "singlecell", "spatial", "multiomics_preview"} <= ids
    assert out["skills"] == []  # root has no leaf skills directly


def test_navigate_into_node_returns_children_and_skills():
    sc = st.navigate_tree("singlecell")
    assert {c["id"] for c in sc["children"]} == {"sc_core", "sc_dynamics", "sc_regulation"}
    preview = st.navigate_tree("multiomics_preview")
    names = {s["name"] for s in preview["skills"]}
    assert "spatial_cell_interaction" in names and "scrna_expression_differential" in names
    assert len(preview["skills"]) == 15


def test_navigate_describe_callback_supplies_descriptions():
    out = st.navigate_tree("statistics", describe=lambda n: f"desc::{n}")
    assert all(s["description"] == f"desc::{s['name']}" for s in out["skills"])


def test_classify_populated_leaf_by_overlap():
    # Strong token overlap with the vcf_core leaf vocabulary -> filed there deterministically.
    assert st.classify_skill("vcf_filter_qc", "VCF variant QC filtering") == "vcf_core"


def test_classify_low_confidence_returns_pending():
    # No meaningful overlap with any leaf -> left for manual review, never mis-filed.
    assert st.classify_skill("zzz_widget", "completely unrelated foobar gizmo") == st.UNCLASSIFIED


def test_classify_injected_classifier_wins_when_valid():
    chosen = st.classify_skill("x", "y", classifier=lambda n, d, cands: "spatial_core")
    assert chosen == "spatial_core"
    # An invalid choice from the classifier falls back to the deterministic vote.
    fallback = st.classify_skill("vcf_filter_qc", "VCF variant QC filtering",
                                 classifier=lambda n, d, cands: "not_a_real_node")
    assert fallback == "vcf_core"


def test_subtree_skills_collects_descendants():
    # genomics_germline has no direct skills; all live in its vcf_core / statgen_postgwas leaves.
    sub = set(st.subtree_skills("genomics_germline"))
    assert {"vcf_qc", "vcf_association", "gwas_proxy", "pathway_enrichment"} <= sub


def test_tree_places_every_known_skill_exactly_once():
    placed = [s for n in m.tree_nodes().values() for s in (n.get("skills") or [])]
    assert len(placed) == len(set(placed)), "a skill is filed into more than one leaf"
    flat = {s for names in m._load()["domains"].values() for s in names}
    assert flat - set(placed) == set(), "a flat-domain skill is missing from the tree"


def test_domain_of_reproduces_literal_domains():
    # Tree-derived domain_of must not move any existing skill out of its legacy domain.
    for dom, names in m._load()["domains"].items():
        for s in names:
            assert m.domain_of(s) == dom


def test_generated_domains_reproduce_literal_memberships():
    gen = m.generated_domains()
    for dom, names in m._load()["domains"].items():
        assert set(names) <= set(gen.get(dom, [])), f"tree drops a {dom} membership"


def test_navigate_degrades_without_tree(monkeypatch):
    monkeypatch.setattr(m, "tree_available", lambda: False)
    out = st.navigate_tree()
    assert out["tree_available"] is False
    assert isinstance(out["children"], list)  # synthesized from flat domain summaries


def test_domain_of_falls_back_to_literal_without_tree(monkeypatch):
    monkeypatch.setattr(m, "tree_nodes", lambda: {})  # no tree -> leaf_of returns None
    assert m.domain_of("vcf_association") == "genomics"  # via literal domains fallback


def test_registry_search_subtree_filter():
    from biobank_agent.registry import autodiscover_skills
    from biobank_agent.core.tools.registry import ToolRegistry

    autodiscover_skills()
    reg = ToolRegistry()
    reg.hydrate_from_legacy()
    q = "vcf variant association covariates"
    all_hits = {h["name"] for h in reg.search(q, k=10)}
    assert "vcf_association" in all_hits  # deferred genomics skill surfaces normally
    sub_hits = {h["name"] for h in reg.search(q, k=10, subtree="cohort_data")}
    assert "vcf_association" not in sub_hits  # restricted to the cohort_data subtree
    # subtree=None is byte-identical to omitting it (back-compat).
    assert {h["name"] for h in reg.search(q, k=10, subtree=None)} == all_hits


def test_navigate_skill_tree_tool_registered():
    from biobank_agent.core.tools.native import build_native_tools

    names = {t.spec().name for t in build_native_tools()}
    assert "navigate_skill_tree" in names and "skill_search" in names
