"""Tests for M14 v1: external GitHub skill ingestion (knowledge-only, trust='external')."""

from __future__ import annotations

from types import SimpleNamespace

from biobank_agent.skills import manifest
from biobank_agent.runtime.skill_ingest import (
    DiscoveredSkill, SkillPackage, discover_skill_md, ingest_knowledge_corpus,
    make_knowledge_skill, parse_skill_md,
)


def test_parse_skill_md_frontmatter_and_body():
    text = (
        "---\n"
        "name: Stereo-seq QC\n"
        'description: "spatial QC for Stereo-seq"\n'
        "---\n"
        "# How to\n"
        "Run the QC steps.\n"
    )
    meta = parse_skill_md(text)
    assert meta["name"] == "Stereo-seq QC"
    assert meta["description"] == "spatial QC for Stereo-seq"
    assert meta["instructions"].startswith("# How to")


def test_parse_skill_md_without_frontmatter_is_all_instructions():
    meta = parse_skill_md("just a body, no frontmatter")
    assert meta["instructions"] == "just a body, no frontmatter"
    assert "name" not in meta


def test_discover_skill_md_finds_and_slugs(tmp_path):
    d = tmp_path / "skills" / "spatial_qc"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: Spatial QC\ndescription: x\n---\nbody\n", encoding="utf-8")
    (tmp_path / "skills" / "noname").mkdir(parents=True)
    (tmp_path / "skills" / "noname" / "SKILL.md").write_text("no frontmatter here\n", encoding="utf-8")

    found = {s.name: s for s in discover_skill_md(tmp_path)}
    assert "spatial_qc" in found and found["spatial_qc"].description == "x"
    assert "noname" in found  # falls back to the folder name as the slug


def test_make_knowledge_skill_returns_guidance_no_code():
    func, schema = make_knowledge_skill(DiscoveredSkill(name="k", description="d", instructions="do x"))
    assert schema["function"]["name"] == "k"
    out = func(ctx=None)
    assert out["status"] == "guidance" and out["instructions"] == "do x"


def test_skill_package_round_trips():
    pkg = SkillPackage(name="c", source_url="u", commit="abc123", license="MIT", skills=["a", "b"])
    assert SkillPackage.from_dict(pkg.to_dict()) == pkg


def test_ingest_registers_classifies_and_tags_external_via_injected_deps():
    registered: dict[str, dict] = {}
    classified: list[str] = []
    tagged: list[str] = []
    sources = [DiscoveredSkill(name="sk1", description="d1"), DiscoveredSkill(name="sk2", description="d2"),
               DiscoveredSkill(name="sk1", description="dup")]  # duplicate name ignored

    out = ingest_knowledge_corpus(
        sources, SkillPackage(name="corpus"),
        register_fn=lambda n, f, s: registered.__setitem__(n, s),
        classify_fn=lambda n, d: classified.append(n) or "pending_classification",
        trust_fn=lambda names: tagged.extend(names),
    )
    assert out["status"] == "ingested" and out["count"] == 2          # dup collapsed
    assert set(registered) == {"sk1", "sk2"}
    assert classified == ["sk1", "sk2"]
    assert tagged == ["sk1", "sk2"]
    assert out["package"]["trust"] == "external"
    assert out["package"]["skills"] == ["sk1", "sk2"]


def test_ingested_skill_is_external_and_excluded_from_promotion():
    from biobank_agent.runtime.curator import SkillUsageStats, recommend_curation
    try:
        ingest_knowledge_corpus([DiscoveredSkill(name="ext_demo", description="d")],
                                SkillPackage(name="corpus"),
                                register_fn=lambda n, f, s: None,
                                classify_fn=lambda n, d: "pending_classification")
        assert manifest.trust_of("ext_demo") == "external"
        # A deferred external skill with strong usage must NOT be auto-promoted.
        usage = {"ext_demo": SkillUsageStats("ext_demo", calls=10, successes=10)}
        recs = recommend_curation(usage, lambda n: "deferred", min_calls=5, trust_of=manifest.trust_of)
        assert "ext_demo" not in recs["promote"]
    finally:
        manifest.clear_external_trust()


def test_ingest_github_skills_flag_off_and_needs_pin():
    from biobank_agent.skills.ingest_skills import ingest_github_skills

    off = ingest_github_skills("https://github.com/x/y", ctx=SimpleNamespace(settings=SimpleNamespace()))
    assert off["status"] == "disabled"

    on = SimpleNamespace(settings=SimpleNamespace(external_skill_ingestion_enabled=True,
                                                  external_skills_dir="./external_skills"))
    needs = ingest_github_skills("https://github.com/x/y", ref="main", ctx=on)
    assert needs["status"] == "needs_commit_pin"


def test_ingest_github_skills_rejects_shell_injection_in_repo_url():
    from biobank_agent.skills.ingest_skills import ingest_github_skills
    on = SimpleNamespace(settings=SimpleNamespace(external_skill_ingestion_enabled=True,
                                                  external_skills_dir="./external_skills"))
    bad = ingest_github_skills("https://x/y; rm -rf ~ #", ref="abc1234", ctx=on)
    assert bad["status"] == "invalid_repo_url"
