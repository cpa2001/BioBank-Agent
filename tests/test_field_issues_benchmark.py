"""Field-issues benchmark, output-path summary, manifest integrity, and external-skill ingest path.

These pin the seven field problems as a regression suite, prove the run-end output summary, and guard
against dead nodes (a manifest entry with no registered skill — exactly the kind that slipped through
before). The external-skill ingest test proves the stereo-seq-skills corpus path is knowledge-only.
"""

from __future__ import annotations

from types import SimpleNamespace

from biobank_agent.eval.field_benchmark import FieldIssuesBenchmark
from biobank_agent.eval.harness import EvalHarness
from biobank_agent.runtime.output_summary import summarize_output_paths


# ── the seven field problems + domain routing, as a deterministic regression suite ──

def test_field_issues_benchmark_all_pass():
    result = EvalHarness().run(FieldIssuesBenchmark(), agent=None)
    failures = [(r.case_id, r.errors) for r in result.results if not r.passed]
    assert not failures, f"field-issue regressions: {failures}"
    assert result.n_total == 11


def test_field_issues_covers_every_problem_and_domain():
    problems = {c.metadata.get("problem") for c in FieldIssuesBenchmark().cases}
    for expected in (
        "problem_1_timeout", "problem_2_monitoring", "problem_3_format_mismatch",
        "problem_4_working_dir", "problem_5_complex_parse", "problem_6_wgs_overdetect",
        "problem_7_output_paths", "domain_ukb", "domain_wgs", "domain_virtualcell",
    ):
        assert expected in problems, f"benchmark missing case for {expected}"


def test_field_issues_suite_is_importable_for_cli():
    # `python -m biobank_agent eval --suite field_issues` imports this from eval.benchmarks.
    from biobank_agent.eval.benchmarks import FieldIssuesBenchmark as FromBenchmarks

    assert FromBenchmarks().name == "field_issues"


# ── problem #7: a finished run reports where its outputs went ──

def test_output_summary_aggregates_dedupes_and_ignores_non_paths():
    steps = [
        {"report_dir": "/w/r", "report_markdown": "/w/r/report.md"},
        {"figure_artifacts": ["/w/r/f1.png", "/w/r/f1.png"]},  # duplicate collapses
        {"log_path": "/w/jobs/x.log"},
        {"status": "ok"},   # no path
        "not-a-dict",       # skipped, never crashes
    ]
    out = summarize_output_paths(steps)
    assert out["count"] == 4
    assert out["paths"].count("/w/r/f1.png") == 1
    assert "/w/r/report.md" in out["message"]


def test_output_summary_empty_run():
    out = summarize_output_paths([{"status": "ok"}, {}, None])
    assert out["count"] == 0 and "did not record" in out["message"]


def test_agent_appends_output_paths_only_when_present():
    from biobank_agent.agent import Agent

    # AnalysisRecord stores the skill result dict in key_results and figures in figure_paths.
    with_path = SimpleNamespace(state=SimpleNamespace(records=[
        SimpleNamespace(key_results={"report_markdown": "/w/r/report.md"}, figure_paths=[]),
    ]))
    appended = Agent._append_output_paths(with_path, "Analysis complete.")
    assert "## Outputs" in appended and "/w/r/report.md" in appended

    # figure_paths must surface too
    with_fig = SimpleNamespace(state=SimpleNamespace(records=[
        SimpleNamespace(key_results={"auc": 0.93}, figure_paths=["/w/r/fig1.png"]),
    ]))
    assert "/w/r/fig1.png" in Agent._append_output_paths(with_fig, "Done.")

    # a numeric-only result (no paths) leaves the text unchanged
    no_path = SimpleNamespace(state=SimpleNamespace(records=[
        SimpleNamespace(key_results={"auc": 0.93, "n_cases": 4821}, figure_paths=[]),
    ]))
    assert Agent._append_output_paths(no_path, "Analysis complete.") == "Analysis complete."


# ── no dead nodes: manifest must not reference a skill that isn't registered ──

def test_manifest_has_no_dead_nodes():
    from biobank_agent.registry import _registry, autodiscover_skills
    from biobank_agent.skills import manifest

    autodiscover_skills()
    issues = manifest.validate_manifest(set(_registry._schemas))
    assert issues == [], f"manifest dead nodes: {issues}"


def test_all_registered_skills_have_valid_schema():
    from biobank_agent.registry import autodiscover_skills, get_registry

    autodiscover_skills()
    bad = []
    for schema in get_registry().tool_schemas():
        fn = schema.get("function", {})
        if not fn.get("name") or not fn.get("description") or fn.get("parameters", {}).get("type") != "object":
            bad.append(fn.get("name"))
    assert not bad, f"skills with invalid schema: {bad}"


# ── external skills (stereo-seq-skills): the ingest path is knowledge-only, trust=external ──

def test_stereo_seq_skills_ingest_path_is_knowledge_only(tmp_path):
    from biobank_agent.runtime.skill_ingest import (
        SkillPackage,
        discover_skill_md,
        ingest_knowledge_corpus,
    )

    corpus = tmp_path / "stereo-seq-skills" / "spatial_domain_discovery"
    corpus.mkdir(parents=True)
    (corpus / "SKILL.md").write_text(
        "---\n"
        "name: Stereo-seq Spatial Domain Discovery\n"
        'description: "Identify spatial domains from Stereo-seq bins"\n'
        "---\n"
        "# How to\n1. Bin the Stereo-seq matrix.\n2. Cluster spatial domains and annotate.\n",
        encoding="utf-8",
    )

    found = discover_skill_md(tmp_path)
    assert any("stereo_seq" in s.name for s in found), [s.name for s in found]

    registered: dict[str, tuple] = {}
    out = ingest_knowledge_corpus(
        found,
        SkillPackage(name="stereo_seq_skills",
                     source_url="https://github.com/fym0503/stereo-seq-skills", commit="abc1234"),
        register_fn=lambda n, f, s: registered.__setitem__(n, (f, s)),
        classify_fn=lambda n, d: "spatial_core",
        trust_fn=lambda names: None,
    )
    assert out["status"] == "ingested" and out["count"] >= 1
    assert out["package"]["trust"] == "external"  # curator never auto-promotes ingested corpora

    # Invoking an ingested skill returns GUIDANCE only — the third-party repo's code never runs.
    name = out["skills"][0]["name"]
    func, _schema = registered[name]
    payload = func(ctx=None)
    assert payload["status"] == "guidance" and "Stereo-seq" in payload["instructions"]
