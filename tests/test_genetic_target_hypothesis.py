"""Tests for genetics-first target hypothesis prioritization."""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import biobank_agent
from biobank_agent.data.genetic_targets import (
    build_genetic_target_hypotheses,
    load_burden_rows,
    therapeutic_direction_from_beta,
)
from biobank_agent.registry import autodiscover_skills, get_registry
from biobank_agent.skills.genetic_target_hypothesis import genetic_target_hypothesis
from biobank_agent.skills.report import _extract_key_findings, _interpret_skill
from biobank_agent.skills.safety_check import safety_check
from biobank_agent.skills.statistical_review import statistical_review
from biobank_agent.state import AnalysisRecord, SessionState


BURDEN_ROWS = [
    {
        "gene": "MC4R",
        "phenotype": "Body mass index",
        "annotation": "pLoF",
        "beta": 0.35,
        "p_value": 6.5e-11,
        "pathways": "G alpha signalling; melanocortin pathway",
        "known_drugs": "setmelanotide",
        "open_targets_direction": "activate",
    },
    {
        "gene": "MC4R",
        "phenotype": "BMI adjusted",
        "annotation": "missense|LC",
        "beta": 0.41,
        "p_value": 2.8e-15,
        "pathways": "G alpha signalling; melanocortin pathway",
    },
    {
        "gene": "IRS1",
        "phenotype": "Body mass index",
        "annotation": "pLoF",
        "beta": -0.18,
        "p_value": 6.59e-11,
        "pathways": "insulin receptor signalling",
        "known_drugs": "aganirsen",
        "independent_direction": "inhibit",
    },
    {
        "gene": "FBXL16",
        "phenotype": "Body mass index",
        "annotation": "pLoF",
        "beta": 0.25,
        "p_value": 1.05e-10,
        "pathways": "ubiquitin ligase complex",
        "tissue_context": "brain nucleus accumbens",
    },
    {
        "gene": "PDE3B",
        "phenotype": "Body mass index",
        "annotation": "missense low confidence",
        "beta": -0.21,
        "p_value": 8e-6,
        "pathways": "G alpha signalling",
    },
    {
        "gene": "FAMOUS",
        "phenotype": "Body mass index",
        "annotation": "pLoF",
        "beta": 0.01,
        "p_value": 9e-5,
        "known_drugs": "drug-a; drug-b; drug-c",
        "literature_count": 5000,
    },
    {
        "gene": "NOISE",
        "phenotype": "Body mass index",
        "annotation": "pLoF",
        "beta": 0.7,
        "p_value": 0.2,
    },
]


class RecordingMemory:
    def __init__(self) -> None:
        self.upserts = []
        self.links = []

    def upsert_node(self, *args, **kwargs):
        self.upserts.append((args, kwargs))
        return ":".join(str(a) for a in args[:2])

    def link_nodes(self, *args, **kwargs):
        self.links.append((args, kwargs))


def test_rare_variant_direction_and_genetics_first_ranking():
    result = build_genetic_target_hypotheses(
        phenotype="BMI",
        burden_rows=BURDEN_ROWS,
        family_size=308_413,
        top_n=10,
    )

    assert therapeutic_direction_from_beta(-0.2) == "inhibit"
    assert therapeutic_direction_from_beta(0.2) == "activate"
    assert result["n_discovery_genes"] == 5
    assert result["targets"][0]["gene"] == "MC4R"
    assert result["targets"][0]["therapeutic_direction"] == "activate"
    assert result["targets"][0]["variant_support"] == "pLoF+missense|LC concordant"

    by_gene = {target["gene"]: target for target in result["targets"]}
    assert by_gene["IRS1"]["therapeutic_direction"] == "inhibit"
    assert by_gene["FAMOUS"]["score"] < by_gene["FBXL16"]["score"]
    assert by_gene["FAMOUS"]["known_drugs"] == "drug-a; drug-b; drug-c"
    assert result["mechanism_clusters"][0]["pathway"] == "G alpha signalling"


def test_variant_support_detects_discordance_across_all_rows():
    result = build_genetic_target_hypotheses(
        phenotype="LDL cholesterol",
        burden_rows=[
            {"gene": "GENE1", "phenotype": "LDL cholesterol", "annotation": "pLoF", "beta": -0.2, "p_value": 1e-8},
            {"gene": "GENE1", "phenotype": "LDL cholesterol", "annotation": "pLoF", "beta": 0.3, "p_value": 2e-8},
            {"gene": "GENE1", "phenotype": "LDL cholesterol", "annotation": "missense|LC", "beta": -0.1, "p_value": 3e-8},
        ],
    )

    assert result["targets"][0]["variant_support"] == "pLoF+missense|LC discordant"


def test_genetic_target_skill_is_registered_writes_artifacts_and_records_memory(tmp_path: Path):
    autodiscover_skills()
    registry = get_registry()
    assert "genetic_target_hypothesis" in {s["name"] for s in registry.list_skills()}

    memory = RecordingMemory()
    ctx = SimpleNamespace(report_dir=tmp_path, memory=memory)
    result = registry.execute(
        "genetic_target_hypothesis",
        {
            "phenotype": "Body mass index",
            "burden_rows": BURDEN_ROWS,
            "family_size": 308_413,
            "top_n": 3,
        },
        ctx=ctx,
    )

    assert result["status"] == "PASS"
    assert result["targets"][0]["gene"] == "MC4R"
    assert Path(result["artifacts"]["markdown"]).exists()
    assert Path(result["artifacts"]["csv"]).exists()
    assert any(call[0][0] == "therapeutic_target_hypothesis" for call in memory.upserts)
    assert any(call[0][0] == "rare_variant_burden_evidence" for call in memory.upserts)
    assert any(call[1].get("relation") == "targets_gene" for call in memory.links)
    assert any(call[1].get("relation") == "supported_by_burden_row" for call in memory.links)


def test_load_burden_rows_from_csv_and_needs_input(tmp_path: Path):
    path = tmp_path / "burden.csv"
    path.write_text(
        "gene,phenotype,annotation,beta,p_value\n"
        "GENE1,LDL cholesterol,pLoF,-0.3,1e-8\n",
        encoding="utf-8",
    )

    rows = load_burden_rows(burden_path=path)
    assert rows[0]["gene"] == "GENE1"

    result = genetic_target_hypothesis("LDL cholesterol", burden_path=str(path), write_report=False)
    assert result["targets"][0]["gene"] == "GENE1"
    assert result["targets"][0]["therapeutic_direction"] == "inhibit"

    needs_input = genetic_target_hypothesis("LDL cholesterol")
    assert needs_input["status"] == "NEEDS_INPUT"
    assert "gene" in needs_input["required_columns"]


def test_phenotype_matching_blocks_mixed_table_contamination():
    result = genetic_target_hypothesis(
        "Alzheimer disease",
        burden_rows=[
            {"gene": "LDLR", "phenotype": "LDL cholesterol", "annotation": "pLoF", "beta": -0.3, "p_value": 1e-9},
            {"gene": "MC4R", "phenotype": "Body mass index", "annotation": "pLoF", "beta": 0.3, "p_value": 1e-9},
        ],
        write_report=False,
    )

    assert result["status"] == "NO_MATCH"
    assert result["n_matched_rows"] == 0
    assert result["targets"] == []
    assert any("refusing to rank" in warning for warning in result["warnings"])


def test_prefiltered_declaration_is_required_for_unlabelled_rows():
    blocked = genetic_target_hypothesis(
        "LDL cholesterol",
        burden_rows=[{"gene": "LDLR", "annotation": "pLoF", "beta": -0.3, "p_value": 1e-9}],
        write_report=False,
    )
    allowed = genetic_target_hypothesis(
        "LDL cholesterol",
        burden_rows=[{"gene": "LDLR", "annotation": "pLoF", "beta": -0.3, "p_value": 1e-9}],
        prefiltered=True,
        write_report=False,
    )

    assert blocked["status"] == "NEEDS_PREFILTERED_DECLARATION"
    assert blocked["targets"] == []
    assert allowed["status"] == "PASS"
    assert allowed["targets"][0]["gene"] == "LDLR"
    assert allowed["multiple_testing_scope"] == "provided_rows_only"
    assert any("explicitly declared" in warning for warning in allowed["warnings"])


def test_invalid_rows_and_bad_family_size_are_not_overstated():
    invalid = genetic_target_hypothesis(
        "LDL cholesterol",
        burden_rows=[{"gene": "BAD", "phenotype": "LDL cholesterol", "annotation": "pLoF", "beta": "bad", "p_value": "bad"}],
        write_report=False,
    )
    bad_family = genetic_target_hypothesis(
        "LDL cholesterol",
        burden_rows=[{"gene": "LDLR", "phenotype": "LDL cholesterol", "annotation": "pLoF", "beta": -0.3, "p_value": 1e-9}],
        family_size=-5,
        write_report=False,
    )

    assert invalid["status"] == "INVALID_INPUT"
    assert invalid["targets"] == []
    assert invalid["row_errors"]
    assert bad_family["status"] == "PASS"
    assert bad_family["multiple_testing_scope"] == "provided_rows_only"
    assert any("family_size must be positive" in warning for warning in bad_family["warnings"])


def test_zero_and_uncertain_beta_are_governed():
    zero_p = genetic_target_hypothesis(
        "LDL cholesterol",
        burden_rows=[{"gene": "LDLR", "phenotype": "LDL cholesterol", "annotation": "pLoF", "beta": -0.3, "p_value": 0}],
        write_report=False,
    )
    uncertain = genetic_target_hypothesis(
        "LDL cholesterol",
        burden_rows=[{"gene": "LDLR", "phenotype": "LDL cholesterol", "annotation": "pLoF", "beta": 0, "p_value": 1e-9}],
        write_report=False,
    )
    record = AnalysisRecord(
        timestamp="2026-05-08T00:02:00",
        skill="genetic_target_hypothesis",
        args={"phenotype": "LDL cholesterol"},
        key_results=uncertain,
        figure_paths=[],
    )
    stats = statistical_review(scope="session", ctx=SimpleNamespace(state=SessionState(records=[record])))

    assert zero_p["targets"][0]["best_p_value"] == 1e-300
    assert uncertain["targets"][0]["therapeutic_direction"] == "uncertain"
    assert uncertain["targets"][0]["claim_status"] == "direction_uncertain"
    assert any(issue["type"] == "uncertain_therapeutic_direction" for issue in stats["issues"])


def test_schema_and_versions_are_release_consistent():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == biobank_agent.__version__

    autodiscover_skills()
    schema = next(
        s for s in get_registry().tool_schemas()
        if s["function"]["name"] == "genetic_target_hypothesis"
    )
    rows_schema = schema["function"]["parameters"]["properties"]["burden_rows"]
    assert rows_schema["items"]["type"] == "object"
    assert rows_schema["items"]["additionalProperties"] is True
    assert "prefiltered" in schema["function"]["parameters"]["properties"]


def test_report_statistical_review_and_safety_check_understand_genetic_targets():
    result = build_genetic_target_hypotheses(
        phenotype="Body mass index",
        burden_rows=BURDEN_ROWS,
        family_size=308_413,
        top_n=5,
    )
    record = AnalysisRecord(
        timestamp="2026-05-08T00:00:00",
        skill="genetic_target_hypothesis",
        args={"phenotype": "Body mass index"},
        key_results=result,
        figure_paths=[],
    )
    ctx = SimpleNamespace(state=SessionState(records=[record]))

    text = _interpret_skill(record)
    findings = _extract_key_findings([record])
    stats = statistical_review(scope="session", ctx=ctx)
    safety = safety_check(scope="session", ctx=ctx)

    assert "MC4R" in text
    assert "filtered_with_family_size" in text
    assert "Genetic target hypothesis" in findings[0]
    assert any(issue["type"] == "filtered_burden_family" for issue in stats["issues"])
    assert any(issue["type"] == "summary_statistic_release_review" for issue in safety["issues"])
    assert "treatment recommendations" in safety["issues"][0]["message"]


def test_statistical_review_blocks_no_match_and_missing_prefilter_declaration():
    no_match_record = AnalysisRecord(
        timestamp="2026-05-08T00:00:00",
        skill="genetic_target_hypothesis",
        args={"phenotype": "Alzheimer disease"},
        key_results=genetic_target_hypothesis(
            "Alzheimer disease",
            burden_rows=[
                {"gene": "LDLR", "phenotype": "LDL cholesterol", "annotation": "pLoF", "beta": -0.3, "p_value": 1e-9},
            ],
            write_report=False,
        ),
        figure_paths=[],
    )
    missing_prefilter_record = AnalysisRecord(
        timestamp="2026-05-08T00:01:00",
        skill="genetic_target_hypothesis",
        args={"phenotype": "LDL cholesterol"},
        key_results=genetic_target_hypothesis(
            "LDL cholesterol",
            burden_rows=[{"gene": "LDLR", "annotation": "pLoF", "beta": -0.3, "p_value": 1e-9}],
            write_report=False,
        ),
        figure_paths=[],
    )
    ctx = SimpleNamespace(state=SessionState(records=[no_match_record, missing_prefilter_record]))

    stats = statistical_review(scope="session", ctx=ctx)
    issue_types = {issue["type"] for issue in stats["issues"]}

    assert "genetic_target_phenotype_no_match" in issue_types
    assert "genetic_target_prefilter_required" in issue_types


def test_statistical_review_blocks_invalid_target_input():
    invalid_record = AnalysisRecord(
        timestamp="2026-05-08T00:03:00",
        skill="genetic_target_hypothesis",
        args={"phenotype": "LDL cholesterol"},
        key_results=genetic_target_hypothesis(
            "LDL cholesterol",
            burden_rows=[{"gene": "BAD", "phenotype": "LDL cholesterol", "annotation": "pLoF", "beta": "bad", "p_value": "bad"}],
            write_report=False,
        ),
        figure_paths=[],
    )
    stats = statistical_review(scope="session", ctx=SimpleNamespace(state=SessionState(records=[invalid_record])))

    assert any(issue["type"] == "genetic_target_invalid_input" for issue in stats["issues"])
