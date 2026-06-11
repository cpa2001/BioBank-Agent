"""VirtualCell/BWhair multimodal data and planner coverage."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pandas as pd


def test_default_wgs_manifest_uses_vcf_stem_sample_ids():
    from biobank_agent.data.virtualcell_multimodal import load_default_wgs_manifest

    df = load_default_wgs_manifest()

    assert len(df) == 28
    assert {"sample_id", "source_sample_id", "donor", "phenotype", "phenotype_group", "vcf_path"}.issubset(df.columns)
    assert "J1-41Y-F" in set(df["sample_id"])
    assert "S1-58Y-F-1" in set(df["sample_id"])
    assert df["sample_id"].equals(df["source_sample_id"])
    assert df["phenotype"].value_counts().to_dict() == {
        "Senile_White": 12,
        "Juvenile_White": 10,
        "Vitiligo_White": 6,
    }
    assert set(df[df["phenotype"] == "Juvenile_White"]["phenotype_group"]) == {"J"}
    assert set(df[df["phenotype"] == "Vitiligo_White"]["phenotype_group"]) == {"V"}


def test_bwhair_manifest_parses_modalities_and_hair_state():
    from biobank_agent.data.virtualcell_multimodal import load_bwhair_manifest

    df = load_bwhair_manifest()

    assert len(df) >= 80
    assert {"spatial", "scrna", "scatac"}.issubset(set(df["modality"]))
    assert {"B", "W", "WB", "G", "GB"}.issubset(set(df["hair_state"]) - {""})
    j1 = df[df["filename"] == "J1-W-41Y-F_SCT.h5ad"].iloc[0]
    assert j1["donor"] == "J1"
    assert j1["hair_state"] == "W"
    assert j1["age"] == 41
    assert j1["sex"] == "F"
    scrna = df[df["filename"] == "BWhair_scRNA.h5ad"].iloc[0]
    assert scrna["modality"] == "scrna"
    assert scrna["donor"] == ""
    assert scrna["cells"] == 764557


def test_virtualcell_link_reports_cross_modal_donors():
    from biobank_agent.data.virtualcell_multimodal import link_wgs_to_bwhair

    link = link_wgs_to_bwhair()

    row = link[link["donor"] == "J1"].iloc[0]
    assert row["has_wgs"] is True or bool(row["has_wgs"]) is True
    assert "spatial" in row["modalities"]
    assert "B" in row["hair_states"]
    assert "W" in row["hair_states"]
    assert "Juvenile_White" in row["phenotype"]


def test_h5ad_missing_path_degrades_without_exception(tmp_path):
    from biobank_agent.data.virtualcell_multimodal import inspect_h5ad_metadata

    result = inspect_h5ad_metadata(tmp_path / "missing.h5ad")

    assert result["status"] == "missing"
    assert result["exists"] is False
    assert result["warnings"]


class _Ctx:
    def __init__(self, tmp_path):
        self.report_dir = tmp_path
        self.state = SimpleNamespace(custom_data={}, figures=[], cohorts={})


def test_virtualcell_inventory_skill_writes_manifest_outputs(tmp_path):
    from biobank_agent.skills.virtualcell_multimodal import virtualcell_data_inventory

    ctx = _Ctx(tmp_path)
    result = virtualcell_data_inventory(inspect_h5ad=False, ctx=ctx)

    assert result["n_wgs_samples"] == 28
    assert result["n_h5ad_files"] >= 80
    assert "spatial" in result["h5ad_modalities"]
    assert (tmp_path / "results" / "00_Cohort" / "virtualcell_wgs_manifest.tsv").exists()
    assert (tmp_path / "results" / "00_Cohort" / "virtualcell_h5ad_manifest.tsv").exists()
    assert "virtualcell_wgs_manifest" in ctx.state.custom_data


def test_virtualcell_link_skill_writes_linkage(tmp_path):
    from biobank_agent.skills.virtualcell_multimodal import virtualcell_multimodal_link

    result = virtualcell_multimodal_link(ctx=_Ctx(tmp_path))

    assert result["n_wgs_donors"] == 28
    assert result["n_linked_donors"] >= 20
    assert result["h5ad_without_wgs"]
    assert (tmp_path / "results" / "00_Cohort" / "virtualcell_multimodal_linkage.tsv").exists()


def test_h5ad_summary_skill_handles_unreadable_real_paths(tmp_path):
    from biobank_agent.skills.virtualcell_multimodal import h5ad_sample_summary

    result = h5ad_sample_summary(modality="spatial", max_files=2, ctx=_Ctx(tmp_path))

    assert result["n_files_summarized"] == 2
    assert result["summary_json"].endswith("h5ad_sample_summary.json")
    assert result["status_counts"]


def test_spatial_and_singlecell_summary_skills(tmp_path):
    from biobank_agent.skills.virtualcell_multimodal import singlecell_modality_summary, spatial_hair_summary

    spatial = spatial_hair_summary(group_by="hair_state", ctx=_Ctx(tmp_path))
    single = singlecell_modality_summary(inspect_h5ad=False, ctx=_Ctx(tmp_path))

    assert spatial["n_spatial_files"] == 79
    assert any(row["hair_state"] == "W" for row in spatial["groups"])
    assert single["n_files"] == 2
    assert {row["modality"] for row in single["modalities"]} == {"scrna", "scatac"}


def test_multimodal_short_prompts_trigger_schema_valid_plan():
    from biobank_agent.planner import LongHorizonPlanner, PlanSchemaValidator
    from biobank_agent.registry import autodiscover_skills, get_registry

    autodiscover_skills()
    registry = get_registry()
    skills = [s["name"] for s in registry.list_skills()]
    schemas = registry.tool_schemas()
    planner = LongHorizonPlanner(tool_schemas=schemas)

    for task in [
        "分析这批黑白发多组学数据",
        "做 VirtualCell BWhair h5ad 单细胞空间整合分析",
        "整合WGS和Stereo-seq scRNA scATAC数据生成报告",
    ]:
        plan = planner.decompose(task, skills, tool_schemas=schemas)
        plan_skills = {step.skill for step in plan.steps}
        assert planner._is_virtualcell_multimodal_goal(task) is True
        assert {
            "virtualcell_data_inventory",
            "virtualcell_multimodal_link",
            "h5ad_sample_summary",
            "spatial_hair_summary",
            "singlecell_modality_summary",
            "statistical_review",
            "safety_check",
            "world_model_audit",
            "generate_report",
        }.issubset(plan_skills)
        assert plan.title == "VirtualCell BWhair Multimodal Analysis"
        assert PlanSchemaValidator(skills, schemas).validate(plan) == []


def test_juvenile_hair_mechanism_short_prompts_trigger_schema_valid_plan():
    from biobank_agent.planner import LongHorizonPlanner, PlanSchemaValidator
    from biobank_agent.registry import autodiscover_skills, get_registry

    autodiscover_skills()
    registry = get_registry()
    skills = [s["name"] for s in registry.list_skills()]
    schemas = registry.tool_schemas()
    planner = LongHorizonPlanner(tool_schemas=schemas)

    task = "Juvenile hair whitening特异性的遗传变异及其表观组、转录组、空间组机制分析"
    plan = planner.decompose(task, skills, tool_schemas=schemas)
    plan_skills = {step.skill for step in plan.steps}

    assert planner._is_juvenile_hair_mechanism_goal(task) is True
    assert plan.title == "Juvenile Hair Whitening Multi-Omics Mechanism Analysis"
    assert {
        "trajectory_profile_match",
        "goal_intent_classifier",
        "jh_variant_discovery",
        "regulatory_variant_annotation",
        "tf_binding_disruption",
        "scatac_peak_overlap",
        "scatac_accessibility_differential",
        "scrna_expression_differential",
        "atac_expression_coupling",
        "spatial_celltype_localization",
        "spatial_cell_interaction",
        "multiomics_mechanism_prioritization",
        "workflow_gap_detector",
        "agent_workflow_evolver",
        "generate_report",
    }.issubset(plan_skills)
    assert PlanSchemaValidator(skills, schemas).validate(plan) == []


def test_juvenile_hair_mechanism_harness_prompt_pool_triggers_mechanism_plan():
    from biobank_agent.planner import LongHorizonPlanner, PlanSchemaValidator
    from biobank_agent.registry import autodiscover_skills, get_registry
    from scripts.virtualcell_multimodal_cli_e2e import SHORT_MECHANISM_TASKS

    autodiscover_skills()
    registry = get_registry()
    skills = [s["name"] for s in registry.list_skills()]
    schemas = registry.tool_schemas()
    planner = LongHorizonPlanner(tool_schemas=schemas)

    required = {
        "trajectory_profile_match",
        "goal_intent_classifier",
        "jh_variant_discovery",
        "tf_binding_disruption",
        "scatac_accessibility_differential",
        "scrna_expression_differential",
        "spatial_cell_interaction",
        "multiomics_mechanism_prioritization",
        "workflow_gap_detector",
        "agent_workflow_evolver",
        "generate_report",
    }
    for task in SHORT_MECHANISM_TASKS:
        plan = planner.decompose(task, skills, tool_schemas=schemas)
        plan_skills = {step.skill for step in plan.steps}
        assert planner._is_juvenile_hair_mechanism_goal(task) is True
        assert required.issubset(plan_skills)
        assert PlanSchemaValidator(skills, schemas).validate(plan) == []


def test_goal_intent_classifier_routes_short_chinese_mechanism_prompt():
    from biobank_agent.skills.goal_intent_classifier import classify_goal_intent, goal_intent_classifier

    task = "少白头的基因组表观组转录组空间组机制分析"
    profile = classify_goal_intent(task)
    skill_profile = goal_intent_classifier(task)

    assert profile["task_family"] == "juvenile_hair_multiomics_mechanism"
    assert profile["confidence"] >= 0.55
    assert {"wgs", "scatac", "scrna", "spatial"}.issubset(set(profile["modalities"]))
    assert skill_profile["task_family"] == profile["task_family"]


def test_goal_intent_classifier_reconciles_conflicting_llm_route():
    from biobank_agent.skills.goal_intent_classifier import classify_goal_intent

    class _BroadRouteLLM:
        def chat(self, *args, **kwargs):
            return type(
                "Resp",
                (),
                {
                    "text": (
                        '{"task_family":"virtualcell_multimodal","modalities":["wgs","scrna","scatac","spatial"],'
                        '"concepts":["hair_whitening"],"confidence":0.9,"rationale":"too broad"}'
                    )
                },
            )()

    task = (
        "少白头的基因组表观组转录组空间组机制分析\n\n"
        "Clarifications:\n"
        "- juvenile_hair_mechanism_policy: Treat this as a Juvenile hair-whitening multi-omics mechanism analysis."
    )
    profile = classify_goal_intent(task, llm=_BroadRouteLLM())

    assert profile["task_family"] == "juvenile_hair_multiomics_mechanism"
    assert profile["source"] in {"clarification_policy_profile", "llm_fallback_ensemble"}


def test_trajectory_profile_match_exports_harness_and_mcp_contracts():
    from biobank_agent.skills.trajectory_profile import match_trajectory_profile, trajectory_profile_match

    task = "少白头的基因组表观组转录组空间组机制分析"
    available = [
        "trajectory_profile_match",
        "goal_intent_classifier",
        "jh_variant_discovery",
        "generate_report",
    ]

    profile = match_trajectory_profile(task, available_skills=available)
    skill_profile = trajectory_profile_match(task, available_skills=",".join(available))

    assert profile["trajectory_id"] == "juvenile_hair_multiomics_mechanism"
    assert profile["confidence"] >= 0.55
    assert "tf_binding_disruption" in profile["required_skills"]
    assert "tf_binding_disruption" in profile["missing_skills"]
    assert profile["artifact_contract"]["mechanism_ranking"]
    assert {item["kind"] for item in profile["mcp_resource_needs"]} >= {"motif_database", "genome_annotation"}
    assert "clarification_policy" in profile["workflow_hooks"]
    assert [q["id"] for q in profile["clarification_questions"]] == [
        "jh_primary_contrast",
        "jh_execution_depth",
        "jh_external_resources",
    ]
    assert skill_profile["trajectory_id"] == profile["trajectory_id"]


def test_clarification_policy_skill_normalizes_interactive_questions():
    from biobank_agent.registry import autodiscover_skills, get_registry
    from biobank_agent.skills.clarification_policy import clarification_policy

    autodiscover_skills()
    skills = [s["name"] for s in get_registry().list_skills()]
    result = clarification_policy(
        "少白头的基因组表观组转录组空间组机制分析",
        available_skills=",".join(skills),
    )

    assert result["n_questions"] == 3
    assert [q["id"] for q in result["questions"]] == [
        "jh_primary_contrast",
        "jh_execution_depth",
        "jh_external_resources",
    ]
    assert result["questions"][0]["source"] == "trajectory_profile"
    assert result["questions"][0]["options"][0]["label"] == "J vs V + W/B"


def test_multimodal_plan_mode_asks_key_clarifications(tmp_path):
    from biobank_agent.planner import PlanMode

    questions = PlanMode(tmp_path).identify_clarifications("分析这批黑白发多组学数据")

    assert [q["id"] for q in questions] == [
        "virtualcell_modality_scope",
        "virtualcell_execution_scope",
        "virtualcell_grouping",
    ]
    assert questions[0]["options"][0]["label"] == "All Modalities"
    assert "backed" in questions[1]["options"][0]["value"].lower()


def test_juvenile_hair_plan_mode_asks_mechanism_clarifications(tmp_path):
    from biobank_agent.planner import PlanMode

    questions = PlanMode(tmp_path).identify_clarifications(
        "Juvenile hair whitening的基因组、表观组、转录组、空间组机制分析"
    )

    assert [q["id"] for q in questions] == [
        "jh_primary_contrast",
        "jh_execution_depth",
        "jh_external_resources",
    ]
    assert questions[0]["options"][0]["label"] == "J vs V + W/B"
    assert "backed" in questions[1]["options"][0]["value"].lower()


def test_juvenile_hair_mechanism_skills_write_artifacts(tmp_path):
    from biobank_agent.skills.juvenile_hair_mechanism import (
        atac_expression_coupling,
        jh_variant_discovery,
        multiomics_mechanism_prioritization,
        regulatory_variant_annotation,
        scatac_accessibility_differential,
        scatac_peak_overlap,
        scrna_expression_differential,
        spatial_cell_interaction,
        spatial_celltype_localization,
        tf_binding_disruption,
        workflow_gap_detector,
    )
    from biobank_agent.skills.agent_workflow_evolver import agent_workflow_evolver

    ctx = _Ctx(tmp_path)
    variants = jh_variant_discovery(ctx=ctx)
    regulatory = regulatory_variant_annotation(ctx=ctx)
    tf = tf_binding_disruption(ctx=ctx)
    overlap = scatac_peak_overlap(max_files=0, ctx=ctx)
    da = scatac_accessibility_differential(ctx=ctx)
    expr = scrna_expression_differential(ctx=ctx)
    coupling = atac_expression_coupling(ctx=ctx)
    loc = spatial_celltype_localization(max_files=1, ctx=ctx)
    interaction = spatial_cell_interaction(ctx=ctx)
    ranked = multiomics_mechanism_prioritization(ctx=ctx)
    gaps = workflow_gap_detector(ctx=ctx)
    evolution = agent_workflow_evolver(goal_profile="juvenile_hair_multiomics_mechanism", scan_roots="biobank_agent/skills", ctx=ctx)

    assert variants["n_candidate_variants"] > 0
    assert regulatory["n_regulatory_hits"] == variants["n_candidate_variants"]
    assert tf["n_tfbs_candidates"] > 0
    assert overlap["n_candidate_variants"] == variants["n_candidate_variants"]
    assert da["n_juvenile_donors"] == 10
    assert expr["n_genes"] > 0
    assert coupling["n_coupled_rows"] > 0
    assert loc["n_spatial_files_inspected"] == 1
    assert interaction["n_samples"] == 1
    assert ranked["n_prioritized_hypotheses"] > 0
    assert gaps["generated_skill_proposals"]
    assert evolution["summary"]["n_rule_surfaces"] >= 0
    assert (tmp_path / "results" / "06_MultiOmics" / "multiomics_mechanism_prioritization.tsv").exists()
    assert (tmp_path / "results" / "07_WorkflowEvolution" / "agent_workflow_evolution_audit.json").exists()
    assert "multiomics_mechanism_prioritization" in ctx.state.custom_data


def test_data_manager_registers_embedded_virtualcell_manifest_without_excel(tmp_path, monkeypatch):
    from biobank_agent.config import Settings
    from biobank_agent.data.loader import DataManager

    monkeypatch.delenv("WGS_SAMPLE_INFO", raising=False)
    settings = Settings(
        bank_id="virtualcell",
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "raw",
        biomarker_parquet=tmp_path / "missing.parquet",
    )
    dm = DataManager(settings)
    df = dm.query("SELECT sample_id, phenotype_group FROM biomarkers")

    assert len(df) == 28
    assert set(df["phenotype_group"]) == {"J", "S", "V"}


def test_virtualcell_manifest_can_be_loaded_as_dataframe():
    from biobank_agent.data.virtualcell_multimodal import load_virtualcell_manifest

    df = load_virtualcell_manifest("wgs,spatial")

    assert isinstance(df, pd.DataFrame)
    assert {"wgs", "spatial"}.issubset(set(df["modality"]))
    assert "scrna" not in set(df["modality"])


def test_multimodal_cli_e2e_auditor_summarizes_plan_artifacts_and_action_graph(tmp_path):
    from scripts import virtualcell_multimodal_cli_e2e as e2e

    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / "plan.md").write_text(
        "\n".join(f"`{skill}({{}})`" for skill in sorted(e2e.REQUIRED_SKILLS)),
        encoding="utf-8",
    )

    run_dir = tmp_path / "reports" / "20260521_000000"
    for patterns in e2e.REQUIRED_ARTIFACT_PATTERNS.values():
        for pattern in patterns:
            path = run_dir / pattern
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x\n", encoding="utf-8")
    (run_dir / "report.md").write_text(
        "# Report\nVirtualCell BWhair multimodal WGS Stereo scRNA scATAC Donor\n",
        encoding="utf-8",
    )
    (run_dir / "report.html").write_text("<html></html>\n", encoding="utf-8")

    memory = tmp_path / "memory"
    memory.mkdir()
    conn = sqlite3.connect(memory / "action_graph.db")
    conn.executescript(
        """
        CREATE TABLE graph_nodes(node_key TEXT);
        CREATE TABLE graph_edges(src_key TEXT);
        INSERT INTO graph_nodes VALUES ('n1');
        INSERT INTO graph_edges VALUES ('e1');
        """
    )
    conn.commit()
    conn.close()

    plan = e2e._plan_summary(plans)
    artifacts = e2e._artifact_summary(run_dir)
    graph = e2e._action_graph_counts(memory)

    assert plan["missing_required_skills"] == []
    assert artifacts["missing_artifact_groups"] == []
    assert artifacts["missing_report_terms"] == []
    assert {"report.md", "report.html"}.issubset(set(artifacts["reports"]))
    assert graph == {"nodes": 1, "edges": 1}
