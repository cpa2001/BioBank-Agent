"""CLI-facing WGS workflow plan tests."""

from __future__ import annotations

import sqlite3
from io import StringIO
from pathlib import Path

from rich.console import Console


WGS_HUMAN_TASK = """
你是一名生物信息学分析专家。现有一套白癜风相关的全基因组测序（WGS）数据，需要对 VCF 变异文件执行标准分析流程。
VCF 文件目录：input/Files/ResultData/VirtualCell_WGS_vcf/
表型信息文件：WGS_Sample_info.xlsx
关键分组字段：Juvenile_White vs Vitiligo_White。
请执行 QC、Annotation、PCA、亲缘关系、关联分析、罕见变异 burden、功能富集、候选基因解读、相关论文调研/论文复现、Action Graph 记录、Markdown/HTML 报告输出。
"""

WGS_SHORT_TASKS = [
    "分析这批白癜风WGS数据",
    "比较青少年白癜风和白癜风组的WGS遗传差异",
    "用现有VirtualCell VCF做白癜风WGS分析",
]


REQUIRED_WGS_SKILLS = {
    "wgs_environment_check",
    "cohort_phenotype_summary",
    "vcf_sample_list",
    "vcf_qc",
    "vcf_annotation",
    "vcf_pca",
    "vcf_kinship",
    "vcf_association",
    "vcf_burden_test",
    "pathway_enrichment",
    "vcf_phenotype_comparison",
    "train_phenotype_model",
    "feature_importance",
    "embedding",
    "deep_research",
    "statistical_review",
    "safety_check",
    "world_model_audit",
    "generate_report",
}


def test_wgs_human_task_uses_deterministic_schema_valid_plan():
    from biobank_agent.planner import LongHorizonPlanner, PlanSchemaValidator
    from biobank_agent.registry import autodiscover_skills, get_registry

    autodiscover_skills()
    registry = get_registry()
    skills = [s["name"] for s in registry.list_skills()]
    schemas = registry.tool_schemas()

    plan = LongHorizonPlanner(tool_schemas=schemas).decompose(
        WGS_HUMAN_TASK,
        skills,
        tool_schemas=schemas,
    )

    plan_skills = [step.skill for step in plan.steps]
    assert REQUIRED_WGS_SKILLS.issubset(set(plan_skills))
    assert plan_skills.index("vcf_qc") < plan_skills.index("vcf_association")
    assert plan_skills.index("vcf_annotation") < plan_skills.index("pathway_enrichment")

    assoc = next(step for step in plan.steps if step.skill == "vcf_association")
    assert assoc.args["case_group"] == "J"
    assert assoc.args["control_group"] == "V"

    model = next(step for step in plan.steps if step.skill == "train_phenotype_model")
    assert model.args["include_classes"] == "J,V"

    issues = PlanSchemaValidator(skills, schemas).validate(plan)
    assert issues == []


def test_short_wgs_tasks_trigger_complete_deterministic_plan():
    from biobank_agent.planner import LongHorizonPlanner, PlanSchemaValidator
    from biobank_agent.registry import autodiscover_skills, get_registry

    autodiscover_skills()
    registry = get_registry()
    skills = [s["name"] for s in registry.list_skills()]
    schemas = registry.tool_schemas()
    planner = LongHorizonPlanner(tool_schemas=schemas)

    for task in WGS_SHORT_TASKS:
        plan = planner.decompose(task, skills, tool_schemas=schemas)
        plan_skills = {step.skill for step in plan.steps}
        assert planner._is_wgs_vitiligo_goal(task) is True
        assert REQUIRED_WGS_SKILLS.issubset(plan_skills)
        assert plan.title == "VirtualCell WGS Vitiligo Case-Control Analysis"
        assert "完整执行" not in plan.goal
        assert PlanSchemaValidator(skills, schemas).validate(plan) == []


def test_wgs_plan_mode_asks_key_clarifications_for_short_prompt(tmp_path):
    from biobank_agent.planner import PlanMode

    questions = PlanMode(tmp_path).identify_clarifications("分析这批白癜风WGS数据")
    assert [q["id"] for q in questions] == [
        "wgs_data_source",
        "wgs_grouping",
        "wgs_execution_scope",
    ]
    assert questions[0]["options"][0]["label"] == "Auto Discover"
    assert "Juvenile_White" in questions[1]["options"][0]["value"]


def test_wgs_cli_e2e_auditor_summarizes_plan_artifacts_and_action_graph(tmp_path):
    from scripts import wgs_cli_e2e

    plans = tmp_path / "plans"
    plans.mkdir()
    plan_md = plans / "plan.md"
    plan_md.write_text(
        "\n".join(f"`{skill}({{}})`" for skill in sorted(REQUIRED_WGS_SKILLS)),
        encoding="utf-8",
    )

    run_dir = tmp_path / "reports" / "20260520_000000"
    for subdir in wgs_cli_e2e.REQUIRED_RESULT_DIRS:
        (run_dir / "results" / subdir).mkdir(parents=True)
        (run_dir / "results" / subdir / "artifact.tsv").write_text("x\n", encoding="utf-8")
    for subdir in wgs_cli_e2e.REQUIRED_RUN_DIRS:
        (run_dir / subdir).mkdir(parents=True)
    for patterns in wgs_cli_e2e.REQUIRED_ARTIFACT_PATTERNS.values():
        for pattern in patterns:
            path = run_dir / pattern.replace("*", "required")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x\n", encoding="utf-8")
    (run_dir / "report.md").write_text(
        "# Report\nJuvenile_White Vitiligo_White WGS QC association pathway literature reproducibility\n",
        encoding="utf-8",
    )
    (run_dir / "report.html").write_text("<html></html>\n", encoding="utf-8")
    (run_dir / "run_reproduce_wgs.sh").chmod(0o755)

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

    plan = wgs_cli_e2e._plan_summary(plans)
    artifacts = wgs_cli_e2e._artifact_summary(run_dir)
    graph = wgs_cli_e2e._action_graph_counts(memory)

    assert plan["missing_required_skills"] == []
    assert artifacts["missing_result_dirs"] == []
    assert artifacts["missing_run_dirs"] == []
    assert artifacts["missing_artifact_groups"] == []
    assert artifacts["missing_report_terms"] == []
    assert {"report.md", "report.html"}.issubset(set(artifacts["reports"]))
    assert graph == {"nodes": 1, "edges": 1}
