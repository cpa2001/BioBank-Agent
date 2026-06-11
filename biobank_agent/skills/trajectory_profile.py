"""Trajectory profile matching for planner and harness routing.

Profiles keep workflow knowledge in one auditable skill instead of scattering
keyword branches through the CLI, planner, and E2E harnesses. The planner uses
the offline matcher for deterministic routing; executed plans can call the
skill with ``use_llm=true`` to let the active model reconcile ambiguous goals.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from biobank_agent.registry import skill


PROFILE_VERSION = "2026-05-21"


TRAJECTORY_PROFILES: dict[str, dict[str, Any]] = {
    "juvenile_hair_multiomics_mechanism": {
        "title": "Juvenile Hair Whitening Multi-Omics Mechanism Analysis",
        "description": "Variant-to-regulation-to-expression-to-spatial mechanism workflow for Juvenile hair whitening.",
        "modalities": ["wgs", "scatac", "scrna", "spatial"],
        "concepts": ["juvenile_or_early_onset", "hair_whitening", "mechanism"],
        "priority": 95,
        "match_groups": [
            ["juvenile", "early-onset", "early onset", "少白头", "青少年", "年轻"],
            ["hair whitening", "white hair", "hair graying", "hair greying", "白发", "黑白发", "毛发变白"],
            ["mechanism", "机制", "影响", "导致", "关系"],
            ["genome", "genomic", "wgs", "基因组", "遗传", "变异"],
            ["epigenome", "scatac", "atac", "chromatin", "表观组", "染色质", "开放性"],
            ["transcriptome", "scrna", "expression", "转录组", "表达"],
            ["spatial", "stereo", "空间组", "空间", "互作"],
        ],
        "min_groups": 3,
        "required_skills": [
            "trajectory_profile_match",
            "goal_intent_classifier",
            "virtualcell_data_inventory",
            "virtualcell_multimodal_link",
            "h5ad_sample_summary",
            "spatial_hair_summary",
            "singlecell_modality_summary",
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
            "statistical_review",
            "safety_check",
            "world_model_audit",
            "generate_report",
        ],
        "report_terms": ["Juvenile", "mechanism", "TF", "ATAC", "expression", "spatial", "WGS"],
        "artifact_contract": {
            "candidate_variants": ["results/06_MultiOmics/jh_candidate_variants.tsv"],
            "tf_binding": ["results/06_MultiOmics/tf_binding_disruption.tsv"],
            "mechanism_ranking": ["results/06_MultiOmics/multiomics_mechanism_prioritization.tsv"],
            "workflow_evolution": ["results/07_WorkflowEvolution/agent_workflow_evolution_audit.json"],
        },
        "mcp_resource_needs": [
            {"kind": "motif_database", "purpose": "sequence-level TF binding delta scoring"},
            {"kind": "genome_annotation", "purpose": "regulatory interval and gene model lookup"},
            {"kind": "literature", "purpose": "hair whitening, melanocyte and immune mechanism context"},
        ],
        "workflow_hooks": ["clarification_policy", "workflow_gap_detector", "agent_workflow_evolver", "academic_report_polisher"],
        "clarification_questions": [
            {
                "id": "jh_primary_contrast",
                "header": "Contrast",
                "question": "Which contrast should anchor Juvenile hair-whitening mechanisms?",
                "options": [
                    {
                        "label": "J vs V + W/B",
                        "value": (
                            "Use Juvenile_White versus Vitiligo_White for WGS candidate variants, "
                            "then use Juvenile donor white-vs-black BWhair samples for ATAC/RNA/spatial mechanism follow-up."
                        ),
                        "description": "Best default for linking the existing WGS and hair-state data.",
                    },
                    {
                        "label": "J vs S",
                        "value": (
                            "Use Juvenile_White versus Senile_White for WGS context and still follow white-vs-black hair states where matched samples exist."
                        ),
                        "description": "Focuses on early-versus-age-associated whitening.",
                    },
                    {
                        "label": "All Context",
                        "value": (
                            "Use Juvenile_White as the primary case group and summarize Vitiligo_White plus Senile_White as context rather than one strict control."
                        ),
                        "description": "Broader hypothesis generation with less clean association framing.",
                    },
                ],
            },
            {
                "id": "jh_execution_depth",
                "header": "Depth",
                "question": "How deeply should large h5ad matrices be read during this run?",
                "options": [
                    {
                        "label": "Backed + Fallback",
                        "value": (
                            "Open h5ad files in backed read-only mode, inspect metadata and emit explicit fallbacks for matrix-level ATAC/RNA/spatial analyses."
                        ),
                        "description": "Recommended for robust CLI and repeated harness testing.",
                    },
                    {
                        "label": "Chunked Matrix",
                        "value": (
                            "Attempt chunked matrix extraction for selected candidate genes/peaks when dependencies and memory allow, otherwise fall back."
                        ),
                        "description": "More biologically complete but slower and dependency-sensitive.",
                    },
                ],
            },
            {
                "id": "jh_external_resources",
                "header": "Resources",
                "question": "How should motif databases, genome annotations and literature tools be used?",
                "options": [
                    {
                        "label": "Local First",
                        "value": (
                            "Use local/embedded resources first, call MCP or web-backed resources only when configured, and record unavailable databases as workflow gaps."
                        ),
                        "description": "Most reproducible default.",
                    },
                    {
                        "label": "Use MCP/Web",
                        "value": (
                            "Prefer configured MCP/web resources for motif databases, regulatory annotations, enrichment and literature, with local fallback."
                        ),
                        "description": "Richer when trusted external resources are available.",
                    },
                ],
            },
        ],
    },
    "virtualcell_multimodal": {
        "title": "VirtualCell BWhair Multimodal Analysis",
        "description": "Donor-linked WGS, Stereo-seq, scRNA-seq and scATAC-seq inventory and integration workflow.",
        "modalities": ["wgs", "spatial", "scrna", "scatac"],
        "concepts": ["bwhair", "multimodal", "hair_state"],
        "priority": 80,
        "match_groups": [
            ["virtualcell", "bwhair", "bw hair", "黑白发"],
            ["multi-omics", "multiomics", "multimodal", "多组学", "多模态", "整合"],
            ["h5ad", "stereo", "spatial", "scrna", "scatac", "单细胞", "空间"],
            ["analyze", "analysis", "summary", "report", "分析", "比较", "读取", "总结", "报告"],
        ],
        "min_groups": 2,
        "required_skills": [
            "trajectory_profile_match",
            "virtualcell_data_inventory",
            "virtualcell_multimodal_link",
            "h5ad_sample_summary",
            "spatial_hair_summary",
            "singlecell_modality_summary",
            "statistical_review",
            "safety_check",
            "world_model_audit",
            "generate_report",
        ],
        "report_terms": ["VirtualCell", "BWhair", "multimodal", "WGS", "Stereo", "scRNA", "scATAC", "Donor"],
        "artifact_contract": {
            "inventory": ["results/00_Cohort/virtualcell_inventory.json"],
            "linkage": ["results/00_Cohort/virtualcell_multimodal_linkage.tsv"],
            "h5ad_summary": ["results/00_Cohort/h5ad_sample_summary.json"],
        },
        "mcp_resource_needs": [
            {"kind": "h5ad_schema", "purpose": "optional external schema documentation or validation"},
            {"kind": "literature", "purpose": "BWhair, melanocyte and spatial single-cell context"},
        ],
        "workflow_hooks": ["clarification_policy", "workflow_gap_detector", "academic_report_polisher"],
        "clarification_questions": [
            {
                "id": "virtualcell_modality_scope",
                "header": "Modalities",
                "question": "Which VirtualCell modalities should anchor this plan?",
                "options": [
                    {
                        "label": "All Modalities",
                        "value": (
                            "Analyze WGS, Stereo-seq, scRNA-seq and scATAC-seq together when readable; "
                            "record missing files or dependencies as structured fallbacks."
                        ),
                        "description": "Best default for BWhair multimodal requests.",
                    },
                    {
                        "label": "h5ad First",
                        "value": "Prioritize Stereo-seq/scRNA/scATAC h5ad inventory and metadata before WGS analysis.",
                        "description": "Useful when the user mainly asks about single-cell/spatial data.",
                    },
                    {
                        "label": "WGS Link Only",
                        "value": "Use WGS primarily as donor/phenotype metadata for multimodal linkage.",
                        "description": "Fastest path when VCF processing is not central.",
                    },
                ],
            },
            {
                "id": "virtualcell_execution_scope",
                "header": "Execution",
                "question": "How much single-cell data should be read during CLI validation?",
                "options": [
                    {
                        "label": "Backed Metadata",
                        "value": "Open h5ad files in backed read-only mode and summarize .obs/.var metadata without loading full matrices.",
                        "description": "Recommended for large h5ad files and repeated end-to-end tests.",
                    },
                    {
                        "label": "Full Single Cell",
                        "value": "Run full single-cell analyses where memory and dependencies allow, with chunking/downsampling fallback.",
                        "description": "Closest to production, may be slow or memory heavy.",
                    },
                ],
            },
            {
                "id": "virtualcell_grouping",
                "header": "Grouping",
                "question": "Which grouping should be primary for summaries?",
                "options": [
                    {
                        "label": "Hair State",
                        "value": "Use B/W/WB/G/GB hair state as the primary h5ad grouping and phenotype as WGS context.",
                        "description": "Matches BWhair sample design.",
                    },
                    {
                        "label": "Phenotype",
                        "value": "Use Juvenile_White/Senile_White/Vitiligo_White phenotype as primary grouping where donor linkage allows.",
                        "description": "Better when comparing clinical WGS groups.",
                    },
                ],
            },
        ],
    },
    "wgs_vitiligo_case_control": {
        "title": "VirtualCell WGS Vitiligo Case-Control Analysis",
        "description": "Juvenile_White versus Vitiligo_White WGS VCF QC, annotation, population structure and association workflow.",
        "modalities": ["wgs"],
        "concepts": ["vitiligo", "case_control", "variant_association"],
        "priority": 75,
        "match_groups": [
            ["wgs", "vcf", "genotyper.vcf", "全基因组", "基因组", "变异"],
            ["vitiligo", "白癜风"],
            ["juvenile_white", "juvenile", "青少年", "年轻"],
            ["association", "case-control", "case control", "关联", "差异", "比较"],
            ["qc", "annotation", "pca", "burden", "enrichment", "质控", "注释", "富集"],
        ],
        "min_groups": 2,
        "required_skills": [
            "trajectory_profile_match",
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
            "statistical_review",
            "safety_check",
            "world_model_audit",
            "generate_report",
        ],
        "report_terms": ["Juvenile_White", "Vitiligo_White", "WGS", "QC", "association", "pathway", "reproducib"],
        "artifact_contract": {
            "qc": ["results/01_QC"],
            "annotation": ["results/02_Annotation"],
            "popgen": ["results/03_PopGen"],
            "association": ["results/04_Association"],
            "enrichment": ["results/05_Enrichment"],
        },
        "mcp_resource_needs": [
            {"kind": "variant_annotation", "purpose": "VEP/ANNOVAR/SnpEff/dbSNP/ClinVar/gnomAD annotation resources"},
            {"kind": "literature", "purpose": "vitiligo GWAS and candidate gene context"},
        ],
        "workflow_hooks": ["clarification_policy", "workflow_gap_detector", "academic_report_polisher"],
        "clarification_questions": [
            {
                "id": "wgs_data_source",
                "header": "Data",
                "question": "Which WGS data source should anchor this plan?",
                "options": [
                    {
                        "label": "Auto Discover",
                        "value": (
                            "Auto-discover WGS_Sample_info.xlsx and VCF directories; if the requested "
                            "input/Files/ResultData/VirtualCell_WGS_vcf path is missing, fall back to "
                            "VC_WGS_VCF_DIR or /Files/ResultData/BW_WGS_vcf and record the fallback."
                        ),
                        "description": "Best default for short human prompts and the VirtualCell benchmark.",
                    },
                    {
                        "label": "Requested Only",
                        "value": "Use only the user-specified input manifest/VCF paths and pause if they are missing.",
                        "description": "Stricter path validation, less robust to local benchmark layout.",
                    },
                ],
            },
            {
                "id": "wgs_grouping",
                "header": "Groups",
                "question": "How should phenotype groups be compared?",
                "options": [
                    {
                        "label": "J vs V",
                        "value": (
                            "Use Juvenile_White as the case group and Vitiligo_White as the main comparison group. "
                            "Summarize and QC Senile_White samples but exclude them from the main association model."
                        ),
                        "description": "Matches the vitiligo WGS task and preserves Senile_White as context only.",
                    },
                    {
                        "label": "Ask If Missing",
                        "value": (
                            "If either Juvenile_White or Vitiligo_White is missing from the manifest, pause before "
                            "association modelling and ask for the intended comparison."
                        ),
                        "description": "More conservative if the manifest differs from the benchmark.",
                    },
                ],
            },
            {
                "id": "wgs_execution_scope",
                "header": "Scope",
                "question": "What execution scope should the CLI validation run use?",
                "options": [
                    {
                        "label": "Tractable WGS",
                        "value": (
                            "Use a chr22/SOX10-bounded WGS validation run for CLI E2E speed, while preserving "
                            "the same skills and report language for full-genome expansion."
                        ),
                        "description": "Recommended for repeated E2E runs and framework regression tests.",
                    },
                    {
                        "label": "Whole Genome",
                        "value": "Attempt whole-genome execution across available VCFs and allow longer runtime/resource use.",
                        "description": "Closer to production, too slow for frequent 50-round tests.",
                    },
                ],
            },
        ],
    },
    "paper_replication": {
        "title": "UKB Paper Replication Plan",
        "description": "Read a supplied paper/PDF/DOI and map a feasible local replication workflow.",
        "modalities": ["literature", "tabular"],
        "concepts": ["paper_replication"],
        "priority": 60,
        "match_groups": [
            ["paper", "pdf", "doi", "论文"],
            ["replicate", "reproduce", "replication", "reproduction", "复现"],
        ],
        "min_groups": 2,
        "required_skills": ["replicate_paper", "read_paper", "deep_research", "paper_replication_compare", "generate_report"],
        "report_terms": ["replication", "methods", "limitations"],
        "artifact_contract": {},
        "mcp_resource_needs": [{"kind": "literature", "purpose": "paper metadata and related work"}],
        "workflow_hooks": ["clarification_policy", "academic_report_polisher"],
        "clarification_questions": [],
    },
    "metabolic_trajectory": {
        "title": "UKB Metabolic Health Trajectory Analysis",
        "description": "Metabolic health, T2D risk and cardiometabolic biomarker trajectory feasibility workflow.",
        "modalities": ["tabular", "longitudinal"],
        "concepts": ["metabolic_health", "trajectory", "prediction"],
        "priority": 50,
        "match_groups": [
            ["biobank", "ukb", "cohort data"],
            ["metabolic", "cardiometabolic", "代谢"],
            ["type 2 diabetes", "t2d", "t2dm", "diabetes risk", "糖尿病"],
            ["biomarker", "biomarkers", "生物标志物"],
            ["trajectory", "trajectories", "progression", "prediction", "轨迹", "进展", "预测"],
        ],
        "min_groups": 4,
        "required_skills": ["ukb_data_inventory", "field_search", "trajectory_tokenize", "train_model", "generate_report"],
        "report_terms": ["trajectory", "Type 2 Diabetes", "biomarker", "limitations"],
        "artifact_contract": {},
        "mcp_resource_needs": [{"kind": "literature", "purpose": "longitudinal biomarker trajectory modelling context"}],
        "workflow_hooks": ["clarification_policy", "academic_report_polisher"],
        "clarification_questions": [],
    },
}


def _fold(text: str) -> str:
    return str(text or "").lower()


def _contains_term(goal_lower: str, goal_text: str, term: str) -> bool:
    term_text = str(term or "")
    if not term_text:
        return False
    if re.search(r"[A-Za-z0-9]", term_text):
        return term_text.lower() in goal_lower
    return term_text in goal_text


def _score_profile(goal: str, profile: dict[str, Any]) -> dict[str, Any]:
    goal_text = str(goal or "")
    goal_lower = _fold(goal_text)
    groups = list(profile.get("match_groups") or [])
    matched_groups: list[dict[str, Any]] = []
    for idx, terms in enumerate(groups):
        hits = [str(term) for term in terms if _contains_term(goal_lower, goal_text, str(term))]
        if hits:
            matched_groups.append({"group_index": idx, "hits": hits[:5]})
    if not groups:
        confidence = 0.0
    else:
        coverage = len(matched_groups) / len(groups)
        min_groups = max(1, int(profile.get("min_groups") or 1))
        floor = min(1.0, len(matched_groups) / min_groups)
        confidence = round(0.25 * coverage + 0.75 * floor, 3)
        if len(matched_groups) < min_groups:
            confidence = round(confidence * 0.55, 3)
    return {
        "confidence": min(confidence, 0.98),
        "matched_groups": matched_groups,
        "n_matched_groups": len(matched_groups),
        "n_groups": len(groups),
    }


def _profile_public_payload(profile_id: str, profile: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(profile)
    payload["trajectory_id"] = profile_id
    payload["task_family"] = profile_id
    payload["profile_version"] = PROFILE_VERSION
    payload.pop("match_groups", None)
    payload.pop("min_groups", None)
    payload.pop("priority", None)
    return payload


def _profile_from_llm(goal: str, available_skills: list[str], llm: Any) -> dict[str, Any] | None:
    if llm is None:
        return None
    prompt = (
        "Match this BioBank Agent request to one trajectory profile. "
        "Return JSON only with keys: trajectory_id, confidence, rationale. "
        f"Allowed trajectory_id values: {', '.join(TRAJECTORY_PROFILES)} or general.\n\n"
        f"Goal: {goal}\n"
        f"Available skills: {', '.join(available_skills[:120])}"
    )
    try:
        response = llm.chat(
            messages=[
                {"role": "system", "content": "You match scientific workflow requests to trajectory profiles. JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=384,
        )
        text = str(getattr(response, "text", "")).strip()
        if "```" in text:
            for part in text.split("```"):
                clean = part.strip().removeprefix("json").strip()
                if clean.startswith("{"):
                    text = clean
                    break
        payload = json.loads(text)
        if not isinstance(payload, dict):
            return None
        profile_id = str(payload.get("trajectory_id") or payload.get("task_family") or "").strip()
        if profile_id not in TRAJECTORY_PROFILES:
            return None
        confidence = max(0.0, min(1.0, float(payload.get("confidence") or 0.0)))
        return {
            "trajectory_id": profile_id,
            "confidence": confidence,
            "source": "llm",
            "rationale": str(payload.get("rationale") or ""),
        }
    except Exception:
        return None


def _best_offline_profile(goal: str) -> dict[str, Any]:
    scored: list[tuple[float, int, str, dict[str, Any]]] = []
    for profile_id, profile in TRAJECTORY_PROFILES.items():
        score = _score_profile(goal, profile)
        scored.append((float(score["confidence"]), int(profile.get("priority") or 0), profile_id, score))
    scored.sort(reverse=True)
    by_id = {profile_id: (confidence, priority, score) for confidence, priority, profile_id, score in scored}
    # If a request explicitly names VCF/WGS plus vitiligo/白癜风, route to the
    # WGS case-control profile even when generic VirtualCell wording is also
    # present. This is a profile-level conflict policy, not planner-local
    # prompt surgery.
    wgs = by_id.get("wgs_vitiligo_case_control")
    multimodal = by_id.get("virtualcell_multimodal")
    mechanism = by_id.get("juvenile_hair_multiomics_mechanism")
    if (
        wgs
        and multimodal
        and wgs[0] >= 0.80
        and multimodal[0] >= wgs[0]
        and not (mechanism and mechanism[0] >= 0.90)
    ):
        scored = [
            item for item in scored
            if item[2] != "wgs_vitiligo_case_control"
        ]
        scored.insert(0, (min(0.99, wgs[0] + 0.08), wgs[1], "wgs_vitiligo_case_control", wgs[2]))
    confidence, _priority, profile_id, score = scored[0]
    if confidence < 0.42:
        return {
            "trajectory_id": "general",
            "task_family": "general",
            "title": "Biobank Analysis Plan",
            "confidence": confidence,
            "source": "trajectory_profile_offline",
            "profile_version": PROFILE_VERSION,
            "matched_groups": score.get("matched_groups", []),
            "modalities": [],
            "concepts": [],
            "required_skills": [],
            "missing_skills": [],
            "clarification_questions": [],
            "artifact_contract": {},
            "report_terms": [],
            "mcp_resource_needs": [],
            "workflow_hooks": [],
            "rationale": "No trajectory profile exceeded the offline matching threshold.",
        }
    payload = _profile_public_payload(profile_id, TRAJECTORY_PROFILES[profile_id])
    payload.update({
        "confidence": confidence,
        "source": "trajectory_profile_offline",
        "matched_groups": score.get("matched_groups", []),
        "rationale": "Matched by profile groups rather than planner-local keyword branches.",
    })
    return payload


def match_trajectory_profile(
    goal: str,
    *,
    available_skills: list[str] | None = None,
    llm: Any = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    """Return a structured trajectory profile for a natural-language goal."""
    available = [str(item) for item in (available_skills or []) if str(item)]
    offline = _best_offline_profile(goal)
    llm_match = _profile_from_llm(goal, available, llm) if use_llm else None
    chosen = offline
    if llm_match and llm_match["trajectory_id"] in TRAJECTORY_PROFILES:
        llm_profile = _profile_public_payload(llm_match["trajectory_id"], TRAJECTORY_PROFILES[llm_match["trajectory_id"]])
        llm_profile.update(llm_match)
        if (
            llm_match["confidence"] >= max(0.55, float(offline.get("confidence") or 0.0) - 0.1)
            or offline.get("trajectory_id") == "general"
        ):
            chosen = llm_profile
        else:
            chosen = dict(offline)
            chosen["source"] = "trajectory_profile_reconciled"
            chosen["rationale"] = (
                "Offline profile evidence was stronger than the LLM match; "
                f"LLM suggested {llm_match['trajectory_id']} at confidence {llm_match['confidence']:.2f}."
            )
    missing = sorted(set(chosen.get("required_skills") or []) - set(available)) if available else []
    chosen["missing_skills"] = missing
    chosen["available_skill_count"] = len(available)
    return chosen


@skill(
    name="trajectory_profile_match",
    description=(
        "Match a short natural-language BioBank Agent request to a structured trajectory profile "
        "with required skills, clarification questions, artifact/report contracts, MCP resource needs "
        "and workflow-evolution hooks. Use this before planner/harness routing."
    ),
    parameters={
        "goal": {"type": "string", "description": "User goal or task prompt"},
        "available_skills": {"type": "string", "description": "Comma-separated available skill names", "default": ""},
        "use_llm": {"type": "boolean", "description": "Ask ctx.llm to reconcile ambiguous profile matching", "default": False},
    },
    required=["goal"],
)
def trajectory_profile_match(
    goal: str,
    available_skills: str = "",
    use_llm: bool = False,
    *,
    ctx=None,
) -> dict:
    skills = [item.strip() for item in str(available_skills or "").split(",") if item.strip()]
    llm = getattr(ctx, "llm", None) if ctx is not None else None
    result = match_trajectory_profile(goal, available_skills=skills, llm=llm, use_llm=bool(use_llm))
    if ctx is not None and hasattr(ctx, "state"):
        try:
            ctx.state.custom_data["trajectory_profile_match"] = result
        except Exception:
            pass
    return result
