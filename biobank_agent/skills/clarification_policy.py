"""Clarification policy skill for interactive planning and CLI selection."""

from __future__ import annotations

from typing import Any

from biobank_agent.registry import skill
from biobank_agent.skills.goal_intent_classifier import classify_goal_intent
from biobank_agent.skills.trajectory_profile import match_trajectory_profile


def _normalize_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for opt in options or []:
        if not isinstance(opt, dict):
            continue
        label = str(opt.get("label") or "").strip()
        if not label:
            continue
        normalized.append({
            "label": label,
            "value": str(opt.get("value") or opt.get("description") or label),
            "description": str(opt.get("description") or ""),
            "default_checked": bool(opt.get("default_checked", False)),
        })
    return normalized


def _question(
    qid: str,
    header: str,
    question: str,
    options: list[dict[str, Any]],
    *,
    multi: bool = False,
    source: str = "trajectory_profile",
) -> dict[str, Any]:
    return {
        "id": qid,
        "header": header,
        "question": question,
        "options": _normalize_options(options),
        "multi": multi,
        "source": source,
    }


def build_clarification_questions(goal: str, *, available_skills: list[str] | None = None) -> list[dict[str, Any]]:
    profile = match_trajectory_profile(goal, available_skills=available_skills, use_llm=False)
    intent = classify_goal_intent(goal, available_skills=available_skills)
    questions: list[dict[str, Any]] = []

    if profile.get("trajectory_id") == "juvenile_hair_multiomics_mechanism":
        questions.extend(list(profile.get("clarification_questions") or []))
        if intent.get("source") in {"clarification_policy_profile", "llm_fallback_ensemble", "llm"}:
            questions.append(
                _question(
                    "jh_prompt_scope",
                    "Scope",
                    "Should the agent treat the prompt as a multi-omics mechanism task even if the wording is short or mixed?",
                    [
                        {
                            "label": "Mechanism First",
                            "value": "Treat this as a Juvenile hair-whitening multi-omics mechanism analysis and use registered skills to infer missing sub-steps.",
                            "description": "Best when the user gives a compact but clearly mechanistic request.",
                        },
                        {
                            "label": "Ask More",
                            "value": "Pause for a stricter clarification before inferring the downstream multi-omics workflow.",
                            "description": "Use this when the wording is too vague to infer the mechanism chain.",
                        },
                    ],
                    source="clarification_policy",
                )
            )
        return questions[:3]

    if profile.get("trajectory_id") == "virtualcell_multimodal":
        questions.extend(list(profile.get("clarification_questions") or []))
        return questions[:3]

    if profile.get("trajectory_id") == "wgs_vitiligo_case_control":
        questions.extend(list(profile.get("clarification_questions") or []))
        return questions[:3]

    if intent.get("task_family") == "juvenile_hair_multiomics_mechanism":
        questions.extend(
            [
                _question(
                    "jh_primary_contrast",
                    "Contrast",
                    "Which comparison should anchor the analysis?",
                    [
                        {
                            "label": "J vs V + W/B",
                            "value": "Use Juvenile_White versus Vitiligo_White for WGS candidate variants, then use Juvenile donor white-vs-black BWhair samples for ATAC/RNA/spatial follow-up.",
                            "description": "Best default for the provided cohort layout.",
                        },
                        {
                            "label": "J vs S",
                            "value": "Use Juvenile_White versus Senile_White for WGS context and still follow white-vs-black hair states where matched samples exist.",
                            "description": "Focus on early versus age-associated whitening.",
                        },
                    ],
                ),
                _question(
                    "jh_execution_depth",
                    "Depth",
                    "How deeply should large matrices be read?",
                    [
                        {
                            "label": "Backed + Fallback",
                            "value": "Open h5ad files in backed read-only mode, inspect metadata and emit explicit fallbacks for matrix-level ATAC/RNA/spatial analyses.",
                            "description": "Recommended for robust CLI and repeated harness testing.",
                        },
                        {
                            "label": "Chunked Matrix",
                            "value": "Attempt chunked matrix extraction for selected candidate genes/peaks when dependencies and memory allow, otherwise fall back.",
                            "description": "More complete but slower and dependency-sensitive.",
                        },
                    ],
                ),
                _question(
                    "jh_external_resources",
                    "Resources",
                    "How should motif databases, genome annotations and literature tools be used?",
                    [
                        {
                            "label": "Local First",
                            "value": "Use local/embedded resources first, call MCP or web-backed resources only when configured, and record unavailable databases as workflow gaps.",
                            "description": "Most reproducible default.",
                        },
                        {
                            "label": "Use MCP/Web",
                            "value": "Prefer configured MCP/web resources for motif databases, regulatory annotations, enrichment and literature, with local fallback.",
                            "description": "Richer when trusted external resources are available.",
                        },
                    ],
                ),
            ]
        )
        return questions[:3]

    if intent.get("task_family") == "virtualcell_multimodal":
        questions.extend(
            [
                _question(
                    "virtualcell_modality_scope",
                    "Modalities",
                    "Which VirtualCell modalities should anchor this plan?",
                    [
                        {
                            "label": "All Modalities",
                            "value": "Analyze WGS, Stereo-seq, scRNA-seq and scATAC-seq together when readable; record missing files or dependencies as structured fallbacks.",
                            "description": "Best default for BWhair multimodal requests.",
                        },
                        {
                            "label": "h5ad First",
                            "value": "Prioritize Stereo-seq/scRNA/scATAC h5ad inventory and metadata before WGS analysis.",
                            "description": "Useful when the user mainly asks about single-cell/spatial data.",
                        },
                    ],
                ),
                _question(
                    "virtualcell_execution_scope",
                    "Execution",
                    "How much single-cell data should be read during CLI validation?",
                    [
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
                ),
                _question(
                    "virtualcell_grouping",
                    "Grouping",
                    "Which grouping should be primary for summaries?",
                    [
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
                ),
            ]
        )
        return questions[:3]

    return questions[:3]


def _normalize_questions(questions: list[dict[str, Any]], *, source: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in questions or []:
        if not isinstance(item, dict):
            continue
        options = _normalize_options(list(item.get("options") or []))
        if not options:
            continue
        normalized.append({
            "id": str(item.get("id") or ""),
            "header": str(item.get("header") or item.get("id") or "Plan"),
            "question": str(item.get("question") or "Clarification needed"),
            "options": options,
            "multi": bool(item.get("multi", False)),
            "source": str(item.get("source") or source),
        })
    return normalized


@skill(
    name="clarification_policy",
    description=(
        "Generate structured interactive clarification questions from a goal, "
        "trajectory profile and available skills so CLI and TUI can ask the user "
        "only the ambiguous decisions that matter."
    ),
    parameters={
        "goal": {"type": "string", "description": "User goal or task prompt"},
        "available_skills": {"type": "string", "description": "Comma-separated available skill names", "default": ""},
    },
    required=["goal"],
)
def clarification_policy(goal: str, available_skills: str = "", *, ctx=None) -> dict:
    skills = [item.strip() for item in str(available_skills or "").split(",") if item.strip()]
    questions = _normalize_questions(
        build_clarification_questions(goal, available_skills=skills),
        source="trajectory_profile",
    )
    return {
        "goal": goal,
        "questions": questions,
        "n_questions": len(questions),
        "available_skills": skills,
        "intents": classify_goal_intent(goal, available_skills=skills),
        "trajectory": match_trajectory_profile(goal, available_skills=skills, use_llm=False),
    }
