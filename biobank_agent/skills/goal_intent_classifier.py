"""Goal intent classification for agentic planner routing."""

from __future__ import annotations

import json
from typing import Any

from biobank_agent.registry import skill


MECHANISM_REQUIRED_SKILLS = {
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
}


def _normalise_profile(payload: Any, goal: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _fallback_profile(goal, source="invalid_llm_payload")
    task_family = str(payload.get("task_family") or "general").strip().lower()
    modalities = payload.get("modalities") if isinstance(payload.get("modalities"), list) else []
    concepts = payload.get("concepts") if isinstance(payload.get("concepts"), list) else []
    confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "task_family": task_family,
        "modalities": [str(x).lower() for x in modalities],
        "concepts": [str(x).lower() for x in concepts],
        "confidence": confidence,
        "source": str(payload.get("source") or "llm"),
        "rationale": str(payload.get("rationale") or ""),
    }


def _fallback_profile(
    goal: str,
    *,
    source: str = "fallback",
    available_skills: list[str] | None = None,
) -> dict[str, Any]:
    """Small semantic fallback for offline tests; keep routing knowledge here, not scattered in planner."""
    text = str(goal or "")
    lower = text.lower()
    modalities = []
    for key, probes in {
        "wgs": ("wgs", "vcf", "genome", "genomic", "基因组", "遗传变异", "变异"),
        "scatac": ("scatac", "atac", "chromatin", "accessibility", "表观组", "染色质", "开放性"),
        "scrna": ("scrna", "transcriptome", "expression", "转录组", "表达"),
        "spatial": ("spatial", "stereo", "空间组", "空间", "互作"),
    }.items():
        if any(token in lower or token in text for token in probes):
            modalities.append(key)
    concepts = []
    if any(token in lower or token in text for token in ("juvenile", "juvenile_white", "early-onset", "早发", "青少年", "年轻", "少白头")):
        concepts.append("juvenile_or_early_onset")
    if any(token in lower or token in text for token in ("hair whitening", "white hair", "hair graying", "白发", "少白头", "黑白发", "毛发变白")):
        concepts.append("hair_whitening")
    if any(token in lower or token in text for token in ("mechanism", "机制", "影响", "导致", "关系")):
        concepts.append("mechanism")
    skill_set = set(available_skills or MECHANISM_REQUIRED_SKILLS)
    has_mechanism_skills = bool(MECHANISM_REQUIRED_SKILLS & skill_set)
    multiomics_context = any(token in lower or token in text for token in ("multi-omics", "multiomics", "多组学"))
    if has_mechanism_skills and multiomics_context and {"juvenile_or_early_onset", "hair_whitening", "mechanism"}.issubset(set(concepts)):
        for modality in ("wgs", "scatac", "scrna", "spatial"):
            if modality not in modalities:
                modalities.append(modality)
    if {"wgs", "scatac", "scrna", "spatial"}.intersection(modalities) and {
        "juvenile_or_early_onset",
        "hair_whitening",
        "mechanism",
    }.issubset(set(concepts)):
        family = "juvenile_hair_multiomics_mechanism"
        confidence = 0.72
    elif {"scatac", "scrna", "spatial"}.intersection(modalities) and "hair_whitening" in concepts:
        family = "virtualcell_multimodal"
        confidence = 0.60
    else:
        family = "general"
        confidence = 0.30
    return {
        "task_family": family,
        "modalities": modalities,
        "concepts": concepts,
        "confidence": confidence,
        "source": source,
        "rationale": "Offline semantic fallback; planner should prefer LLM intent profiles when available.",
    }


def _is_mechanism_profile(profile: dict[str, Any]) -> bool:
    return (
        profile.get("task_family") == "juvenile_hair_multiomics_mechanism"
        and float(profile.get("confidence") or 0.0) >= 0.55
    )


def _reconcile_profiles(goal: str, llm_profile: dict[str, Any], fallback_profile: dict[str, Any]) -> dict[str, Any]:
    """Blend LLM and offline profiles without scattering routing rules through the planner."""
    text = str(goal or "").lower()
    if _is_mechanism_profile(llm_profile):
        return llm_profile
    if _is_mechanism_profile(fallback_profile):
        if "juvenile_hair_mechanism_policy" in text:
            merged = dict(fallback_profile)
            merged["source"] = "clarification_policy_profile"
            merged["rationale"] = (
                "Interactive planning already confirmed the Juvenile hair-whitening mechanism workflow; "
                "preserving that explicit profile over an inconsistent LLM route."
            )
            return merged
        if llm_profile.get("task_family") in {"general", "virtualcell_multimodal", ""}:
            merged = dict(fallback_profile)
            merged["source"] = "llm_fallback_ensemble"
            merged["rationale"] = (
                "Fallback semantic profile detected the full WGS/scATAC/scRNA/spatial Juvenile hair-whitening "
                "mechanism trajectory while the LLM route was broader or less specific."
            )
            return merged
    return llm_profile


def classify_goal_intent(goal: str, *, llm=None, available_skills: list[str] | None = None) -> dict[str, Any]:
    """Return a compact task profile for planner/harness routing."""
    fallback_profile = _fallback_profile(goal, available_skills=available_skills)
    if "juvenile_hair_mechanism_policy" in str(goal or "").lower() and _is_mechanism_profile(fallback_profile):
        fallback_profile = dict(fallback_profile)
        fallback_profile["source"] = "clarification_policy_profile"
        return fallback_profile
    if llm is not None:
        prompt = (
            "Classify this BioBank Agent user request for workflow routing. "
            "Return JSON only with keys: task_family, modalities, concepts, confidence, rationale. "
            "Allowed task_family values include juvenile_hair_multiomics_mechanism, virtualcell_multimodal, "
            "wgs_vitiligo_case_control, paper_replication, metabolic_trajectory, general. "
            "Prefer juvenile_hair_multiomics_mechanism when the user asks about early/juvenile hair whitening "
            "or Chinese 少白头 with genomics/epigenomics/transcriptomics/spatial mechanism analysis.\n\n"
            f"Goal: {goal}\n"
            f"Available skills include: {', '.join((available_skills or [])[:80])}"
        )
        try:
            response = llm.chat(
                messages=[
                    {"role": "system", "content": "You are a concise scientific workflow intent classifier. JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
            )
            text = str(getattr(response, "text", "")).strip()
            if "```" in text:
                for part in text.split("```"):
                    clean = part.strip().removeprefix("json").strip()
                    if clean.startswith("{"):
                        text = clean
                        break
            profile = _normalise_profile(json.loads(text), goal)
            profile["source"] = "llm"
            return _reconcile_profiles(goal, profile, fallback_profile)
        except Exception:
            pass
    return fallback_profile


@skill(
    name="goal_intent_classifier",
    description=(
        "Classify a natural-language user goal into a structured BioBank Agent workflow intent profile "
        "for planner routing and harness evaluation."
    ),
    parameters={
        "goal": {"type": "string", "description": "User goal or task prompt"},
    },
    required=["goal"],
)
def goal_intent_classifier(goal: str, *, ctx=None) -> dict:
    return classify_goal_intent(goal, available_skills=sorted(MECHANISM_REQUIRED_SKILLS))
