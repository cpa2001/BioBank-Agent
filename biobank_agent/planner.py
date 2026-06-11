"""Plan mode engine — structured planning workflow with human-in-the-loop.

Implements a full Plan → Review → Approve → Execute → Report lifecycle
with iterative refinement, progress tracking, and pause/resume support.

State machine: INACTIVE → PLANNING → REVIEW ⇄ REFINING → APPROVED → EXECUTING ⇄ PAUSED → DONE

DEPRECATED (legacy engine): the live ``biobank`` CLI does NOT use this module. It
uses the v3 ``InteractiveShell`` (``biobank_agent/cli/interactive.py``) backed by
``RuntimePlanner`` (``biobank_agent/runtime/planner.py``). ``LongHorizonPlanner`` /
``PlanMode`` / ``_specialized_default_plan`` / the WGS templates here are reached
only by the Textual TUI. Fix live-CLI planning/routing in the v3 engine, not here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMClient
    from .study_spec import StudySpec

from .skills.goal_intent_classifier import classify_goal_intent
from .skills.trajectory_profile import match_trajectory_profile
from .skills.clarification_policy import build_clarification_questions

logger = logging.getLogger(__name__)


# ── Plan state enum ──────────────────────────────────────────────


class PlanState(str, Enum):
    """Plan mode lifecycle states."""

    INACTIVE = "INACTIVE"       # No plan active
    PLANNING = "PLANNING"       # LLM is decomposing goal into steps
    REVIEW = "REVIEW"           # Plan displayed, awaiting user approval/edits
    REFINING = "REFINING"       # LLM updating plan based on user feedback
    APPROVED = "APPROVED"       # User approved, ready to execute (transient)
    EXECUTING = "EXECUTING"     # Steps being executed
    PAUSED = "PAUSED"           # Execution paused (issue or user interrupt)
    DONE = "DONE"               # All steps complete


# ── Long-horizon plan structures ──────────────────────────────


@dataclass
class PlanStep:
    """A single step in a long-horizon plan with dependencies."""

    id: str
    skill: str
    args: dict = field(default_factory=dict)
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    can_parallelize: bool = False
    status: str = "pending"   # pending | running | done | failed | skipped
    result: dict = field(default_factory=dict)
    error: str = ""
    criticality: str = "required"  # required | optional | diagnostic
    repair_hints: list[str] = field(default_factory=list)
    skip_reason: str = ""

    @property
    def is_done(self) -> bool:
        return self.status in ("done", "skipped")

    @property
    def is_blocked(self) -> bool:
        return self.status == "pending" and bool(self.depends_on)


@dataclass(frozen=True)
class PlanValidationIssue:
    """A concrete schema/dependency problem that blocks plan approval."""

    step_id: str
    skill: str
    message: str

    def format(self) -> str:
        prefix = f"{self.step_id} `{self.skill}`" if self.step_id else f"`{self.skill}`"
        return f"{prefix}: {self.message}"


@dataclass
class LongHorizonPlan:
    """A dependency-aware multi-step analysis plan.

    Steps form a DAG where each step lists its dependencies.
    The plan executor runs steps in topological order, parallelizing
    independent branches where possible.
    """

    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    base_plan_markdown: str = ""
    planning_council: list[dict] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    clarifications: list[dict] = field(default_factory=list)
    title: str = ""

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def done_steps(self) -> int:
        return sum(1 for s in self.steps if s.is_done)

    @property
    def failed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == "failed")

    def progress(self) -> float:
        """0.0 to 1.0 completion ratio."""
        return self.done_steps / self.total_steps if self.total_steps else 0.0

    def next_runnable(self) -> list[PlanStep]:
        """Return all steps whose dependencies are satisfied."""
        done_ids = {s.id for s in self.steps if s.is_done}
        return [
            s for s in self.steps
            if s.status == "pending"
            and set(s.depends_on).issubset(done_ids)
        ]

    def mark_done(self, step_id: str, result: dict) -> None:
        """Mark a step as completed with its result."""
        for s in self.steps:
            if s.id == step_id:
                s.status = "done"
                s.result = result
                return

    def mark_failed(self, step_id: str, error: str) -> None:
        """Mark a step as failed and skip dependents."""
        for s in self.steps:
            if s.id == step_id:
                s.status = "failed"
                s.error = error
                break
        # Skip all transitive dependents
        failed_ids = {step_id}
        changed = True
        while changed:
            changed = False
            for s in self.steps:
                if s.status == "pending" and set(s.depends_on) & failed_ids:
                    s.status = "skipped"
                    s.error = f"Skipped: dependency {step_id} failed"
                    failed_ids.add(s.id)
                    changed = True

    def mark_skipped(self, step_id: str, reason: str = "User skipped") -> None:
        """Mark a step as skipped by user choice."""
        for s in self.steps:
            if s.id == step_id:
                s.status = "skipped"
                s.error = reason
                s.skip_reason = reason
                return

    def completed_steps(self) -> list[PlanStep]:
        """Return all steps that are done or skipped."""
        return [s for s in self.steps if s.is_done]

    def remaining_steps(self) -> list[PlanStep]:
        """Return all steps that are not yet done."""
        return [s for s in self.steps if not s.is_done and s.status != "failed"]

    def summary(self) -> str:
        """One-line progress summary."""
        return (
            f"Plan: {self.done_steps}/{self.total_steps} done, "
            f"{self.failed_steps} failed, "
            f"{len(self.next_runnable())} ready"
        )

    def to_markdown(self) -> str:
        """Render plan as numbered markdown checklist."""
        heading = self.title or self.goal
        lines = [f"# Plan: {heading}\n"]
        if self.title and self.goal and self.goal != self.title:
            lines.extend(["## Original Goal", self.goal, ""])
        if self.assumptions:
            lines.append("## Assumptions")
            for assumption in self.assumptions:
                lines.append(f"- {assumption}")
            lines.append("")
        for i, s in enumerate(self.steps, 1):
            check = "x" if s.is_done else ("!" if s.status == "failed" else " ")
            deps = f" (after: {', '.join(s.depends_on)})" if s.depends_on else ""
            id_tag = f" `{s.id}`" if s.id else ""
            lines.append(f"{i}. [{check}]{id_tag} `{s.skill}({s.args})`{deps}")
            if s.description:
                lines.append(f"   {s.description}")
            if s.criticality != "required":
                lines.append(f"   Criticality: {s.criticality}")
            if s.error:
                lines.append(f"   **Error:** {s.error}")
            if s.skip_reason:
                lines.append(f"   **Skip reason:** {s.skip_reason}")
        lines.append(f"\n**Progress:** {self.progress():.0%}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize plan to a JSON-serializable dict."""
        return {
            "goal": self.goal,
            "base_plan_markdown": self.base_plan_markdown,
            "planning_council": self.planning_council,
            "assumptions": self.assumptions,
            "clarifications": self.clarifications,
            "title": self.title,
            "steps": [
                {
                    "id": s.id,
                    "skill": s.skill,
                    "args": s.args,
                    "description": s.description,
                    "depends_on": s.depends_on,
                    "can_parallelize": s.can_parallelize,
                    "status": s.status,
                    "result": s.result,
                    "error": s.error,
                    "criticality": s.criticality,
                    "repair_hints": s.repair_hints,
                    "skip_reason": s.skip_reason,
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LongHorizonPlan":
        """Deserialize plan from a dict."""
        steps = [
            PlanStep(
                id=s["id"],
                skill=s.get("skill", "think"),
                args=s.get("args", {}),
                description=s.get("description", ""),
                depends_on=s.get("depends_on", []),
                can_parallelize=s.get("can_parallelize", False),
                status=s.get("status", "pending"),
                result=s.get("result", {}),
                error=s.get("error", ""),
                criticality=s.get("criticality", "required"),
                repair_hints=list(s.get("repair_hints", []) or []),
                skip_reason=s.get("skip_reason", ""),
            )
            for s in data.get("steps", [])
        ]
        return cls(
            goal=data.get("goal", ""),
            steps=steps,
            base_plan_markdown=data.get("base_plan_markdown", ""),
            planning_council=list(data.get("planning_council", []) or []),
            assumptions=list(data.get("assumptions", []) or []),
            clarifications=list(data.get("clarifications", []) or []),
            title=str(data.get("title", "") or ""),
        )


def _schema_name(schema: dict) -> str:
    return str(schema.get("function", {}).get("name", "") or "")


def _schema_parameters(schema: dict) -> dict:
    return schema.get("function", {}).get("parameters", {}) or {}


def _schema_properties(schema: dict) -> dict:
    return _schema_parameters(schema).get("properties", {}) or {}


def _schema_required(schema: dict) -> list[str]:
    required = _schema_parameters(schema).get("required", []) or []
    return [str(item) for item in required]


def _schemas_by_name(tool_schemas: list[dict] | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for schema in tool_schemas or []:
        name = _schema_name(schema)
        if name:
            out[name] = schema
    return out


def _format_arg_signature(schema: dict) -> str:
    props = _schema_properties(schema)
    required = set(_schema_required(schema))
    parts = []
    for name, spec in props.items():
        if name in required:
            parts.append(name)
        elif "default" in spec:
            parts.append(f"{name}={spec.get('default')!r}")
        else:
            parts.append(f"{name}=optional")
    return f"{_schema_name(schema)}({', '.join(parts)})"


def _format_tool_schema_summary(
    available_skills: list[str],
    tool_schemas: list[dict] | None,
    *,
    limit: int = 50,
) -> str:
    """Compact tool schema summary for planner prompts."""
    schema_map = _schemas_by_name(tool_schemas)
    names = [name for name in available_skills if name]
    lines: list[str] = []

    common = ["ukb_data_inventory", "ukb_field_resolve", "field_search", "cohort_summary",
              "train_model", "smart_plot", "statistical_review", "safety_check",
              "world_model_audit", "generate_report"]
    common_lines = [
        f"- {_format_arg_signature(schema_map[name])}"
        for name in common
        if name in schema_map and name in names
    ]
    if common_lines:
        lines.append("Common exact signatures:")
        lines.extend(common_lines)

    schema_lines: list[str] = []
    for name in names[:limit]:
        schema = schema_map.get(name)
        if not schema:
            schema_lines.append(f"- {name}")
            continue
        desc = str(schema.get("function", {}).get("description", "") or "").strip()
        if len(desc) > 120:
            desc = desc[:117] + "..."
        props = _schema_properties(schema)
        required = set(_schema_required(schema))
        arg_bits = []
        for arg_name, arg_spec in props.items():
            arg_type = arg_spec.get("type", "any")
            marker = "required" if arg_name in required else f"default={arg_spec.get('default')!r}" if "default" in arg_spec else "optional"
            arg_bits.append(f"{arg_name}:{arg_type} {marker}")
        args = "; ".join(arg_bits) if arg_bits else "no args"
        schema_lines.append(f"- {name}: {desc} Args: {args}")
    if schema_lines:
        lines.append("Available tool schemas:")
        lines.extend(schema_lines)

    return "\n".join(lines)


class PlanSchemaValidator:
    """Validate plan steps against the registered skill schemas."""

    def __init__(
        self,
        available_skills: list[str] | None = None,
        tool_schemas: list[dict] | None = None,
    ) -> None:
        self.available_skills = set(available_skills or [])
        self.schemas = _schemas_by_name(tool_schemas)
        if not self.available_skills:
            self.available_skills = set(self.schemas)

    def validate(self, plan: LongHorizonPlan | None) -> list[PlanValidationIssue]:
        if not plan:
            return [PlanValidationIssue("", "", "Plan is empty.")]

        issues: list[PlanValidationIssue] = []
        seen_ids: set[str] = set()
        all_ids = {s.id for s in plan.steps}

        for step in plan.steps:
            step.criticality = _normalize_criticality(step.criticality)
            if not step.id:
                issues.append(PlanValidationIssue("", step.skill, "Step id is required."))
            elif step.id in seen_ids:
                issues.append(PlanValidationIssue(step.id, step.skill, "Duplicate step id."))
            seen_ids.add(step.id)

            if self.available_skills and step.skill not in self.available_skills:
                issues.append(PlanValidationIssue(
                    step.id,
                    step.skill,
                    "Unknown skill; choose one of the registered tools.",
                ))

            bad_deps = [dep for dep in step.depends_on if dep not in all_ids]
            if bad_deps:
                issues.append(PlanValidationIssue(
                    step.id,
                    step.skill,
                    f"Unknown dependencies: {', '.join(bad_deps)}.",
                ))
            if step.id and step.id in step.depends_on:
                issues.append(PlanValidationIssue(
                    step.id,
                    step.skill,
                    "Step cannot depend on itself.",
                ))

            if not isinstance(step.args, dict):
                issues.append(PlanValidationIssue(step.id, step.skill, "args must be a JSON object."))
                continue

            schema = self.schemas.get(step.skill)
            if not schema:
                continue

            props = _schema_properties(schema)
            allowed_args = set(props)
            supplied_args = set(step.args)
            extra_args = sorted(supplied_args - allowed_args)
            if extra_args:
                issues.append(PlanValidationIssue(
                    step.id,
                    step.skill,
                    f"Unsupported args: {', '.join(extra_args)}. Allowed args: {', '.join(sorted(allowed_args)) or 'none'}.",
                ))

            missing = [arg for arg in _schema_required(schema) if arg not in step.args]
            if missing:
                issues.append(PlanValidationIssue(
                    step.id,
                    step.skill,
                    f"Missing required args: {', '.join(missing)}.",
                ))

        for cycle in _dependency_cycles(plan.steps):
            if cycle:
                first = cycle[0]
                skill = next((s.skill for s in plan.steps if s.id == first), "")
                issues.append(PlanValidationIssue(
                    first,
                    skill,
                    f"Dependency cycle detected: {' -> '.join(cycle)}.",
                ))

        return issues


def _normalize_criticality(value: Any) -> str:
    """Normalize LLM criticality labels into the executor's strict enum."""
    text = str(value or "required").strip().lower().replace("-", "_")
    if text in {"required", "optional", "diagnostic"}:
        return text
    if text in {"nice_to_have", "nice", "exploratory"}:
        return "optional"
    # Labels such as high/medium/low are importance scores, not skip policy.
    return "required"


def _dependency_map(steps: list[PlanStep]) -> dict[str, list[str]]:
    ids = {step.id for step in steps}
    return {
        step.id: [dep for dep in step.depends_on if dep in ids and dep != step.id]
        for step in steps
        if step.id
    }


def _has_dependency_path(steps: list[PlanStep], start: str, target: str) -> bool:
    """Return True if ``start`` already depends on ``target`` transitively."""
    graph = _dependency_map(steps)
    seen: set[str] = set()
    stack = list(graph.get(start, []))
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.get(node, []))
    return False


def _dependency_cycles(steps: list[PlanStep]) -> list[list[str]]:
    """Detect cycles in the plan dependency graph."""
    graph = _dependency_map(steps)
    cycles: list[list[str]] = []
    visiting: list[str] = []
    visited: set[str] = set()
    emitted: set[tuple[str, ...]] = set()

    def canonical(cycle: list[str]) -> tuple[str, ...]:
        body = cycle[:-1]
        if not body:
            return tuple(cycle)
        rotations = [tuple(body[i:] + body[:i]) for i in range(len(body))]
        return min(rotations)

    def dfs(node: str) -> None:
        if node in visiting:
            idx = visiting.index(node)
            cycle = visiting[idx:] + [node]
            key = canonical(cycle)
            if key not in emitted:
                emitted.add(key)
                cycles.append(cycle)
            return
        if node in visited:
            return
        visiting.append(node)
        for dep in graph.get(node, []):
            dfs(dep)
        visiting.pop()
        visited.add(node)

    for node in graph:
        dfs(node)
    return cycles


class LongHorizonPlanner:
    """Decompose complex goals into dependency-aware step graphs.

    Uses LLM to analyze a goal and produce a LongHorizonPlan with
    skill-level steps and dependency ordering.

    Usage::

        planner = LongHorizonPlanner(llm)
        plan = planner.decompose("Discover T2DM biomarkers", available_skills)
        while plan.next_runnable():
            for step in plan.next_runnable():
                result = registry.execute(step.skill, step.args, ctx)
                plan.mark_done(step.id, result)
    """

    def __init__(self, llm: Optional["LLMClient"] = None, tool_schemas: list[dict] | None = None) -> None:
        self.llm = llm
        self.tool_schemas = tool_schemas or []

    @staticmethod
    def _is_metabolic_showcase_goal(goal: str) -> bool:
        """Detect the broad UKB metabolic/T2D/biomarker showcase question."""
        lower = str(goal or "").lower()
        data_context = any(token in lower for token in ("ukb", "biobank", "biobank data", "cohort data"))
        return (
            data_context
            and any(token in lower for token in ("metabolic health", "cardiometabolic", "cardio-metabolic"))
            and any(token in lower for token in ("type 2 diabetes", "t2d", "t2dm", "diabetes risk"))
            and any(token in lower for token in ("biomarker", "biomarkers", "actionable"))
            and any(token in lower for token in ("trajectory", "trajectories", "progression", "risk prediction", "prediction"))
        )

    @staticmethod
    def _is_wgs_vitiligo_goal(goal: str) -> bool:
        """Detect the VirtualCell WGS vitiligo case-control task family."""
        lower = str(goal or "").lower()
        text = str(goal or "")
        has_wgs_data = any(
            token in lower
            for token in (
                "wgs",
                "vcf",
                "genotyper.vcf",
                "wgs_sample_info",
                "virtualcell_wgs_vcf",
                "bw_wgs_vcf",
                "全基因组",
                "基因组",
                "变异",
            )
        )
        has_grouping = (
            "juvenile_white" in lower
            and "vitiligo_white" in lower
        ) or (
            "青少年白癜风" in text
            and ("白癜风相关" in goal or "vitiligo" in lower)
        ) or (
            "白癜风" in text
            and any(token in text for token in ("青少年", "年轻", "juvenile"))
        ) or (
            "白癜风" in text
            and "Clarifications:" in text
            and "Juvenile_White" in text
            and "Vitiligo_White" in text
        )
        has_analysis_intent = any(
            token in lower
            for token in (
                "association",
                "case-control",
                "case control",
                "pca",
                "annotation",
                "qc",
                "burden",
                "enrichment",
                "plink",
                "snpeff",
                "vep",
                "关联",
                "注释",
                "质控",
                "富集",
                "群体遗传",
            )
        ) or any(token in text for token in ("分析", "比较", "差异", "遗传", "研究"))
        vitiligo_context = "vitiligo" in lower or "白癜风" in text
        return has_wgs_data and vitiligo_context and (has_grouping or has_analysis_intent)

    @staticmethod
    def _is_virtualcell_multimodal_goal(goal: str) -> bool:
        """Detect VirtualCell/BWhair multi-omics h5ad analysis requests."""
        lower = str(goal or "").lower()
        text = str(goal or "")
        data_context = any(
            token in lower
            for token in (
                "virtualcell",
                "bwhair",
                "bw hair",
                "h5ad",
                "stereo",
                "stereo-seq",
                "scrna",
                "scatac",
                "single-cell",
                "single cell",
                "spatial",
                "multi-omics",
                "multiomics",
                "multimodal",
                "多组学",
                "多模态",
                "单细胞",
                "空间",
                "黑白发",
                "白发",
                "黑发",
            )
        )
        modality_context = any(
            token in lower
            for token in (
                "h5ad",
                "stereo",
                "scrna",
                "scatac",
                "spatial",
                "multi-omics",
                "multimodal",
                "single-cell",
                "single cell",
            )
        ) or any(token in text for token in ("多组学", "多模态", "单细胞", "空间", "黑白发"))
        analysis_intent = any(
            token in lower
            for token in ("analy", "summary", "inventory", "link", "integrat", "compare", "workflow", "report")
        ) or any(token in text for token in ("分析", "比较", "整合", "读取", "总结", "报告", "关联"))
        return data_context and modality_context and analysis_intent

    def _goal_intent_profile(self, goal: str, available_skills: list[str] | None = None) -> dict[str, Any]:
        # Planner routing must not consume the decomposition LLM call. The
        # executable plan can still invoke ``goal_intent_classifier`` as a
        # first-class skill, and harnesses can enable LLM-backed matching there.
        return classify_goal_intent(goal, available_skills=available_skills)

    def _trajectory_profile(self, goal: str, available_skills: list[str] | None = None) -> dict[str, Any]:
        return match_trajectory_profile(goal, available_skills=available_skills, use_llm=False)

    def _is_juvenile_hair_mechanism_goal(self, goal: str, available_skills: list[str] | None = None) -> bool:
        """Detect Juvenile hair-whitening mechanism tasks through trajectory profiles."""
        profile = self._trajectory_profile(goal, available_skills)
        if profile.get("trajectory_id") == "juvenile_hair_multiomics_mechanism":
            return float(profile.get("confidence") or 0.0) >= 0.55
        profile = self._goal_intent_profile(goal, available_skills)
        return (
            profile.get("task_family") == "juvenile_hair_multiomics_mechanism"
            and float(profile.get("confidence") or 0.0) >= 0.55
        )

    @staticmethod
    def _default_title_for_goal(goal: str) -> str:
        """Generate a concise human-facing plan title without requiring user input."""
        lower = str(goal or "").lower()
        profile = match_trajectory_profile(goal)
        if profile.get("trajectory_id") != "general" and profile.get("title"):
            return str(profile["title"])
        if LongHorizonPlanner._is_virtualcell_multimodal_goal(goal):
            return "VirtualCell BWhair Multimodal Analysis"
        if LongHorizonPlanner._is_wgs_vitiligo_goal(goal):
            return "VirtualCell WGS Vitiligo Case-Control Analysis"
        if LongHorizonPlanner._is_metabolic_showcase_goal(goal):
            return "UKB Metabolic Health Trajectory Analysis"
        if "paper" in lower or ".pdf" in lower or "论文" in str(goal or ""):
            return "UKB Paper Replication Plan"
        if "e11" in lower or "type 2 diabetes" in lower or "t2d" in lower:
            return "UKB Type 2 Diabetes Analysis"
        cleaned = " ".join(str(goal or "Biobank analysis").split())
        cleaned = cleaned[:80].strip(" -:;")
        return cleaned or "Biobank Analysis Plan"

    def decompose(
        self,
        goal: str,
        available_skills: list[str],
        context: str = "",
        spec: Optional["StudySpec"] = None,
        tool_schemas: list[dict] | None = None,
    ) -> LongHorizonPlan:
        """Use LLM to decompose a goal into a step graph.

        Falls back to a sensible default plan if LLM fails.
        After decomposition, checks temporal safety rules.

        Args:
            goal: The research goal to decompose
            available_skills: List of available skill names
            context: Additional context string
            spec: Optional StudySpec object to constrain planning.
                  If provided, filters skills to spec.modalities and
                  enforces spec.tool_budget as maximum steps.
        """
        original_available_skills = list(available_skills or [])
        pre_spec_template = self._specialized_default_plan(goal, original_available_skills)
        if pre_spec_template is not None:
            return self._check_temporal_safety(
                self._enforce_report_contract(pre_spec_template, original_available_skills)
            )

        # Apply StudySpec constraints if provided
        if spec is not None:
            try:
                available_skills = spec.constrain_skills(available_skills)
                tool_budget = getattr(spec, "tool_budget", 20)
            except (AttributeError, TypeError):
                tool_budget = 20
        else:
            tool_budget = 50  # Default budget without spec

        tool_schemas = tool_schemas if tool_schemas is not None else self.tool_schemas
        schema_map = _schemas_by_name(tool_schemas)
        tool_schemas = [schema_map[name] for name in available_skills if name in schema_map]

        template_plan = self._specialized_default_plan(goal, available_skills)
        if template_plan is not None:
            return self._check_temporal_safety(self._enforce_report_contract(template_plan, available_skills))

        if not self.llm:
            return self._check_temporal_safety(self._enforce_report_contract(self._default_plan(goal, available_skills), available_skills))

        schema_summary = _format_tool_schema_summary(available_skills, tool_schemas)
        prompt = f"""Decompose this biobank research goal into concrete analysis steps.

**Goal:** {goal}
{f"**Context:** {context[:500]}" if context else ""}

**Available tools:** {', '.join(available_skills[:50])}

Use only the exact tool names and argument names below. Do not invent aliases,
extra keys, or report sections arguments.

{schema_summary}

For predictive modelling, prefer `train_model` with `model_type: "auto"` unless
the user explicitly requests a specific model family. Auto mode records the
candidate models considered and the selection rationale for downstream review.

Output a JSON array where each step has:
- "id": unique step ID (e.g., "s1", "s2")
- "skill": tool name from the list above
- "args": dict of arguments
	- "description": what this step does (clear, concise, human-readable)
	- "depends_on": array of step IDs that must complete first
	- "can_parallelize": true if this can run alongside other ready steps
	- "criticality": "required" for scientific/report gates, "optional" or
	  "diagnostic" only for steps that may be explicitly skipped without
	  invalidating the final answer

Example:
[
  {{"id": "s1", "skill": "prevalence", "args": {{"top_n": 10}}, "description": "Check disease prevalence", "depends_on": [], "can_parallelize": false}},
  {{"id": "s2", "skill": "train_model", "args": {{"icd10_code": "E11", "model_type": "auto"}}, "description": "Train predictor with automatic model selection", "depends_on": ["s1"], "can_parallelize": false}}
]

Respond with ONLY the JSON array."""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "You are a biobank research planner. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
            )

            text = response.text.strip()
            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    clean = part.strip().removeprefix("json").strip()
                    if clean.startswith("["):
                        text = clean
                        break

            data = json.loads(text)
            if not isinstance(data, list):
                return self._check_temporal_safety(
                    self._enforce_report_contract(self._default_plan(goal, available_skills), available_skills)
                )

            steps = []
            for item in data:
                steps.append(PlanStep(
                    id=item.get("id", f"s{len(steps)+1}"),
                    skill=item.get("skill", "think"),
                    args=item.get("args", {}),
	                    description=item.get("description", ""),
	                    depends_on=item.get("depends_on", []),
	                    can_parallelize=item.get("can_parallelize", False),
	                    criticality=item.get("criticality", "required"),
	                    repair_hints=list(item.get("repair_hints", []) or []),
	                ))

            # Enforce tool_budget from StudySpec
            if len(steps) > tool_budget:
                logger.warning(
                    "Plan has %d steps but tool_budget is %d; truncating.",
                    len(steps), tool_budget
                )
                steps = steps[:tool_budget]

            plan = LongHorizonPlan(goal=goal, steps=steps, title=self._default_title_for_goal(goal))
            return self._check_temporal_safety(self._enforce_report_contract(plan, available_skills))

        except Exception as e:
            logger.warning("LLM plan decomposition failed: %s. Using default.", e)
            return self._check_temporal_safety(self._enforce_report_contract(self._default_plan(goal, available_skills), available_skills))

    def refine(
        self,
        plan: LongHorizonPlan,
        feedback: str,
        available_skills: list[str] | None = None,
        tool_schemas: list[dict] | None = None,
        spec: Optional["StudySpec"] = None,
    ) -> LongHorizonPlan:
        """Re-plan incorporating user feedback while preserving completed steps.

        Args:
            plan: Current plan (may have completed steps)
            feedback: User's natural language modification request
            available_skills: Available skill names for the revised plan

        Returns:
            Updated LongHorizonPlan with modifications applied
        """
        if not self.llm:
            # Without LLM, we can't refine — return plan unchanged
            return plan

        completed = [s for s in plan.steps if s.is_done]
        remaining = [s for s in plan.steps if not s.is_done and s.status != "failed"]

        available_skills = available_skills or []
        tool_budget = 50
        if spec is not None:
            try:
                available_skills = spec.constrain_skills(available_skills)
                tool_budget = getattr(spec, "tool_budget", tool_budget)
            except (AttributeError, TypeError):
                pass
        tool_schemas = tool_schemas if tool_schemas is not None else self.tool_schemas
        schema_map = _schemas_by_name(tool_schemas)
        tool_schemas = [schema_map[name] for name in available_skills if name in schema_map]
        skills_str = ", ".join(available_skills[:50]) if available_skills else "think, prevalence, train_model, correlation, cohort, plot"
        schema_summary = _format_tool_schema_summary(available_skills, tool_schemas)

        prompt = f"""Revise this research plan based on user feedback.

**Original goal:** {plan.goal}

**Completed steps (DO NOT modify these):**
{json.dumps([{"id": s.id, "skill": s.skill, "description": s.description, "status": s.status} for s in completed], indent=2) if completed else "None yet"}

**Remaining steps (modify/replace/reorder as needed):**
{json.dumps([{"id": s.id, "skill": s.skill, "args": s.args, "description": s.description, "depends_on": s.depends_on} for s in remaining], indent=2)}

**User feedback:** {feedback}

**Available tools:** {skills_str}

Use only these exact tool names and argument schemas. Do not invent aliases,
extra keys, or report sections arguments.

{schema_summary}

For predictive modelling, prefer `train_model` with `model_type: "auto"` unless
the user explicitly asks for a specific model family.

Output a JSON array of the REVISED remaining steps only (completed steps are kept automatically).
	Each step needs: "id", "skill", "args", "description", "depends_on", "can_parallelize".
	Include "criticality" and "repair_hints" when useful.
	Use new sequential IDs starting after the last completed step.

Respond with ONLY the JSON array."""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "You are a biobank research planner. Revise the plan per user feedback. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
            )

            text = response.text.strip()
            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    clean = part.strip().removeprefix("json").strip()
                    if clean.startswith("["):
                        text = clean
                        break

            data = json.loads(text)
            if not isinstance(data, list):
                return plan  # Failed to parse, return unchanged

            new_steps = []
            for item in data:
                new_steps.append(PlanStep(
                    id=item.get("id", f"s{len(completed) + len(new_steps) + 1}"),
                    skill=item.get("skill", "think"),
                    args=item.get("args", {}),
	                    description=item.get("description", ""),
	                    depends_on=item.get("depends_on", []),
	                    can_parallelize=item.get("can_parallelize", False),
	                    criticality=item.get("criticality", "required"),
	                    repair_hints=list(item.get("repair_hints", []) or []),
	                ))

            if len(completed) + len(new_steps) > tool_budget:
                new_steps = new_steps[: max(0, tool_budget - len(completed))]

            # Merge: completed steps + revised remaining steps
            revised_plan = LongHorizonPlan(
                goal=plan.goal,
                steps=completed + new_steps,
            )
            return self._check_temporal_safety(self._enforce_report_contract(revised_plan, available_skills))

        except Exception as e:
            logger.warning("Plan refinement failed: %s. Keeping original.", e)
            return plan

    def _default_plan(self, goal: str, available_skills: list[str] | None = None) -> LongHorizonPlan:
        """Sensible default plan for common biobank analyses."""
        specialized = self._specialized_default_plan(goal, available_skills)
        if specialized is not None:
            return specialized
        return LongHorizonPlan(
            goal=goal,
            steps=[
                PlanStep(id="s1", skill="think", args={"reasoning": f"Planning: {goal}"}, description="Analyze the goal"),
                PlanStep(id="s2", skill="prevalence", args={"top_n": 10}, description="Check disease prevalence", depends_on=["s1"]),
            ],
        )

    def _specialized_default_plan(
        self,
        goal: str,
        available_skills: list[str] | None = None,
    ) -> LongHorizonPlan | None:
        """Return a deterministic workflow template for high-value task classes."""
        lower = goal.lower()
        available = set(available_skills or [])

        def has_required(required: set[str]) -> bool:
            return not available or required.issubset(available)

        paper_required = {
            "read_paper",
            "deep_research",
            "field_search",
            "cohort_summary",
            "missing_data",
            "train_model",
            "evaluate_model",
            "calibration",
            "feature_importance",
            "paper_replication_compare",
            "statistical_review",
            "safety_check",
            "world_model_audit",
            "generate_report",
        }
        discovery_required = paper_required - {"read_paper"}
        invalid_request_required = discovery_required | {"critical_thinking", "think"}
        trajectory_required = {
            "deep_research",
            "field_search",
            "cohort_summary",
            "cohort_card",
            "trajectory_tokenize",
            "missing_data",
            "train_model",
            "evaluate_model",
            "calibration",
            "feature_importance",
            "statistical_review",
            "safety_check",
            "world_model_audit",
            "generate_report",
        }
        trajectory_optional = {"project_doc", "ukb_data_inventory", "ukb_field_resolve", "bank_data_probe", "smart_plot"}
        bank_readiness_required = {"bank_data_readiness"}
        concise_e11_required = {
            "field_search",
            "cohort_summary",
            "statistical_review",
            "safety_check",
            "world_model_audit",
            "generate_report",
        }
        wgs_required = {
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
        }
        multimodal_required = {
            "virtualcell_data_inventory",
            "virtualcell_multimodal_link",
            "h5ad_sample_summary",
            "spatial_hair_summary",
            "singlecell_modality_summary",
            "statistical_review",
            "safety_check",
            "world_model_audit",
            "generate_report",
        }
        juvenile_mechanism_required = multimodal_required | {
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
        }
        intent_profile = self._goal_intent_profile(goal, list(available) if available else None)
        has_pdf = ".pdf" in lower
        paper_negated = any(
            phrase in lower
            for phrase in (
                "without using a supplied paper",
                "without a supplied paper",
                "without using a paper",
                "without paper",
                "no supplied paper",
                "no paper",
                "not using a paper",
            )
        )
        supplied_paper = any(
            phrase in lower
            for phrase in (
                "paper pdf",
                "supplied paper",
                "this paper",
                "given paper",
                "giving you this paper",
                "read the pdf",
            )
        )
        replication_intent = any(token in lower for token in ("replicate", "reproduce", "replication", "reproduction")) or "复现" in goal
        paper_subject = "paper" in lower or "论文" in goal
        invalid_or_unsafe = any(
            token in lower
            for token in (
                "causal proof",
                "prove caus",
                "patient-level id",
                "patient level id",
                "patient ids",
                "patient identifiers",
            )
        )
        if has_required(invalid_request_required) and invalid_or_unsafe:
            return self._invalid_request_default_plan(goal)
        trajectory_profile = self._trajectory_profile(goal, list(available) if available else None)
        if trajectory_profile.get("trajectory_id") == "juvenile_hair_multiomics_mechanism":
            return self._juvenile_hair_mechanism_default_plan(goal, available_skills)
        if (
            intent_profile.get("task_family") == "juvenile_hair_multiomics_mechanism"
            and float(intent_profile.get("confidence") or 0.0) >= 0.55
        ):
            return self._juvenile_hair_mechanism_default_plan(goal, available_skills)
        if has_required(multimodal_required) and (
            trajectory_profile.get("trajectory_id") == "virtualcell_multimodal"
            or self._is_virtualcell_multimodal_goal(goal)
        ):
            return self._virtualcell_multimodal_default_plan(goal, available_skills)
        if has_required(wgs_required) and (
            trajectory_profile.get("trajectory_id") == "wgs_vitiligo_case_control"
            or self._is_wgs_vitiligo_goal(goal)
        ):
            return self._wgs_vitiligo_default_plan(goal, available_skills)
        if has_required(paper_required) and (
            has_pdf or (not paper_negated and (supplied_paper or (replication_intent and paper_subject)))
        ):
            return self._paper_replication_default_plan(goal, available_skills)
        trajectory_intent = any(
            token in lower
            for token in (
                "trajectory",
                "trajectories",
                "longitudinal",
                "healthformer",
                "forecast over time",
                "over time",
                "time-ordered",
                "time ordered",
            )
        )
        metabolic_showcase_intent = self._is_metabolic_showcase_goal(goal)
        ports_explicitly_deferred = any(
            token in lower
            for token in (
                "future ports only",
                "future port",
                "future ports",
                "ukb-only",
                "ukb only",
                "out of scope",
            )
        ) and any(token in lower for token in ("hpp", "ckb", "rap"))
        if has_required(trajectory_required) and (trajectory_intent or metabolic_showcase_intent):
            allow_tabular_fallback = not any(
                phrase in lower
                for phrase in (
                    "do not train a tabular fallback",
                    "trajectory-only",
                    "trajectory only",
                    "without tabular fallback",
                )
            )
            return self._trajectory_default_plan(
                goal,
                include_bank_probe=(not available or "bank_data_probe" in available),
                include_project_doc=(not available or "project_doc" in available),
                include_summary_plot=allow_tabular_fallback and (not available or "smart_plot" in available),
                include_model_branch=allow_tabular_fallback,
            )
        readiness_intent = any(
            token in lower
            for token in (
                "bank readiness",
                "data readiness",
                "readiness probe",
                "credential",
                "credentialed aggregate data path",
                "configured data path",
                "validate data path",
                "probe data path",
            )
        )
        if has_required(bank_readiness_required) and readiness_intent and not ports_explicitly_deferred:
            return self._bank_readiness_default_plan(goal)
        advanced_analysis_intent = any(
            token in lower
            for token in (
                "train",
                "model",
                "predict",
                "prediction",
                "evaluate",
                "calibration",
                "feature_importance",
                "feature importance",
                "missingness",
                "missing data",
                "trajectory",
                "trajectories",
                "longitudinal",
                "healthformer",
                "gemini",
                "deep research",
                "literature",
                "related work",
                "paper",
                "replicate",
                "reproduce",
                "custom skill",
                "readiness card",
                "planning council",
                "codex",
                "claude",
            )
        )
        concise_e11_report = (
            any(token in lower for token in ("e11", "type 2 diabetes", "t2d"))
            and any(token in lower for token in ("report", "workflow", "summarize", "summarise"))
            and not advanced_analysis_intent
            and (
                "concise" in lower
                or "field_search" in lower
                or "cohort_summary" in lower
                or any(token in lower for token in ("search field", "search ukb field", "field catalog", "field catalogue"))
                or any(token in lower for token in ("cohort count", "cohort summary", "case count", "counts"))
            )
        )
        if has_required(concise_e11_required) and concise_e11_report:
            return self._concise_e11_report_default_plan(goal)
        broad_incomplete_discovery = any(
            token in lower
            for token in (
                "intentionally incomplete",
                "choose the endpoint",
                "publication quality",
                "diabetes/obesity progression",
                "diabetes obesity progression",
            )
        )
        discovery_intent = any(
            token in lower for token in ("related work", "literature", "natural language discovery", "biomarker", "cardiometabolic")
        )
        if has_required(discovery_required) and (discovery_intent or broad_incomplete_discovery):
            plan = self._literature_discovery_default_plan(goal)
            if broad_incomplete_discovery:
                plan.steps[0].description = "Find related work and use it to choose a conservative feasible endpoint"
                plan.steps[-1].args["title"] = "UKB diabetes and obesity progression report"
                plan.steps[-1].description = "Generate paired technical and Nature-style progression reports"
            return plan
        return None

    def _virtualcell_multimodal_default_plan(
        self,
        goal: str,
        available_skills: list[str] | None = None,
    ) -> LongHorizonPlan:
        """Deterministic plan for VirtualCell/BWhair WGS+h5ad workflows."""
        available = set(available_skills or [])

        def present(skill: str) -> bool:
            return not available or skill in available

        steps: list[PlanStep] = [
            PlanStep(
                id="s1",
                skill="virtualcell_data_inventory",
                args={"modalities": "all", "inspect_h5ad": True, "max_h5ad_files": 3},
                description="Inventory WGS VCF, Stereo-seq, scRNA-seq and scATAC-seq resources and file readiness",
            ),
            PlanStep(
                id="s2",
                skill="virtualcell_multimodal_link",
                args={"include_senile": True},
                description="Link WGS donors to BWhair h5ad modalities using Donor as the cross-modal key",
                depends_on=["s1"],
            ),
            PlanStep(
                id="s3",
                skill="h5ad_sample_summary",
                args={"modality": "all", "sample_query": "", "max_files": 5, "include_obs_summary": True},
                description="Inspect representative h5ad files in backed mode and summarize obs/var metadata",
                depends_on=["s1"],
            ),
            PlanStep(
                id="s4",
                skill="spatial_hair_summary",
                args={"group_by": "hair_state"},
                description="Summarize Stereo-seq samples by black/white/gray hair state and donor coverage",
                depends_on=["s1", "s2"],
            ),
            PlanStep(
                id="s5",
                skill="singlecell_modality_summary",
                args={"modality": "all", "inspect_h5ad": False},
                description="Summarize scRNA-seq and scATAC-seq h5ad modalities with manifest-first bounded metadata policy",
                depends_on=["s1"],
            ),
        ]

        if present("wgs_environment_check"):
            steps.append(PlanStep(
                id="s6",
                skill="wgs_environment_check",
                args={},
                description="Check WGS toolchain and VCF readiness for cross-modal donor analyses",
                depends_on=["s1"],
                criticality="diagnostic",
            ))
        if present("cohort_phenotype_summary"):
            steps.append(PlanStep(
                id="s7",
                skill="cohort_phenotype_summary",
                args={"group_by": "phenotype_group"},
                description="Summarize WGS donor phenotype, age and sex distributions for cross-modal interpretation",
                depends_on=["s1"],
                criticality="diagnostic",
            ))
        if present("deep_research"):
            steps.append(PlanStep(
                id="s8",
                skill="deep_research",
                args={
                    "topic": "black white hair graying vitiligo single-cell spatial transcriptomics scATAC WGS melanocyte immune aging",
                    "max_sources": 8,
                },
                description="Collect related literature for black/white hair, vitiligo, melanocyte and immune multi-omics",
                depends_on=["s1"],
                criticality="diagnostic",
            ))

        if present("trajectory_profile_match"):
            for step in steps:
                if step.id == "s1":
                    step.depends_on = ["s0p"]
                elif "s1" in step.depends_on and "s0p" not in step.depends_on:
                    step.depends_on = ["s0p" if dep == "s1" else dep for dep in step.depends_on]
            steps.insert(0, PlanStep(
                id="s0p",
                skill="trajectory_profile_match",
                args={"goal": goal, "available_skills": ",".join(sorted(available)), "use_llm": False},
                description="Match the short request to an auditable trajectory profile with skill, artifact, report and MCP contracts",
            ))

        analysis_ids = [step.id for step in steps]
        steps.extend([
            PlanStep(
                id="s9",
                skill="statistical_review",
                args={"scope": "session"},
                description="Review multimodal linkage, h5ad metadata limits, sample size and feasible inference boundaries",
                depends_on=analysis_ids,
            ),
            PlanStep(
                id="s10",
                skill="safety_check",
                args={"scope": "session", "k": 5},
                description="Check privacy and small-cell reporting constraints for donor-linked multimodal summaries",
                depends_on=["s9"],
            ),
            PlanStep(
                id="s11",
                skill="world_model_audit",
                args={
                    "task": "VirtualCell BWhair multimodal donor-linked WGS, spatial, scRNA and scATAC workflow",
                    "simulation_type": "multimodal_association_feasibility",
                    "input_modalities": "WGS VCF, Stereo-seq h5ad, scRNA h5ad, scATAC h5ad, phenotype manifest",
                    "available_tokens": 28,
                    "training_distribution_coverage": 0.2,
                    "calibration_status": "unknown",
                    "external_validation_status": "not_validated",
                },
                description="Audit claim boundaries for a small donor-linked multimodal cohort",
                depends_on=["s10"],
            ),
            PlanStep(
                id="s12",
                skill="generate_report",
                args={"title": "VirtualCell BWhair Multimodal Analysis", "format": "dual"},
                description="Generate Markdown and HTML reports with inventory, linkage, h5ad summaries, literature context and reproducibility notes",
                depends_on=["s11"],
            ),
        ])
        plan = LongHorizonPlan(goal=goal, steps=steps, title=self._default_title_for_goal(goal))
        plan.assumptions = [
            "Donor is the primary key linking WGS to Stereo-seq, scRNA-seq and scATAC-seq data.",
            "h5ad files are inspected in backed read-only mode first; full matrix operations must be chunked or downsampled.",
            "WGS phenotype metadata comes from the embedded or discovered WGS manifest; h5ad-specific metadata comes from manifest columns and .obs when files are readable.",
            "Missing h5ad files, missing anndata/scanpy, missing VCF indexes or unavailable annotation databases are recorded as degradations rather than causing the plan to crash.",
        ]
        return plan

    def _juvenile_hair_mechanism_default_plan(
        self,
        goal: str,
        available_skills: list[str] | None = None,
    ) -> LongHorizonPlan:
        """Deterministic workflow for Juvenile hair-whitening mechanism tasks."""
        available = set(available_skills or [])

        def present(skill: str) -> bool:
            return not available or skill in available

        steps: list[PlanStep] = [
            *([
                PlanStep(
                    id="s0p",
                    skill="trajectory_profile_match",
                    args={"goal": goal, "available_skills": ",".join(sorted(available)), "use_llm": False},
                    description="Match the request to a trajectory profile with skill, artifact, report and MCP contracts",
                )
            ] if present("trajectory_profile_match") else []),
            PlanStep(
                id="s0",
                skill="goal_intent_classifier",
                args={"goal": goal},
                description="Classify the short natural-language request into a structured Juvenile hair-whitening mechanism workflow intent",
                depends_on=["s0p"] if present("trajectory_profile_match") else [],
            ),
            PlanStep(
                id="s1",
                skill="virtualcell_data_inventory",
                args={"modalities": "all", "inspect_h5ad": True, "max_h5ad_files": 4},
                description="Inventory WGS, scATAC, scRNA and Stereo-seq data and verify readable files",
                depends_on=["s0"],
            ),
            PlanStep(
                id="s2",
                skill="virtualcell_multimodal_link",
                args={"include_senile": True},
                description="Build Donor-key linkage across WGS, white/black hair h5ad samples and phenotype groups",
                depends_on=["s1"],
            ),
            PlanStep(
                id="s3",
                skill="h5ad_sample_summary",
                args={"modality": "all", "sample_query": "J", "max_files": 6, "include_obs_summary": True},
                description="Inspect Juvenile donor h5ad metadata in backed mode for cell-type, peak, gene and spatial coordinate fields",
                depends_on=["s1"],
            ),
            PlanStep(
                id="s3a",
                skill="spatial_hair_summary",
                args={"group_by": "hair_state"},
                description="Summarize Stereo-seq hair-state coverage across donors before spatial mechanism analysis",
                depends_on=["s1", "s2"],
            ),
            PlanStep(
                id="s3b",
                skill="singlecell_modality_summary",
                args={"modality": "all", "inspect_h5ad": False},
                description="Summarize scRNA-seq and scATAC-seq h5ad files before expression/accessibility mechanism analysis using manifest-first bounded metadata policy",
                depends_on=["s1"],
            ),
        ]

        if present("wgs_environment_check"):
            steps.append(PlanStep(
                id="s4",
                skill="wgs_environment_check",
                args={},
                description="Check WGS/VCF and standard annotation tool readiness",
                depends_on=["s1"],
                criticality="diagnostic",
            ))
        whole_genome_requested = any(
            token in str(goal or "").lower()
            for token in (
                "whole genome",
                "full wgs",
                "full genome",
                "全基因组执行",
                "完整wgs",
                "全量wgs",
            )
        )
        if whole_genome_requested and present("vcf_annotation"):
            steps.append(PlanStep(
                id="s5",
                skill="vcf_annotation",
                args={"genes": "TYR,OCA2,SLC45A2,MC1R,HLA,NLRP1,PAX3,SOX10,MITF,TYRP1,DCT,IRF4,TNF,IFNG,CXCL10", "min_qual": 30.0, "max_variants_per_gene": 500},
                description="Annotate candidate pigmentation and immune loci for WGS variant anchors when full WGS execution is explicitly requested",
                depends_on=["s4"] if present("wgs_environment_check") else ["s1"],
                criticality="diagnostic",
            ))
        if whole_genome_requested and present("vcf_association"):
            steps.append(PlanStep(
                id="s6",
                skill="vcf_association",
                args={"case_group": "J", "control_group": "V", "maf_min": 0.01, "chromosomes": "chr22", "max_variants": 1000},
                description="Run a bounded exploratory Juvenile_White versus Vitiligo_White WGS association when full WGS execution is explicitly requested",
                depends_on=["s4"] if present("wgs_environment_check") else ["s1"],
                criticality="diagnostic",
            ))

        wgs_deps = [sid for sid in ("s5", "s6") if any(step.id == sid for step in steps)]
        if not wgs_deps:
            wgs_deps = ["s2"]
        steps.extend([
            PlanStep(
                id="s7",
                skill="jh_variant_discovery",
                args={"case_group": "J", "control_groups": "V,S", "max_variants": 30},
                description="Prioritize Juvenile hair-whitening candidate variants or curated locus anchors from WGS evidence",
                depends_on=wgs_deps,
            ),
            PlanStep(
                id="s8",
                skill="regulatory_variant_annotation",
                args={"window_bp": 100000},
                description="Classify candidate variants by regulatory, nearby-gene, immune and pigmentation context",
                depends_on=["s7"],
            ),
            PlanStep(
                id="s9",
                skill="tf_binding_disruption",
                args={"motif_database": "auto", "top_n": 30},
                description="Assess candidate TF binding disruption with motif database fallback tracking",
                depends_on=["s8"],
            ),
            PlanStep(
                id="s10",
                skill="scatac_peak_overlap",
                args={"max_files": 1},
                description="Check candidate variants against scATAC peak metadata and exact-overlap readiness",
                depends_on=["s7", "s3", "s3b"],
            ),
            PlanStep(
                id="s11",
                skill="scatac_accessibility_differential",
                args={"case_hair_state": "W", "control_hair_state": "B"},
                description="Assess Juvenile white-vs-black hair chromatin accessibility differential readiness and cell-type metadata",
                depends_on=["s2", "s3", "s3b", "s10"],
            ),
            PlanStep(
                id="s12",
                skill="scrna_expression_differential",
                args={"genes": ""},
                description="Assess scRNA expression differential readiness for variant-linked genes and cell types",
                depends_on=["s7", "s3", "s3b"],
            ),
            PlanStep(
                id="s13",
                skill="atac_expression_coupling",
                args={"top_n": 30},
                description="Integrate variant-linked accessibility and expression evidence into peak-gene mechanism rows",
                depends_on=["s10", "s11", "s12"],
            ),
            PlanStep(
                id="s14",
                skill="spatial_celltype_localization",
                args={"max_files": 4, "sample_query": "J"},
                description="Inspect Juvenile Stereo-seq cell-type and coordinate metadata for spatial localization analysis",
                depends_on=["s2", "s3", "s3a", "s13"],
            ),
            PlanStep(
                id="s15",
                skill="spatial_cell_interaction",
                args={"neighborhood_radius": 50.0},
                description="Assess readiness for spatial neighbor and cell-cell interaction shifts linked to prioritized cell states",
                depends_on=["s14"],
            ),
            PlanStep(
                id="s16",
                skill="multiomics_mechanism_prioritization",
                args={"top_n": 20},
                description="Rank variant-to-TF-to-accessibility-to-expression-to-spatial mechanism hypotheses",
                depends_on=["s9", "s13", "s15"],
            ),
            PlanStep(
                id="s17",
                skill="workflow_gap_detector",
                args={"trusted_harness_mode": False},
                description="Audit missing databases, MCP tools and generated-skill proposals for future automatic evolution",
                depends_on=["s16"],
            ),
            PlanStep(
                id="s17a",
                skill="agent_workflow_evolver",
                args={"goal_profile": "juvenile_hair_multiomics_mechanism", "trusted_harness_mode": False},
                description="Audit planner, harness and MCP evolution opportunities so repeated runs can migrate brittle rules into skills and trajectory profiles",
                depends_on=["s17"],
                criticality="diagnostic",
            ),
        ])

        if present("deep_research"):
            steps.append(PlanStep(
                id="s18",
                skill="deep_research",
                args={
                    "topic": "juvenile hair whitening WGS scATAC scRNA spatial transcriptomics melanocyte transcription factor chromatin accessibility",
                    "max_sources": 8,
                },
                description="Collect literature context for hair whitening, melanocyte biology, immune loci and multi-omics mechanism validation",
                depends_on=["s7"],
                criticality="diagnostic",
            ))
            review_deps = ["s17a", "s18"]
        else:
            review_deps = ["s17a"]

        steps.extend([
            PlanStep(
                id="s19",
                skill="statistical_review",
                args={"scope": "session"},
                description="Review small-sample, multi-omics and fallback limitations before reporting",
                depends_on=review_deps,
            ),
            PlanStep(
                id="s20",
                skill="safety_check",
                args={"scope": "session", "k": 5},
                description="Check donor-linked multi-omics privacy, small-cell reporting and claim-safety constraints",
                depends_on=["s19"],
            ),
            PlanStep(
                id="s21",
                skill="world_model_audit",
                args={
                    "task": "Juvenile hair whitening WGS-to-epigenome-to-transcriptome-to-spatial mechanism workflow",
                    "simulation_type": "multiomics_mechanism_hypothesis_generation",
                    "input_modalities": "WGS VCF, scATAC h5ad, scRNA h5ad, Stereo-seq h5ad, BWhair phenotype manifest",
                    "available_tokens": 28,
                    "training_distribution_coverage": 0.2,
                    "calibration_status": "not_validated",
                    "external_validation_status": "not_validated",
                },
                description="Audit whether the evidence supports mechanism hypotheses rather than causal claims",
                depends_on=["s20"],
            ),
            PlanStep(
                id="s22",
                skill="generate_report",
                args={"title": "Juvenile Hair Whitening Multi-Omics Mechanism Analysis", "format": "dual"},
                description="Generate Markdown and HTML reports with the four task trajectories, Action Graph trace and reproducibility notes",
                depends_on=["s21"],
            ),
        ])

        plan = LongHorizonPlan(goal=goal, steps=steps, title=self._default_title_for_goal(goal))
        plan.assumptions = [
            "Primary biological target is Juvenile_White donor-linked hair whitening; WGS comparison defaults to Juvenile_White versus Vitiligo_White with Senile_White as context.",
            "White-vs-black hair analyses use matched BWhair donor/sample metadata where available and degrade to readiness/fallback reports when matrix-level contrasts are unavailable.",
            "Default CLI planning does not run heavy WGS annotation/association unless full WGS execution is explicitly requested; candidate variants are anchored by upstream WGS results when present, otherwise curated pigmentation/autoimmune loci are explicitly labelled as fallback hypotheses.",
            "TF binding, scATAC peak overlap, scRNA differential expression and Stereo spatial interactions require external databases or matrix extraction for quantitative claims; missing pieces are recorded as workflow gaps.",
            "Generated skill proposals are review-gated unless a trusted harness explicitly enables activation.",
        ]
        return plan

    def _wgs_vitiligo_default_plan(
        self,
        goal: str,
        available_skills: list[str] | None = None,
    ) -> LongHorizonPlan:
        """Deterministic CLI plan for the VirtualCell WGS vitiligo task."""
        available = set(available_skills or [])

        def present(skill: str) -> bool:
            return not available or skill in available

        steps: list[PlanStep] = [
            *([
                PlanStep(
                    id="s0p",
                    skill="trajectory_profile_match",
                    args={"goal": goal, "available_skills": ",".join(sorted(available)), "use_llm": False},
                    description="Match the WGS request to a trajectory profile with skill, artifact, report and MCP contracts",
                )
            ] if present("trajectory_profile_match") else []),
            PlanStep(
                id="s1",
                skill="wgs_environment_check",
                args={},
                description="Check WGS toolchain, annotation databases, VCF discovery and standard-workflow readiness",
                depends_on=["s0p"] if present("trajectory_profile_match") else [],
            ),
            PlanStep(
                id="s2",
                skill="cohort_phenotype_summary",
                args={"group_by": "phenotype_group"},
                description="Summarize WGS samples by phenotype group, age, sex and sampling site",
                depends_on=["s1"],
            ),
            PlanStep(
                id="s3",
                skill="vcf_sample_list",
                args={"check_index": True},
                description="List available single-sample WGS VCF files and index status",
                depends_on=["s1"],
            ),
            PlanStep(
                id="s4",
                skill="vcf_qc",
                args={
                    "min_qual": 30.0,
                    "min_dp": 10,
                    "min_gq": 20,
                    "min_call_rate": 0.90,
                    "output_filtered": True,
                    "region": "chr22",
                    "max_variants_per_sample": 200000,
                },
                description="Run VCF-level and sample-level QC with chr22 smoke window for CLI validation",
                depends_on=["s2", "s3"],
            ),
            PlanStep(
                id="s5",
                skill="vcf_annotation",
                args={"genes": "SOX10", "min_qual": 30.0, "max_variants_per_gene": 5000},
                description="Annotate vitiligo candidate variants with standard SnpEff/VEP fallback where available",
                depends_on=["s4"],
            ),
            PlanStep(
                id="s6",
                skill="vcf_pca",
                args={"maf_threshold": 0.05, "n_components": 5, "max_variants": 1000, "chromosomes": "chr22"},
                description="Compute common-variant PCA and phenotype-colored population-structure plots",
                depends_on=["s4"],
            ),
            PlanStep(
                id="s7",
                skill="vcf_kinship",
                args={"maf_threshold": 0.05, "max_snps": 1000, "kinship_threshold": 0.1, "region": "chr22"},
                description="Estimate pairwise kinship and flag potentially related WGS samples",
                depends_on=["s4"],
            ),
            PlanStep(
                id="s8",
                skill="vcf_association",
                args={
                    "case_group": "J",
                    "control_group": "V",
                    "maf_min": 0.01,
                    "chromosomes": "chr22",
                    "max_variants": 2000,
                },
                description="Run Juvenile_White versus Vitiligo_White single-variant association testing",
                depends_on=["s4", "s6", "s7"],
            ),
            PlanStep(
                id="s9",
                skill="vcf_burden_test",
                args={
                    "case_group": "J",
                    "control_group": "V",
                    "maf_max": 0.05,
                    "min_variants_per_gene": 1,
                    "gene_list": "SOX10",
                },
                description="Run rare-variant burden testing for vitiligo candidate genes",
                depends_on=["s5", "s8"],
            ),
            PlanStep(
                id="s10",
                skill="pathway_enrichment",
                args={"gene_list": "TYR,OCA2,MC1R,HLA,NLRP1,PAX3,SOX10", "database": "all", "top_n": 20},
                description="Run pathway enrichment focused on melanogenesis, autoimmunity and oxidative stress",
                depends_on=["s9"],
            ),
        ]

        if present("vcf_phenotype_comparison"):
            steps.append(PlanStep(
                id="s11",
                skill="vcf_phenotype_comparison",
                args={"region": "chr22", "group_by": "phenotype_group", "max_variants_per_sample": 50000},
                description="Compare variant burden and genotype quality summaries across phenotype groups",
                depends_on=["s4"],
            ))
            model_dep = "s11"
        else:
            model_dep = "s4"

        if present("train_phenotype_model"):
            steps.append(PlanStep(
                id="s12",
                skill="train_phenotype_model",
                args={
                    "target": "phenotype_group",
                    "positive_class": "J",
                    "model_type": "auto",
                    "features": "all_numeric",
                    "n_folds": 0,
                    "include_classes": "J,V",
                },
                description="Train a small-sample phenotype classifier for Juvenile_White versus Vitiligo_White only",
                depends_on=[model_dep],
                criticality="diagnostic",
            ))
            if present("feature_importance"):
                steps.append(PlanStep(
                    id="s13",
                    skill="feature_importance",
                    args={"top_n": 10, "method": "tree"},
                    description="Extract feature importance for the diagnostic phenotype model",
                    depends_on=["s12"],
                    criticality="diagnostic",
                ))
            if present("embedding"):
                steps.append(PlanStep(
                    id="s14",
                    skill="embedding",
                    args={"method": "tsne", "sample_size": 0},
                    description="Visualize model feature space with a two-dimensional embedding",
                    depends_on=["s12"],
                    criticality="diagnostic",
                ))

        if present("deep_research"):
            steps.append(PlanStep(
                id="s15",
                skill="deep_research",
                args={
                    "topic": "vitiligo genome-wide association HLA TYR OCA2 MC1R NLRP1 PAX3 SOX10 rare variant burden",
                    "max_sources": 8,
                },
                description="Collect related literature for vitiligo genetics, GWAS, HLA and candidate genes",
                depends_on=["s1"],
                criticality="diagnostic",
            ))
        elif present("web_search"):
            steps.append(PlanStep(
                id="s15",
                skill="web_search",
                args={"query": "vitiligo GWAS HLA TYR OCA2 NLRP1 candidate genes", "max_results": 8},
                description="Search related literature for vitiligo genetic associations",
                depends_on=["s1"],
                criticality="diagnostic",
            ))

        analysis_ids = [step.id for step in steps]
        steps.extend([
            PlanStep(
                id="s16",
                skill="statistical_review",
                args={"scope": "session"},
                description="Review WGS QC, PCA, kinship, association, burden and model limitations",
                depends_on=analysis_ids,
            ),
            PlanStep(
                id="s17",
                skill="safety_check",
                args={"scope": "session", "k": 5},
                description="Check privacy, small-cell and reporting constraints before final output",
                depends_on=["s16"],
            ),
            PlanStep(
                id="s18",
                skill="world_model_audit",
                args={
                    "task": "VirtualCell WGS Juvenile_White versus Vitiligo_White vitiligo genetics workflow",
                    "simulation_type": "association_conditioned_forecast",
                    "input_modalities": "WGS VCF, phenotype manifest, age, sex",
                    "available_tokens": 28,
                    "training_distribution_coverage": 0.2,
                    "calibration_status": "unknown",
                    "external_validation_status": "not_validated",
                },
                description="Audit claim boundaries for a small exploratory WGS cohort",
                depends_on=["s17"],
            ),
            PlanStep(
                id="s19",
                skill="generate_report",
                args={"title": "VirtualCell WGS Vitiligo Case-Control Analysis", "format": "dual"},
                description="Generate Markdown and HTML reports with WGS methods, figures, top hits, caveats and reproducibility notes",
                depends_on=["s18"],
            ),
        ])

        plan = LongHorizonPlan(
            goal=goal,
            steps=steps,
            title=self._default_title_for_goal(goal),
        )
        plan.assumptions = [
            "Juvenile_White is encoded as phenotype_group J and is the case group.",
            "Vitiligo_White is encoded as phenotype_group V and is the main comparison/control group.",
            "Senile_White samples are summarized and QC-screened but excluded from the main J versus V case-control tests.",
            "The CLI validation template uses chr22/SOX10-bounded WGS steps for tractable end-to-end testing; full-genome runs can broaden chromosomes/genes with the same skill interfaces.",
        ]
        return plan

    def _concise_e11_report_default_plan(self, goal: str) -> LongHorizonPlan:
        """Short deterministic E11 report plan used when LLM planning degrades."""
        return LongHorizonPlan(
            goal=goal,
            steps=[
                PlanStep(
                    id="s1",
                    skill="field_search",
                    args={"query": "E11 Type 2 Diabetes", "limit": 20},
                    description="Search UKB field catalogue for E11 Type 2 Diabetes related fields",
                ),
                PlanStep(
                    id="s2",
                    skill="cohort_summary",
                    args={"icd10_code": "E11", "controls_ratio": 0},
                    description="Summarize cohort counts for E11 Type 2 Diabetes",
                    depends_on=["s1"],
                ),
                PlanStep(
                    id="s3",
                    skill="statistical_review",
                    args={"scope": "session"},
                    description="Review session analyses for statistical issues",
                    depends_on=["s2", "s1"],
                ),
                PlanStep(
                    id="s4",
                    skill="safety_check",
                    args={"scope": "last", "k": 5},
                    description="Check privacy and safety compliance of recent analyses",
                    depends_on=["s3"],
                ),
                PlanStep(
                    id="s5",
                    skill="world_model_audit",
                    args={
                        "task": goal,
                        "calibration_status": "unknown",
                        "external_validation_status": "not_validated",
                    },
                    description="Audit claim boundaries before final reporting",
                    depends_on=["s4"],
                ),
                PlanStep(
                    id="s6",
                    skill="generate_report",
                    args={"title": "E11 Type 2 Diabetes Analysis", "format": "dual"},
                    description="Generate final dual-format report with paths",
                    depends_on=["s5"],
                ),
            ],
        )

    def _goal_requests_dual_report(self, goal: str) -> bool:
        lower = (goal or "").lower()
        return any(
            token in lower
            for token in (
                "format=\"dual\"",
                "format='dual'",
                "dual report",
                "dual technical",
                "technical plus nature",
                "technical and nature",
                "nature-style",
                "nature style",
                "publication quality",
                "publication-quality",
            )
        )

    def _enforce_report_contract(
        self,
        plan: LongHorizonPlan,
        available_skills: list[str] | None = None,
    ) -> LongHorizonPlan:
        """Normalize report steps and enforce required pre-report reviews."""
        available = set(available_skills or [])
        report_steps = [step for step in plan.steps if step.skill == "generate_report"]
        if not report_steps:
            return plan

        for step in plan.steps:
            if step.skill != "generate_report":
                continue
            if self._goal_requests_dual_report(plan.goal):
                args = dict(step.args or {})
                args["format"] = "dual"
                if not args.get("title"):
                    args["title"] = "UKB analysis report"
                step.args = args

        def skill_available(skill: str) -> bool:
            return not available or skill in available

        def next_step_id() -> str:
            existing = {s.id for s in plan.steps}
            i = len(plan.steps) + 1
            while f"s{i}" in existing or f"r{i}" in existing:
                i += 1
            return f"s{i}"

        def add_unique_deps(step: PlanStep, deps: list[str]) -> None:
            step.depends_on = list(dict.fromkeys([*step.depends_on, *[d for d in deps if d and d != step.id]]))

        def can_add_deps(step: PlanStep, deps: list[str]) -> bool:
            """Adding step -> dep edges is safe only if dep does not already depend on step."""
            for dep in deps:
                if not dep or dep == step.id:
                    return False
                if _has_dependency_path(plan.steps, dep, step.id):
                    return False
            return True

        def find_last(skill: str) -> PlanStep | None:
            matches = [s for s in plan.steps if s.skill == skill]
            return matches[-1] if matches else None

        def insert_before_report(report_step: PlanStep, guard: PlanStep) -> None:
            try:
                insert_at = plan.steps.index(report_step)
            except ValueError:
                insert_at = len(plan.steps)
            plan.steps.insert(insert_at, guard)

        required = [
            ("statistical_review", {"scope": "session"}, "Review statistical validity before reporting"),
            ("safety_check", {"scope": "session"}, "Check privacy and governance before reporting"),
            (
                "world_model_audit",
                {
                    "task": plan.goal or "Biobank analysis report",
                    "calibration_status": "unknown",
                    "external_validation_status": "not_validated",
                },
                "Audit claim boundaries before final reporting",
            ),
        ]

        for report_step in report_steps:
            analysis_deps = [
                s.id for s in plan.steps
                if s.id != report_step.id
                and s.skill not in {"generate_report", "statistical_review", "safety_check", "world_model_audit"}
            ]
            current_deps = list(analysis_deps)
            guard_chain: list[str] = []
            for skill, args, description in required:
                guard = find_last(skill)
                required_deps = current_deps
                if guard is not None and not can_add_deps(guard, required_deps):
                    # Reusing an upstream guard would create a cycle. Add a
                    # fresh final guard for the report gate instead.
                    guard = None
                if guard is None and skill_available(skill):
                    guard = PlanStep(
                        id=next_step_id(),
                        skill=skill,
                        args=dict(args),
                        description=description,
                        depends_on=list(required_deps),
                        criticality="required",
                    )
                    insert_before_report(report_step, guard)
                if guard is None:
                    continue
                if required_deps:
                    add_unique_deps(guard, required_deps)
                guard_chain.append(guard.id)
                current_deps = [guard.id]
            add_unique_deps(report_step, [*analysis_deps, *guard_chain])
        return plan

    def _invalid_request_default_plan(self, goal: str) -> LongHorizonPlan:
        """Plan for unsafe or scientifically invalid requests: push back then analyze safely."""
        return LongHorizonPlan(
            goal=goal,
            steps=[
                PlanStep(
                    id="s1",
                    skill="safety_check",
                    args={"scope": "session"},
                    description="Check privacy and governance constraints before responding to the unsafe request",
                ),
                PlanStep(
                    id="s2",
                    skill="critical_thinking",
                    args={
                        "claim": "A request for causal proof from observational UKB data and patient-level identifiers",
                        "context": (
                            "Observational UK Biobank analyses cannot prove causality without a valid causal design, "
                            "and patient-level identifiers must not be disclosed in reports. Reframe to aggregate, "
                            "associational and predictive evidence with explicit limitations."
                        ),
                    },
                    description="Push back on invalid causal and privacy requirements",
                    depends_on=["s1"],
                ),
                PlanStep(
                    id="s3",
                    skill="think",
                    args={
                        "reasoning": (
                            "Use the safest valid alternative: aggregate UKB-only association and prediction for E11, "
                            "with no patient-level identifiers and no causal claims."
                        )
                    },
                    description="Design the safest feasible aggregate analysis",
                    depends_on=["s2"],
                ),
                PlanStep(id="s4", skill="field_search", args={"query": "HbA1c glycated haemoglobin glucose BMI Type 2 Diabetes", "limit": 20}, description="Find UKB fields related to HbA1c, diabetes and cardiometabolic biomarkers", depends_on=["s3"]),
                PlanStep(id="s5", skill="cohort_summary", args={"icd10_code": "E11"}, description="Summarize Type 2 Diabetes case/control counts", depends_on=["s4"]),
                PlanStep(id="s6", skill="missing_data", args={}, description="Assess missing biomarker data before modelling using the full biomarker table", depends_on=["s5"]),
                PlanStep(id="s7", skill="train_model", args={"icd10_code": "E11", "model_type": "auto", "n_folds": 5}, description="Train an automatically selected predictive model for the safe aggregate endpoint", depends_on=["s6"]),
                PlanStep(id="s8", skill="evaluate_model", args={}, description="Evaluate model discrimination", depends_on=["s7"]),
                PlanStep(id="s9", skill="calibration", args={}, description="Evaluate calibration", depends_on=["s7"]),
                PlanStep(id="s10", skill="feature_importance", args={"top_n": 20, "method": "tree"}, description="Quantify aggregate biomarker importance without exposing identifiers", depends_on=["s7"]),
                PlanStep(id="s11", skill="statistical_review", args={"scope": "session"}, description="Review leakage, confounding and causal-claim limits", depends_on=["s8", "s9", "s10"]),
                PlanStep(id="s12", skill="safety_check", args={"scope": "session"}, description="Confirm no patient-level identifiers or small-cell risks before reporting", depends_on=["s11"]),
                PlanStep(
                    id="s13",
                    skill="world_model_audit",
                    args={
                        "task": "Safe aggregate HbA1c and Type 2 Diabetes association/prediction analysis; no causal proof or patient-level identifiers",
                        "input_modalities": "blood,bmi,diagnoses",
                        "calibration_status": "unknown",
                        "external_validation_status": "not_validated",
                    },
                    description="Audit allowed claim type and external validity limits",
                    depends_on=["s12"],
                ),
                PlanStep(
                    id="s14",
                    skill="generate_report",
                    args={"title": "Aggregate HbA1c and Type 2 Diabetes analysis", "format": "dual"},
                    description="Generate paired technical and Nature-style reports with explicit pushback and limitations",
                    depends_on=["s13"],
                ),
            ],
        )

    def _bank_readiness_default_plan(self, goal: str) -> LongHorizonPlan:
        """Plan credential-gated aggregate readiness probes for configured banks."""
        lower = goal.lower()
        banks: list[str] = []
        if any(token in lower for token in ("all bank", "all configured", "ukb/hpp/ckb", "ukb, hpp, ckb")):
            banks = ["ukb", "hpp", "ckb", "ukb_rap"]
        else:
            if "ukb" in lower and "ukb-rap" not in lower and "ukb rap" not in lower:
                banks.append("ukb")
            if "hpp" in lower:
                banks.append("hpp")
            if "ckb" in lower:
                banks.append("ckb")
            if any(token in lower for token in ("ukb-rap", "ukb rap", "rap")):
                banks.append("ukb_rap")
        if not banks:
            banks = ["ukb", "hpp", "ckb", "ukb_rap"]
        banks_arg = ",".join(dict.fromkeys(banks))
        return LongHorizonPlan(
            goal=goal,
            steps=[
                PlanStep(
                    id="s1",
                    skill="bank_data_readiness",
                    args={
                        "banks": banks_arg,
                        "icd10_code": "E11",
                        "probe_fields": "hba1c,bmi,glucose,systolic_bp,diastolic_bp",
                    },
                    description="Run aggregate credential-gated readiness probes for configured biobank data paths",
                ),
            ],
        )

    def _trajectory_default_plan(
        self,
        goal: str,
        *,
        include_bank_probe: bool = False,
        include_project_doc: bool = False,
        include_summary_plot: bool = False,
        include_model_branch: bool = True,
    ) -> LongHorizonPlan:
        """Plan explicit longitudinal/trajectory requests as feasibility-first workflows."""
        cohort_deps = ["s2a"]
        probe_step = []
        if include_bank_probe:
            probe_step = [
                PlanStep(
                    id="s2b",
                    skill="bank_data_probe",
                    args={"icd10_code": "E11", "probe_fields": "hba1c,bmi,glucose,systolic_bp,diastolic_bp"},
                    description="Verify the active bank adapter, diagnosis filter and biomarker data path before cohort construction",
                    depends_on=["s2a"],
                )
            ]
            cohort_deps.append("s2b")
        doc_step = []
        deep_research_deps = ["s0"]
        if include_project_doc:
            doc_step = [
                PlanStep(
                    id="sdoc",
                    skill="project_doc",
                    args={"mode": "search", "query": "UKB data reference external agents plan mode report generation", "limit": 8},
                    description="Inspect curated project documentation before choosing the executable research workflow",
                )
            ]
            deep_research_deps.append("sdoc")
        summary_plot_step = []
        final_review_deps = ["s5"]
        report_deps = ["s13"]
        model_branch_steps = []
        if include_model_branch:
            model_branch_steps = [
                PlanStep(
                    id="s6",
                    skill="missing_data",
                    args={},
                    description="Assess biomarker missingness before tabular predictive fallback modelling",
                    depends_on=["s2a", "s3"],
                ),
                PlanStep(
                    id="s7",
                    skill="train_model",
                    args={"icd10_code": "E11", "model_type": "auto", "n_folds": 5},
                    description="Train an automatically selected tabular prediction model as a feasible fallback to longitudinal forecasting",
                    depends_on=["s6"],
                ),
                PlanStep(
                    id="s8",
                    skill="evaluate_model",
                    args={},
                    description="Evaluate discrimination for the selected tabular model",
                    depends_on=["s7"],
                ),
                PlanStep(
                    id="s9",
                    skill="calibration",
                    args={},
                    description="Assess calibration for the selected tabular model",
                    depends_on=["s7"],
                ),
                PlanStep(
                    id="s10",
                    skill="feature_importance",
                    args={"top_n": 20, "method": "tree"},
                    description="Summarize important biomarkers for the selected prediction model",
                    depends_on=["s7"],
                ),
            ]
            final_review_deps.extend(["s8", "s9", "s10"])
        if include_summary_plot:
            summary_plot_step = [
                PlanStep(
                    id="s10a",
                    skill="smart_plot",
                    args={
                        "plot_type": "summary",
                        "data_source": "session",
                        "style": "auto",
                        "title": "Trajectory feasibility and E11 model diagnostics",
                    },
                    description="Create a combined figure summarising trajectory feasibility, model discrimination, calibration, and key biomarkers",
                    depends_on=["s5", "s8", "s9", "s10"],
                    criticality="diagnostic",
                )
            ]
            final_review_deps.append("s10a")
            report_deps.append("s10a")
        return LongHorizonPlan(
            goal=goal,
            steps=[
                *doc_step,
                PlanStep(
                    id="s0",
                    skill="ukb_data_inventory",
                    args={},
                    description="Inspect available UKB parquet, raw CSV and full feature-store coverage before planning analysis",
                ),
                PlanStep(
                    id="s1",
                    skill="deep_research",
                    args={
                        "topic": "UK Biobank longitudinal biomarker trajectories HealthFormer disease progression forecast",
                        "max_sources": 15,
                    },
                    description="Collect related work for longitudinal biomarker trajectory modelling",
                    depends_on=deep_research_deps,
                ),
                PlanStep(
                    id="s2",
                    skill="field_search",
                    args={"query": "longitudinal repeated measures BMI HbA1c glucose blood pressure Type 2 Diabetes", "limit": 20},
                    description="Find UKB fields that could support a longitudinal trajectory layer",
                    depends_on=["s1", "s0"],
                ),
                PlanStep(
                    id="s2a",
                    skill="ukb_field_resolve",
                    args={
                        "query": "BMI HbA1c glucose systolic blood pressure diastolic blood pressure LDL HDL cholesterol diabetes medication age diabetes diagnosed",
                        "field_ids": "21001,30750,30740,4080,4079,30780,30760,2443,2976,6153,6177",
                        "limit": 30,
                    },
                    description="Resolve cardiometabolic and diabetes fields to concrete UKB data sources, including raw main CSV fields",
                    depends_on=["s2"],
                ),
                *probe_step,
                PlanStep(
                    id="s3",
                    skill="cohort_summary",
                    args={"icd10_code": "E11"},
                    description="Check Type 2 Diabetes cohort counts before trajectory interpretation",
                    depends_on=cohort_deps,
                ),
                PlanStep(
                    id="s4",
                    skill="cohort_card",
                    args={"query": goal, "endpoint": "E11", "banks": "ukb"},
                    description="Create an auditable trajectory cohort design card",
                    depends_on=["s3"],
                ),
                PlanStep(
                    id="s5",
                    skill="trajectory_tokenize",
                    args={"max_bins": 20, "target_modality": "bmi", "target_timestamp": "2030-01-01"},
                    description="Prepare available longitudinal rows as a HealthFormer-style trajectory layer or record why it is partial",
                    depends_on=["s4"],
                ),
                *model_branch_steps,
                *summary_plot_step,
                PlanStep(
                    id="s11",
                    skill="statistical_review",
                    args={"scope": "session"},
                    description="Review trajectory feasibility, model validity, empty-token risks and unsupported temporal claims",
                    depends_on=final_review_deps,
                ),
                PlanStep(
                    id="s12",
                    skill="safety_check",
                    args={"scope": "session"},
                    description="Check privacy and governance constraints before reporting trajectory results",
                    depends_on=["s11"],
                ),
                PlanStep(
                    id="s13",
                    skill="world_model_audit",
                    args={
                        "task": "UKB longitudinal cardiometabolic trajectory forecast",
                        "simulation_type": "association_conditioned_forecast",
                        "input_modalities": "blood,bmi,diagnoses",
                        "calibration_status": "unknown",
                        "external_validation_status": "not_validated",
                    },
                    description="Audit whether trajectory outputs support only association-conditioned forecasts",
                    depends_on=["s12"],
                ),
                PlanStep(
                    id="s14",
                    skill="generate_report",
                    args={"title": "UKB longitudinal trajectory feasibility report", "format": "dual"},
                    description="Generate paired technical and Nature-style reports with trajectory limitations and model comparison",
                    depends_on=report_deps,
                ),
            ],
        )

    def _extract_pdf_path(self, goal: str) -> str:
        """Extract the first PDF path-like token or DOI from a natural-language goal."""
        import re

        match = re.search(r"(/[^\s]+\.pdf)", goal)
        if match:
            return match.group(1).rstrip(".,;)")
        match = re.search(r"([^\s]+\.pdf)", goal)
        if match:
            return match.group(1).rstrip(".,;)")
        match = re.search(r"(10\.\d{4,9}/[^\s,;)\]]+)", goal, flags=re.IGNORECASE)
        return match.group(1).rstrip(".,;)") if match else goal

    def _paper_replication_default_plan(
        self,
        goal: str,
        available_skills: list[str] | None = None,
    ) -> LongHorizonPlan:
        """Fallback plan for paper-to-replication workflows when LLM planning fails."""
        paper_path = self._extract_pdf_path(goal)
        available = set(available_skills or [])
        include_replication_design = not available or "replicate_paper" in available
        replicate_args = (
            {"source": paper_path, "source_type": "path"}
            if paper_path.lower().endswith(".pdf")
            else {"source": goal, "source_type": "text"}
        )
        steps: list[PlanStep] = []
        if include_replication_design:
            steps.append(
                PlanStep(
                    id="s1",
                    skill="replicate_paper",
                    args=replicate_args,
                    description="Create an auditable paper-to-UKB replication design card and review artifacts",
                )
            )
        first_read_id = "s2" if include_replication_design else "s1"
        first_read_deps = ["s1"] if include_replication_design else []
        offset = 1 if include_replication_design else 0

        def sid(base: int) -> str:
            return f"s{base + offset}"

        steps.extend([
            PlanStep(
                id=first_read_id,
                skill="read_paper",
                args={"paper_path_or_doi": paper_path, "focus": "ukb-relevance"},
                description="Read the supplied paper and extract UKB-relevant methods and replication targets",
                depends_on=first_read_deps,
            ),
            PlanStep(
                id=sid(2),
                skill="deep_research",
                args={"topic": "MILTON UK Biobank disease prediction multi-omics biomarkers replication 10.1038/s41588-024-01898-1", "max_sources": 8},
                description="Collect related work for comparison context",
                depends_on=[first_read_id],
            ),
            PlanStep(
                id=sid(3),
                skill="field_search",
                args={"query": "E11 type 2 diabetes HbA1c glucose BMI cholesterol blood pressure biomarkers UK Biobank", "limit": 20},
                description="Find UKB fields for a feasible disease-prediction replication slice",
                depends_on=[first_read_id],
            ),
            PlanStep(
                id=sid(4),
                skill="cohort_summary",
                args={"icd10_code": "E11"},
                description="Summarize the Type 2 Diabetes replication cohort before modelling",
                depends_on=[sid(3)],
            ),
            PlanStep(
                id=sid(5),
                skill="missing_data",
                args={},
                description="Assess missing biomarker data before model training",
                depends_on=[sid(4)],
            ),
            PlanStep(
                id=sid(6),
                skill="train_model",
                args={"icd10_code": "E11", "model_type": "auto", "n_folds": 5},
                description="Train an automatically selected UKB biomarker model for the feasible replication slice",
                depends_on=[sid(5)],
            ),
            PlanStep(id=sid(7), skill="evaluate_model", args={}, description="Evaluate model discrimination", depends_on=[sid(6)]),
            PlanStep(id=sid(8), skill="calibration", args={}, description="Evaluate model calibration", depends_on=[sid(6)]),
            PlanStep(
                id=sid(9),
                skill="feature_importance",
                args={"top_n": 20, "method": "tree"},
                description="Rank biomarkers contributing to the replication model",
                depends_on=[sid(6)],
            ),
            PlanStep(
                id=sid(10),
                skill="paper_replication_compare",
                args={"scope": "session"},
                description="Compare paper targets with local UKB approximation, model diagnostics and unavailable elements",
                depends_on=[sid(7), sid(8), sid(9)],
            ),
            PlanStep(id=sid(11), skill="statistical_review", args={"scope": "session"}, description="Review statistical validity", depends_on=[sid(10)]),
            PlanStep(id=sid(12), skill="safety_check", args={"scope": "session"}, description="Check privacy and governance constraints", depends_on=[sid(11)]),
            PlanStep(
                id=sid(13),
                skill="world_model_audit",
                args={
                    "task": "Replicate MILTON-style UKB biomarker disease prediction for E11 and compare with the supplied paper",
                    "input_modalities": "blood,bmi,diagnoses",
                    "calibration_status": "unknown",
                    "external_validation_status": "not_validated",
                },
                description="Audit claims and comparison limits before final synthesis",
                depends_on=[sid(12)],
            ),
            PlanStep(
                id=sid(14),
                skill="generate_report",
                args={"title": "UKB paper replication report", "format": "dual"},
                description="Generate paired technical and Nature-style replication reports",
                depends_on=[sid(2), sid(13)],
            ),
        ])
        return LongHorizonPlan(
            goal=goal,
            steps=steps,
        )

    def _literature_discovery_default_plan(self, goal: str) -> LongHorizonPlan:
        """Fallback plan for open-ended natural-language discovery workflows."""
        return LongHorizonPlan(
            goal=goal,
            steps=[
                PlanStep(
                    id="s1",
                    skill="deep_research",
                    args={"topic": "UK Biobank biomarkers Type 2 Diabetes obesity cardiometabolic progression prediction recent literature", "max_sources": 10},
                    description="Find related work and comparison papers",
                ),
                PlanStep(
                    id="s2",
                    skill="field_search",
                    args={"query": "E11 type 2 diabetes obesity BMI HbA1c glucose LDL HDL triglycerides CRP creatinine eGFR blood pressure", "limit": 25},
                    description="Find local UKB fields for the discovery workflow",
                    depends_on=["s1"],
                ),
                PlanStep(id="s3", skill="cohort_summary", args={"icd10_code": "E11"}, description="Summarize Type 2 Diabetes case/control counts", depends_on=["s2"]),
                PlanStep(id="s4", skill="missing_data", args={}, description="Assess missing biomarker data before model training using the full biomarker table", depends_on=["s3"]),
                PlanStep(id="s5", skill="train_model", args={"icd10_code": "E11", "model_type": "auto", "n_folds": 5}, description="Train and select a predictive model", depends_on=["s4"]),
                PlanStep(id="s6", skill="evaluate_model", args={}, description="Evaluate discrimination", depends_on=["s5"]),
                PlanStep(id="s7", skill="calibration", args={}, description="Evaluate calibration", depends_on=["s5"]),
                PlanStep(id="s8", skill="feature_importance", args={"top_n": 20, "method": "tree"}, description="Extract biomarker importance insights", depends_on=["s5"]),
                PlanStep(id="s9", skill="statistical_review", args={"scope": "session"}, description="Review statistical validity", depends_on=["s6", "s7", "s8"]),
                PlanStep(id="s10", skill="safety_check", args={"scope": "session"}, description="Check privacy and governance constraints", depends_on=["s9"]),
                PlanStep(
                    id="s11",
                    skill="world_model_audit",
                    args={
                        "task": "UKB biomarker prediction of Type 2 Diabetes and cardiometabolic progression",
                        "input_modalities": "blood,bmi,diagnoses",
                        "calibration_status": "unknown",
                        "external_validation_status": "not_validated",
                    },
                    description="Audit biological and external-validity claims",
                    depends_on=["s10"],
                ),
                PlanStep(
                    id="s12",
                    skill="generate_report",
                    args={"title": "UKB biomarker discovery report", "format": "dual"},
                    description="Generate paired technical and Nature-style discovery reports",
                    depends_on=["s1", "s11"],
                ),
            ],
        )

    def _check_temporal_safety(self, plan: LongHorizonPlan) -> LongHorizonPlan:
        """Check temporal safety rules and log warnings for violations.

        Does NOT block execution — only logs warnings so the agent can
        self-correct or the user is informed of ordering risks.
        """
        try:
            from .guardrails import TemporalSafetyChecker
            checker = TemporalSafetyChecker()
            plan_steps = [{"skill": s.skill, "id": s.id} for s in plan.steps]
            violations = checker.check_plan(plan_steps)
            for v in violations:
                logger.warning("Temporal safety: %s", v.description)
        except Exception as e:
            logger.debug("Temporal safety check skipped: %s", e)
        return plan


# ── Plan Mode Controller ─────────────────────────────────────────


class PlanMode:
    """Human-in-the-loop planning workflow with Rich progress display.

    Lifecycle: INACTIVE → PLANNING → REVIEW ⇄ REFINING → APPROVED → EXECUTING ⇄ PAUSED → DONE

    Key features:
    - Single-command entry: /plan <goal> immediately starts planning
    - Visible plan generation with spinner
    - User approval gate before execution
    - Iterative refinement via natural language
    - Step-by-step execution with live progress
    - Pause/resume on failure or user request
    """

    def __init__(
        self,
        plans_dir: Path,
        llm: Optional["LLMClient"] = None,
        available_skills: list[str] | None = None,
        tool_schemas: list[dict] | None = None,
        study_spec_compiler: Any | None = None,
    ) -> None:
        self.plans_dir = plans_dir
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.available_skills = available_skills or []
        self.tool_schemas = tool_schemas or []
        self.planner = LongHorizonPlanner(llm, tool_schemas=self.tool_schemas)
        self.validator = PlanSchemaValidator(self.available_skills, self.tool_schemas)
        if study_spec_compiler is not None:
            self.study_spec_compiler = study_spec_compiler
        else:
            try:
                from .study_spec import StudySpecCompiler
                self.study_spec_compiler = StudySpecCompiler(llm=None)
            except Exception:
                self.study_spec_compiler = None

        # State
        self.state: PlanState = PlanState.INACTIVE
        self.plan: LongHorizonPlan | None = None
        self.goal: str = ""
        self.revision: int = 0
        self.pause_reason: str = ""
        self.validation_issues: list[PlanValidationIssue] = []
        self.repair_log: list[dict] = []
        self.clarification_log: list[dict] = []
        self.report_dir: str = ""
        self.current_study_spec: Any | None = None

        # Execution tracking
        from .progress import StepResult
        self.execution_log: list[StepResult] = []

        # File persistence (for plan history)
        self.current_plan_file: Path | None = None

    @property
    def is_active(self) -> bool:
        """Whether plan mode is currently active (not INACTIVE or DONE)."""
        return self.state not in (PlanState.INACTIVE, PlanState.DONE)

    @property
    def status(self) -> str:
        """Current state as string (backward compat)."""
        return self.state.value

    def identify_clarifications(self, goal: str, max_questions: int = 3) -> list[dict]:
        """Return only critical questions that materially affect execution.

        The goal is not to turn `/plan` into a questionnaire. We ask only when
        the current request would otherwise force a high-impact assumption such
        as endpoint selection, trajectory fallback policy, or model-selection
        strategy.
        """
        try:
            policy_questions = build_clarification_questions(goal, available_skills=self.available_skills)
            if policy_questions:
                return policy_questions[:max_questions]
        except Exception as e:
            logger.debug("Clarification policy skill fallback: %s", e)

        lower = (goal or "").lower()
        questions: list[dict] = []

        if self.planner._is_metabolic_showcase_goal(goal):
            return []

        profile = self.planner._trajectory_profile(goal, self.available_skills)
        profile_questions = list(profile.get("clarification_questions") or [])
        if profile.get("trajectory_id") in {
            "juvenile_hair_multiomics_mechanism",
            "virtualcell_multimodal",
            "wgs_vitiligo_case_control",
        } and profile_questions:
            return profile_questions[:max_questions]

        def has_any(tokens: tuple[str, ...]) -> bool:
            return any(token in lower for token in tokens)

        endpoint_explicit = has_any((" e11", "type 2 diabetes", "t2d", "t2dm", "icd10", "endpoint"))
        if not endpoint_explicit and has_any((
            "diabetes",
            "cardiometabolic",
            "obesity",
            "progression",
            "metabolic",
            "糖尿病",
            "肥胖",
            "代谢",
            "进展",
        )):
            questions.append({
                "id": "endpoint",
                "header": "Endpoint",
                "question": "Which endpoint should anchor the executable UKB workflow?",
                "options": [
                    {
                        "label": "E11 T2D",
                        "value": "Use ICD-10 E11 Type 2 Diabetes as the primary endpoint.",
                        "description": "Most reproducible default for cohort counts and model training.",
                    },
                    {
                        "label": "Cardiometabolic",
                        "value": "Use a broad cardiometabolic endpoint if fields support it.",
                        "description": "Broader but may require more feasibility checks.",
                    },
                    {
                        "label": "Ask Later",
                        "value": "Pause before endpoint-specific modelling if the endpoint is still ambiguous.",
                        "description": "More conservative but slower.",
                    },
                ],
            })

        if has_any(("trajectory", "longitudinal", "healthformer", "forecast over time", "over time", "轨迹", "纵向")):
            questions.append({
                "id": "trajectory_policy",
                "header": "Trajectory",
                "question": "How should the agent handle incomplete longitudinal trajectory support?",
                "options": [
                    {
                        "label": "Feasible Fallback",
                        "value": "Attempt trajectory feasibility first, then fall back to governed tabular prediction if tokens are sparse.",
                        "description": "Best default for producing a report without overstating temporal claims.",
                    },
                    {
                        "label": "Trajectory Only",
                        "value": "Do not train a tabular fallback model unless trajectory data are sufficient.",
                        "description": "Stricter but may end with a feasibility-only report.",
                    },
                ],
            })

        if has_any(("model", "prediction", "predict", "train", "forecast", "auc", "模型", "预测", "训练")) and not has_any(
            ("logistic", "xgboost", "random forest", "lightgbm", "auto")
        ):
            questions.append({
                "id": "model_policy",
                "header": "Model",
                "question": "How should model choice be handled?",
                "options": [
                    {
                        "label": "Auto Select",
                        "value": "Use train_model model_type=auto and report candidate models, metrics, and rationale.",
                        "description": "Lets the agent compare candidates and recover if the first model is weak.",
                    },
                    {
                        "label": "Interpretable",
                        "value": "Prefer an interpretable baseline unless performance is clearly inadequate.",
                        "description": "Simpler scientific story with less modelling breadth.",
                    },
                ],
            })

        return questions[:max_questions]

    def record_clarification_answers(self, answers: list[dict]) -> None:
        """Persist clarification answers onto the current plan."""
        safe_answers: list[dict] = []
        assumptions: list[str] = []
        for answer in answers or []:
            if not isinstance(answer, dict):
                continue
            safe = {
                "id": str(answer.get("id", "")),
                "question": str(answer.get("question", ""))[:500],
                "answer": str(answer.get("answer", ""))[:1000],
                "label": str(answer.get("label", ""))[:200],
            }
            safe_answers.append(safe)
            if safe["answer"]:
                assumptions.append(safe["answer"])

        self.clarification_log = safe_answers
        if self.plan:
            self.plan.clarifications = safe_answers
            existing = list(self.plan.assumptions or [])
            for assumption in assumptions:
                if assumption not in existing:
                    existing.append(assumption)
            self.plan.assumptions = existing
            self._save_plan_file()

    def start(self, goal: str) -> str:
        """Start plan mode: decompose goal into steps, enter REVIEW state.

        This is the single-command entry point. It:
        1. Sets state to PLANNING
        2. Calls LongHorizonPlanner.decompose()
        3. Sets state to REVIEW
        4. Returns status message

        Args:
            goal: Natural language research goal

        Returns:
            Status message for the CLI to display
        """
        if self.is_active:
            return f"Plan mode already active (state: {self.state.value}). Use /plan-exit first."

        self.goal = goal
        self.state = PlanState.PLANNING
        self.revision = 1
        self.execution_log = []
        self.pause_reason = ""
        self.repair_log = []
        self.clarification_log = []
        self.current_study_spec = self._compile_study_spec(goal)

        # Decompose goal into steps
        self.plan = self.planner.decompose(
            goal=goal,
            available_skills=self.available_skills,
            tool_schemas=self.tool_schemas,
            spec=self.current_study_spec,
        )
        if self.plan and not self.plan.title:
            self.plan.title = LongHorizonPlanner._default_title_for_goal(goal)
        self._attach_study_spec_metadata()
        self._validate_current_plan()
        self.state = PlanState.REVIEW

        # Save plan file for history
        self._save_plan_file()

        logger.info("Plan mode entered: %s (%d steps)", goal[:50], self.plan.total_steps)
        if self.validation_issues:
            return (
                f"Plan generated with {self.plan.total_steps} steps, but schema validation failed. "
                f"Fix before approval:\n{self.validation_summary()}"
            )
        return f"Plan generated with {self.plan.total_steps} steps. Awaiting your review."

    def rename(self, title: str) -> str:
        """Rename the active plan without changing the original goal."""
        clean = " ".join(str(title or "").split()).strip(" -:;")
        if not clean:
            return "Usage: /plan-title <new title>"
        if not self.plan:
            return "No active plan to rename."
        self.plan.title = clean[:200]
        for step in self.plan.steps:
            if step.skill == "generate_report":
                args = dict(step.args or {})
                args["title"] = self.plan.title
                step.args = args
        self.revision += 1
        self._save_plan_file()
        return f"Plan title updated to: {self.plan.title}"

    def refine(self, feedback: str) -> str:
        """Refine the plan based on user feedback.

        Can be called in REVIEW or PAUSED state.

        Args:
            feedback: Natural language modification request

        Returns:
            Status message
        """
        if self.state not in (PlanState.REVIEW, PlanState.PAUSED):
            return f"Cannot refine: state is {self.state.value} (need REVIEW or PAUSED)."

        if not self.plan:
            return "No plan to refine."

        old_plan = self.plan
        self.state = PlanState.REFINING

        new_plan = self.planner.refine(
            plan=self.plan,
            feedback=feedback,
            available_skills=self.available_skills,
            tool_schemas=self.tool_schemas,
            spec=self.current_study_spec,
        )

        # Only increment revision if the plan actually changed
        if new_plan is not old_plan:
            self.plan = new_plan
            self._attach_study_spec_metadata()
            self.revision += 1
            self._validate_current_plan()
            self.state = PlanState.REVIEW
            self._save_plan_file()
            logger.info("Plan refined (v%d): %s", self.revision, feedback[:50])
            if self.validation_issues:
                return (
                    f"Plan updated (v{self.revision}), but schema validation failed. "
                    f"Fix before approval:\n{self.validation_summary()}"
                )
            return f"Plan updated (v{self.revision}). Review the changes."
        else:
            # LLM refinement failed silently — plan unchanged
            self._validate_current_plan()
            self.state = PlanState.REVIEW
            logger.warning("Plan refinement returned unchanged plan for: %s", feedback[:50])
            return "Could not refine plan (LLM error or no changes needed). Plan unchanged."

    def approve(self) -> str:
        """Approve the plan for execution.

        Returns:
            Status message
        """
        if self.state != PlanState.REVIEW:
            return f"Cannot approve: state is {self.state.value} (need REVIEW)."

        if not self.plan or not self.plan.steps:
            return "Cannot approve: plan has no steps."

        self._validate_current_plan()
        if self.validation_issues:
            return f"Cannot approve: plan schema validation failed:\n{self.validation_summary()}"

        self.state = PlanState.APPROVED
        logger.info("Plan approved for execution")
        return "Plan approved. Starting execution..."

    def pause(self, reason: str = "User requested") -> str:
        """Pause plan execution.

        Args:
            reason: Why execution was paused

        Returns:
            Status message
        """
        if self.state != PlanState.EXECUTING:
            return f"Cannot pause: state is {self.state.value} (need EXECUTING)."

        self.state = PlanState.PAUSED
        self.pause_reason = reason
        logger.info("Plan paused: %s", reason)
        return f"Plan execution paused: {reason}"

    def resume(self) -> str:
        """Resume paused plan execution.

        Returns:
            Status message
        """
        if self.state != PlanState.PAUSED:
            return f"Cannot resume: state is {self.state.value} (need PAUSED)."

        self.state = PlanState.EXECUTING
        self.pause_reason = ""
        logger.info("Plan execution resumed")
        return "Resuming plan execution..."

    def complete(self) -> str:
        """Mark plan as done.

        Returns:
            Status message
        """
        self.state = PlanState.DONE
        self._save_plan_file()
        logger.info("Plan completed")
        return "Plan execution complete."

    def exit(self) -> str:
        """Exit plan mode from any state.

        Returns:
            Status message
        """
        if not self.is_active and self.state != PlanState.DONE:
            return "Not in plan mode."

        old_state = self.state
        self._reset()
        return f"Exited plan mode (was: {old_state.value})."

    def set_executing(self) -> None:
        """Transition to EXECUTING state (called by executor)."""
        self.state = PlanState.EXECUTING

    def get_plan_content(self) -> str:
        """Get current plan as markdown (backward compat)."""
        if self.plan:
            validation = f"\n\n## Validation errors\n{self.validation_summary()}" if self.validation_issues else ""
            return self.plan.to_markdown() + validation
        return "(no active plan)"

    def validation_summary(self) -> str:
        """Human-readable validation errors for review/approval gates."""
        if not self.validation_issues:
            return "No validation errors."
        return "\n".join(f"- {issue.format()}" for issue in self.validation_issues)

    def validate_current_plan(self) -> list[PlanValidationIssue]:
        """Public validation hook for tests and CLI integrations."""
        return self._validate_current_plan()

    def refresh_tools(self, registry: Any) -> int:
        """Refresh tool schemas after dynamic skill activation."""
        try:
            self.available_skills = [s["name"] for s in registry.list_skills()]
            self.tool_schemas = registry.tool_schemas()
            self.planner.tool_schemas = self.tool_schemas
            self.validator = PlanSchemaValidator(self.available_skills, self.tool_schemas)
            self._validate_current_plan()
            logger.info("Plan tools refreshed: %d skills", len(self.available_skills))
        except Exception as e:
            logger.warning("Failed to refresh plan tools: %s", e)
        return len(self.available_skills)

    def _validate_current_plan(self) -> list[PlanValidationIssue]:
        """Validate and annotate the active plan without changing lifecycle state."""
        if self.plan:
            for step in self.plan.steps:
                if step.error.startswith("Validation:"):
                    step.error = ""
        self.validation_issues = self.validator.validate(self.plan)
        if self.plan and self.validation_issues:
            by_step: dict[str, list[str]] = {}
            for issue in self.validation_issues:
                by_step.setdefault(issue.step_id, []).append(issue.message)
            for step in self.plan.steps:
                messages = by_step.get(step.id)
                if messages:
                    step.error = "Validation: " + "; ".join(messages)
        return self.validation_issues

    def capture_base_plan_snapshot(self) -> None:
        """Store the Biobank-only plan before external planning advice is merged."""
        if self.plan and not self.plan.base_plan_markdown:
            self.plan.base_plan_markdown = self.plan.to_markdown()

    def attach_planning_council(self, records: list[dict]) -> None:
        """Attach read-only external planning records to the current plan."""
        if not self.plan:
            return
        safe_records = []
        for record in records or []:
            if not isinstance(record, dict):
                continue
            safe_records.append({
                "agent": str(record.get("agent", "")),
                "status": str(record.get("status", "")),
                "available": bool(record.get("available", True)),
                "elapsed_s": float(record.get("elapsed_s") or 0.0),
                "summary": str(record.get("summary", ""))[:2000],
                "stdout": str(record.get("stdout", ""))[:4000],
                "stderr": str(record.get("stderr", ""))[:4000],
                "error": str(record.get("error", ""))[:1000],
                "remediation": str(record.get("remediation", ""))[:1000],
                "command_display": str(record.get("command_display", ""))[:1000],
                "prompt_hash": str(record.get("prompt_hash", ""))[:128],
            })
        self.plan.planning_council = safe_records

    def merge_external_plans(self) -> str:
        """Merge external planning advice using deterministic schema-safe gates."""
        if not self.plan:
            return "No plan to merge."
        before = [(s.skill, dict(s.args), list(s.depends_on)) for s in self.plan.steps]
        self.plan = self._ensure_goal_critical_steps(self.plan)
        after = [(s.skill, dict(s.args), list(s.depends_on)) for s in self.plan.steps]
        if after != before:
            self.revision += 1
            self._validate_current_plan()
            self._save_plan_file()
            return f"Planning council merged into executable plan (v{self.revision})."
        self._validate_current_plan()
        self._save_plan_file()
        return "Planning council attached; executable plan already covered required gates."

    def _ensure_goal_critical_steps(self, plan: LongHorizonPlan) -> LongHorizonPlan:
        """Add missing high-level gates that external planners commonly request."""
        goal = (self.goal or plan.goal or "").lower()
        council_text = " ".join(
            f"{item.get('summary', '')} {item.get('error', '')}"
            for item in (plan.planning_council or [])
            if isinstance(item, dict)
        ).lower()
        negative_model = any(
            phrase in goal or phrase in council_text
            for phrase in (
                "no model",
                "no model training",
                "without model",
                "without modelling",
                "do not train",
                "don't train",
                "skip model",
                "skip modelling",
                "descriptive workflow",
            )
        )
        goal_needs_model = any(
            token in goal
            for token in (
                "predict",
                "prediction",
                "risk model",
                "train_model",
                "model_type",
                "auc",
                "calibration",
                "best possible model",
                "best feasible model",
            )
        )
        council_needs_model = (
            not negative_model
            and any(
                token in council_text
                for token in (
                    "train_model",
                    "model_type=\"auto\"",
                    "model_type='auto'",
                    "evaluate_model",
                    "feature_importance",
                    "calibration",
                )
            )
        )
        needs_model = (goal_needs_model or council_needs_model) and not negative_model
        if not needs_model or any(s.skill == "train_model" for s in plan.steps):
            return plan

        skills = {s.skill for s in plan.steps}
        available = set(self.available_skills or [])

        def skill_available(name: str) -> bool:
            return not available or name in available

        def next_id() -> str:
            existing = {s.id for s in plan.steps}
            i = len(plan.steps) + 1
            while f"s{i}" in existing:
                i += 1
            return f"s{i}"

        cohort_ids = [s.id for s in plan.steps if s.skill == "cohort_summary"] or [
            s.id for s in plan.steps if s.skill == "field_search"
        ]
        deps = cohort_ids[-1:] if cohort_ids else []
        inserted_ids: list[str] = []

        def add(skill: str, args: dict, description: str, depends_on: list[str]) -> str:
            sid = next_id()
            plan.steps.append(PlanStep(id=sid, skill=skill, args=args, description=description, depends_on=depends_on))
            inserted_ids.append(sid)
            return sid

        missing_id = ""
        if "missing_data" not in skills and skill_available("missing_data"):
            missing_id = add(
                "missing_data",
                {},
                "Assess missing biomarker data before model training",
                deps,
            )
        train_dep = [missing_id] if missing_id else deps
        train_id = ""
        if skill_available("train_model"):
            train_id = add(
                "train_model",
                {"icd10_code": "E11", "model_type": "auto", "n_folds": 5},
                "Train automatically selected candidate models for the feasible prediction endpoint",
                train_dep,
            )
        if train_id and skill_available("evaluate_model"):
            add("evaluate_model", {}, "Evaluate discrimination for the selected model", [train_id])
        if train_id and skill_available("calibration"):
            add("calibration", {}, "Assess calibration for the selected model", [train_id])
        if train_id and skill_available("feature_importance"):
            add(
                "feature_importance",
                {"top_n": 20, "method": "tree"},
                "Summarize important biomarkers for the selected model",
                [train_id],
            )

        if inserted_ids:
            for step in plan.steps:
                if step.skill in {"statistical_review", "safety_check", "world_model_audit", "generate_report"}:
                    step.depends_on = list(dict.fromkeys([*step.depends_on, *inserted_ids]))
        return plan

    def _reset(self) -> None:
        """Reset all state."""
        self.state = PlanState.INACTIVE
        self.plan = None
        self.goal = ""
        self.revision = 0
        self.execution_log = []
        self.pause_reason = ""
        self.validation_issues = []
        self.repair_log = []
        self.clarification_log = []
        self.current_plan_file = None
        self.report_dir = ""
        self.current_study_spec = None

    def _compile_study_spec(self, goal: str) -> Any | None:
        compiler = getattr(self, "study_spec_compiler", None)
        if compiler is None:
            return None
        try:
            return compiler.compile(goal)
        except Exception as e:
            logger.debug("Plan StudySpec compilation skipped: %s", e)
            return None

    def _attach_study_spec_metadata(self) -> None:
        if not self.plan or self.current_study_spec is None:
            return
        try:
            spec = self.current_study_spec
            marker = (
                f"StudySpec gate: hash={spec.spec_hash()}, "
                f"design={getattr(spec.design, 'value', spec.design)}, "
                f"tool_budget={getattr(spec, 'tool_budget', '?')}"
            )
            assumptions = list(self.plan.assumptions or [])
            if marker not in assumptions:
                assumptions.append(marker)
            self.plan.assumptions = assumptions
        except Exception:
            pass

    def _save_plan_file(self) -> None:
        """Save current plan state to a markdown file."""
        if not self.plan:
            return
        now = datetime.now()
        if not self.current_plan_file:
            slug_source = (self.plan.title or self.goal)[:60]
            slug = slug_source.lower().replace(" ", "-").replace("/", "-")
            slug = "".join(c for c in slug if c.isalnum() or c == "-")
            plan_id = now.strftime(f"%Y-%m-%d-%H%M-{slug}")
            self.current_plan_file = self.plans_dir / f"{plan_id}.md"

        validation_block = ""
        if self.validation_issues:
            validation_block = f"## Validation errors\n{self.validation_summary()}\n\n"

        content = (
            f"# Plan: {(self.plan.title if self.plan and self.plan.title else self.goal)}\n\n"
            f"- Original goal: {self.goal}\n"
            f"- Status: {self.state.value}\n"
            f"- Revision: v{self.revision}\n"
            f"- Updated: {now.strftime('%Y-%m-%d %H:%M')}\n\n"
            f"{validation_block}"
            f"{self.plan.to_markdown()}\n"
            f"{self._planning_council_markdown()}"
        )
        self.current_plan_file.write_text(content)

    def _planning_council_markdown(self) -> str:
        if not self.plan or not (self.plan.base_plan_markdown or self.plan.planning_council):
            return ""
        lines = ["\n\n## Planning Council Appendix\n"]
        if self.plan.base_plan_markdown:
            lines.extend(["\n### Biobank Agent Initial Plan\n", self.plan.base_plan_markdown, "\n"])
        if self.plan.planning_council:
            lines.append("\n### External Planner Outputs\n")
            for item in self.plan.planning_council:
                agent = item.get("agent", "external")
                status = item.get("status", "unknown")
                elapsed = item.get("elapsed_s", 0.0)
                lines.append(f"\n#### {agent} ({status}, {elapsed:.1f}s)\n")
                if item.get("command_display"):
                    lines.append(f"`{item['command_display']}`\n")
                if item.get("summary"):
                    lines.append(str(item["summary"]).strip() + "\n")
                if item.get("error"):
                    lines.append(f"\nError: {item['error']}\n")
                if item.get("stderr") and item.get("stderr") != item.get("summary"):
                    lines.append(f"\nStderr:\n\n```text\n{str(item['stderr']).strip()[:2000]}\n```\n")
                if item.get("remediation"):
                    lines.append(f"\nNext action: {item['remediation']}\n")
                if item.get("prompt_hash"):
                    lines.append(f"\nPrompt hash: `{item['prompt_hash']}`\n")
        lines.append("\n### Merged Executable Plan\n")
        lines.append(self.plan.to_markdown())
        lines.append("\n")
        return "\n".join(lines)

    def list_plans(self) -> list[dict]:
        """List all plan files in the plans directory."""
        plans = []
        for f in sorted(self.plans_dir.glob("*.md"), reverse=True):
            content = f.read_text()
            status = "UNKNOWN"
            for line in content.split("\n"):
                if line.startswith("- Status:"):
                    status = line.split(":", 1)[1].strip()
                    break
            plans.append({"file": f.name, "status": status, "path": str(f)})
        return plans
