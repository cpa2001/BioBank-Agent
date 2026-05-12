"""Paper replicator (M3 half-automatic workflow).

Pipeline:

    paper(text|pdf) → extract_study_design()  -> StudySpec dict
                   → planner.decompose(spec=...)
                   → user approves (Textual modal in M2.5)
                   → execute via legacy plan_executor
                   → diff figure/table claims against reproduced numbers

The extractor is a pattern-matching first pass; it does not try to
fully understand prose. Heuristics:

* "Methods" section → cohort criteria, exposure/outcome variables
* Table 1 / Table 2 captions → outcome counts
* Figure 1 / Figure 2 captions → primary plots to reproduce

The important contract is that this module returns an auditable,
schema-safe replication plan and explicit comparison checklist. It
does **not** claim the paper has been reproduced until the proposed
steps have actually run and a report has been generated.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Heuristic extractor ─────────────────────────────────────


_SECTION_HEADERS = re.compile(
    r"\n\s*(?:abstract|introduction|methods?|results?|discussion|"
    r"conclusion|references)\s*\n",
    re.IGNORECASE,
)

_COHORT_PATTERNS = (
    re.compile(r"(\d{3,7})\s+(?:participants?|subjects?|individuals?|patients?)", re.I),
    re.compile(r"n\s*=\s*(\d{3,7})", re.I),
)

_AGE_PATTERN = re.compile(r"(\d{2})[–\-]\s*(\d{2})\s*(?:years?|yrs)", re.I)

_DESIGN_PATTERNS = {
    "prediction_modeling": re.compile(r"prediction|predictive model|risk model|disease prediction", re.I),
    "case_control": re.compile(r"case[\s-]control", re.I),
    "cohort": re.compile(r"prospective cohort|retrospective cohort", re.I),
    "cross_sectional": re.compile(r"cross[\s-]sectional", re.I),
    "rct": re.compile(r"randomi[sz]ed (?:controlled )?trial", re.I),
    "mendelian_randomization": re.compile(r"mendelian randomi[sz]ation|MR analysis", re.I),
}

_OUTCOME_PATTERNS = (
    re.compile(r"(?:primary )?outcome[s]?(?:\s+was|\s+were)?\s+([^.]{6,200})", re.I),
    re.compile(r"defined as\s+([^.]{6,200})", re.I),
)

_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s,;)\]]+)", re.I)
_URL_RE = re.compile(r"https?://\S+", re.I)
_TABLE_CAPTION_RE = re.compile(r"\b(Table\s+\d+[.:]\s*[^\n]{0,300})", re.I)
_FIGURE_CAPTION_RE = re.compile(r"\b(Fig(?:ure)?\.?\s+\d+[.:]\s*[^\n]{0,300})", re.I)

_BIOMARKER_TERMS = {
    "HbA1c": ("hba1c", "glycated haemoglobin", "glycated hemoglobin"),
    "glucose": ("glucose",),
    "BMI": ("bmi", "body mass index"),
    "blood pressure": ("blood pressure", "systolic", "diastolic"),
    "cholesterol": ("cholesterol", "ldl", "hdl", "lipid", "triglyceride"),
    "CRP": ("crp", "c-reactive protein"),
    "creatinine": ("creatinine",),
}

_DISEASE_TARGETS = (
    ("E11", ("type 2 diabetes", "t2d", "diabetes mellitus", "e11")),
    ("I10", ("hypertension", "high blood pressure", "i10")),
    ("I25", ("coronary", "ischemic heart", "ischaemic heart", "i25")),
    ("E66", ("obesity", "e66")),
)


def _extract_first(patterns, text):
    for p in patterns:
        m = p.search(text)
        if m:
            return m.group(1) if m.lastindex else m.group(0)
    return ""


def extract_study_design(text: str) -> dict[str, Any]:
    """Heuristic extraction of a paper's study design.

    Returns a dict shaped as a draft ``StudySpec``. None of the keys
    are required — downstream callers should treat each as a hint.
    """
    text = text or ""
    out: dict[str, Any] = {
        "design": "",
        "designs_matched": [],
        "n": None,
        "age_range": None,
        "outcomes": [],
        "doi": "",
        "biomarkers": [],
        "icd10_code": "",
        "tables": [],
        "figures": [],
    }
    doi = _DOI_RE.search(text)
    if doi:
        out["doi"] = doi.group(1).rstrip(".")
    for design, pattern in _DESIGN_PATTERNS.items():
        if pattern.search(text):
            out["designs_matched"].append(design)
    if out["designs_matched"]:
        out["design"] = out["designs_matched"][0]
    n_match = _extract_first(_COHORT_PATTERNS, text)
    if n_match:
        try:
            out["n"] = int(n_match)
        except ValueError:
            pass
    age = _AGE_PATTERN.search(text)
    if age:
        try:
            out["age_range"] = (float(age.group(1)), float(age.group(2)))
        except ValueError:
            pass
    for p in _OUTCOME_PATTERNS:
        m = p.search(text)
        if m:
            out["outcomes"].append(m.group(1).strip())
            break
    lower = text.lower()
    for label, aliases in _BIOMARKER_TERMS.items():
        if any(alias in lower for alias in aliases):
            out["biomarkers"].append(label)
    for code, aliases in _DISEASE_TARGETS:
        if any(alias in lower for alias in aliases):
            out["icd10_code"] = code
            break
    out["tables"] = _unique_captions(_TABLE_CAPTION_RE.findall(text))
    out["figures"] = _unique_captions(_FIGURE_CAPTION_RE.findall(text))
    if not out["outcomes"] and out["icd10_code"]:
        label = {
            "E11": "Type 2 Diabetes",
            "I10": "hypertension",
            "I25": "coronary heart disease",
            "E66": "obesity",
        }.get(str(out["icd10_code"]), str(out["icd10_code"]))
        out["outcomes"].append(f"Local UKB ICD-10 {out['icd10_code']} approximation for {label}")
    return out


# ── Replicator ──────────────────────────────────────────────


@dataclass
class PaperReplicationOutcome:
    paper_path: str
    extracted_design: dict[str, Any] = field(default_factory=dict)
    proposed_spec: dict[str, Any] = field(default_factory=dict)
    plan: list[dict[str, Any]] = field(default_factory=list)
    awaiting_user_approval: bool = True
    replication_targets: list[dict[str, Any]] = field(default_factory=list)
    feasibility_map: list[dict[str, Any]] = field(default_factory=list)
    comparison_checklist: list[str] = field(default_factory=list)
    reproduced_results: dict[str, Any] = field(default_factory=dict)
    diff: dict[str, Any] = field(default_factory=dict)
    table_figure_diff: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class PaperReplicator:
    """Coordinator for the half-automatic replication flow."""

    def __init__(self, *, planner: Optional[Any] = None) -> None:
        self.planner = planner

    def from_text(self, paper_text: str, *, paper_path: str = "") -> PaperReplicationOutcome:
        design = extract_study_design(paper_text)
        source_ref = paper_path or design.get("doi") or "<inline>"
        icd10_code = design.get("icd10_code") or "E11"
        biomarkers = list(design.get("biomarkers") or [])
        if not biomarkers:
            biomarkers = ["HbA1c", "glucose", "BMI", "blood pressure", "cholesterol"]
        spec = {
            "design": design.get("design"),
            "n": design.get("n"),
            "age_range": design.get("age_range"),
            "outcomes": list(design.get("outcomes", [])),
            "icd10_code": icd10_code,
            "biomarkers": biomarkers,
            "modalities": ["biomarker", "diagnosis"],
            "tool_budget": 12,
            "source": source_ref,
            "replication_mode": "closest_feasible_ukb_slice",
        }
        targets = _replication_targets(design, biomarkers, icd10_code)
        feasibility = _feasibility_map(targets)
        plan: list[dict[str, Any]] = []
        if self.planner is not None:
            try:
                goal = (
                    "Reproduce the closest feasible UKB-only slice of "
                    f"{source_ref} for {icd10_code} using biomarkers and guarded reporting."
                )
                plan_result = self.planner.decompose(goal=goal, available_skills=[], spec=spec)
                if hasattr(plan_result, "to_dict"):
                    plan = plan_result.to_dict().get("steps", [])
                elif isinstance(plan_result, list):
                    plan = list(plan_result)
            except Exception as e:
                logger.debug("planner.decompose failed: %s", e)
        if not plan:
            plan = _default_replication_plan(
                source_ref=source_ref,
                icd10_code=icd10_code,
                biomarkers=biomarkers,
                goal_text=paper_text,
            )
        return PaperReplicationOutcome(
            paper_path=paper_path,
            extracted_design=design,
            proposed_spec=spec,
            plan=plan,
            awaiting_user_approval=True,
            replication_targets=targets,
            feasibility_map=feasibility,
            comparison_checklist=_comparison_checklist(icd10_code),
            diff={
                "status": "not_run",
                "reason": "The proposed plan is awaiting approval; no reproduction results exist yet.",
            },
            table_figure_diff=_table_figure_diff_targets(design),
            notes=[
                "Half-automatic replication plan created; user approval is required before execution.",
                "Final reports must distinguish exact paper methods from the feasible UKB-only approximation.",
            ],
        )

    def from_path(self, path: Path) -> PaperReplicationOutcome:
        text = self._read_paper(path)
        return self.from_text(text, paper_path=str(path))

    def _read_paper(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            try:
                import fitz  # PyMuPDF
            except ImportError:  # pragma: no cover
                logger.warning("PyMuPDF not installed; falling back to bytes-decode")
                return path.read_bytes().decode("utf-8", errors="ignore")
            doc = fitz.open(str(path))
            try:
                pages = [page.get_text("text") for page in doc]
            finally:
                doc.close()
            return "\n".join(pages)
        return path.read_text(encoding="utf-8", errors="ignore")


def _replication_targets(
    design: dict[str, Any],
    biomarkers: list[str],
    icd10_code: str,
) -> list[dict[str, Any]]:
    outcomes = list(design.get("outcomes") or [])
    if not outcomes:
        outcomes = [f"Incident or prevalent {icd10_code} phenotype"]
    return [
        {
            "target": "Paper cohort and endpoint definition",
            "paper_hint": outcomes[0],
            "local_proxy": f"UKB ICD-10 {icd10_code} diagnosis-centered cohort",
            "required_evidence": ["cohort_summary", "field_search"],
        },
        {
            "target": "Disease prediction model",
            "paper_hint": "Use the strongest feasible local biomarker feature set",
            "local_proxy": f"train_model(icd10_code='{icd10_code}', model_type='auto')",
            "required_evidence": ["train_model", "evaluate_model", "calibration"],
        },
        {
            "target": "Biomarker interpretation",
            "paper_hint": ", ".join(biomarkers),
            "local_proxy": "feature_importance(top_n=20, method='tree')",
            "required_evidence": ["feature_importance", "statistical_review"],
        },
    ]


def _feasibility_map(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        rows.append({
            "target": target["target"],
            "status": "feasible_approximation",
            "reason": (
                "Local UKB fields and diagnosis codes can test an approximation, "
                "but exact replication depends on feature availability, follow-up "
                "definitions and access to the original paper feature set."
            ),
            "local_proxy": target["local_proxy"],
        })
    return rows


def _field_query(biomarkers: list[str], icd10_code: str) -> str:
    terms = ["UK Biobank", icd10_code, "Type 2 Diabetes"]
    terms.extend(biomarkers)
    return " ".join(dict.fromkeys(str(t) for t in terms if t))


def _deep_research_topic(source_ref: str, icd10_code: str) -> str:
    if source_ref and source_ref not in {"<inline>", ""}:
        return f"{source_ref} UK Biobank disease prediction biomarkers replication {icd10_code}"
    return f"UK Biobank disease prediction biomarkers replication {icd10_code}"


def _default_replication_plan(
    *,
    source_ref: str,
    icd10_code: str,
    biomarkers: list[str],
    goal_text: str,
) -> list[dict[str, Any]]:
    """Return a schema-valid plan dict for human review or downstream execution."""
    steps: list[dict[str, Any]] = []
    next_id = 1

    def add(skill: str, args: dict[str, Any], description: str, deps: list[str] | None = None) -> str:
        nonlocal next_id
        step_id = f"s{next_id}"
        next_id += 1
        steps.append({
            "id": step_id,
            "skill": skill,
            "args": args,
            "description": description,
            "depends_on": deps or [],
            "can_parallelize": False,
            "criticality": "required",
        })
        return step_id

    paper_step = ""
    if source_ref and source_ref != "<inline>":
        paper_step = add(
            "read_paper",
            {"paper_path_or_doi": source_ref, "focus": "ukb-relevance"},
            "Read the target paper and extract UKB-relevant methods, fields and endpoints",
        )

    research = add(
        "deep_research",
        {"topic": _deep_research_topic(source_ref, icd10_code), "max_sources": 8},
        "Collect related work and context for the replication target",
        [paper_step] if paper_step else [],
    )
    fields = add(
        "field_search",
        {"query": _field_query(biomarkers, icd10_code), "limit": 30},
        "Map paper biomarkers and disease endpoints to local UKB fields",
        [paper_step] if paper_step else [],
    )
    cohort = add(
        "cohort_summary",
        {"icd10_code": icd10_code},
        "Define and count the UKB diagnosis-centered replication cohort",
        [fields],
    )
    missing = add(
        "missing_data",
        {},
        "Measure biomarker missingness on the available local feature set",
        [fields, cohort],
    )
    model = add(
        "train_model",
        {"icd10_code": icd10_code, "model_type": "auto", "n_folds": 5},
        "Train the best feasible local tabular prediction model",
        [missing],
    )
    eval_step = add("evaluate_model", {}, "Evaluate discrimination for the selected model", [model])
    cal = add("calibration", {}, "Evaluate calibration for the selected model", [model])
    feat = add(
        "feature_importance",
        {"top_n": 20, "method": "tree"},
        "Rank biomarkers and compare interpretation with the paper's feature emphasis",
        [model],
    )
    plot = add(
        "smart_plot",
        {
            "plot_type": "summary",
            "data_source": "session",
            "style": "auto",
            "title": "Paper replication summary",
        },
        "Create a summary figure for model performance, calibration and biomarker importance",
        [eval_step, cal, feat],
    )
    compare = add(
        "paper_replication_compare",
        {"scope": "session"},
        "Compare paper targets with local UKB approximation, model diagnostics and unavailable elements",
        [eval_step, cal, feat],
    )
    stat = add(
        "statistical_review",
        {"scope": "session"},
        "Review model validity, missingness, and paper-comparison limits",
        [compare],
    )
    safety = add("safety_check", {"scope": "session"}, "Check governance and reporting constraints", [stat])
    audit = add(
        "world_model_audit",
        {
            "task": goal_text[:240] or f"UKB paper replication for {icd10_code}",
            "simulation_type": "association_conditioned_forecast",
            "input_modalities": "blood,bmi,diagnoses",
            "calibration_status": "unknown",
            "external_validation_status": "not_validated",
        },
        "Audit claim boundaries before final replication report",
        [safety],
    )
    add(
        "generate_report",
        {"title": "UKB paper replication report", "format": "dual"},
        "Generate paired technical and Nature-style reports with exact replication gaps",
        [research, plot, compare, audit],
    )
    return steps


def _comparison_checklist(icd10_code: str) -> list[str]:
    return [
        "State whether the paper full text was read, abstract-only, or unavailable.",
        f"State the local endpoint and ICD-10 code used for approximation: {icd10_code}.",
        "List paper features that are available locally and features that are unavailable.",
        "Report the trained model family selected by auto mode and why it won.",
        "Report discrimination, calibration, missingness and top biomarkers.",
        "Do not claim exact replication unless paper cohort, feature set, endpoint and evaluation protocol match.",
        "Run statistical_review, safety_check and world_model_audit before generate_report(format='dual').",
    ]


def _unique_captions(captions: list[str], *, limit: int = 10) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for caption in captions:
        clean = " ".join(str(caption).split()).strip()
        key = clean.lower()
        if not clean or key in seen:
            continue
        seen.add(key)
        out.append(clean[:300])
        if len(out) >= limit:
            break
    return out


def _table_figure_diff_targets(design: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for caption in design.get("tables") or []:
        rows.append({
            "target_type": "table",
            "paper_caption": caption,
            "local_artifact": "",
            "comparison_method": "numeric/table diff after execution",
            "status": "awaiting_execution",
            "limitation": "No local table has been generated yet; exact row/column matching requires the approved plan to run.",
        })
    for caption in design.get("figures") or []:
        rows.append({
            "target_type": "figure",
            "paper_caption": caption,
            "local_artifact": "",
            "comparison_method": "visual/pixel diff after execution",
            "status": "awaiting_execution",
            "limitation": "No local figure has been generated yet; pixel-level comparison requires a generated local figure.",
        })
    if not rows:
        rows.append({
            "target_type": "paper_outputs",
            "paper_caption": "No explicit Table/Figure caption detected",
            "local_artifact": "",
            "comparison_method": "comparison matrix only",
            "status": "not_available",
            "limitation": "The source text did not expose concrete table or figure captions to compare.",
        })
    return rows


def render_replication_checklist(outcome: PaperReplicationOutcome) -> str:
    """Render the review artifact shown to a human before execution."""
    lines = [
        "# Paper Replication Review",
        "",
        "## Proposed Spec",
    ]
    for key, value in outcome.proposed_spec.items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Feasibility Map"])
    for row in outcome.feasibility_map:
        lines.append(f"- **{row['target']}**: {row['status']} via {row['local_proxy']}")
        lines.append(f"  - Reason: {row['reason']}")
    lines.extend(["", "## Proposed Plan"])
    for step in outcome.plan:
        deps = f" after {', '.join(step.get('depends_on', []))}" if step.get("depends_on") else ""
        lines.append(f"- `{step.get('id')}` `{step.get('skill')}`{deps}: {step.get('description')}")
    lines.extend(["", "## Comparison Checklist"])
    for item in outcome.comparison_checklist:
        lines.append(f"- [ ] {item}")
    lines.extend(["", "## Table/Figure Diff Targets"])
    for row in outcome.table_figure_diff:
        lines.append(
            f"- **{row['target_type']}** `{row['status']}`: "
            f"{row['paper_caption']} ({row['comparison_method']})"
        )
        lines.append(f"  - Limitation: {row['limitation']}")
    return "\n".join(lines) + "\n"


__all__ = [
    "PaperReplicator",
    "PaperReplicationOutcome",
    "extract_study_design",
    "render_replication_checklist",
]
