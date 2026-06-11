"""Runtime-native, multi-agent council planner.

``RuntimePlanner`` replaces the static ``RuntimeEngine.build_plan`` template with
a real, objective-specific, LLM-backed plan produced by a council:

    Clarification?  ->  Planning (N parallel candidates, diverse personas)
                     -> External council (parallel critics score each candidate)
                     -> Merge (synthesize the best plan)
                     -> Validation (schema -> runtime-native PlanState)

It returns a runtime-native :class:`PlanState` (NOT the legacy skill/args
``LongHorizonPlan``), and **fails loudly** (``CouncilError``) rather than silently
falling back to a generic template when planning genuinely fails.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from typing import Any, Callable

from biobank_agent.runtime.council import (
    CouncilContext,
    CouncilError,
    CouncilJob,
    CouncilResult,
    extract_json,
    run_parallel,
)
from biobank_agent.runtime.adversarial_council import adversarial_plan
from biobank_agent.runtime.types import PlanState, PlanStatus, PlanStep, ProviderRole

logger = logging.getLogger(__name__)

# Clarifier presents one question and returns the chosen answer text (or None).
ClarifierFn = Callable[[dict[str, Any]], "str | None"]


def _ctx_now(ctx: CouncilContext) -> float:
    try:
        return float(ctx.clock())
    except Exception:
        return 0.0


def _as_str_list(value: Any) -> list[str]:
    """Coerce an LLM-provided field into a list of strings WITHOUT splitting a
    bare string into characters (``"qc"`` -> ``["qc"]``, not ``["q", "c"]``)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x).strip()]
    return [str(value)]

_PERSONAS = [
    "a meticulous biostatistician who prioritizes data QC, correct statistical "
    "methodology, multiple-testing correction, and confounder control",
    "a pragmatic bioinformatics engineer who prioritizes reproducible tooling, "
    "explicit data/file flow, and runnable, well-scoped steps",
    "a domain geneticist who prioritizes biological interpretation, prior "
    "literature, hypothesis framing, and clear, auditable reporting",
]

# Short, display-friendly tags shown live in the planning dashboard so each
# parallel candidate is identifiable ("kimi · drafting · biostatistician").
_PERSONA_TAGS = ["biostatistician", "bioinformatics", "geneticist"]

# Distinct provider ROLES per draft slot so each candidate resolves to a
# different model (model_for_role: PLANNER=kimi, PRIMARY_EXECUTOR=deepseek,
# CRITIC=glm). Cycled by slot index so num_candidates > 3 wraps deterministically.
# This is what makes the dashboard show three distinct models instead of 3×kimi.
_DRAFT_ROLES = [ProviderRole.PLANNER, ProviderRole.PRIMARY_EXECUTOR, ProviderRole.CRITIC]

_PLAN_SCHEMA_HINT = """Return ONLY a JSON object with this exact shape:
{
  "title": "<=80 char title",
  "summary": "one-paragraph summary of the approach",
  "steps": [
    {
      "id": "short_snake_case_id",
      "title": "imperative step title",
      "purpose": "what this step accomplishes and why",
      "dependencies": ["ids of steps that must finish first"],
      "tool_scope": ["tool names this step would use, from the available tools"],
      "file_scope": ["files/dirs/datasets this step reads or writes"],
      "verification": ["how to confirm this step succeeded"],
      "risks": ["what could go wrong"]
    }
  ],
  "risks": ["overall risks"],
  "verification_plan": ["how the whole objective is verified end-to-end"],
  "required_approvals": ["approvals needed before mutating/running anything"],
  "proposed_tool_scope": ["union of tools the plan may use"],
  "proposed_file_scope": ["union of files/data the plan may touch"],
  "open_questions": ["unresolved questions, if any"]
}
Make steps concrete and specific to the objective (4-9 steps typically). Use ONLY
tool names from the provided list in tool_scope. No prose outside the JSON."""


# ── Inspection fast-path (issue #6) ────────────────────────────────────────────
# A pure "read first N rows / show columns / head / preview" request must NOT be
# routed through the council, which over-plans a heavyweight genomics workflow for
# any genomics-flavored filename (e.g. a CSV named *gwas_results*). We detect such
# requests deterministically and answer with a 1-step read-only preview.

_INSPECTION_RE = re.compile(
    r"(first\s+\d+\s+rows?|first\s+few\s+rows?|\bhead\b|show\s+(the\s+)?columns?|"
    r"column\s+names?|list\s+(the\s+)?columns?|\bpreview\b|\bpeek\b|read\s+the\s+first|"
    r"print\s+(the\s+)?columns?|what\s+(are\s+the\s+)?columns?|\bdtypes?\b|schema\s+of|"
    r"inspect\s+the\s+file|describe\s+the\s+(file|table|data)|list\s+(the\s+)?files|"
    r"前\s*\d+\s*行|看(一下)?列名|列名|表头|预览|有哪些列|查看.*文件)",
    re.I,
)
# Analysis/heavy imperatives that DISQUALIFY the fast path. Tested only against the
# objective AFTER file paths/names are stripped, so a token inside a filename
# (e.g. "gwas_analysis_results.csv") never blocks a genuine inspection request.
_ANALYSIS_RE = re.compile(
    r"(\bassociat|run\s+gwas|\bperform\b|\bcomput|calculat|analy[sz]e|\banalysis\b|\btrain\b|"
    r"\bmodel\b|predict|classif|cluster|regress|\bpca\b|kinship|burden|annotat|enrich|pathway|"
    r"\bqc\b|quality\s+control|variant\s+call|imputation|\bprs\b|polygenic|characteristic\s+gene|"
    r"differential|分析|关联|建模|训练|预测|分类|聚类|回归|富集|注释|质控|变异|评分|特征基因|差异)",
    re.I,
)
_PATH_RE = re.compile(
    # absolute or relative path with at least one "/" (optionally a leading dir
    # component, so "data/x.csv" keeps its prefix), OR a bare filename with a known
    # data extension.
    r"((?:[~\w.\-]+)?(?:/[\w.\-]+)+|[\w.\-]+\.(?:csv|tsv|txt|parquet|xlsx?|json|vcf|gz|h5ad))",
    re.I,
)
_ROWS_RE = re.compile(r"first\s+(\d+)\s+rows?|前\s*(\d+)\s*行|head\s+(\d+)|\b(\d+)\s+rows?\b", re.I)


def _strip_paths(text: str) -> str:
    return _PATH_RE.sub(" ", str(text or ""))


def extract_path(objective: str) -> str:
    """First file path / filename referenced in the objective (or "")."""
    match = _PATH_RE.search(str(objective or ""))
    return match.group(0) if match else ""


def extract_row_count(objective: str) -> int | None:
    match = _ROWS_RE.search(str(objective or ""))
    if not match:
        return None
    for group in match.groups():
        if group:
            try:
                return int(group)
            except ValueError:
                return None
    return None


def is_pure_inspection_objective(objective: str) -> bool:
    """True when the objective only wants to look at a file (read rows / show
    columns / head / preview) with no analysis imperative. Requires a file
    reference so vague prompts still go to the council."""
    text = str(objective or "")
    if not extract_path(text):
        return False
    prose = _strip_paths(text)  # ignore filename tokens when judging intent
    return bool(_INSPECTION_RE.search(prose)) and not bool(_ANALYSIS_RE.search(prose))


# Generic tabular analysis (classify / cluster / find characteristic features of a
# delimited file). Used only as a fallback when the council fails to parse a plan,
# so a "classify 243 traits and find each class's characteristic genes" request
# yields a workable plan instead of an error.
_TABULAR_ANALYSIS_RE = re.compile(
    r"(classif|cluster|\bgroup\b|grouping|segment|categor|characteristic|signature|"
    r"\bmarker|differential|distinguish|profile|分类|聚类|分组|归类|特征|差异|标志)",
    re.I,
)
_TABULAR_EXT_RE = re.compile(r"\.(?:csv|tsv|txt|parquet|xlsx?)\b", re.I)


def is_tabular_analysis_objective(objective: str) -> bool:
    """True when the goal analyzes a delimited/tabular file (classify rows, find
    characteristic features) — the shape that should get a deterministic analysis
    fallback if the council can't produce a plan."""
    text = str(objective or "")
    if not _TABULAR_EXT_RE.search(text):
        return False
    prose = _strip_paths(text)
    return bool(_TABULAR_ANALYSIS_RE.search(prose))


class RuntimePlanner:
    """Council-backed planner returning runtime-native ``PlanState``."""

    def __init__(
        self,
        provider_router: Any,
        config: Any | None = None,
        *,
        num_candidates: int = 3,
        timeout_s: float = 360.0,
        max_workers: int = 6,
        clock: Callable[[], float] | None = None,
        enable_clarification: bool = True,
        enable_debate: bool = True,
        debate_rounds: int = 2,
        consensus_threshold: float = 0.85,
        confidence_floor: float = 0.45,
    ) -> None:
        self.provider_router = provider_router
        self.config = config
        self.num_candidates = max(1, int(num_candidates))
        self.timeout_s = float(timeout_s)
        self.max_workers = int(max_workers)
        self._clock = clock or (lambda: 0.0)
        self.enable_clarification = bool(enable_clarification)

        # Debate knobs prefer ``config`` (RuntimeConfig) when present, else the
        # explicit kwargs. The whole debate degrades to the single-shot pipeline
        # when disabled, rounds<=0, <2 candidates, or <2 distinct models.
        def _cfg(attr: str, default: Any) -> Any:
            return getattr(config, attr, default) if config is not None else default

        self.enable_debate = bool(_cfg("enable_debate", enable_debate))
        self.debate_rounds = max(0, int(_cfg("debate_rounds", debate_rounds)))
        self.consensus_threshold = float(_cfg("consensus_threshold", consensus_threshold))
        self.confidence_floor = float(_cfg("debate_confidence_floor", confidence_floor))

    # ------------------------------------------------------------------ public
    def build_plan(
        self,
        objective: str,
        *,
        previous: PlanState | None = None,
        refinement: str = "",
        context: str = "",
        tool_names: list[str] | None = None,
        emit: Callable[..., None] | None = None,
        clarifier: ClarifierFn | None = None,
        interactive: bool = False,
        session_id: str = "plan",
        turn_id: str = "plan",
        stream: bool = False,
        cancel_event: Any = None,
    ) -> PlanState:
        clean_objective = " ".join(str(objective or "").split())
        if not clean_objective:
            raise CouncilError("cannot plan an empty objective")
        ctx = CouncilContext(
            provider_router=self.provider_router,
            session_id=session_id,
            turn_id=turn_id,
            emit=emit,
            max_workers=self.max_workers,
            timeout_s=self.timeout_s,
            clock=self._clock,
            stream=bool(stream),
            cancel_event=cancel_event,
        )
        tools = list(tool_names or [])
        open_questions: list[str] = []
        ctx.emit_event("Preflight", status="success", message=f"{len(tools)} tools available; council planner ready")

        # 0) Inspection fast-path (issue #6): a pure "read first N rows / show
        # columns / preview <file>" request is answered with a deterministic
        # read-only preview instead of the council, which would otherwise over-plan
        # a heavyweight genomics workflow for a genomics-flavored filename.
        if is_pure_inspection_objective(clean_objective):
            ctx.emit_event("Planning", status="success",
                           message="file-inspection request — read-only preview (council skipped)")
            plan = self._inspection_plan(clean_objective, tools, previous=previous, refinement=refinement)
            ctx.emit_event("Review", status="success", message="inspection plan ready for review")
            return plan

        # 1) Clarification gate (before decomposition).
        context = self._clarify(ctx, clean_objective, context, clarifier, interactive, open_questions)

        # 2-5) Council pipeline. If it genuinely fails to produce a valid plan, fall
        # back to a deterministic tabular-analysis plan for analyze-a-file goals
        # (issue #5) instead of erroring — so "classify N traits and find each
        # class's characteristic genes" yields a workable plan. Otherwise re-raise.
        try:
            if self.config is not None and getattr(self.config, "adversarial_council_enabled", False):
                # 2') Adversarial-game council: proposer drafts, red-team attacks,
                # referee adjudicates the load-bearing flaws into a revised plan.
                merged = self._adversarial_plan_pipeline(ctx, clean_objective, context, refinement, tools)
            else:
                # 2) Planning: parallel candidate drafts.
                ctx.emit_event("Planning", status="running", message=f"drafting {self.num_candidates} candidate plans")
                candidate_jobs = [
                    CouncilJob(
                        role=_DRAFT_ROLES[i % len(_DRAFT_ROLES)],
                        messages=self._candidate_messages(clean_objective, context, refinement, tools, _PERSONAS[i % len(_PERSONAS)]),
                        label=f"candidate-{i + 1}",
                        stage="Planning",
                        metadata={
                            "persona": i % len(_PERSONAS),
                            "persona_tag": _PERSONA_TAGS[i % len(_PERSONA_TAGS)],
                            "slot": i,
                            "activity": "drafting",
                        },
                    )
                    for i in range(self.num_candidates)
                ]
                candidate_results = run_parallel(ctx, candidate_jobs)
                nodes = self._parse_candidates(candidate_results, candidate_jobs)
                if not nodes:
                    ctx.emit_event("Planning", status="error", message="no candidate plan parsed")
                    raise CouncilError(
                        "Council planning failed: no model produced a valid plan. "
                        "Check the planner model/credentials, then retry /plan."
                    )
                ctx.emit_event("Planning", status="success", message=f"{len(nodes)}/{self.num_candidates} candidate plan(s) parsed")

                # 3) External council: critique each candidate in parallel (scores nodes).
                nodes = self._critique(ctx, clean_objective, nodes)

                # 3b) Debate: bounded multi-round cross-pollination with confidence-based
                # consensus pruning (CONCAT/EVOCHAMBER). No-op unless >=2 distinct models.
                nodes = self._debate(ctx, clean_objective, context, refinement, tools, nodes)

                # 4) Merge: orchestrator synthesizes from the debate winners (or uses the
                # single best surviving candidate).
                merged = self._merge(ctx, clean_objective, context, refinement, tools, nodes)

            # 5) Validation: build + validate runtime-native PlanState.
            ctx.emit_event("Validation", status="running", message="validating merged plan schema")
            plan = self._plan_state_from_json(merged, clean_objective, previous=previous,
                                              refinement=refinement, available_tools=tools)
        except CouncilError:
            if is_tabular_analysis_objective(clean_objective):
                ctx.emit_event("Planning", status="success",
                               message="council could not plan; using a deterministic tabular-analysis plan")
                plan = self._tabular_analysis_plan(clean_objective, tools, previous=previous, refinement=refinement)
                for question in open_questions:
                    if question not in plan.open_questions:
                        plan.open_questions.append(question)
                ctx.emit_event("Review", status="success", message="tabular-analysis plan ready for review")
                return plan
            # Never hard-fail — degrade to a labeled deterministic scaffold plan so a
            # transient council failure does not dead-end the user (issue #5). The fallback
            # is clearly marked so a genuine model/credential outage stays visible.
            ctx.emit_event("Planning", status="success",
                           message="council could not plan; using a deterministic scaffold plan")
            plan = self._scaffold_plan(clean_objective, tools, previous=previous, refinement=refinement)
            for question in open_questions:
                if question not in plan.open_questions:
                    plan.open_questions.append(question)
            ctx.emit_event("Review", status="success", message="scaffold plan ready for review")
            return plan
        for question in open_questions:
            if question not in plan.open_questions:
                plan.open_questions.append(question)
        self._validate_plan(plan)
        ctx.emit_event("Validation", status="success", message=f"{len(plan.steps)} steps validated")
        ctx.emit_event("Review", status="success", message="plan ready for review")
        return plan

    def _inspection_plan(
        self,
        objective: str,
        tools: list[str],
        *,
        previous: PlanState | None = None,
        refinement: str = "",
    ) -> PlanState:
        """Deterministic 1-step read-only preview plan (issue #6 fast path)."""
        tool = "python_exec" if (not tools or "python_exec" in tools) else (
            "shell_exec" if "shell_exec" in tools else "python_exec"
        )
        path = extract_path(objective)
        rows = extract_row_count(objective) or 5
        target = path or "the file referenced in the request"
        name = path.rsplit("/", 1)[-1] if path else "file"
        purpose = (
            f"Read-only preview of {target}. Use {tool} to load the file and print "
            f"(1) its column names / header and (2) the first {rows} rows, then give a "
            f"brief final answer that lists the columns and the output location. Do NOT "
            f"run any analysis, association, QC, annotation, scoring, or variant pipeline "
            f"— this is a file-inspection task only."
        )
        data = {
            "title": f"Preview {name}"[:80],
            "summary": f"Read-only file inspection: {objective[:160]}",
            "steps": [{
                "id": "s1",
                "title": "Preview file (columns + first rows)",
                "purpose": purpose,
                "tool_scope": [tool],
                "file_scope": [path] if path else [],
                "verification": ["The column names and the requested rows are printed."],
            }],
            "proposed_tool_scope": [tool],
            "required_approvals": ["Read-only inspection; no file mutation expected."],
        }
        plan = self._plan_state_from_json(data, objective, previous=previous, refinement=refinement)
        plan.context_gathering = [
            "Deterministic read-only file preview; council planning skipped for a pure "
            "inspection request (issue #6)."
        ]
        plan.audit_summary = "Inspection fast-path plan (no council)."
        return plan

    def _tabular_analysis_plan(
        self,
        objective: str,
        tools: list[str],
        *,
        previous: PlanState | None = None,
        refinement: str = "",
    ) -> PlanState:
        """Deterministic multi-step plan to analyze a tabular file (issue #5 fallback).

        Used only when the council fails to produce a plan for an analyze-a-file
        goal (e.g. 'classify N traits and find each class's characteristic genes').
        Uses the generic python_exec skill (no domain dependency); the user can
        refine it via /plan-edit."""
        has_report = (not tools) or ("generate_report" in tools)
        path = extract_path(objective) or "the input file"
        steps: list[dict[str, Any]] = [
            {
                "id": "load",
                "title": "Load and profile the file",
                "purpose": (
                    f"Use python_exec (pandas) to load {path}; print its shape, column names, "
                    f"dtypes, head, and basic per-column summary stats so the structure is clear."
                ),
                "tool_scope": ["python_exec"],
                "file_scope": [path] if extract_path(objective) else [],
                "verification": ["Shape, columns and head are printed."],
            },
            {
                "id": "analyze",
                "title": "Perform the requested classification/analysis",
                "purpose": (
                    f"Use python_exec to carry out the analysis the user asked for on {path} "
                    f"(original request: {objective[:200]}). Choose a defensible method, print the "
                    f"resulting groups/classes with their sizes, and SAVE outputs (assignments + any "
                    f"tables) to files under the workspace; report the output paths."
                ),
                "tool_scope": ["python_exec"],
                "dependencies": ["load"],
                "verification": ["Classes/groups and their sizes are printed; outputs saved with paths shown."],
            },
            {
                "id": "characterize",
                "title": "Find each class's characteristic features",
                "purpose": (
                    "Use python_exec to identify, for each class/group from the previous step, the "
                    "characteristic/distinguishing features (e.g. top genes/markers per class) with a "
                    "stated criterion; print a compact per-class table and save it to the workspace."
                ),
                "tool_scope": ["python_exec"],
                "dependencies": ["analyze"],
                "verification": ["A per-class characteristic-feature table is produced and saved."],
            },
        ]
        if has_report:
            steps.append({
                "id": "report",
                "title": "Summarize findings in a report",
                "purpose": "Summarize the classes and their characteristic features, with method, "
                           "assumptions and output paths, into a concise report.",
                "tool_scope": ["generate_report"],
                "dependencies": ["characterize"],
                "verification": ["A report is written and its path is shown."],
            })
        data = {
            "title": f"Tabular analysis: {objective[:60]}"[:80],
            "summary": f"Deterministic tabular-file analysis (council fallback): {objective[:160]}",
            "steps": steps,
            "required_approvals": ["User approval before writing outputs."],
        }
        plan = self._plan_state_from_json(data, objective, previous=previous, refinement=refinement,
                                          available_tools=tools)
        plan.context_gathering = [
            "Deterministic tabular-analysis scaffold used because council planning did not "
            "produce a valid plan (issue #5). Refine with /plan-edit if needed."
        ]
        plan.audit_summary = "Tabular-analysis fallback plan (council failed)."
        return plan

    def _scaffold_plan(
        self,
        objective: str,
        tools: list[str],
        *,
        previous: PlanState | None = None,
        refinement: str = "",
    ) -> PlanState:
        """Deterministic minimal plan used when council planning fails for any goal,
        so planning never hard-fails. A labeled gather→execute→report scaffold the
        user can refine via /plan-edit. The degradation is surfaced (audit_summary +
        context_gathering) so a real model/credential failure is never silently hidden."""
        has_report = (not tools) or ("generate_report" in tools)
        path = extract_path(objective)
        steps: list[dict[str, Any]] = [
            {
                "id": "gather",
                "title": "Understand the request and inspect inputs",
                "purpose": (
                    f"Restate the goal in one line and gather what is needed to act on it "
                    f"(original request: {objective[:200]}). If a data file/path is referenced, use "
                    f"python_exec to load and profile it (shape, columns, dtypes, head); otherwise use "
                    f"field_search/python_exec to locate the relevant inputs."
                ),
                "tool_scope": ["python_exec"],
                "file_scope": [path] if path else [],
                "verification": ["Key inputs are identified and printed."],
            },
            {
                "id": "execute",
                "title": "Carry out the requested task",
                "purpose": (
                    f"Perform the task the user asked for (\"{objective[:200]}\") with a defensible "
                    f"method. Print the key results and SAVE all outputs (tables/figures) to files under "
                    f"the workspace; report the output paths. If the data or tooling does not match "
                    f"expectations, call pause_and_ask rather than silently simplifying the analysis."
                ),
                "tool_scope": ["python_exec"],
                "dependencies": ["gather"],
                "verification": ["Key results are printed and outputs saved with paths shown."],
            },
        ]
        if has_report:
            steps.append({
                "id": "report",
                "title": "Summarize findings in a report",
                "purpose": "Summarize the method, assumptions, key results and output paths into a concise report.",
                "tool_scope": ["generate_report"],
                "dependencies": ["execute"],
                "verification": ["A report is written and its path is shown."],
            })
        data = {
            "title": f"Plan: {objective[:60]}"[:80],
            "summary": f"Deterministic scaffold plan (council fallback): {objective[:160]}",
            "steps": steps,
            "required_approvals": ["User approval before writing outputs."],
        }
        plan = self._plan_state_from_json(data, objective, previous=previous, refinement=refinement,
                                          available_tools=tools)
        plan.context_gathering = [
            "Deterministic scaffold plan used because council planning did not produce a valid plan "
            "(robust-council fallback) — check the planner model/credentials if this recurs. "
            "Refine with /plan-edit if needed."
        ]
        plan.audit_summary = "Scaffold fallback plan (council failed)."
        return plan

    # ------------------------------------------------------------ stage: clarify
    def _clarify(
        self,
        ctx: CouncilContext,
        objective: str,
        context: str,
        clarifier: ClarifierFn | None,
        interactive: bool,
        open_questions: list[str],
    ) -> str:
        if not self.enable_clarification:
            return context
        ctx.emit_event("Clarification", status="running", message="checking for critical ambiguities")
        questions = self._identify_clarifications(ctx, objective, context)
        if not questions:
            ctx.emit_event("Clarification", status="success", message="no blocking questions")
            return context
        if not interactive or clarifier is None:
            # Non-interactive: record the questions; never block.
            for q in questions:
                open_questions.append(str(q.get("question") or "").strip())
            ctx.emit_event("Clarification", status="success", message=f"{len(questions)} question(s) deferred (non-interactive)")
            return context
        answered: list[str] = []
        remaining = list(questions)
        while remaining:
            q = remaining.pop(0)
            try:
                answer = clarifier(q)
            except Exception as exc:
                # Do NOT silently record None for every question (that produced
                # the "questions blast past, 0 resolved" symptom when the
                # selector raised a Rich LiveError). Surface the failure and stop
                # asking; defer this and the remaining questions as open.
                # (KeyboardInterrupt is a BaseException and is intentionally not
                # caught here, so Ctrl-C cancels the whole plan.)
                logger.warning("clarifier raised; stopping clarification", exc_info=True)
                ctx.emit_event("Clarification", status="error", message=f"clarifier error: {exc}")
                open_questions.append(str(q.get("question") or "").strip())
                open_questions.extend(str(r.get("question") or "").strip() for r in remaining)
                break
            if answer:
                answered.append(f"Q: {q.get('question')}\nA: {answer}")
            else:
                open_questions.append(str(q.get("question") or "").strip())
        ctx.emit_event("Clarification", status="success", message=f"{len(answered)} clarification(s) resolved")
        if answered:
            context = (context + "\n\nClarifications:\n" + "\n".join(answered)).strip()
        return context

    def _identify_clarifications(self, ctx: CouncilContext, objective: str, context: str) -> list[dict[str, Any]]:
        prompt = (
            "You are scoping a biobank research task. If — and ONLY if — the "
            "objective is genuinely ambiguous in a way that would change the plan, "
            "ask up to 3 critical multiple-choice clarifying questions. If it is "
            "clear enough to plan, return an empty list.\n\n"
            f"Objective: {objective}\n"
            f"{('Context: ' + context[:800]) if context else ''}\n\n"
            'Return ONLY JSON: {"questions": [{"id": "q1", "header": "<=12 chars", '
            '"question": "...", "options": [{"label": "...", "description": "..."}]}]}'
        )
        results = run_parallel(
            ctx,
            [CouncilJob(role=ProviderRole.PLANNER, messages=[
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": prompt},
            ], label="clarify", stage="Clarification")],
        )
        if not results or not results[0].ok:
            return []
        try:
            data = extract_json(results[0].text)
        except Exception:
            return []
        # Tolerant extraction: models vary wildly in the shape they emit. Accept the
        # questions list under several keys (or a bare top-level list); accept each
        # option as a plain STRING or a dict with the label under any common key;
        # accept the question text under question/text/prompt/title. Without this, a
        # model that returns options as strings (very common — e.g. kimi emits
        # "options":["A. ...","B. ..."] with a "text" field) yields ZERO valid
        # questions and the entire clarification step silently no-ops on a real run.
        questions: Any = None
        if isinstance(data, dict):
            for key in ("questions", "clarifications", "clarifying_questions", "items"):
                if isinstance(data.get(key), list):
                    questions = data[key]
                    break
        elif isinstance(data, list):
            questions = data
        if not isinstance(questions, list):
            return []

        def _first_str(d: dict[str, Any], keys: tuple[str, ...]) -> str:
            for k in keys:
                v = d.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""

        def _option(o: Any) -> dict[str, str] | None:
            if isinstance(o, str):
                label, desc = o.strip(), ""
            elif isinstance(o, dict):
                label = _first_str(o, ("label", "text", "value", "name", "option", "title", "answer"))
                desc = _first_str(o, ("description", "detail", "desc", "explanation", "note"))
            else:
                return None
            return {"label": label[:200], "description": desc[:200]} if label else None

        cleaned: list[dict[str, Any]] = []
        for item in questions[:3]:
            if not isinstance(item, dict):
                continue
            options = [opt for opt in (_option(o) for o in (item.get("options") or [])) if opt]
            if not options:
                continue
            cleaned.append({
                "id": str(item.get("id") or f"q{len(cleaned) + 1}"),
                "header": (_first_str(item, ("header", "category", "topic")) or "Clarify")[:12],
                "question": _first_str(item, ("question", "text", "prompt", "q", "title")) or "Clarification needed",
                # Cap options per question so a verbose model cannot produce a
                # menu taller than a typical viewport (the selector also windows
                # defensively, but bounding at the source keeps menus scannable).
                "options": options[:6],
            })
        return cleaned

    # ----------------------------------------------------------- stage: critique
    def _critique(self, ctx: CouncilContext, objective: str, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score each candidate node in parallel; attach ``score`` and return the
        nodes sorted best-first. Each node keeps its plan + provenance (slot /
        role / model / persona) so debate can re-dispatch the SAME model."""
        if len(nodes) == 1:
            nodes[0]["score"] = 100.0
            return nodes
        ctx.emit_event("External council", status="running", message=f"critiquing {len(nodes)} candidate(s)")
        jobs = [
            CouncilJob(
                role=ProviderRole.CRITIC,
                messages=self._critic_messages(objective, node["plan"]),
                label=f"critic-{i + 1}",
                stage="External council",
                metadata={"candidate": i},
            )
            for i, node in enumerate(nodes)
        ]
        results = run_parallel(ctx, jobs)
        scores: dict[int, float] = {}
        for result in results:
            tail = result.label.split("-")[-1]
            idx = int(tail) - 1 if tail.isdigit() else None
            if idx is None or not result.ok:
                continue
            try:
                data = extract_json(result.text)
                scores[idx] = float(data.get("score", 50)) if isinstance(data, dict) else 50.0
            except Exception:
                scores[idx] = 50.0
        for i, node in enumerate(nodes):
            node["score"] = scores.get(i, 50.0)
        nodes.sort(key=lambda n: n["score"], reverse=True)
        ctx.emit_event("External council", status="success", message=f"best candidate scored {nodes[0]['score']:.0f}")
        return nodes

    # --------------------------------------------------------------- stage: debate
    def _debate(
        self,
        ctx: CouncilContext,
        objective: str,
        context: str,
        refinement: str,
        tools: list[str],
        nodes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Bounded multi-round debate: each surviving model sees its peers' plans
        and revises (reporting a self-confidence); homogeneous/low-confidence
        nodes are pruned between rounds (CONCAT/EVOCHAMBER). Stops on consensus
        (the MAD "don't debate when they agree" rule), round cap, ≤1 leader,
        cancel, or time budget. Degrades to a no-op when disabled or <2 distinct
        models. Never returns empty (fail-loud is preserved downstream)."""
        if not self.enable_debate or self.debate_rounds <= 0 or len(nodes) < 2:
            return nodes
        distinct_models = {n.get("model") for n in nodes if n.get("model")}
        if len(distinct_models) < 2:
            ctx.emit_event("Debate", status="success", message="single distinct model; debate skipped")
            return nodes

        # Seed self-confidence from the critic score so round-1 pruning is meaningful.
        for n in nodes:
            if not n.get("confidence"):
                n["confidence"] = max(0.0, min(1.0, float(n.get("score", 50.0)) / 100.0))

        seed = list(nodes)  # ultimate fallback; never return empty
        deadline = _ctx_now(ctx) + float(self.timeout_s)
        max_calls = max(1, self.num_candidates * self.debate_rounds)
        calls = 0
        for r in range(1, self.debate_rounds + 1):
            if self._cancelled(ctx):
                ctx.emit_event("Debate", status="warning", message="cancelled; stopping debate")
                break
            remaining = deadline - _ctx_now(ctx)
            if self.timeout_s > 0 and _ctx_now(ctx) and remaining <= 0:
                ctx.emit_event("Debate", status="warning", message="debate time budget exhausted")
                break
            if self._converged(nodes):
                ctx.emit_event("Debate", status="success", message=f"consensus reached before R{r}; debate short-circuited")
                break
            if calls >= max_calls:
                ctx.emit_event("Debate", status="warning", message="debate call ceiling reached")
                break
            ctx.emit_event("Debate", status="running", message=f"round {r}: {len(nodes)} model(s) revising")
            jobs = [self._revise_job(node, [o for o in nodes if o is not node], r) for node in nodes]
            # Round-scoped timeout so the whole debate can't exceed the budget.
            round_timeout = float(self.timeout_s)
            if self.timeout_s > 0 and _ctx_now(ctx):
                round_timeout = max(1.0, min(round_timeout, remaining))
            round_ctx = replace(ctx, timeout_s=round_timeout)
            results = run_parallel(round_ctx, jobs)
            calls += len(jobs)
            nodes = self._apply_revisions(nodes, results)
            nodes = self._prune(ctx, nodes, r)
            if len(nodes) <= 1:
                ctx.emit_event("Debate", status="success", message=f"converged to {len(nodes)} leader after R{r}")
                break
        if not nodes:
            ctx.emit_event("Debate", status="warning", message="all debate nodes dropped; using best initial draft")
            nodes = seed[:1]
        nodes.sort(key=lambda n: (n.get("confidence", 0.0), n.get("score", 0.0)), reverse=True)
        return nodes

    def _cancelled(self, ctx: CouncilContext) -> bool:
        ce = getattr(ctx, "cancel_event", None)
        return ce is not None and bool(getattr(ce, "is_set", lambda: False)())

    @staticmethod
    def _slot_of_label(label: str) -> int | None:
        # "debate-r2-s1" -> 1
        tail = str(label).rsplit("-s", 1)
        return int(tail[-1]) if len(tail) == 2 and tail[-1].isdigit() else None

    def _revise_job(self, node: dict[str, Any], peers: list[dict[str, Any]], r: int) -> CouncilJob:
        role = node["role"] if isinstance(node["role"], ProviderRole) else ProviderRole(str(node["role"]))
        return CouncilJob(
            role=role,
            messages=self._revise_messages(node, peers),
            label=f"debate-r{r}-s{node['slot']}",
            stage="Debate",
            metadata={
                "persona": node.get("persona", 0),
                "persona_tag": node.get("persona_tag", ""),
                "slot": node["slot"],
                "round": r,
                "activity": f"revising R{r}",
            },
        )

    def _revise_messages(self, node: dict[str, Any], peers: list[dict[str, Any]]) -> list[dict[str, str]]:
        persona = _PERSONAS[int(node.get("persona", 0)) % len(_PERSONAS)]
        own = json.dumps(node["plan"], ensure_ascii=False)[:2500]
        peer_blocks = "\n\n".join(
            f"Peer {i + 1} ({p.get('persona_tag', 'expert')}) plan:\n{json.dumps(p['plan'], ensure_ascii=False)[:1800]}"
            for i, p in enumerate(peers)
        ) or "(no peer plans)"
        user = (
            f"Your current plan (JSON):\n{own}\n\n"
            f"{peer_blocks}\n\n"
            "Improve YOUR plan by grafting the peers' best ideas and fixing the "
            "weaknesses you see. Keep what is genuinely better in yours. Return ONLY a "
            "JSON object with the SAME schema PLUS two extra top-level keys: "
            '"confidence" (a number 0-1: how confident you are this plan is the best) '
            'and "debate_rationale" (<=160 chars: what you changed and why).\n\n'
            + _PLAN_SCHEMA_HINT
        )
        return [
            {"role": "system", "content": f"You are {persona}. You are revising a biobank research plan in a multi-expert debate. Output ONLY valid JSON."},
            {"role": "user", "content": user},
        ]

    def _apply_revisions(self, nodes: list[dict[str, Any]], results: list[CouncilResult]) -> list[dict[str, Any]]:
        """Fold each round's revision back into its node. A failed/garbage round
        KEEPS the prior plan (confidence decays ×0.9) — a transient error must not
        delete a model. Never drops a node (only ``_prune`` does)."""
        by_slot = {n["slot"]: n for n in nodes}
        answered: set[int] = set()
        for result in results:
            slot = self._slot_of_label(result.label)
            node = by_slot.get(slot) if slot is not None else None
            if node is None:
                continue
            answered.add(slot)
            data = None
            if result.ok:
                try:
                    data = extract_json(result.text)
                except Exception:
                    data = None
            steps = data.get("steps") if isinstance(data, dict) else None
            if isinstance(steps, list) and any(isinstance(s, dict) for s in steps):
                node["plan"] = data
                try:
                    node["confidence"] = max(0.0, min(1.0, float(data.get("confidence", node.get("confidence", 0.5)))))
                except (TypeError, ValueError):
                    pass
                node["rationale"] = str(data.get("debate_rationale", node.get("rationale", "")))[:160]
            else:
                node["confidence"] = max(0.0, float(node.get("confidence", 0.5)) * 0.9)
        for n in nodes:
            if n["slot"] not in answered:
                n["confidence"] = max(0.0, float(n.get("confidence", 0.5)) * 0.9)
        return nodes

    def _converged(self, nodes: list[dict[str, Any]]) -> bool:
        if len(nodes) < 2:
            return True
        sims = [
            self._similarity(a["plan"], b["plan"])
            for i, a in enumerate(nodes)
            for b in nodes[i + 1:]
        ]
        if not sims:
            return False
        if all(s >= self.consensus_threshold for s in sims):
            return True
        if all(n.get("confidence", 0.0) >= 0.8 for n in nodes) and all(s >= (self.consensus_threshold - 0.1) for s in sims):
            return True
        return False

    def _prune(self, ctx: CouncilContext, nodes: list[dict[str, Any]], r: int) -> list[dict[str, Any]]:
        """Drop homogeneous (redundant) and low-confidence nodes; always keep the
        leader. Each prune emits a Debate/warning event (mapped status) so the
        decision is visible and no live row is left dangling."""
        ordered = sorted(nodes, key=lambda n: (n.get("confidence", 0.0), n.get("score", 0.0)), reverse=True)
        keep: list[dict[str, Any]] = []
        dropped: list[tuple[dict[str, Any], str]] = []
        for node in ordered:
            redundant = any(self._similarity(node["plan"], k["plan"]) >= self.consensus_threshold for k in keep)
            if redundant:
                dropped.append((node, "homogeneous with a higher-confidence peer"))
            else:
                keep.append(node)
        survivors: list[dict[str, Any]] = []
        for i, node in enumerate(keep):
            if i > 0 and float(node.get("confidence", 0.0)) < self.confidence_floor:
                dropped.append((node, f"confidence {node.get('confidence', 0.0):.2f} < floor {self.confidence_floor:.2f}"))
            else:
                survivors.append(node)
        if not survivors:
            survivors = keep[:1] or ordered[:1]
        for node, reason in dropped:
            tag = node.get("persona_tag") or node.get("model") or node["label"]
            ctx.emit_event("Debate", status="warning", message=f"R{r}: pruned {node['label']} ({tag}) — {reason}")
            logger.info("debate R%d pruned %s: %s", r, node["label"], reason)
        return survivors

    def _plan_signature(self, plan: Any) -> set[str]:
        """Structured fingerprint of a plan for similarity: step ids, title tokens,
        and tool/file/dependency sets — the execution-relevant structure, not the
        cosmetic prose. More stable than a raw SequenceMatcher ratio."""
        sig: set[str] = set()
        steps = plan.get("steps") if isinstance(plan, dict) else None
        for s in steps or []:
            if not isinstance(s, dict):
                continue
            sid = str(s.get("id", "")).strip().lower()
            if sid:
                sig.add(f"id:{sid}")
            for w in re.split(r"\W+", str(s.get("title", "")).lower()):
                if len(w) > 2:
                    sig.add(f"w:{w}")
            for t in _as_str_list(s.get("tool_scope")):
                sig.add(f"t:{t.lower()}")
            for f in _as_str_list(s.get("file_scope")):
                sig.add(f"f:{f.lower()}")
            for d in _as_str_list(s.get("dependencies")):
                sig.add(f"d:{d.lower()}")
        return sig

    def _similarity(self, a: Any, b: Any) -> float:
        sa, sb = self._plan_signature(a), self._plan_signature(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    # -------------------------------------------------------------- stage: merge
    def _merge(
        self,
        ctx: CouncilContext,
        objective: str,
        context: str,
        refinement: str,
        tools: list[str],
        nodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        best = nodes[0]["plan"]
        if len(nodes) == 1:
            return best
        ctx.emit_event("Merge", status="running", message="synthesizing the strongest plan")
        results = run_parallel(
            ctx,
            [CouncilJob(role=ProviderRole.SUMMARIZER, messages=self._merge_messages(objective, context, refinement, tools, nodes), label="merge", stage="Merge")],
        )
        if results and results[0].ok:
            try:
                merged = extract_json(results[0].text)
                merged_steps = merged.get("steps") if isinstance(merged, dict) else None
                if isinstance(merged_steps, list) and any(isinstance(s, dict) for s in merged_steps):
                    ctx.emit_event("Merge", status="success", message="merged plan synthesized")
                    return merged
            except Exception:
                logger.debug("merge parse failed; using best candidate", exc_info=True)
        ctx.emit_event("Merge", status="success", message="merge unavailable; using top-scored candidate")
        return best

    # --------------------------------------------------------------- prompt bits
    def _candidate_messages(self, objective: str, context: str, refinement: str, tools: list[str], persona: str) -> list[dict[str, str]]:
        tool_line = ", ".join(tools[:80]) if tools else "(no tool registry provided)"
        context_block = f"Context:\n{context[:1500]}\n" if context else ""
        refine_block = f"Requested refinement: {refinement[:600]}\n" if refinement else ""
        user = (
            f"Objective: {objective}\n"
            f"{context_block}"
            f"{refine_block}\n"
            f"Available tools: {tool_line}\n\n"
            f"{_PLAN_SCHEMA_HINT}"
        )
        return [
            {"role": "system", "content": f"You are {persona}. You are a biobank research planner. Output ONLY valid JSON."},
            {"role": "user", "content": user},
        ]

    def _critic_messages(self, objective: str, candidate: dict[str, Any]) -> list[dict[str, str]]:
        user = (
            f"Objective: {objective}\n\n"
            f"Candidate plan (JSON):\n{json.dumps(candidate, ensure_ascii=False)[:4000]}\n\n"
            "Critique this plan for correctness, completeness, methodology, and "
            "feasibility. Return ONLY JSON: "
            '{"score": <0-100>, "strengths": ["..."], "weaknesses": ["..."], "missing": ["..."]}'
        )
        return [
            {"role": "system", "content": "You are a rigorous scientific plan reviewer. Output only valid JSON."},
            {"role": "user", "content": user},
        ]

    def _merge_messages(self, objective: str, context: str, refinement: str, tools: list[str], nodes: list[dict[str, Any]]) -> list[dict[str, str]]:
        blocks = []
        for i, node in enumerate(nodes, 1):
            meta = f"critic {float(node.get('score', 0.0)):.0f}"
            if node.get("confidence"):
                meta += f", self-confidence {float(node['confidence']):.2f}"
            if node.get("rationale"):
                meta += f", rationale: {node['rationale']}"
            blocks.append(f"Candidate {i} ({meta}):\n{json.dumps(node['plan'], ensure_ascii=False)[:2500]}")
        tool_line = ", ".join(tools[:80]) if tools else "(no tool registry provided)"
        context_block = f"Context:\n{context[:1000]}\n" if context else ""
        refine_block = f"Requested refinement: {refinement[:600]}\n" if refinement else ""
        user = (
            f"Objective: {objective}\n"
            f"{context_block}"
            f"{refine_block}\n"
            f"Available tools: {tool_line}\n\n"
            + "\n\n".join(blocks)
            + "\n\nThese plans survived a multi-expert debate. Synthesize the single "
            "strongest plan, grafting the best ideas (favoring higher-confidence "
            "candidates) and fixing the weaknesses the critics found.\n\n"
            + _PLAN_SCHEMA_HINT
        )
        return [
            {"role": "system", "content": "You are a senior research lead merging plan proposals. Output ONLY valid JSON."},
            {"role": "user", "content": user},
        ]

    # ------------------------------------------------------- adversarial council
    def _adversarial_plan_pipeline(
        self, ctx: CouncilContext, objective: str, context: str, refinement: str, tools: list[str]
    ) -> dict[str, Any]:
        """Adversarial-game planning: a proposer drafts a plan, a red-team attacks it, and a
        referee folds the load-bearing flaws into a revised plan — each bound to a distinct council
        model. Returns the final plan dict for ``_plan_state_from_json``; raises ``CouncilError``
        (caught by ``build_plan``, which then degrades to a scaffold) if no plan ever parses."""
        first_draft: dict[str, str] = {"text": ""}

        # Each role goes through run_parallel (even for one job) so the as_completed
        # timeout in council.py bounds a hung provider — run_one would not.
        def proposer_fn(obj: str) -> str:
            res = run_parallel(ctx, [CouncilJob(
                role=ProviderRole.PLANNER,
                messages=self._candidate_messages(obj, context, refinement, tools, _PERSONAS[0]),
                label="adversarial-proposer", stage="Adversarial")])[0]
            if not res.ok:
                raise CouncilError(f"adversarial proposer produced no plan: {res.error or 'empty response'}")
            first_draft["text"] = res.text
            return res.text

        def redteam_fn(obj: str, draft: str) -> list[dict[str, Any]]:
            res = run_parallel(ctx, [CouncilJob(
                role=ProviderRole.CRITIC, messages=self._redteam_messages(obj, draft),
                label="adversarial-redteam", stage="Adversarial")])[0]
            if not res.ok:
                return []
            try:
                data = extract_json(res.text)
            except ValueError:
                return []
            flaws = data.get("flaws") if isinstance(data, dict) else data
            return [f for f in flaws if isinstance(f, dict)] if isinstance(flaws, list) else []

        def referee_fn(obj: str, draft: str, flaws: list[dict[str, str]]) -> dict[str, Any]:
            res = run_parallel(ctx, [CouncilJob(
                role=ProviderRole.SUMMARIZER, messages=self._referee_messages(obj, draft, flaws),
                label="adversarial-referee", stage="Adversarial")])[0]
            if not res.ok:
                return {"plan": draft, "accepted": [], "dismissed": list(flaws), "notes": "referee unavailable"}
            try:
                verdict = extract_json(res.text)
            except ValueError:
                return {"plan": draft, "accepted": [], "dismissed": [], "notes": "referee returned no JSON"}
            if not isinstance(verdict, dict):
                return {"plan": draft, "accepted": [], "dismissed": [], "notes": ""}
            # Keep the running draft a JSON STRING: adversarial_plan does ``str(verdict["plan"])``,
            # so a dict/list plan would become an unparseable Python repr next round.
            plan_val = verdict.get("plan")
            if isinstance(plan_val, (dict, list)):
                verdict["plan"] = json.dumps(plan_val, ensure_ascii=False)
            return verdict

        ctx.emit_event("Adversarial", status="running",
                       message="proposer drafting; red-team + referee to adjudicate")
        out = adversarial_plan(objective, proposer_fn=proposer_fn, redteam_fn=redteam_fn,
                               referee_fn=referee_fn, rounds=self.debate_rounds)
        ctx.emit_event("Adversarial", status="success",
                       message=(f"converged after R{out.rounds}; red-team "
                                f"{out.game_score['redteam']} / proposer {out.game_score['proposer']}"),
                       game_score=out.game_score, rounds=out.rounds,
                       addressed=[f.to_dict() for f in out.addressed])
        # Prefer the adjudicated plan; fall back to the proposer's first draft. Require a
        # non-empty steps list so a stepless verdict cannot slip past _plan_state_from_json.
        for candidate in (out.plan, first_draft["text"]):
            try:
                merged = extract_json(candidate)
            except ValueError:
                continue
            if isinstance(merged, dict) and merged.get("steps"):
                return merged
        raise CouncilError("adversarial council produced no parseable plan with steps")

    def _redteam_messages(self, objective: str, draft: str) -> list[dict[str, str]]:
        user = (
            f"Objective: {objective}\n\n"
            f"Proposed plan (JSON):\n{str(draft)[:4000]}\n\n"
            "You are an adversarial red-team. Attack this plan: find missing data or cohort "
            "definitions, methodology sins (uncorrected multiple testing, data leakage, missing "
            "covariates, underpowered groups, causal overreach), infeasible or mis-ordered steps, "
            "and hidden assumptions. Severity 'block' invalidates the result; 'major' is serious but "
            "fixable; 'minor' is a nitpick. Return ONLY JSON: "
            '{"flaws": [{"severity": "block"|"major"|"minor", "kind": "<short tag>", '
            '"claim": "what is wrong", "fix": "the concrete correction"}]}'
        )
        return [
            {"role": "system", "content": "You are a rigorous adversarial red-team for biobank research plans. Output only valid JSON."},
            {"role": "user", "content": user},
        ]

    def _referee_messages(self, objective: str, draft: str, flaws: list[dict[str, Any]]) -> list[dict[str, str]]:
        user = (
            f"Objective: {objective}\n\n"
            f"Current plan (JSON):\n{str(draft)[:3500]}\n\n"
            f"Red-team flaws:\n{json.dumps(flaws, ensure_ascii=False)[:2500]}\n\n"
            "You are the referee. Accept the flaws that are genuinely load-bearing (block/major — "
            "they would invalidate or block the result) and dismiss nitpicks or misunderstandings. "
            "Fold every accepted fix into a fully revised plan that still follows the schema below. "
            'Return ONLY JSON: {"plan": <full revised plan object>, "accepted": [<accepted flaw '
            'objects>], "dismissed": [<dismissed flaw objects>], "notes": "<one line>"}\n\n'
            f"{_PLAN_SCHEMA_HINT}"
        )
        return [
            {"role": "system", "content": "You are an impartial referee adjudicating a plan red-team. Output only valid JSON."},
            {"role": "user", "content": user},
        ]

    # ------------------------------------------------------------- parse/validate
    @staticmethod
    def _repair_json_object(text: str) -> dict[str, Any] | None:
        """Best-effort salvage of a JSON object from a noisy LLM response.
        Trims to the outermost brace pair, balances a truncated tail, and drops
        trailing commas. Returns the parsed dict or None when unrecoverable."""
        import json
        import re as _re

        if not text:
            return None
        s = str(text)
        start = s.find("{")
        if start == -1:
            return None
        end = s.rfind("}")
        raw_candidates: list[str] = []
        if end > start:
            raw_candidates.append(s[start:end + 1])  # complete object, trailing prose stripped
        raw_candidates.append(s[start:])             # truncated tail (balance below)
        attempts: list[str] = []
        for snippet in raw_candidates:
            opens = snippet.count("{") - snippet.count("}")
            if opens > 0:
                snippet += "}" * opens
            snippet = _re.sub(r",\s*([}\]])", r"\1", snippet)  # drop trailing commas
            attempts.append(snippet)
            attempts.append(snippet.replace("'", '"'))
        for candidate in attempts:
            try:
                obj = json.loads(candidate)
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
        return None

    def _parse_candidates(self, results: list[CouncilResult], jobs: list[CouncilJob]) -> list[dict[str, Any]]:
        """Parse each successful candidate into a debate NODE carrying the plan
        plus its provenance (slot / role / model / persona). Matched back to the
        source job by label — completion order != dispatch order, so never trust
        the positional index (Codex's misattribution guard)."""
        by_label = {job.label: job for job in jobs}
        nodes: list[dict[str, Any]] = []
        for result in results:
            if not result.ok:
                continue
            try:
                data = extract_json(result.text)
            except Exception:
                data = self._repair_json_object(result.text)  # salvage malformed JSON
                if data is None:
                    logger.debug("candidate %s did not parse as JSON (even after repair)",
                                 result.label, exc_info=True)
                    continue
            steps = data.get("steps") if isinstance(data, dict) else None
            if not (isinstance(steps, list) and any(isinstance(s, dict) for s in steps)):
                continue
            job = by_label.get(result.label)
            meta = (job.metadata if job else {}) or {}
            nodes.append({
                "plan": data,
                "slot": int(meta.get("slot", len(nodes))),
                "role": job.role if job else ProviderRole.PLANNER,
                "model": result.model or "",
                "persona": int(meta.get("persona", 0)),
                "persona_tag": str(meta.get("persona_tag", "")),
                "label": result.label,
                "score": 0.0,
                "confidence": 0.0,
                "rationale": "",
            })
        return nodes

    def _plan_state_from_json(self, data: dict[str, Any], objective: str, *, previous: PlanState | None,
                              refinement: str, available_tools: list[str] | None = None) -> PlanState:
        if not isinstance(data, dict):
            raise CouncilError("planner returned a non-object plan")
        steps_raw = data.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            raise CouncilError("planner returned no valid steps list")
        steps: list[PlanStep] = []
        seen_ids: set[str] = set()
        dropped_tools: set[str] = set()
        known_tools = set(available_tools or [])
        for i, item in enumerate(steps_raw):
            if not isinstance(item, dict):
                continue
            sid = str(item.get("id") or f"s{i + 1}").strip() or f"s{i + 1}"
            base = sid
            n = 2
            while sid in seen_ids:
                sid = f"{base}_{n}"
                n += 1
            seen_ids.add(sid)
            steps.append(
                PlanStep(
                    id=sid,
                    title=str(item.get("title") or sid),
                    purpose=str(item.get("purpose") or ""),
                    dependencies=_as_str_list(item.get("dependencies")),
                    tool_scope=_as_str_list(item.get("tool_scope")),
                    file_scope=_as_str_list(item.get("file_scope")),
                    verification=_as_str_list(item.get("verification")),
                    risks=_as_str_list(item.get("risks")),
                )
            )
        # Drop tool_scope entries that name tools which are not actually available
        # (a council can hallucinate skill names) so the plan stays honest (issue #5).
        if known_tools:
            for step in steps:
                kept = [t for t in step.tool_scope if t in known_tools]
                dropped_tools.update(t for t in step.tool_scope if t not in known_tools)
                step.tool_scope = kept
        revision = (previous.revision + 1) if previous else 1
        title = str(data.get("title") or "").strip() or objective[:80]
        plan = PlanState(
            objective=objective,
            title=title,
            status=PlanStatus.DRAFT,
            summary=str(data.get("summary") or f"Council plan for: {objective}"),
            context_gathering=[
                "Council-drafted plan from parallel candidate proposals, critique, and merge.",
                "Read-only context gathering precedes any workspace mutation.",
            ],
            steps=steps,
            risks=_as_str_list(data.get("risks")),
            verification_plan=_as_str_list(data.get("verification_plan")),
            required_approvals=_as_str_list(data.get("required_approvals"))
            or ["User approval before mutating files or running shell commands."],
            proposed_tool_scope=_as_str_list(data.get("proposed_tool_scope"))
            or sorted({t for step in steps for t in step.tool_scope}),
            proposed_file_scope=_as_str_list(data.get("proposed_file_scope"))
            or sorted({f for step in steps for f in step.file_scope}),
            open_questions=_as_str_list(data.get("open_questions")),
            audit_summary="Plan generated by the multi-agent council; only final plan state is stored.",
            revision=revision,
        )
        if refinement:
            plan.risks.append(f"Refinement requested by user: {refinement[:240]}")
        if dropped_tools:
            note = f"Planner referenced unavailable tools (ignored): {', '.join(sorted(dropped_tools))}"
            plan.risks.append(note)
            plan.open_questions.append(note)
        return plan

    @staticmethod
    def _validate_plan(plan: PlanState) -> None:
        if not plan.steps:
            raise CouncilError("merged plan has no steps")
        ids = {step.id for step in plan.steps}
        for step in plan.steps:
            for dep in step.dependencies:
                if dep not in ids:
                    # Drop dangling dependencies rather than failing the whole plan.
                    step.dependencies = [d for d in step.dependencies if d in ids]
