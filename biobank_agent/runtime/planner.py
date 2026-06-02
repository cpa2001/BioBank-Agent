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
# DIFFERENT model (model_for_role: PLANNER=kimi, PRIMARY_EXECUTOR=deepseek,
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

        # 1) Clarification gate (before decomposition).
        context = self._clarify(ctx, clean_objective, context, clarifier, interactive, open_questions)

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
        plan = self._plan_state_from_json(merged, clean_objective, previous=previous, refinement=refinement)
        for question in open_questions:
            if question not in plan.open_questions:
                plan.open_questions.append(question)
        self._validate_plan(plan)
        ctx.emit_event("Validation", status="success", message=f"{len(plan.steps)} steps validated")
        ctx.emit_event("Review", status="success", message="plan ready for review")
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
        cosmetic prose. (Codex-recommended over raw SequenceMatcher.)"""
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

    # ------------------------------------------------------------- parse/validate
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
                logger.debug("candidate %s did not parse as JSON", result.label, exc_info=True)
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

    def _plan_state_from_json(self, data: dict[str, Any], objective: str, *, previous: PlanState | None, refinement: str) -> PlanState:
        if not isinstance(data, dict):
            raise CouncilError("planner returned a non-object plan")
        steps_raw = data.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            raise CouncilError("planner returned no valid steps list")
        steps: list[PlanStep] = []
        seen_ids: set[str] = set()
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
