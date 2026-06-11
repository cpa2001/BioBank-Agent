"""Multi-model orchestrator — routes hard tasks to collaborative model pools.

Upgrades:
  - Returns OrchestrationResult (claims, evidence links, debate trace, safety)
  - Supports MAS-v2 style red/blue debate with anonymous voting
  - Enforces execution-grounded safety checks before finalising output
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any

from .complexity import Strategy, classify_complexity
from .llm import LLMClient, LLMResponse, ToolCall

logger = logging.getLogger(__name__)


class SubagentRole(Enum):
    """Role-based subagent specialization (leaf agents — never re-delegate)."""

    EXPLORER = "explorer"
    WORKER = "worker"
    VERIFIER = "verifier"


@dataclass
class SubagentCall:
    """A structured subagent dispatch request."""

    role: SubagentRole
    task: str
    scope: list[str] = field(default_factory=list)
    owned_files: list[str] = field(default_factory=list)
    off_limits: list[str] = field(default_factory=list)
    output_format: str = "text"


_ROLE_SYSTEM_PROMPTS = {
    "explorer": (
        "You are an Explorer agent — a read-only search specialist. "
        "You MUST NOT modify any files or data. You MUST NOT spawn sub-agents. "
        "Return only concrete findings with file:line references. Be terse."
    ),
    "worker": (
        "You are a Worker agent — a one-shot execution specialist. "
        "Execute the provided specification exactly. Run the narrowest validation test. "
        "If spec is ambiguous, STOP and report the gap. You MUST NOT spawn sub-agents."
    ),
    "verifier": (
        "You are a Verifier agent — an adversarial reviewer. "
        "Assume every change has a bug until proven otherwise. "
        "Check correctness, boundaries, null safety, and type conversions. "
        "Report structured PASS/FAIL/PARTIAL verdict. You MUST NOT modify files."
    ),
}


@dataclass
class ModelSpec:
    """A model available in the pool."""

    model_id: str
    role: str = "generalist"
    strengths: list[str] = field(default_factory=list)
    priority: int = 0


@dataclass
class DebateProposal:
    """One model's proposal in a debate round."""

    model_id: str
    response: LLMResponse
    score: float = 0.0
    critique: str = ""


@dataclass
class Claim:
    """Atomic claim extracted from orchestration output."""

    claim_id: str
    text: str
    confidence: float = 0.0
    source_model: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class EvidenceLink:
    """Evidence item linked to a claim."""

    claim_id: str
    evidence_type: str
    evidence_id: str
    relation: str = "supports"
    score: float = 0.0
    source_model: str = ""
    snippet: str = ""


@dataclass
class OrchestrationResult:
    """Structured orchestrator return payload."""

    final_answer: str = ""
    claims: list[Claim] = field(default_factory=list)
    evidence_links: list[EvidenceLink] = field(default_factory=list)
    debate_trace: dict[str, Any] = field(default_factory=dict)
    safety_status: str = "PASS"
    llm_response: LLMResponse = field(default_factory=LLMResponse)

    @property
    def text(self) -> str:
        return self.final_answer or self.llm_response.text

    @text.setter
    def text(self, value: str) -> None:
        self.final_answer = value
        self.llm_response.text = value

    @property
    def tool_calls(self):
        return self.llm_response.tool_calls

    @property
    def has_tool_calls(self) -> bool:
        return self.llm_response.has_tool_calls

    @property
    def usage(self) -> dict:
        return self.llm_response.usage

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_answer": self.text,
            "claims": [c.__dict__ for c in self.claims],
            "evidence_links": [e.__dict__ for e in self.evidence_links],
            "debate_trace": self.debate_trace,
            "safety_status": self.safety_status,
        }


class MultiModelOrchestrator:
    """Routes tasks to one or more models based on complexity."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str,
        model_pool: list[ModelSpec] | None = None,
        complexity_threshold: float = 0.7,
        debate_rounds: int = 2,
        max_workers: int = 3,
        parallel_timeout_s: float = 90.0,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.default_model = default_model
        self.model_pool = model_pool or [ModelSpec(default_model, "generalist")]
        self.complexity_threshold = complexity_threshold
        self.debate_rounds = debate_rounds
        self.max_workers = max_workers
        self.parallel_timeout_s = parallel_timeout_s
        self._clients: dict[str, LLMClient] = {}
        self._clients_lock = Lock()
        self.last_result: OrchestrationResult | None = None

    def get_client(self, model_id: str) -> LLMClient:
        """Get or create an LLMClient for a specific model (thread-safe)."""

        with self._clients_lock:
            if model_id not in self._clients:
                self._clients[model_id] = LLMClient(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    model=model_id,
                )
            return self._clients[model_id]

    @property
    def default_client(self) -> LLMClient:
        return self.get_client(self.default_model)

    def route(
        self,
        query: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        records: list | None = None,
        force_strategy: Strategy | None = None,
    ) -> OrchestrationResult:
        """Classify and route a query to the appropriate strategy."""

        if force_strategy:
            strategy = force_strategy
            logger.info("Forced strategy: %s", strategy.value)
        else:
            result = classify_complexity(query, records, self.complexity_threshold)
            strategy = result.strategy
            logger.info(
                "Complexity score=%.2f strategy=%s reason=%s",
                result.score,
                strategy.value,
                result.reason,
            )

        records = records or []
        out = self._run_strategy(
            strategy=strategy,
            query=query,
            messages=messages,
            tools=tools,
            records=records,
        )
        self._attach_execution_evidence(out, records=records)
        self._apply_execution_judge(out, records=records)
        if out.safety_status != "PASS" and not out.has_tool_calls:
            out = self._rollback_on_gate_fail(
                failed_strategy=strategy,
                query=query,
                messages=messages,
                tools=tools,
                records=records,
                failed_result=out,
            )
        self.last_result = out
        return out

    def _run_strategy(
        self,
        strategy: Strategy,
        query: str,
        messages: list[dict],
        tools: list[dict] | None,
        records: list,
    ) -> OrchestrationResult:
        """Execute one specific strategy without fallback logic."""
        if strategy == Strategy.SINGLE or len(self.model_pool) < 2:
            raw = self.default_client.chat(messages, tools)
            return self._wrap_llm_response(raw, strategy=Strategy.SINGLE, model_id=self.default_model)
        if strategy == Strategy.DEBATE:
            return self.debate(query=query, messages=messages, tools=tools, records=records)
        if strategy == Strategy.SUPERVISOR:
            return self.supervisor(query=query, messages=messages, tools=tools)
        if strategy == Strategy.ENSEMBLE:
            return self.ensemble(messages=messages, tools=tools)
        raw = self.default_client.chat(messages, tools)
        return self._wrap_llm_response(raw, strategy=Strategy.SINGLE, model_id=self.default_model)

    def _rollback_on_gate_fail(
        self,
        failed_strategy: Strategy,
        query: str,
        messages: list[dict],
        tools: list[dict] | None,
        records: list,
        failed_result: OrchestrationResult,
    ) -> OrchestrationResult:
        """Auto rollback strategy when safety gate fails."""
        rollback_candidates: list[Strategy] = []
        if len(self.model_pool) > 1 and failed_strategy != Strategy.ENSEMBLE:
            rollback_candidates.append(Strategy.ENSEMBLE)
        if failed_strategy != Strategy.SINGLE:
            rollback_candidates.append(Strategy.SINGLE)
        if not rollback_candidates:
            return failed_result

        history = [failed_strategy.value]
        best = failed_result
        for candidate in rollback_candidates:
            try:
                out = self._run_strategy(
                    strategy=candidate,
                    query=query,
                    messages=messages,
                    tools=tools,
                    records=records,
                )
                self._attach_execution_evidence(out, records=records)
                self._apply_execution_judge(out, records=records)
                trace = out.debate_trace if isinstance(out.debate_trace, dict) else {}
                trace["rollback_from"] = failed_strategy.value
                trace["rollback_chain"] = history + [candidate.value]
                out.debate_trace = trace
                history.append(candidate.value)
                best = out
                if out.safety_status == "PASS" or out.has_tool_calls:
                    return out
            except Exception as e:
                logger.warning("Rollback strategy %s failed: %s", candidate.value, e)
        return best

    def _wrap_llm_response(
        self,
        raw: LLMResponse,
        strategy: Strategy,
        model_id: str,
        claims: list[Claim] | None = None,
        evidence_links: list[EvidenceLink] | None = None,
        debate_trace: dict[str, Any] | None = None,
        safety_status: str = "PASS",
    ) -> OrchestrationResult:
        claims = claims or self._extract_claims(raw.text, source_model=model_id)
        return OrchestrationResult(
            final_answer=raw.text or "",
            claims=claims,
            evidence_links=evidence_links or [],
            debate_trace=debate_trace or {"strategy": strategy.value},
            safety_status=safety_status,
            llm_response=raw,
        )

    def _extract_claims(self, text: str, source_model: str = "") -> list[Claim]:
        """Best-effort claim extraction from free text."""

        if not text:
            return []
        cleaned = re.sub(r"\s+", " ", text).strip()
        parts = re.split(r"(?<=[.!?。！？])\s+", cleaned)
        claims: list[Claim] = []
        for sentence in parts:
            s = sentence.strip()
            if len(s) < 30 or len(s) > 400:
                continue
            cid = "claim_" + hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
            claims.append(Claim(
                claim_id=cid,
                text=s,
                confidence=0.55,
                source_model=source_model,
            ))
            if len(claims) >= 5:
                break
        return claims

    def _evidence_from_records(self, claims: list[Claim], records: list) -> list[EvidenceLink]:
        """Create lightweight claim→execution evidence links from past records."""

        if not claims or not records:
            return []
        evidence: list[EvidenceLink] = []
        recent = records[-12:]
        adjudication_skills = {
            "statistical_review",
            "safety_check",
            "web_search",
            "field_search",
            "recall_session",
            "read_paper",
            "fetch_paper",
            "web_fetch",
        }
        for claim in claims:
            claim_terms = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", claim.text)}
            for rec in recent:
                key_results = getattr(rec, "key_results", {}) or {}
                if "error" in key_results:
                    continue
                if getattr(rec, "skill", "") in adjudication_skills:
                    rec_id = f"{getattr(rec, 'timestamp', 'na')}:{getattr(rec, 'skill', 'unknown')}"
                    evidence.append(EvidenceLink(
                        claim_id=claim.claim_id,
                        evidence_type="execution_record",
                        evidence_id=rec_id,
                        relation="adjudicates",
                        score=0.45,
                        snippet=json.dumps(key_results, default=str)[:220],
                    ))
                    if len([e for e in evidence if e.claim_id == claim.claim_id]) >= 2:
                        break
                    continue
                blob = " ".join([
                    getattr(rec, "skill", ""),
                    json.dumps(getattr(rec, "args", {}), default=str),
                    json.dumps(key_results, default=str),
                ]).lower()
                overlap = sum(1 for t in claim_terms if t in blob)
                if overlap == 0:
                    continue
                rec_id = f"{getattr(rec, 'timestamp', 'na')}:{getattr(rec, 'skill', 'unknown')}"
                evidence.append(EvidenceLink(
                    claim_id=claim.claim_id,
                    evidence_type="execution_record",
                    evidence_id=rec_id,
                    relation="supports",
                    score=min(1.0, 0.2 + 0.1 * overlap),
                    snippet=blob[:220],
                ))
                if len([e for e in evidence if e.claim_id == claim.claim_id]) >= 2:
                    break
        return evidence

    @staticmethod
    def _is_execution_evidence_link(link: EvidenceLink) -> bool:
        et = str(getattr(link, "evidence_type", "") or "").lower()
        return et.startswith("execution") or et in {"tool_execution", "analysis_record"}

    def _has_execution_evidence(self, result: OrchestrationResult) -> bool:
        return any(self._is_execution_evidence_link(e) for e in result.evidence_links)

    def _attach_execution_evidence(self, result: OrchestrationResult, records: list) -> None:
        """Attach execution-grounded evidence links from tool records."""

        if not result.claims or not records:
            return
        new_links = self._evidence_from_records(result.claims, records)
        if not new_links:
            return

        dedup: set[tuple[str, str, str, str]] = set()
        merged: list[EvidenceLink] = []
        for link in [*result.evidence_links, *new_links]:
            key = (link.claim_id, link.evidence_type, link.evidence_id, link.relation)
            if key in dedup:
                continue
            dedup.add(key)
            merged.append(link)
        result.evidence_links = merged

    def _apply_execution_judge(self, result: OrchestrationResult, records: list) -> None:
        """Final safety gate before returning output to the agent loop."""

        reasons: list[str] = []
        text_lower = result.text.lower()
        strategy = str(result.debate_trace.get("strategy", "single")).lower()
        has_execution_evidence = self._has_execution_evidence(result)
        high_risk_assertion = any(
            marker in text_lower for marker in (
                "p <",
                "p=",
                "95% ci",
                "confidence interval",
                "odds ratio",
                "hazard ratio",
                "significant",
                "causal",
                "causality",
                "leads to",
                "reduces risk",
                "increases risk",
            )
        )
        scientific_conclusion_like = high_risk_assertion or any(
            marker in text_lower for marker in (
                "we conclude",
                "our conclusion",
                "therefore",
                "evidence suggests",
                "associated with",
                "显著",
                "结论",
            )
        )
        evidence_mandatory = (
            strategy in {"debate", "supervisor", "ensemble"}
            or high_risk_assertion
            or bool(result.debate_trace.get("disagreement"))
        )
        if evidence_mandatory and result.claims and not has_execution_evidence:
            reasons.append("No execution-grounded evidence attached for current scientific claims.")

        low_sample_hits = 0
        for rec in records[-20:]:
            kr = getattr(rec, "key_results", {}) or {}
            n_cases = kr.get("n_cases")
            if isinstance(n_cases, (int, float)) and n_cases < 100:
                low_sample_hits += 1
        if low_sample_hits > 0:
            reasons.append(f"{low_sample_hits} result(s) violate n_cases >= 100 guardrail.")

        causal_like = any(k in text_lower for k in ("causal", "causality", "prove that", "leads to"))
        confounder_mentions = any(k in text_lower for k in ("confound", "age-adjusted", "sex-adjusted", "covariate"))
        if causal_like and not confounder_mentions:
            reasons.append("Causal wording detected without confounder adjustment evidence.")

        if result.debate_trace.get("disagreement") and not has_execution_evidence:
            reasons.append("Model disagreement unresolved by tool-grounded evidence.")

        if reasons:
            result.safety_status = "PARTIAL"
            hard_block = evidence_mandatory and not has_execution_evidence and scientific_conclusion_like
            if hard_block:
                tool_hint = ""
                if result.has_tool_calls:
                    names = ", ".join(tc.name for tc in result.tool_calls[:4])
                    tool_hint = f"\nPending adjudication tools: {names}"
                result.text = (
                    "[Adjudication Required]\n"
                    "Scientific conclusion is blocked until execution evidence is produced and verified."
                    + tool_hint
                    + "\nRun the requested tools and re-evaluate before publishing a conclusion."
                    + "\n\n[Safety Gate: PARTIAL]\n"
                    + "\n".join(f"- {r}" for r in reasons[:5])
                )
            else:
                caution = (
                    "\n\n[Safety Gate: PARTIAL]\n"
                    + "\n".join(f"- {r}" for r in reasons[:5])
                    + "\nPlease execute evidence-producing tools before final scientific conclusion."
                )
                result.text = (result.text or "") + caution

    def debate(
        self,
        query: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        records: list | None = None,
    ) -> OrchestrationResult:
        """MAS-v2 debate: propose → anonymous vote → critique → judge synth."""

        proposers = [m for m in self.model_pool if m.role != "judge"][:3]
        judge_spec = next((m for m in self.model_pool if m.role == "judge"), self.model_pool[0])

        proposals = self._parallel_chat(proposers, messages, tools)
        logger.info("Debate round 1: %d proposals collected", len(proposals))
        if len(proposals) < 2:
            raw = proposals[0].response if proposals else self.default_client.chat(messages, tools)
            return self._wrap_llm_response(raw, strategy=Strategy.DEBATE, model_id=self.default_model)

        votes = self._anonymous_initial_votes(proposers, proposals)
        vote_choices = [v.get("vote", "") for v in votes if v.get("vote")]
        disagreement = len(set(vote_choices)) > 1 if vote_choices else False
        reretrieval_queries: list[str] = []
        if disagreement:
            reretrieval_queries = self._suggest_reretrieval_queries(query, proposals)

        for i, prop in enumerate(proposals):
            other_texts = [
                f"[Proposal {j+1}]: {p.response.text[:1000]}"
                for j, p in enumerate(proposals) if j != i
            ]
            critique_prompt = (
                "You are the BLUE TEAM skeptic. Critique these alternatives rigorously.\n\n"
                + "\n\n".join(other_texts)
                + "\n\nYour own proposal was:\n"
                + prop.response.text[:1000]
                + "\n\nReturn concise critique: major flaw, missing evidence, and one fix."
            )
            try:
                client = self.get_client(prop.model_id)
                critique_resp = client.chat(
                    messages + [{"role": "user", "content": critique_prompt}],
                    max_tokens=768,
                )
                prop.critique = critique_resp.text
            except Exception as e:
                logger.warning("Critique from %s failed: %s", prop.model_id, e)
                prop.critique = "(critique unavailable)"

        judge_raw = self._judge_select(judge_spec, proposals, messages, tools)
        adjudication_forced = False
        adjudication_already_requested = self._adjudication_already_requested(messages)
        if disagreement and not judge_raw.has_tool_calls and not adjudication_already_requested:
            forced_calls = self._build_adjudication_tool_calls(
                tools=tools,
                query=query,
                reretrieval_queries=reretrieval_queries,
            )
            if forced_calls:
                judge_raw.tool_calls = forced_calls
                judge_raw.text = (
                    "[Adjudication Required]\n"
                    "Debate disagreement detected. Final scientific conclusion is deferred.\n"
                    "Execute the adjudication tools first, then re-synthesize with execution evidence."
                )
                adjudication_forced = True
        claims = self._extract_claims(judge_raw.text, source_model=judge_spec.model_id)
        evidence_links = self._evidence_from_proposals(claims, proposals)

        debate_trace = {
            "strategy": Strategy.DEBATE.value,
            "roles": ["Explorer", "Skeptic", "Executor", "Judge"],
            "anonymous_votes": votes,
            "disagreement": disagreement,
            "reretrieval_triggered": bool(reretrieval_queries),
            "reretrieval_queries": reretrieval_queries,
            "participants": [p.model_id for p in proposals],
            "judge": judge_spec.model_id,
            "forced_adjudication": adjudication_forced,
            "adjudication_already_requested": adjudication_already_requested,
        }
        safety_status = "PARTIAL" if disagreement else "PASS"
        result = self._wrap_llm_response(
            raw=judge_raw,
            strategy=Strategy.DEBATE,
            model_id=judge_spec.model_id,
            claims=claims,
            evidence_links=evidence_links,
            debate_trace=debate_trace,
            safety_status=safety_status,
        )
        return result

    def _adjudication_already_requested(self, messages: list[dict]) -> bool:
        """Detect if this conversation already requested adjudication tool execution."""

        if not messages:
            return False
        adjudication_tools = {"statistical_review", "safety_check", "web_search", "field_search", "recall_session"}
        for msg in reversed(messages[-12:]):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = str(tc.get("id", ""))
                fn = tc.get("function") or {}
                name = str(fn.get("name", tc.get("name", "")))
                if tc_id.startswith("adjudicate_") or name in adjudication_tools:
                    return True
        return False

    def _build_adjudication_tool_calls(
        self,
        tools: list[dict] | None,
        query: str,
        reretrieval_queries: list[str] | None = None,
    ) -> list[ToolCall]:
        """Build forced tool-call plan for disagreement adjudication."""

        available: set[str] = set()
        for tool in tools or []:
            try:
                if tool.get("type") == "function":
                    fn = tool.get("function") or {}
                    name = str(fn.get("name", "")).strip()
                    if name:
                        available.add(name)
            except Exception:
                continue

        if not available:
            return []

        calls: list[ToolCall] = []
        trace_id = hashlib.md5((query or "adjudicate").encode("utf-8")).hexdigest()[:8]

        def _append(name: str, args: dict[str, Any]) -> None:
            if name not in available:
                return
            calls.append(ToolCall(
                id=f"adjudicate_{trace_id}_{len(calls) + 1}",
                name=name,
                args=args,
            ))

        _append("statistical_review", {"scope": "session"})
        _append("safety_check", {"scope": "session", "k": 5})
        reretrieval_queries = reretrieval_queries or []
        search_query = reretrieval_queries[0] if reretrieval_queries else f"{query} confounder adjustment replication"
        _append("web_search", {"query": search_query[:240], "max_results": 8})
        _append("recall_session", {"query": query[:240], "limit": 5})
        _append("field_search", {"query": query[:120], "limit": 15})

        # Keep adjudication focused and bounded.
        return calls[:3]

    def _anonymous_initial_votes(
        self,
        voters: list[ModelSpec],
        proposals: list[DebateProposal],
    ) -> list[dict[str, Any]]:
        """Collect anonymous votes to detect premature consensus/sycophancy."""

        labels = [chr(ord("A") + i) for i in range(len(proposals))]
        prompt_lines = [
            "Anonymous multi-agent vote. Do NOT infer model identity.",
            "Choose the strongest proposal label and confidence.",
            "Return JSON only: {\"vote\":\"A\",\"confidence\":0.0-1.0,\"reason\":\"...\"}",
            "",
        ]
        for i, prop in enumerate(proposals):
            prompt_lines.append(f"Proposal {labels[i]}:\n{prop.response.text[:1500]}\n")
        prompt = "\n".join(prompt_lines)

        votes: list[dict[str, Any]] = []
        for spec in voters:
            client = self.get_client(spec.model_id)
            try:
                resp = client.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                parsed = self._parse_json_loose(resp.text)
                vote = {
                    "voter_model": spec.model_id,
                    "vote": str(parsed.get("vote", "")).strip().upper()[:1],
                    "confidence": float(parsed.get("confidence", 0.5)),
                    "reason": str(parsed.get("reason", ""))[:240],
                }
            except Exception:
                vote = {"voter_model": spec.model_id, "vote": "", "confidence": 0.0, "reason": "vote_unavailable"}
            votes.append(vote)
        return votes

    def _suggest_reretrieval_queries(self, query: str, proposals: list[DebateProposal]) -> list[str]:
        """Generate retrieval queries when debate disagreement is detected."""

        prompt = (
            "These proposals disagree. Generate up to 3 retrieval queries that can falsify or validate the disagreement. "
            "Return JSON array of strings only.\n\n"
            f"User query: {query}\n\n"
        )
        for i, prop in enumerate(proposals, 1):
            prompt += f"Proposal {i}: {prop.response.text[:800]}\n\n"

        try:
            resp = self.default_client.chat([{"role": "user", "content": prompt}], max_tokens=256)
            arr = self._parse_json_loose(resp.text)
            if isinstance(arr, list):
                return [str(x)[:200] for x in arr if str(x).strip()][:3]
        except Exception:
            pass
        return [f"{query} validation study", f"{query} confounder adjustment", f"{query} replication cohort"]

    def _parallel_chat(
        self,
        models: list[ModelSpec],
        messages: list[dict],
        tools: list[dict] | None,
    ) -> list[DebateProposal]:
        """Call multiple models in parallel and collect their responses."""

        proposals: list[DebateProposal] = []
        pool = ThreadPoolExecutor(max_workers=self.max_workers)
        future_to_model = {}
        for spec in models:
            client = self.get_client(spec.model_id)
            future = pool.submit(client.chat, messages, tools)
            future_to_model[future] = spec
        try:
            for future in as_completed(future_to_model, timeout=self.parallel_timeout_s):
                spec = future_to_model[future]
                try:
                    resp = future.result()
                    proposals.append(DebateProposal(model_id=spec.model_id, response=resp))
                except Exception as e:
                    logger.warning("Model %s failed in debate: %s", spec.model_id, e)
        except FuturesTimeoutError:
            logger.warning("Parallel chat timeout after %.1fs; collecting partial proposals.", self.parallel_timeout_s)
        finally:
            for future, spec in future_to_model.items():
                if not future.done():
                    future.cancel()
                    logger.warning("Cancelled slow model response: %s", spec.model_id)
            pool.shutdown(wait=False, cancel_futures=True)
        proposals.sort(
            key=lambda p: next((m.priority for m in self.model_pool if m.model_id == p.model_id), 0),
            reverse=True,
        )
        return proposals

    def _judge_select(
        self,
        judge_spec: ModelSpec,
        proposals: list[DebateProposal],
        messages: list[dict],
        tools: list[dict] | None,
    ) -> LLMResponse:
        """Judge synthesises the best response from debate proposals."""

        judge_prompt_parts = ["You are the final judge in a red/blue scientific debate."]
        judge_prompt_parts.append("Prioritise evidence, statistical validity, and uncertainty disclosure.\n")
        for i, p in enumerate(proposals, 1):
            judge_prompt_parts.append(f"### Proposal {i}\n{p.response.text[:1800]}\n")
            if p.critique:
                judge_prompt_parts.append(f"Critique: {p.critique[:600]}\n")
        judge_prompt_parts.append(
            "\nReturn a single integrated answer. "
            "If evidence is insufficient, state uncertainty and request concrete verification steps."
        )

        judge_messages = messages + [{"role": "user", "content": "\n".join(judge_prompt_parts)}]
        judge_client = self.get_client(judge_spec.model_id)
        result = judge_client.chat(judge_messages, tools)
        prefix = f"*[Multi-model debate: {', '.join(p.model_id for p in proposals)} → judged by {judge_spec.model_id}]*\n\n"
        result.text = prefix + (result.text or "")
        return result

    def _evidence_from_proposals(
        self,
        claims: list[Claim],
        proposals: list[DebateProposal],
    ) -> list[EvidenceLink]:
        """Create soft evidence links from debate proposal content."""

        links: list[EvidenceLink] = []
        for claim in claims:
            terms = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", claim.text)}
            if not terms:
                continue
            for prop in proposals:
                text = (prop.response.text or "").lower()
                overlap = sum(1 for t in terms if t in text)
                if overlap <= 0:
                    continue
                links.append(EvidenceLink(
                    claim_id=claim.claim_id,
                    evidence_type="debate_proposal",
                    evidence_id=f"proposal:{prop.model_id}",
                    relation="supports",
                    score=min(1.0, 0.2 + 0.1 * overlap),
                    source_model=prop.model_id,
                    snippet=(prop.response.text or "")[:220],
                ))
                if len([x for x in links if x.claim_id == claim.claim_id]) >= 2:
                    break
        return links

    def supervisor(
        self,
        query: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> OrchestrationResult:
        """Supervisor decomposes task → specialists execute → merge."""

        supervisor_client = self.get_client(self.default_model)
        decompose_prompt = (
            "Break down this complex query into 2-4 independent sub-tasks that can be solved in parallel.\n\n"
            f"Query: {query}\n\n"
            "Return JSON array only: [{\"task\":\"...\", \"focus\":\"...\"}]"
        )
        decompose_resp = supervisor_client.chat(
            [{"role": "user", "content": decompose_prompt}],
            max_tokens=768,
        )
        parsed = self._parse_json_loose(decompose_resp.text)
        if not isinstance(parsed, list) or len(parsed) < 2:
            raw = self.default_client.chat(messages, tools)
            return self._wrap_llm_response(raw, strategy=Strategy.SUPERVISOR, model_id=self.default_model)
        sub_tasks = parsed[:4]

        specialists = [m for m in self.model_pool if m.role != "judge"] or self.model_pool[:1]
        sub_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for i, st in enumerate(sub_tasks):
                spec = specialists[i % len(specialists)]
                client = self.get_client(spec.model_id)
                task_text = st.get("task", str(st)) if isinstance(st, dict) else str(st)
                focus = st.get("focus", "scientific validity") if isinstance(st, dict) else "scientific validity"
                sub_messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "You are an execution specialist. Return concise result.\n"
                            f"Sub-task: {task_text}\nFocus: {focus}"
                        ),
                    },
                ]
                futures[pool.submit(client.chat, sub_messages, tools)] = (spec.model_id, task_text, focus)
            for future in as_completed(futures):
                model_id, task_text, focus = futures[future]
                try:
                    resp = future.result(timeout=120)
                    sub_results.append({
                        "model": model_id,
                        "task": task_text,
                        "focus": focus,
                        "response": (resp.text or "")[:1800],
                    })
                except Exception as e:
                    logger.warning("Specialist %s failed: %s", model_id, e)

        merge_prompt = "Merge these specialist outputs into one coherent answer with uncertainty notes:\n\n"
        for sr in sub_results:
            merge_prompt += f"Task: {sr['task']}\nModel: {sr['model']}\n{sr['response']}\n\n"
        merge_resp = supervisor_client.chat(messages + [{"role": "user", "content": merge_prompt}], tools)
        claims = self._extract_claims(merge_resp.text, source_model=self.default_model)
        trace = {
            "strategy": Strategy.SUPERVISOR.value,
            "subtasks": sub_tasks,
            "specialist_results": [{"model": x["model"], "task": x["task"]} for x in sub_results],
        }
        return self._wrap_llm_response(
            raw=merge_resp,
            strategy=Strategy.SUPERVISOR,
            model_id=self.default_model,
            claims=claims,
            evidence_links=[],
            debate_trace=trace,
        )

    def ensemble(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> OrchestrationResult:
        """All models answer in parallel → pick best or merge."""

        proposals = self._parallel_chat(self.model_pool[:3], messages, tools)
        if not proposals:
            raw = self.default_client.chat(messages, tools)
            return self._wrap_llm_response(raw, strategy=Strategy.ENSEMBLE, model_id=self.default_model)
        if len(proposals) == 1:
            return self._wrap_llm_response(
                proposals[0].response,
                strategy=Strategy.ENSEMBLE,
                model_id=proposals[0].model_id,
            )

        tool_proposals = [p for p in proposals if p.response.has_tool_calls]
        if tool_proposals:
            chosen = tool_proposals[0]
            trace = {
                "strategy": Strategy.ENSEMBLE.value,
                "selection_reason": "tool_calls_present",
                "participants": [p.model_id for p in proposals],
                "selected_model": chosen.model_id,
            }
            return self._wrap_llm_response(
                raw=chosen.response,
                strategy=Strategy.ENSEMBLE,
                model_id=chosen.model_id,
                debate_trace=trace,
            )

        merge_prompt = "Synthesize these model responses into one robust answer:\n\n"
        for i, p in enumerate(proposals, 1):
            merge_prompt += f"Response {i} ({p.model_id}):\n{(p.response.text or '')[:1500]}\n\n"
        merge_resp = self.default_client.chat(messages + [{"role": "user", "content": merge_prompt}], tools)
        claims = self._extract_claims(merge_resp.text, source_model=self.default_model)
        links = self._evidence_from_proposals(claims, proposals)
        trace = {
            "strategy": Strategy.ENSEMBLE.value,
            "participants": [p.model_id for p in proposals],
        }
        return self._wrap_llm_response(
            raw=merge_resp,
            strategy=Strategy.ENSEMBLE,
            model_id=self.default_model,
            claims=claims,
            evidence_links=links,
            debate_trace=trace,
        )

    def _parse_json_loose(self, text: str) -> Any:
        """Parse JSON in strict or fenced format."""

        if not text:
            return None
        candidate = text.strip()
        if "```" in candidate:
            chunks = candidate.split("```")
            for chunk in chunks:
                clean = chunk.strip().removeprefix("json").strip()
                if clean.startswith("{") or clean.startswith("["):
                    candidate = clean
                    break
        try:
            return json.loads(candidate)
        except Exception:
            return None

    def _build_self_contained_message(
        self,
        role: SubagentRole,
        task: str,
        context_pack: dict | None = None,
    ) -> list[dict]:
        """Build a self-contained message list with zero parent history."""

        system_prompt = _ROLE_SYSTEM_PROMPTS.get(role.value, "You are a helpful assistant.")
        if context_pack:
            ctx_str = "\n".join(f"- {k}: {v}" for k, v in context_pack.items())
            system_prompt += f"\n\n## Context\n{ctx_str}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

    def dispatch_subagent(
        self,
        call: SubagentCall,
        context_pack: dict | None = None,
        model_id: str | None = None,
    ) -> LLMResponse:
        """Dispatch a subagent with role-based constraints."""

        messages = self._build_self_contained_message(
            role=call.role,
            task=call.task,
            context_pack=context_pack,
        )
        client = self.get_client(model_id or self.default_model)
        return client.chat(messages)
