"""Multi-model orchestrator — routes hard tasks to collaborative model pools.

Strategies:
  - SINGLE:     One model (status quo, default for simple queries)
  - DEBATE:     N models propose independently → cross-critique → judge selects
  - SUPERVISOR: Planner model decomposes → specialist models execute → merge
  - ENSEMBLE:   All models answer in parallel → vote / merge results

All models are accessed via the same OpenAI-compatible relay (api.shubiaobiao.cn).
Each model gets its own LLMClient instance sharing the same base_url.

Thread safety: parallel model calls use ThreadPoolExecutor.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Optional

from .complexity import Strategy, ComplexityScore, classify_complexity
from .llm import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


# ── Subagent roles (ported from heathcliff233/my_codex) ──────

class SubagentRole(Enum):
    """Role-based subagent specialization (leaf agents — never re-delegate)."""
    EXPLORER = "explorer"    # read-only search specialist
    WORKER = "worker"        # one-shot implementation executor
    VERIFIER = "verifier"    # adversarial review specialist


@dataclass
class SubagentCall:
    """A structured subagent dispatch request."""
    role: SubagentRole
    task: str
    scope: list[str] = field(default_factory=list)      # directories/files in scope
    owned_files: list[str] = field(default_factory=list) # worker only: files it can modify
    off_limits: list[str] = field(default_factory=list)  # files other workers own
    output_format: str = "text"                           # text | json | verdict


# ── Role system prompts ──────────────────────────────────────

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
    model_id: str              # e.g. "claude-opus-4-7", "gpt-4o", "gemini-2.5-pro"
    role: str = "generalist"   # generalist | reasoning | coding | judge
    strengths: list[str] = field(default_factory=list)
    priority: int = 0          # higher = preferred for tiebreaks


@dataclass
class DebateProposal:
    """One model's proposal in a debate round."""
    model_id: str
    response: LLMResponse
    score: float = 0.0         # assigned by judge
    critique: str = ""         # from cross-critique round


class MultiModelOrchestrator:
    """Routes tasks to one or more models based on complexity.

    Usage::

        orch = MultiModelOrchestrator(base_url, api_key, "claude-opus-4-7", pool)
        result = orch.route(query, messages, tools, records)
        # result is an LLMResponse (same interface as single-model)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str,
        model_pool: list[ModelSpec] | None = None,
        complexity_threshold: float = 0.7,
        debate_rounds: int = 2,
        max_workers: int = 3,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.default_model = default_model
        self.model_pool = model_pool or [ModelSpec(default_model, "generalist")]
        self.complexity_threshold = complexity_threshold
        self.debate_rounds = debate_rounds
        self.max_workers = max_workers

        # Lazily created clients (one per model, thread-safe)
        self._clients: dict[str, LLMClient] = {}
        self._clients_lock = Lock()

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
    ) -> LLMResponse:
        """Classify and route a query to the appropriate strategy.

        Parameters
        ----------
        query : str
            User query.
        messages : list[dict]
            Full conversation history (including system prompt).
        tools : list[dict], optional
            OpenAI-format tool schemas.
        records : list, optional
            Recent AnalysisRecord list for failure detection.
        force_strategy : Strategy, optional
            Override auto-classification (e.g. from /debate command).

        Returns
        -------
        LLMResponse — merged/selected response from the chosen strategy.
        """
        if force_strategy:
            strategy = force_strategy
            logger.info("Forced strategy: %s", strategy.value)
        else:
            result = classify_complexity(query, records, self.complexity_threshold)
            strategy = result.strategy
            logger.info(
                "Complexity score=%.2f strategy=%s reason=%s",
                result.score, strategy.value, result.reason,
            )

        if strategy == Strategy.SINGLE or len(self.model_pool) < 2:
            return self.default_client.chat(messages, tools)

        elif strategy == Strategy.DEBATE:
            return self.debate(messages, tools)

        elif strategy == Strategy.SUPERVISOR:
            return self.supervisor(query, messages, tools)

        elif strategy == Strategy.ENSEMBLE:
            return self.ensemble(messages, tools)

        # Fallback
        return self.default_client.chat(messages, tools)

    # ── DEBATE strategy ────────────────────────────────────

    def debate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Multi-agent debate: propose → critique → judge select.

        Round 1: Each model independently generates a response.
        Round 2: Each model critiques the other proposals.
        Round 3: Judge model selects or synthesises the best answer.
        """
        proposers = [m for m in self.model_pool if m.role != "judge"][:3]
        judge_spec = next(
            (m for m in self.model_pool if m.role == "judge"),
            self.model_pool[0],  # default model is judge
        )

        # ── Round 1: Independent proposals (parallel) ─────
        proposals = self._parallel_chat(proposers, messages, tools)
        logger.info("Debate round 1: %d proposals collected", len(proposals))

        if len(proposals) < 2:
            # Not enough models — return single best
            return proposals[0].response if proposals else self.default_client.chat(messages, tools)

        # ── Round 2: Cross-critique ───────────────────────
        for i, prop in enumerate(proposals):
            other_texts = [
                f"[{p.model_id}]: {p.response.text[:1000]}"
                for j, p in enumerate(proposals) if j != i
            ]
            critique_prompt = (
                "You are reviewing alternative analyses. Here are other proposals:\n\n"
                + "\n\n".join(other_texts)
                + "\n\nYour original response was:\n" + prop.response.text[:1000]
                + "\n\nProvide a brief critique: what are the strengths and weaknesses of each approach? "
                "What would you change or keep?"
            )
            try:
                client = self.get_client(prop.model_id)
                critique_resp = client.chat(
                    messages + [{"role": "user", "content": critique_prompt}],
                    max_tokens=1024,
                )
                prop.critique = critique_resp.text
            except Exception as e:
                logger.warning("Critique from %s failed: %s", prop.model_id, e)
                prop.critique = "(critique unavailable)"

        logger.info("Debate round 2: critiques collected")

        # ── Round 3: Judge synthesis ──────────────────────
        return self._judge_select(judge_spec, proposals, messages, tools)

    def _parallel_chat(
        self,
        models: list[ModelSpec],
        messages: list[dict],
        tools: list[dict] | None,
    ) -> list[DebateProposal]:
        """Call multiple models in parallel and collect their responses."""
        proposals: list[DebateProposal] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_model = {}
            for spec in models:
                client = self.get_client(spec.model_id)
                future = pool.submit(client.chat, messages, tools)
                future_to_model[future] = spec

            for future in as_completed(future_to_model):
                spec = future_to_model[future]
                try:
                    resp = future.result(timeout=120)
                    proposals.append(DebateProposal(model_id=spec.model_id, response=resp))
                except Exception as e:
                    logger.warning("Model %s failed in debate: %s", spec.model_id, e)

        return proposals

    def _judge_select(
        self,
        judge_spec: ModelSpec,
        proposals: list[DebateProposal],
        messages: list[dict],
        tools: list[dict] | None,
    ) -> LLMResponse:
        """Judge synthesises the best response from debate proposals."""
        judge_prompt_parts = ["You are the judge in a multi-model debate.\n"]
        judge_prompt_parts.append("## Proposals and Critiques\n")

        for i, p in enumerate(proposals, 1):
            judge_prompt_parts.append(f"### Proposal {i} (from {p.model_id})\n")
            judge_prompt_parts.append(p.response.text[:2000] + "\n")
            if p.critique:
                judge_prompt_parts.append(f"**Critique:** {p.critique[:500]}\n")

        judge_prompt_parts.append(
            "\n## Your task\n"
            "Synthesise the best answer by combining the strongest elements from all proposals. "
            "If one proposal is clearly superior, adopt it. If they complement each other, merge them. "
            "Produce a single, coherent, high-quality response."
        )

        judge_messages = messages + [
            {"role": "user", "content": "\n".join(judge_prompt_parts)}
        ]

        judge_client = self.get_client(judge_spec.model_id)
        result = judge_client.chat(judge_messages, tools)

        # Add metadata about the debate
        if not result.text:
            result.text = ""
        result.text = (
            f"*[Multi-model debate: {', '.join(p.model_id for p in proposals)} "
            f"→ judged by {judge_spec.model_id}]*\n\n"
            + result.text
        )
        return result

    # ── SUPERVISOR strategy ────────────────────────────────

    def supervisor(
        self,
        query: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Supervisor decomposes task → specialists execute → merge.

        Step 1: Supervisor model creates a task decomposition.
        Step 2: Sub-tasks assigned to specialist models (parallel).
        Step 3: Supervisor merges results into final response.
        """
        supervisor_client = self.get_client(self.default_model)

        # Step 1: Decompose
        decompose_prompt = (
            "You are a task planner. Break down this complex query into 2-4 independent sub-tasks "
            "that can be executed in parallel by different AI models.\n\n"
            f"Query: {query}\n\n"
            "Output a JSON array of sub-tasks, each with 'task' (description) and 'focus' "
            "(what aspect to prioritize). Example:\n"
            '[{"task": "Analyze prevalence data", "focus": "statistical accuracy"}, ...]\n'
            "Respond with ONLY the JSON array."
        )

        decompose_resp = supervisor_client.chat(
            [{"role": "user", "content": decompose_prompt}],
            max_tokens=1024,
        )

        try:
            sub_tasks = json.loads(decompose_resp.text.strip().strip("```json").strip("```"))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Supervisor decomposition failed, falling back to debate")
            return self.debate(messages, tools)

        if not isinstance(sub_tasks, list) or len(sub_tasks) < 2:
            return self.default_client.chat(messages, tools)

        # Step 2: Execute sub-tasks in parallel
        specialists = [m for m in self.model_pool if m.role != "judge"]
        sub_results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for i, st in enumerate(sub_tasks[:4]):
                spec = specialists[i % len(specialists)]
                client = self.get_client(spec.model_id)
                sub_messages = messages + [
                    {"role": "user", "content": f"Focus on this sub-task: {st.get('task', str(st))}\nFocus: {st.get('focus', 'accuracy')}"}
                ]
                future = pool.submit(client.chat, sub_messages, tools)
                futures[future] = (spec.model_id, st)

            for future in as_completed(futures):
                model_id, sub_task = futures[future]
                try:
                    resp = future.result(timeout=120)
                    sub_results.append({
                        "model": model_id,
                        "task": sub_task,
                        "response": resp.text[:2000],
                    })
                except Exception as e:
                    logger.warning("Specialist %s failed: %s", model_id, e)

        # Step 3: Merge
        merge_prompt = (
            "You are merging results from specialist models. "
            "Combine these sub-task results into a single coherent response:\n\n"
        )
        for sr in sub_results:
            merge_prompt += f"### Sub-task: {sr['task']}\n**Model:** {sr['model']}\n{sr['response']}\n\n"
        merge_prompt += "Produce a unified, comprehensive answer."

        merge_resp = supervisor_client.chat(
            messages + [{"role": "user", "content": merge_prompt}],
            tools,
        )
        return merge_resp

    # ── ENSEMBLE strategy ──────────────────────────────────

    def ensemble(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """All models answer in parallel → pick best or merge.

        Simpler than debate — no critique round. Best for moderate complexity
        where we want robustness but not the full debate overhead.
        """
        proposals = self._parallel_chat(self.model_pool[:3], messages, tools)

        if not proposals:
            return self.default_client.chat(messages, tools)
        if len(proposals) == 1:
            return proposals[0].response

        # If any proposal has tool_calls, prefer that one
        # (tool calls are more actionable for the agent loop)
        tool_proposals = [p for p in proposals if p.response.has_tool_calls]
        if tool_proposals:
            # Prefer the proposal from the highest-priority model
            tool_proposals.sort(
                key=lambda p: next(
                    (m.priority for m in self.model_pool if m.model_id == p.model_id), 0
                ),
                reverse=True,
            )
            return tool_proposals[0].response

        # No tool calls — merge text responses
        if len(proposals) >= 2:
            # Use default model to merge
            merge_prompt = "Multiple AI models produced these responses. Synthesise them:\n\n"
            for i, p in enumerate(proposals, 1):
                merge_prompt += f"### Response {i} ({p.model_id})\n{p.response.text[:1500]}\n\n"
            merge_prompt += "Produce a single, best answer."

            merge_resp = self.default_client.chat(
                messages + [{"role": "user", "content": merge_prompt}],
                tools,
            )
            return merge_resp

        return proposals[0].response

    # ── Subagent dispatch (my_codex patterns) ──────────────

    def _build_self_contained_message(
        self,
        role: SubagentRole,
        task: str,
        context_pack: dict | None = None,
    ) -> list[dict]:
        """Build a self-contained message list with ZERO parent history.

        Ported from my_codex: "Subagents start with NO parent conversation
        history. Every child message must be self-contained."
        """
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
        """Dispatch a subagent with role-based constraints.

        The subagent gets a self-contained message (no parent history),
        uses a role-specific system prompt, and is a leaf (the prompt
        explicitly forbids re-delegation).
        """
        messages = self._build_self_contained_message(
            role=call.role,
            task=call.task,
            context_pack=context_pack,
        )
        client = self.get_client(model_id or self.default_model)
        return client.chat(messages)
