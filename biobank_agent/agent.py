"""ReAct agent loop — orchestrates LLM + skill execution.

Uses native OpenAI-compatible tool_use (function calling), not text parsing.
Includes a think tool for internal reasoning traces.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import Settings
from .data.catalog import FieldCatalog
from .data.loader import DataManager
from .llm import LLMClient, LLMResponse
from .memory import LongTermMemory
from .registry import autodiscover_skills, discover_custom_skills, get_registry
from .state import AnalysisRecord, Provenance, SessionState

# Multi-model & reasoning imports (lazy-friendly)
from .complexity import Strategy
from .orchestrator import MultiModelOrchestrator, ModelSpec, OrchestrationResult
from .reflexion import ReflexionEngine
from .verdict import VerdictEngine
from .guardrails import DelegationGuardrails
from .retrieval import AgenticRAG
from .tool_learner import ToolLearner
from .interfaces.multimodal import SimpleMultimodalGrounder
from .core.safety.nli_causal_check import maybe_inject_disclaimer
from .core.compaction import structured_extract

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are **Biobank Agent**, an autonomous scientific discovery system for biobank research.
You combine biomedical expertise with computational analysis to discover disease biomarkers,
build predictive models, and generate publication-quality reports.

## Data source: {biobank_name}
{data_description}

## Analysis principles
1. Always check cohort sizes before modelling. Refuse to train if n_cases < 100.
2. Report 95% confidence intervals alongside point estimates (AUC, OR, HR).
3. For any new diagnosis code, run prevalence check first.
4. Prefer established biomarker groups (metabolic, haematological, anthropometric) as baseline features.
5. Use the `think` tool for multi-step reasoning before complex analyses.
6. All figures must be publication-quality (Nature style: Arial, 300 dpi, no top/right spines).
7. When reporting results, use precise scientific language suitable for a Nature paper.
8. For high-risk multi-step plans or publication-facing reports, consider local
   external review skills (`codex_plan`, `claude_plan`, `codex_check_execution`,
   `claude_check_execution`) after checking `external_agent_status`.

## Current session state
{session_state}
"""


_UNSAFE_EXECUTIVE_PATTERNS = (
    r"/Users/",
    r"\btraceback\b",
    r"\bstdout\b|\bstderr\b|\breturncode\b",
    r"\bTODO\b|\bFIXME\b",
    r"```",
    r"\bpytest\b|\bbenchmark\b",
    r"\bcodex\b|\bclaude\b",
    r"\breview(ed|er| loop)?\b",
    r"\bapi[_ -]?key\b|\bauth\b|\blogin\b",
    r"\bpath:\b|\breport path\b",
    r"\banalysis completed\b",
)
_UNQUALIFIED_CAUSAL_PATTERN = re.compile(
    r"\b(causes?|causal effect|prevents?|treats?|cures?|reduces risk|protects against|therapy recommendation)\b",
    re.IGNORECASE,
)
_CAUSAL_QUALIFIER_PATTERN = re.compile(
    r"\b(associat(?:ed|ion)|hypothes(?:is|ize)|observational|not causal|cannot infer|does not establish|requires validation|may)\b",
    re.IGNORECASE,
)


def _safe_executive_finding(line: str) -> bool:
    """Return whether a final-answer line is safe for report lead placement."""
    if len(line) < 20:
        return False
    if line.endswith(":") and len(line) < 80:
        return False
    for pattern in _UNSAFE_EXECUTIVE_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return False
    if _UNQUALIFIED_CAUSAL_PATTERN.search(line) and not _CAUSAL_QUALIFIER_PATTERN.search(line):
        return False
    return True


def _sanitize_executive_findings(text: str, *, max_items: int = 8) -> list[str]:
    """Extract concise, human-facing findings from a final answer."""
    findings: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", line)
        line = " ".join(line.strip(" -\t").split())
        if not _safe_executive_finding(line):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        findings.append(line[:450].rstrip())
        if len(findings) >= max_items:
            break
    return findings


def _safe_final_text(text: str) -> str:
    try:
        return maybe_inject_disclaimer(text or "")
    except Exception:
        return text or ""


class Agent:
    """The main agent loop: user query → LLM → tool calls → result."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_dirs()

        # Data layer
        self.dm = DataManager(settings)
        self.catalog = FieldCatalog(settings.field_txt, settings.category_txt)

        # State
        self.state = SessionState(duckdb_conn=self.dm.conn)
        self.memory = LongTermMemory(settings.memory_dir)

        # LLM
        self.llm = LLMClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
        self.llm.tool_call_content_mode = settings.tool_call_content_mode

        # Discover relay-supported models (best effort) for richer multi-agent routing.
        self.available_models: list[str] = self._discover_available_models()

        # Multi-model orchestrator
        model_pool = self._build_model_pool(settings, self.available_models)
        self.orchestrator = MultiModelOrchestrator(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            default_model=settings.llm_model,
            model_pool=model_pool,
            complexity_threshold=settings.complexity_threshold,
            debate_rounds=settings.debate_rounds,
        )

        # Reflexion engine (structured self-correction)
        self.reflexion = ReflexionEngine(
            llm=self.llm,
            memory=self.memory,
        ) if settings.enable_reflexion else None

        # Verdict engine (PASS/FAIL/PARTIAL verification)
        self.verdict_engine = VerdictEngine(self.llm)

        # Delegation guardrails (anti-pattern enforcement)
        self.guardrails = DelegationGuardrails()

        # Agentic RAG (autonomous retrieval on uncertainty)
        self.rag = AgenticRAG(llm=self.llm, memory=self.memory)

        # Tool learner (skill performance tracking)
        self.tool_learner = ToolLearner(memory=self.memory)
        self.multimodal_grounder = SimpleMultimodalGrounder()

        # Difficulty estimator (DAAO: learned routing with heuristic fallback)
        from .difficulty import DifficultyEstimator
        self.difficulty_estimator = DifficultyEstimator(
            history_path=settings.memory_dir / "difficulty",
        )

        # Reproducibility harness (NeuroClaw pattern: SHA-256 checkpoints)
        from .reproducibility import ReproducibilityHarness
        self.reproducibility = ReproducibilityHarness(
            checkpoint_dir=settings.reports_dir / "checkpoints",
        )

        # StudySpec compiler (schema-gated execution boundary)
        # Uses heuristic-only until planner integration consumes the spec.
        # Switch to llm=self.llm when planner.decompose(spec=...) is wired in.
        from .study_spec import StudySpecCompiler
        self._study_spec_compiler = StudySpecCompiler(llm=None)
        self._current_study_spec = None  # Set per-run if compilation succeeds

        # Skills
        autodiscover_skills()
        discover_custom_skills(settings.custom_skills_dir)
        self.registry = get_registry()
        logger.info("Loaded %d skills", len(self.registry))

        # Conversation
        self.messages: list[dict] = []
        self._active_query_id: str | None = None
        self._last_orchestration_strategy: str = "single"

        # Background review
        self._review_lock = threading.Lock()
        self._review_interval = 5  # trigger review every N records

    @staticmethod
    def _split_model_csv(raw: str) -> list[str]:
        """Split comma-separated model IDs and drop empties."""
        return [m.strip() for m in raw.split(",") if m.strip()]

    @classmethod
    def _build_model_pool(
        cls,
        settings: Settings,
        available_models: list[str] | None = None,
    ) -> list[ModelSpec]:
        """Build model pool from config + discovered relay models."""
        pool = [ModelSpec(settings.llm_model, "generalist", priority=10)]
        seen = {settings.llm_model}
        available_set = set(available_models or [])

        # 1) Explicit model pool from env has highest priority.
        explicit_pool = bool(settings.model_pool)
        if explicit_pool:
            candidates = cls._split_model_csv(settings.model_pool)
        else:
            # 2) Auto pool from preferred models if relay discovery is enabled.
            preferred = cls._split_model_csv(settings.preferred_multi_models)
            if available_set:
                candidates = [m for m in preferred if m in available_set]
                if not candidates:
                    candidates = [m for m in available_models or [] if m != settings.llm_model]
            else:
                candidates = []

        max_models = 999 if explicit_pool else max(1, settings.max_auto_model_pool)
        for idx, model_id in enumerate(candidates):
            if model_id in seen:
                continue
            priority = max(1, 8 - idx)
            pool.append(ModelSpec(model_id, role="generalist", priority=priority))
            seen.add(model_id)
            if len(pool) >= max_models:
                break
        return pool

    def _discover_available_models(self) -> list[str]:
        """Fetch relay model list (non-fatal on failure)."""
        if not self.settings.auto_discover_models:
            return []
        api_key = (self.settings.llm_api_key or "").strip()
        if not api_key or api_key.startswith("your-"):
            return []
        try:
            return self.llm.list_models(refresh=False)
        except Exception as e:  # defensive: never break startup on discovery
            logger.warning("Model discovery failed: %s", e)
            return []

    def refresh_available_models(self) -> list[str]:
        """Refresh relay model list and rebuild auto pool when applicable."""
        api_key = (self.settings.llm_api_key or "").strip()
        if not api_key or api_key.startswith("your-"):
            return list(self.available_models)
        try:
            self.available_models = self.llm.list_models(refresh=True)
        except Exception as e:
            logger.warning("Model refresh failed: %s", e)
            self.available_models = self.available_models or []

        if self.settings.auto_discover_models and not self.settings.model_pool:
            self.orchestrator.model_pool = self._build_model_pool(
                self.settings,
                self.available_models,
            )
        return list(self.available_models)

    def switch_model(self, model_id: str) -> None:
        """Switch default model at runtime and keep orchestrator in sync."""
        model_id = model_id.strip()
        if not model_id:
            return
        self.settings.llm_model = model_id
        self.llm.model = model_id
        self.orchestrator.default_model = model_id

        if all(spec.model_id != model_id for spec in self.orchestrator.model_pool):
            self.orchestrator.model_pool.insert(0, ModelSpec(model_id, "generalist", priority=10))

    @staticmethod
    def _plan_mode_status(query: str) -> str | None:
        """Extract plan-mode status from CLI-wrapped query context."""
        match = re.search(r"\[PLAN MODE - Status:\s*([A-Z]+)\]", query)
        if not match:
            return None
        return match.group(1).strip().upper()

    def _forced_strategy(self, query: str) -> Strategy | None:
        """Force multi-agent strategy for early planning stages."""
        status = self._plan_mode_status(query)
        if status in {"INTAKE", "ALIGNMENT"}:
            return Strategy.SUPERVISOR
        if status == "EXECUTION":
            return Strategy.ENSEMBLE
        return None

    def _build_data_description(self) -> str:
        """Generate dynamic data description from live data layer."""
        parts = []
        try:
            n_subjects = self.dm.count_subjects()
            parts.append(f"- **{n_subjects:,} participants** with biomarker data (parquet, fast)")
        except Exception:
            parts.append(f"- Participant data available via {self.settings.biobank_name}")

        try:
            n_diag = self.dm.conn.execute("SELECT COUNT(*) FROM diagnoses").fetchone()[0]
            parts.append(f"- **{n_diag:,} diagnosis records** with coded diagnoses")
        except Exception:
            parts.append("- Diagnosis records available")

        try:
            n_deaths = self.dm.conn.execute("SELECT COUNT(*) FROM deaths").fetchone()[0]
            parts.append(f"- **{n_deaths:,} death records** with cause-of-death codes")
        except Exception:
            pass

        try:
            n_fields = len(self.catalog.fields)
            parts.append(f"- **{n_fields:,} field definitions** in the catalogue")
        except Exception:
            pass

        return "\n".join(parts) if parts else "Data available via configured biobank."

    def _system_message(self) -> dict:
        mem_summary = self.memory.summary()
        content = SYSTEM_PROMPT.format(
            biobank_name=self.settings.biobank_name,
            data_description=self._build_data_description(),
            session_state=self.state.context_summary(),
        )
        if mem_summary:
            content += f"\n\n## Long-term Memory\n{mem_summary}"

        tool_learning = self.tool_learner.summary() if hasattr(self, "tool_learner") else ""
        if tool_learning:
            content += f"\n\n## Tool Learning\n{tool_learning}"

        content += (
            "\n\n## Self-Improvement Policy\n"
            "When an approved plan cannot be completed with current tools, first repair arguments "
            "or add missing prerequisite analysis steps. If a genuinely missing low-risk analysis "
            "capability blocks the goal, use `create_skill` to generate a review artifact under "
            "reports/generated_skills/. Do not activate generated skills unless an explicit user "
            "approval flow enables it. Do not modify repository source code automatically."
        )

        # Inject domain knowledge (prose, frozen snapshot)
        domain_text = self.memory.domain.summary(max_chars=2000)
        if domain_text:
            content += f"\n\n## Domain Knowledge\n{domain_text}"

        # Inject user profile
        user_text = self.memory.user.summary()
        if user_text:
            content += f"\n\n## Researcher Profile\n{user_text}"

        return {"role": "system", "content": content}

    def run(self, user_query: str) -> str:
        """Execute a full agent turn: user query → final text response.

        The agent will call tools in a loop until the LLM produces a
        text-only response (no more tool calls).

        When ``settings.async_runtime_enabled`` is set, the call is
        delegated to ``biobank_agent.core.runtime.AsyncAgent`` so the
        new streaming event pipeline drives the same legacy skill
        execution path with non-blocking LLM streaming, smart
        compaction, automatic ReproducibilityHarness checkpoints, and
        StudySpec gate emissions. The synchronous facade is preserved
        for CLI / eval / benchmark / test callers.
        """
        if getattr(self.settings, "async_runtime_enabled", False):
            import asyncio as _asyncio

            from .core.runtime import AsyncAgent as _AsyncAgent

            safe, reason = _AsyncAgent.is_safe_for_async(self)
            if not safe:
                logger.warning(
                    "async_runtime_enabled but unsafe to use; falling back "
                    "to synchronous loop. Reason: %s",
                    reason,
                )
            else:
                runtime = _AsyncAgent(self)
                try:
                    return _asyncio.run(runtime.run_to_text(user_query))
                except RuntimeError as e:
                    # Already-running loop (e.g. inside Jupyter or another
                    # asyncio context). Fall back to the legacy sync loop.
                    logger.warning(
                        "async_runtime_enabled but no asyncio.run available "
                        "(%s); falling back to legacy synchronous loop.",
                        e,
                    )

        self.messages.append({"role": "user", "content": user_query})
        self._active_query_id = f"q_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{abs(hash(user_query)) % 1000000}"
        try:
            self.memory.upsert_node(
                node_type="query",
                node_id=self._active_query_id,
                payload={"text": user_query, "timestamp": datetime.now().isoformat()},
                score=1.0,
            )
        except Exception:
            pass

        # Compute report_dir once per run() call for consistency
        _report_dir = self.settings.reports_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        _report_dir.mkdir(parents=True, exist_ok=True)

        # Attempt StudySpec compilation (schema-gated execution boundary).
        # On success, the spec constrains planning; on failure, proceed untyped.
        self._current_study_spec = None
        try:
            spec = self._study_spec_compiler.compile(user_query)
            self._current_study_spec = spec
            logger.info("StudySpec compiled: design=%s, modalities=%s, budget=%d",
                        spec.design.value, [m.value for m in spec.modalities], spec.tool_budget)
        except Exception as e:
            logger.debug("StudySpec compilation skipped: %s", e)
        turn_orchestrations: list[dict] = []

        for round_n in range(self.settings.max_tool_rounds):
            if self.state.interrupted:
                return "[Interrupted by user]"

            # Call LLM (with optional multi-model routing)
            all_messages = [self._system_message()] + self.messages
            orchestration_result: OrchestrationResult | None = None

            force_strategy = self._forced_strategy(user_query)
            # Stability guard: use heavy multi-model orchestration primarily on the first turn.
            # Subsequent tool rounds run on a single model to prevent repeated supervisor fan-out,
            # timeout cascades, and context bloat on long workflows.
            if (
                self.settings.multi_model_enabled
                and len(self.orchestrator.model_pool) > 1
                and round_n == 0
            ):
                orchestration_result = self.orchestrator.route(
                    query=user_query,
                    messages=all_messages,
                    tools=self.registry.tool_schemas() or None,
                    records=self.state.records,
                    force_strategy=force_strategy,
                )
                response = orchestration_result
                # Track actual strategy for difficulty estimator
                self._last_orchestration_strategy = orchestration_result.debate_trace.get("strategy", "single") if orchestration_result and orchestration_result.debate_trace else "ensemble"
            else:
                llm_raw = self.llm.chat(
                    messages=all_messages,
                    tools=self.registry.tool_schemas() or None,
                )
                orchestration_result = self.orchestrator._wrap_llm_response(
                    raw=llm_raw,
                    strategy=Strategy.SINGLE,
                    model_id=self.settings.llm_model,
                )
                try:
                    self.orchestrator._attach_execution_evidence(orchestration_result, records=self.state.records)
                    self.orchestrator._apply_execution_judge(orchestration_result, records=self.state.records)
                except Exception:
                    pass
                response = orchestration_result

            # Expose current routing status for CLI diagnostics/eval harness.
            try:
                trace_payload = orchestration_result.to_dict() if orchestration_result else {}
                if trace_payload:
                    turn_orchestrations.append(trace_payload)
                    trace_payload = dict(trace_payload)
                    trace_payload["turn_orchestrations"] = list(turn_orchestrations)
                    first_trace = turn_orchestrations[0].get("debate_trace", {}) if turn_orchestrations else {}
                    last_trace = turn_orchestrations[-1].get("debate_trace", {}) if turn_orchestrations else {}
                    trace_payload["initial_strategy"] = first_trace.get("strategy", "single")
                    trace_payload["final_strategy"] = last_trace.get("strategy", "single")
                self.state.last_orchestration = trace_payload
            except Exception:
                self.state.last_orchestration = {}

            # Track token usage
            if response.usage:
                self.state.token_usage.update(response.usage)

            # If no tool calls → final answer
            if not response.has_tool_calls:
                safe_text = _safe_final_text(response.text)
                self.messages.append({"role": "assistant", "content": safe_text})
                try:
                    # Normal routing/wrapping always yields an orchestration result here.
                    if orchestration_result:  # pragma: no branch
                        self._record_orchestration_graph(orchestration_result)
                except Exception:
                    pass
                self._record_executive_findings(user_query, safe_text, orchestration_result)
                self._post_run(user_query)
                return safe_text

            # Process tool calls
            # First, add the assistant message with tool_calls
            # content is set to None when empty — _sanitize_messages() handles relay compat
            assistant_msg = {"role": "assistant", "content": response.text or None}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                }
                for tc in response.tool_calls
            ]
            self.messages.append(assistant_msg)

            # Execute each tool call
            for tc in response.tool_calls:
                logger.info("Tool call: %s(%s)", tc.name, tc.args)
                execution = self._execute_skill_and_record(
                    tc.name,
                    tc.args,
                    _report_dir,
                    allow_retry=True,
                )

                # Add tool result message
                compacted_tool_result = structured_extract(
                    execution.get("result", {}),
                    target_tokens=2000,
                )
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": compacted_tool_result.text,
                })

                logger.info("Tool %s completed in %.1fs", tc.name, execution["elapsed_s"])

        logger.warning("Max tool rounds (%d) reached for query: %s",
                       self.settings.max_tool_rounds, user_query[:100])
        self.messages.append({"role": "assistant", "content": "[Max tool rounds reached]"})
        self._post_run(user_query)
        return "[Max tool rounds reached]"

    def _execute_skill_and_record(
        self,
        skill_name: str,
        args: dict,
        report_dir: Path,
        *,
        allow_retry: bool = True,
    ) -> dict:
        """Execute a skill and write the same audit trail used by the agent loop."""
        t0 = time.time()
        figs_before = len(self.state.figures)
        ctx = self._build_ctx(report_dir)
        clean_args = {k: v for k, v in (args or {}).items() if k != "ctx"}

        try:
            result = self.registry.execute(skill_name, args or {}, ctx=ctx)
            result_str = json.dumps(result, default=str, ensure_ascii=False)
        except Exception as e:
            logger.error("Tool %s failed: %s", skill_name, e)

            try:
                self.memory.record_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    skill_name=skill_name,
                    context={"args": {k: str(v)[:100] for k, v in clean_args.items()}},
                )
            except Exception:
                pass

            suggestions = self.memory.get_error_suggestions(type(e).__name__, skill_name)
            error_info = {"error": str(e)}
            if suggestions:
                error_info["known_fixes"] = suggestions[:3]

            retried = False
            if allow_retry:
                try:
                    if self.reflexion and self.reflexion.should_retry(e, skill_name):
                        logger.info("Reflexion: analyzing failure of %s...", skill_name)
                        reflection = self.reflexion.reflect(
                            skill_name=skill_name,
                            args=args or {},
                            error=e,
                            context=self.state.context_summary()[:500],
                        )
                        if reflection.retry_recommended and reflection.corrections:
                            retry_args = {**(args or {}), **reflection.corrected_args}
                            logger.info(
                                "Reflexion: retrying %s with corrections: %s (confidence=%.2f)",
                                skill_name, reflection.corrected_args, reflection.confidence,
                            )
                            result = self.registry.execute(skill_name, retry_args, ctx=ctx)
                            if isinstance(result, dict):
                                result["_reflexion"] = {
                                    "root_cause": reflection.root_cause,
                                    "corrections": {c.param: c.new_value for c in reflection.corrections},
                                    "confidence": reflection.confidence,
                                }
                            result_str = json.dumps(result, default=str, ensure_ascii=False)
                            clean_args = {k: v for k, v in retry_args.items() if k != "ctx"}
                            retried = True
                    elif not self.reflexion:
                        from .skills.retry import should_retry_on_error
                        if should_retry_on_error(e):
                            logger.info("Retrying %s with reduced params (no reflexion)...", skill_name)
                            retry_args = dict(args or {})
                            for key in ("n_folds", "top_n"):
                                if key in retry_args and isinstance(retry_args[key], int):
                                    retry_args[key] = max(2, retry_args[key] // 2)
                            result = self.registry.execute(skill_name, retry_args, ctx=ctx)
                            if isinstance(result, dict):
                                result["_retried_with"] = {
                                    k: v for k, v in retry_args.items()
                                    if k in ("n_folds", "top_n") and retry_args[k] != (args or {}).get(k)
                                }
                            result_str = json.dumps(result, default=str, ensure_ascii=False)
                            clean_args = {k: v for k, v in retry_args.items() if k != "ctx"}
                            retried = True
                except Exception as retry_err:
                    logger.warning("Retry of %s also failed: %s", skill_name, retry_err)

            if not retried:
                result = error_info
                result_str = json.dumps(error_info, default=str, ensure_ascii=False)

        elapsed = time.time() - t0
        is_error = isinstance(result, dict) and "error" in result

        if not is_error:
            for arg_val in clean_args.values():
                if isinstance(arg_val, str) and arg_val.replace("-", "").isdigit():
                    try:
                        self.memory.record_field_usage(arg_val)
                    except Exception:
                        pass

        try:
            self.tool_learner.record(
                skill_name,
                clean_args,
                result if isinstance(result, dict) else {},
                elapsed,
            )
        except Exception:
            pass

        if not is_error and skill_name not in ("think", "generate_report"):
            try:
                verdict = self.verdict_engine.verify_skill_result(
                    skill_name,
                    clean_args,
                    result if isinstance(result, dict) else {},
                )
                if verdict.n_blockers > 0:
                    logger.warning("Verdict FAIL for %s: %s", skill_name, verdict.summary())
            except Exception:
                pass

        try:
            self.reproducibility.create_audit_log(
                skill_name=skill_name,
                inputs=clean_args,
                outputs=result if isinstance(result, dict) else {"raw": str(result)[:500]},
                duration_ms=elapsed * 1000 if isinstance(elapsed, (int, float)) else 0.0,
                status="success" if not is_error else "failed",
            )
        except Exception:
            pass

        new_figs = [str(p) for p in self.state.figures[figs_before:]]
        ts = datetime.now().isoformat()
        key_res = result if isinstance(result, dict) else {"result": str(result)[:200]}
        self.state.add_record(AnalysisRecord(
            timestamp=ts,
            skill=skill_name,
            args=clean_args,
            key_results=key_res,
            figure_paths=new_figs,
        ))

        try:
            self._record_tool_graph(
                skill=skill_name,
                args=clean_args,
                key_results=key_res,
                figure_paths=new_figs,
                timestamp=ts,
            )
        except Exception:
            pass

        if not is_error:
            prov = Provenance(
                provenance_id=Provenance.make_id(skill_name, clean_args, ts),
                skill=skill_name,
                args=clean_args,
                timestamp=ts,
                bank_id=self.settings.bank_id,
                result_hash=Provenance.compute_hash(key_res),
                parent_ids=[p.provenance_id for p in self.state.provenances[-3:]],
            )
            self.state.provenances.append(prov)

        return {
            "result": key_res,
            "result_str": result_str,
            "elapsed_s": elapsed,
            "new_figures": new_figs,
            "is_error": is_error,
            "args": clean_args,
        }

    def _record_executive_findings(
        self,
        user_query: str,
        final_text: str,
        orchestration_result: OrchestrationResult | None,
    ) -> None:
        """Persist final-answer insights so reports can surface them prominently."""
        text = (final_text or "").strip()
        if not text:
            return

        claims = []
        evidence_links = []
        safety_status = "UNKNOWN"
        strategy = self._last_orchestration_strategy
        if orchestration_result is not None:
            try:
                claims = [c.__dict__ for c in orchestration_result.claims[:8]]
                evidence_links = [e.__dict__ for e in orchestration_result.evidence_links[:12]]
                safety_status = orchestration_result.safety_status
                strategy = orchestration_result.debate_trace.get("strategy", strategy)
            except Exception:
                pass

        recent_valid_records = [
            r for r in self.state.records[-20:]
            if isinstance(getattr(r, "key_results", None), dict)
            and "error" not in (getattr(r, "key_results", {}) or {})
            and getattr(r, "skill", "") not in {"think"}
        ]
        if not claims and not recent_valid_records:
            return

        summary_items = _sanitize_executive_findings(text)
        if not summary_items:
            return
        payload = {
            "timestamp": datetime.now().isoformat(),
            "query": user_query[:1000],
            "summary": summary_items,
            "final_text": text[:4000],
            "claim_type": "execution_grounded" if claims or self.state.records else "hypothesis",
            "safety_status": safety_status,
            "strategy": strategy,
            "claims": claims,
            "evidence_links": evidence_links,
            "gated": True,
        }
        self.state.executive_findings.append(payload)

    def _post_run(self, user_query: str) -> None:
        """Post-run housekeeping: index session, trigger background review."""
        # Difficulty estimator: record query-level outcome (once per query, not per tool call)
        try:
            # Determine if this query had any errors
            recent_records = self.state.records[-10:]
            query_succeeded = not any(
                isinstance(r.key_results, dict) and "error" in r.key_results
                for r in recent_records
                if r.timestamp >= (datetime.now().isoformat()[:10])  # today only
            )
            self.difficulty_estimator.record_outcome(
                query=user_query,
                strategy_used=self._last_orchestration_strategy,
                succeeded=query_succeeded,
            )
        except Exception:
            pass  # non-critical

        # Index this turn for cross-session recall
        try:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.memory.sessions.index_turn(
                session_id=session_id,
                user_query=user_query,
                records=self.state.records[-20:],
            )
        except Exception as e:
            logger.debug("Session indexing failed (non-critical): %s", e)

        # Trigger background review every N records
        n = len(self.state.records)
        if n > 0 and n % self._review_interval == 0:
            t = threading.Thread(target=self._background_review, daemon=True)
            t.start()

    def _record_orchestration_graph(self, result: OrchestrationResult) -> None:
        """Persist orchestration-level claim/evidence structure into Action Graph."""
        if not self._active_query_id:
            return
        query_id = self._active_query_id

        # Record claims.
        for claim in result.claims:
            self.memory.upsert_node(
                node_type="claim",
                node_id=claim.claim_id,
                payload={
                    "text": claim.text,
                    "source_model": claim.source_model,
                    "confidence": claim.confidence,
                    "tags": claim.tags,
                    "safety_status": result.safety_status,
                },
                score=float(claim.confidence or 0.5),
            )
            self.memory.link_nodes(
                src_type="query",
                src_id=query_id,
                dst_type="claim",
                dst_id=claim.claim_id,
                relation="produced_claim",
                weight=float(claim.confidence or 0.5),
            )

        # Record claim→evidence links from orchestrator trace.
        for ev in result.evidence_links:
            self.memory.upsert_node(
                node_type=ev.evidence_type,
                node_id=ev.evidence_id,
                payload={"snippet": ev.snippet, "source_model": ev.source_model},
                score=float(ev.score or 0.3),
            )
            self.memory.link_nodes(
                src_type="claim",
                src_id=ev.claim_id,
                dst_type=ev.evidence_type,
                dst_id=ev.evidence_id,
                relation=ev.relation or "supports",
                weight=float(ev.score or 0.3),
                evidence={"snippet": ev.snippet, "source_model": ev.source_model},
            )

        # Keep routing trace as a result node for reproducibility.
        trace_id = f"trace_{datetime.now().strftime('%H%M%S')}"
        self.memory.upsert_node(
            node_type="result",
            node_id=trace_id,
            payload={
                "strategy": result.debate_trace.get("strategy", "single"),
                "safety_status": result.safety_status,
                "debate_trace": result.debate_trace,
            },
            score=1.0,
        )
        self.memory.link_nodes(
            src_type="query",
            src_id=query_id,
            dst_type="result",
            dst_id=trace_id,
            relation="has_orchestration_trace",
            weight=1.0,
        )

    def _record_tool_graph(
        self,
        skill: str,
        args: dict,
        key_results: dict,
        figure_paths: list[str],
        timestamp: str,
    ) -> None:
        """Persist tool execution artifacts into Action Graph."""
        if not self._active_query_id:
            return
        query_id = self._active_query_id
        result_id = f"{timestamp}:{skill}"

        self.memory.upsert_node(
            node_type="result",
            node_id=result_id,
            payload={
                "skill": skill,
                "args": args,
                "key_results": key_results,
                "timestamp": timestamp,
            },
            score=1.0,
        )
        self.memory.link_nodes(
            src_type="query",
            src_id=query_id,
            dst_type="result",
            dst_id=result_id,
            relation="executed",
            weight=1.0,
        )

        self.memory.upsert_node(
            node_type="tool",
            node_id=skill,
            payload={"name": skill},
            score=1.0,
        )
        self.memory.link_nodes(
            src_type="result",
            src_id=result_id,
            dst_type="tool",
            dst_id=skill,
            relation="produced_by",
            weight=1.0,
        )

        # Link field references.
        for v in args.values():
            if isinstance(v, str) and v.replace("-", "").replace(".", "").isdigit():
                field_id = v.strip()
                self.memory.upsert_node("field", field_id, payload={"field_id": field_id}, score=0.8)
                self.memory.link_nodes(
                    src_type="result",
                    src_id=result_id,
                    dst_type="field",
                    dst_id=field_id,
                    relation="uses_field",
                    weight=0.7,
                )

        # Link figures.
        for fig in figure_paths:
            fig_id = Path(fig).name
            self.memory.upsert_node(
                node_type="figure",
                node_id=fig_id,
                payload={"path": fig, "timestamp": timestamp},
                score=0.9,
            )
            self.memory.link_nodes(
                src_type="result",
                src_id=result_id,
                dst_type="figure",
                dst_id=fig_id,
                relation="renders",
                weight=0.9,
            )
            # Cross-modal grounding MVP: extract lightweight structural signals.
            try:
                tensor = self.multimodal_grounder.figure_to_tensor(fig)
                signals = self.multimodal_grounder.extract_structural_signals(
                    table=None,
                    modality_tensors={"figure": tensor},
                )
                for s in signals[:4]:
                    signal_id = f"{fig_id}:{s.signal_type}"
                    self.memory.upsert_node(
                        node_type="signal",
                        node_id=signal_id,
                        payload={
                            "modality": s.modality,
                            "signal_type": s.signal_type,
                            "value": s.value,
                            "detail": s.detail,
                            "figure": fig_id,
                        },
                        score=float(min(1.0, abs(s.value))),
                    )
                    self.memory.link_nodes(
                        src_type="figure",
                        src_id=fig_id,
                        dst_type="signal",
                        dst_id=signal_id,
                        relation="has_structure_signal",
                        weight=0.6,
                    )
            except Exception:
                pass

        # Auto-create soft claim nodes for interpretable scalar findings.
        scalar_items = []
        for k, v in key_results.items():
            if isinstance(v, (int, float, str)) and k != "error":
                scalar_items.append((k, v))
        for key, val in scalar_items[:3]:
            claim_text = f"{skill}.{key}={val}"
            claim_id = f"auto_{abs(hash(claim_text)) % 10_000_000}"
            self.memory.upsert_node(
                node_type="claim",
                node_id=claim_id,
                payload={"text": claim_text, "source": "tool_result"},
                score=0.4,
            )
            self.memory.link_nodes(
                src_type="query",
                src_id=query_id,
                dst_type="claim",
                dst_id=claim_id,
                relation="derived_claim",
                weight=0.4,
            )
            self.memory.link_nodes(
                src_type="claim",
                src_id=claim_id,
                dst_type="result",
                dst_id=result_id,
                relation="supports",
                weight=0.8,
                evidence={"key": key, "value": str(val)},
            )

    def _background_review(self) -> None:
        """Background sweep: extract findings, detect patterns, update memory."""
        if not self._review_lock.acquire(blocking=False):
            return
        try:
            recent = self.state.records[-10:]
            self._save_domain_findings(recent)
            self._detect_pipeline_patterns()
            self._update_user_profile()
        except Exception as e:
            logger.debug("Background review non-critical failure: %s", e)
        finally:
            self._review_lock.release()

    def _save_domain_findings(self, records) -> None:
        """Extract significant findings to domain memory."""
        for r in records:
            if r.skill == "train_model":
                auc = r.key_results.get("mean_auc", r.key_results.get("auc", 0))
                if isinstance(auc, (int, float)) and auc > 0.70:
                    disease = r.args.get("icd10_code", r.args.get("icd_code", "unknown"))
                    model = r.args.get("model_type", "model")
                    features = r.key_results.get("top_features", [])
                    feat_str = ", ".join(str(f) for f in features[:5]) if features else "N/A"
                    body = (
                        f"- Disease: {disease}, Model: {model}, AUC: {auc:.3f}\n"
                        f"- Top predictors: {feat_str}"
                    )
                    self.memory.domain.append_finding(f"Predictor: {disease}", body)

                    if auc > 0.95:
                        self.memory.domain.append_finding(
                            f"WARNING: Possible leakage — {disease}",
                            f"AUC={auc:.3f} is suspiciously high. Check for label leakage.",
                        )

            if r.skill == "survival":
                p = r.key_results.get("log_rank_p", r.key_results.get("p_value", 1.0))
                if isinstance(p, (int, float)) and p < 0.01:
                    disease = r.args.get("icd10_code", r.args.get("icd_code", "unknown"))
                    self.memory.domain.append_finding(
                        f"Survival: {disease}",
                        f"Significant log-rank test (p={p:.4f})",
                    )

    def _detect_pipeline_patterns(self) -> None:
        """Auto-save repeated skill sequences as pipelines."""
        skills = [r.skill for r in self.state.records if r.skill != "think"]
        if len(skills) < 6:
            return
        trigrams: dict[tuple, int] = {}
        for i in range(len(skills) - 2):
            tri = (skills[i], skills[i + 1], skills[i + 2])
            trigrams[tri] = trigrams.get(tri, 0) + 1
        for trigram, count in trigrams.items():
            if count >= 2:
                name = "_".join(trigram)
                if name not in self.memory.list_pipelines():
                    # The trigram was counted from this same skills list, so one window must match.
                    for i in range(len(skills) - 2):  # pragma: no branch
                        if tuple(skills[i : i + 3]) == trigram:
                            steps = [
                                {"skill": r.skill, "args": r.args}
                                for r in self.state.records[i : i + 3]
                            ]
                            self.memory.save_pipeline(name, steps)
                            logger.info("Auto-saved pipeline: %s", name)
                            break

    def _update_user_profile(self) -> None:
        """Infer researcher preferences from session history."""
        all_diseases = [
            str(v) for r in self.state.records
            for k, v in r.args.items()
            if k in ("icd10_code", "icd_code", "icd10", "disease_code")
        ]
        if all_diseases:
            top = Counter(all_diseases).most_common(5)
            top_str = ", ".join(f"{d}(n={c})" for d, c in top)
            self.memory.user.upsert_preference("commonly_studied_diseases", top_str)

        model_types = [
            r.args.get("model_type")
            for r in self.state.records
            if r.skill == "train_model" and r.args.get("model_type")
        ]
        if model_types:
            top_model = Counter(model_types).most_common(1)[0][0]
            self.memory.user.upsert_preference("preferred_model", top_model)

    def _build_ctx(self, report_dir: Optional[Path] = None):
        """Build the context object passed to skills."""

        class SkillContext:
            pass

        ctx = SkillContext()
        ctx.dm = self.dm
        ctx.catalog = self.catalog
        ctx.state = self.state
        ctx.state.memory = self.memory  # Expose memory to skills
        ctx.settings = self.settings
        ctx.memory = self.memory
        # Use provided report_dir (computed once per run) — never create a new one
        ctx.report_dir = report_dir or self.settings.reports_dir

        def emit_progress(phase: str = "", message: str = "", metadata: Optional[dict] = None) -> None:
            event = {
                "phase": str(phase or ""),
                "message": str(message or ""),
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            try:
                self.state.custom_data.setdefault("skill_progress_events", []).append(event)
                self.state.custom_data["skill_progress_events"] = self.state.custom_data["skill_progress_events"][-200:]
            except Exception:
                pass
            callback = None
            try:
                callback = self.state.custom_data.get("plan_progress_callback")
            except Exception:
                callback = None
            if callable(callback):
                try:
                    callback(event)
                except Exception:
                    pass

        ctx.emit_progress = emit_progress
        return ctx
