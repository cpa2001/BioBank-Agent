"""ReAct agent loop — orchestrates LLM + skill execution.

Uses native OpenAI-compatible tool_use (function calling), not text parsing.
Includes a think tool for internal reasoning traces.
"""

from __future__ import annotations

import json
import logging
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
from .registry import SkillRegistry, autodiscover_skills, discover_custom_skills, get_registry
from .state import AnalysisRecord, Provenance, SessionState

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

## Current session state
{session_state}
"""


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

        # Skills
        autodiscover_skills()
        discover_custom_skills(settings.custom_skills_dir)
        self.registry = get_registry()
        logger.info("Loaded %d skills", len(self.registry))

        # Conversation
        self.messages: list[dict] = []

        # Background review
        self._review_lock = threading.Lock()
        self._review_interval = 5  # trigger review every N records

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
        """
        self.messages.append({"role": "user", "content": user_query})

        # Compute report_dir once per run() call for consistency
        _report_dir = self.settings.reports_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        _report_dir.mkdir(parents=True, exist_ok=True)

        for round_n in range(self.settings.max_tool_rounds):
            if self.state.interrupted:
                return "[Interrupted by user]"

            # Build messages with fresh system prompt
            all_messages = [self._system_message()] + self.messages

            # Call LLM
            response = self.llm.chat(
                messages=all_messages,
                tools=self.registry.tool_schemas() or None,
            )

            # Track token usage
            if response.usage:
                self.state.token_usage.update(response.usage)

            # If no tool calls → final answer
            if not response.has_tool_calls:
                self.messages.append({"role": "assistant", "content": response.text})
                self._post_run(user_query)
                return response.text

            # Process tool calls
            # First, add the assistant message with tool_calls
            assistant_msg = {"role": "assistant", "content": response.text or ""}
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
                t0 = time.time()
                figs_before = len(self.state.figures)
                ctx = self._build_ctx(_report_dir)
                try:
                    result = self.registry.execute(tc.name, tc.args, ctx=ctx)
                    result_str = json.dumps(result, default=str, ensure_ascii=False)
                except Exception as e:
                    logger.error("Tool %s failed: %s", tc.name, e)

                    # Track error in long-term memory
                    try:
                        self.memory.record_error(
                            error_type=type(e).__name__,
                            error_message=str(e),
                            skill_name=tc.name,
                            context={"args": {k: str(v)[:100] for k, v in tc.args.items()}},
                        )
                    except Exception:
                        pass  # Don't let error tracking break the agent

                    # Get suggestions from error history
                    suggestions = self.memory.get_error_suggestions(type(e).__name__, tc.name)
                    error_info = {"error": str(e)}
                    if suggestions:
                        error_info["known_fixes"] = suggestions[:3]

                    # Auto-retry once for retryable errors (not ValueError/KeyError/etc.)
                    retried = False
                    try:
                        from .skills.retry import should_retry_on_error
                        if should_retry_on_error(e):
                            logger.info("Retrying %s with adjusted params...", tc.name)
                            retry_args = dict(tc.args)
                            # Simple parameter adjustments
                            for key in ("n_folds", "top_n"):
                                if key in retry_args and isinstance(retry_args[key], int):
                                    retry_args[key] = max(2, retry_args[key] // 2)
                            for key in ("sample_size",):
                                if key in retry_args and isinstance(retry_args[key], int):
                                    retry_args[key] = retry_args[key] // 2
                            result = self.registry.execute(tc.name, retry_args, ctx=ctx)
                            # Mark result as retried so LLM knows params changed
                            if isinstance(result, dict):
                                result["_retried_with"] = {
                                    k: v for k, v in retry_args.items()
                                    if k in ("n_folds", "top_n", "sample_size") and retry_args[k] != tc.args.get(k)
                                }
                            result_str = json.dumps(result, default=str, ensure_ascii=False)
                            retried = True
                    except Exception as retry_err:
                        logger.warning("Retry of %s also failed: %s", tc.name, retry_err)

                    if not retried:
                        result_str = json.dumps(error_info, default=str)
                        result = error_info

                elapsed = time.time() - t0

                # Track field usage only for successful data queries
                is_error = isinstance(result, dict) and "error" in result
                if not is_error:
                    for arg_val in tc.args.values():
                        if isinstance(arg_val, str) and arg_val.replace("-", "").isdigit():
                            self.memory.record_field_usage(arg_val)

                # Record — only figures produced by THIS tool call
                new_figs = [str(p) for p in self.state.figures[figs_before:]]
                ts = datetime.now().isoformat()
                clean_args = {k: v for k, v in tc.args.items() if k != "ctx"}
                key_res = result if isinstance(result, dict) else {"result": str(result)[:200]}

                self.state.add_record(AnalysisRecord(
                    timestamp=ts,
                    skill=tc.name,
                    args=clean_args,
                    key_results=key_res,
                    figure_paths=new_figs,
                ))

                # Record provenance for reproducibility
                if not is_error:
                    prov = Provenance(
                        provenance_id=Provenance.make_id(tc.name, clean_args, ts),
                        skill=tc.name,
                        args=clean_args,
                        timestamp=ts,
                        bank_id=self.settings.bank_id,
                        result_hash=Provenance.compute_hash(key_res),
                        parent_ids=[p.provenance_id for p in self.state.provenances[-3:]],
                    )
                    self.state.provenances.append(prov)

                # Add tool result message
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str[:8000],  # Truncate large results
                })

                logger.info("Tool %s completed in %.1fs", tc.name, elapsed)

        logger.warning("Max tool rounds (%d) reached for query: %s",
                       self.settings.max_tool_rounds, user_query[:100])
        self.messages.append({"role": "assistant", "content": "[Max tool rounds reached]"})
        self._post_run(user_query)
        return "[Max tool rounds reached]"

    def _post_run(self, user_query: str) -> None:
        """Post-run housekeeping: index session, trigger background review."""
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
                    for i in range(len(skills) - 2):
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
        return ctx
