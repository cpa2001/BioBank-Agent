"""ReAct agent loop — orchestrates LLM + skill execution.

Uses native OpenAI-compatible tool_use (function calling), not text parsing.
Includes a think tool for internal reasoning traces.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import Settings
from .data.catalog import FieldCatalog
from .data.loader import DataManager
from .llm import LLMClient, LLMResponse
from .memory import LongTermMemory
from .registry import SkillRegistry, autodiscover_skills, get_registry
from .state import AnalysisRecord, SessionState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are **Biobank Agent (bb)**, an expert biomedical data analyst for UK Biobank research.

## Data available
- **502,370 participants** with 2,031 biomarker columns (parquet, fast)
- **6.9 million ICD10 diagnosis records** (hospital episode statistics)
- **112,917 death records** with ICD10 cause-of-death codes
- **4,953 additional field IDs** accessible via raw CSV fallback
- **11,821 field definitions** in the UKB catalogue

## Analysis principles
1. Always check cohort sizes before modelling. Refuse to train if n_cases < 100.
2. Report 95% confidence intervals alongside point estimates (AUC, OR, HR).
3. For any new ICD10 code, run prevalence check first.
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
        self.registry = get_registry()
        logger.info("Loaded %d skills", len(self.registry))

        # Conversation
        self.messages: list[dict] = []

    def _system_message(self) -> dict:
        mem_summary = self.memory.summary()
        content = SYSTEM_PROMPT.format(
            session_state=self.state.context_summary()
        )
        if mem_summary:
            content += f"\n\n{mem_summary}"
        return {"role": "system", "content": content}

    def run(self, user_query: str) -> str:
        """Execute a full agent turn: user query → final text response.

        The agent will call tools in a loop until the LLM produces a
        text-only response (no more tool calls).
        """
        self.messages.append({"role": "user", "content": user_query})

        # Compute report_dir once per run() call for consistency
        _report_dir = self.settings.reports_dir / datetime.now().strftime("%Y%m%d_%H%M%S")

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

            # If no tool calls → final answer
            if not response.has_tool_calls:
                self.messages.append({"role": "assistant", "content": response.text})
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

                    # Auto-retry once for retryable errors
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
                            result_str = json.dumps(result, default=str, ensure_ascii=False)
                            retried = True
                    except Exception:
                        pass

                    if not retried:
                        result_str = json.dumps(error_info, default=str)
                        result = error_info

                elapsed = time.time() - t0

                # Track field usage for data queries
                for arg_val in tc.args.values():
                    if isinstance(arg_val, str) and arg_val.replace("-", "").isdigit():
                        self.memory.record_field_usage(arg_val)

                # Record
                self.state.add_record(AnalysisRecord(
                    timestamp=datetime.now().isoformat(),
                    skill=tc.name,
                    args={k: v for k, v in tc.args.items() if k != "ctx"},
                    key_results=result if isinstance(result, dict) else {"result": str(result)[:200]},
                    figure_paths=[str(p) for p in self.state.figures[-5:]],
                ))

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
        return "[Max tool rounds reached]"

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
