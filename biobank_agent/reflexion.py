"""Structured Reflexion engine — LLM-driven self-correction on failure.

Replaces the blind retry logic (agent.py "halve parameters") with:
1. Root cause analysis — LLM reasons about why the skill failed
2. Error catalog lookup — check long-term memory for known fixes
3. Reasoned correction — generate specific, justified parameter changes
4. Retry with confidence — execute corrected call, track outcome

Reference: Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement
Learning" (NeurIPS 2023). Extended with error catalog integration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMClient
    from .memory import LongTermMemory

logger = logging.getLogger(__name__)


@dataclass
class Correction:
    """A single parameter correction with justification."""
    param: str
    old_value: Any
    new_value: Any
    reason: str


@dataclass
class ReflectionResult:
    """Structured output from failure reflection."""
    root_cause: str                          # what went wrong
    error_category: str                      # data | parameter | logic | external
    retry_recommended: bool                  # should we retry?
    corrections: list[Correction] = field(default_factory=list)
    confidence: float = 0.0                  # 0.0-1.0 confidence in fix
    known_fix_used: bool = False             # was this from error catalog?
    summary: str = ""

    @property
    def corrected_args(self) -> dict:
        """Return dict of only the changed parameters."""
        return {c.param: c.new_value for c in self.corrections}


# ── Error categories that are NOT retryable ─────────────────

_NON_RETRYABLE = {
    "ValueError",      # bad argument type/value
    "KeyError",        # missing key — won't appear on retry
    "TypeError",       # wrong argument types
    "ImportError",     # missing package
    "ModuleNotFoundError",
    "AttributeError",  # wrong attribute
    "SyntaxError",
    "PermissionError",
}


class ReflexionEngine:
    """LLM-driven self-correction with error catalog integration.

    Usage::

        engine = ReflexionEngine(llm_client, memory)
        if engine.should_retry(error, skill_name):
            reflection = engine.reflect(skill_name, args, error, context)
            if reflection.retry_recommended:
                new_args = {**args, **reflection.corrected_args}
                result = registry.execute(skill_name, new_args, ctx=ctx)
    """

    def __init__(
        self,
        llm: LLMClient,
        memory: Optional[LongTermMemory] = None,
        max_reflection_tokens: int = 512,
    ):
        self.llm = llm
        self.memory = memory
        self.max_reflection_tokens = max_reflection_tokens

    def should_retry(self, error: Exception, skill_name: str = "") -> bool:
        """Quick check: is this error type worth retrying?

        Returns False for deterministic errors (ValueError, KeyError, etc.)
        that will fail identically on retry.
        """
        error_type = type(error).__name__
        if error_type in _NON_RETRYABLE:
            return False

        error_msg = str(error).lower()
        # Don't retry on clear data issues
        if any(s in error_msg for s in ("not found", "does not exist", "no such")):
            return False

        return True

    def reflect(
        self,
        skill_name: str,
        args: dict,
        error: Exception,
        context: str = "",
    ) -> ReflectionResult:
        """Analyze a skill failure and generate a correction plan.

        Parameters
        ----------
        skill_name : str
            Name of the failed skill.
        args : dict
            Arguments that were passed to the skill.
        error : Exception
            The exception that was raised.
        context : str, optional
            Additional context (e.g. recent analysis history).

        Returns
        -------
        ReflectionResult with root cause, corrections, and confidence.
        """
        error_type = type(error).__name__
        error_msg = str(error)

        # ── Step 1: Check error catalog for known fixes ────
        known_fixes = []
        if self.memory:
            known_fixes = self.memory.get_error_suggestions(error_type, skill_name)

        # ── Step 2: Fast-path for common patterns ──────────
        fast_result = self._fast_path_fix(skill_name, args, error_type, error_msg)
        if fast_result:
            fast_result.known_fix_used = bool(known_fixes)
            return fast_result

        # ── Step 3: LLM-driven reflection ──────────────────
        reflection_prompt = self._build_reflection_prompt(
            skill_name, args, error_type, error_msg, known_fixes, context,
        )

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": (
                        "You are a debugging assistant for a biobank analysis agent. "
                        "Analyze skill failures and suggest specific parameter corrections. "
                        "Be concise and precise. Output valid JSON."
                    )},
                    {"role": "user", "content": reflection_prompt},
                ],
                max_tokens=self.max_reflection_tokens,
            )
            return self._parse_reflection(response.text, args, error_type, error_msg)

        except Exception as e:
            logger.warning("LLM reflection failed: %s. Falling back to heuristics.", e)
            return self._heuristic_fallback(skill_name, args, error_type, error_msg)

    def _build_reflection_prompt(
        self,
        skill_name: str,
        args: dict,
        error_type: str,
        error_msg: str,
        known_fixes: list[str],
        context: str,
    ) -> str:
        """Build the prompt for LLM-driven reflection."""
        args_str = json.dumps(
            {k: str(v)[:200] for k, v in args.items() if k != "ctx"},
            indent=2,
        )
        prompt = f"""The skill `{skill_name}` failed.

**Error:** {error_type}: {error_msg[:500]}
**Arguments:** {args_str}
"""
        if known_fixes:
            prompt += f"\n**Previously successful fixes:**\n"
            for fix in known_fixes[:3]:
                prompt += f"  - {fix}\n"

        if context:
            prompt += f"\n**Context:** {context[:300]}\n"

        prompt += """
Analyze and respond with this exact JSON format:
{
  "root_cause": "brief explanation",
  "category": "data|parameter|logic|external",
  "retry": true/false,
  "confidence": 0.0-1.0,
  "corrections": [
    {"param": "param_name", "new_value": "corrected_value", "reason": "why"}
  ]
}"""
        return prompt

    def _parse_reflection(
        self,
        llm_text: str,
        original_args: dict,
        error_type: str,
        error_msg: str,
    ) -> ReflectionResult:
        """Parse LLM reflection output into a ReflectionResult."""
        # Extract JSON from LLM response (may be wrapped in ```json...```)
        text = llm_text.strip()
        if "```" in text:
            # Extract content between code fences
            parts = text.split("```")
            for part in parts:
                clean = part.strip().removeprefix("json").strip()
                if clean.startswith("{"):
                    text = clean
                    break

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM reflection JSON, using heuristic")
            return self._heuristic_fallback("", original_args, error_type, error_msg)

        corrections = []
        for c in data.get("corrections", []):
            param = c.get("param", "")
            if param and param in original_args:
                corrections.append(Correction(
                    param=param,
                    old_value=original_args[param],
                    new_value=c.get("new_value"),
                    reason=c.get("reason", ""),
                ))

        return ReflectionResult(
            root_cause=data.get("root_cause", "unknown"),
            error_category=data.get("category", "unknown"),
            retry_recommended=data.get("retry", False),
            corrections=corrections,
            confidence=float(data.get("confidence", 0.5)),
            summary=f"Reflexion: {data.get('root_cause', 'unknown')}",
        )

    def _fast_path_fix(
        self,
        skill_name: str,
        args: dict,
        error_type: str,
        error_msg: str,
    ) -> Optional[ReflectionResult]:
        """Handle common error patterns without LLM call."""
        # Insufficient data — reduce folds or sample
        if "insufficient" in error_msg.lower() or "too few" in error_msg.lower():
            corrections = []
            if "n_folds" in args and isinstance(args["n_folds"], int) and args["n_folds"] > 2:
                corrections.append(Correction(
                    param="n_folds",
                    old_value=args["n_folds"],
                    new_value=max(2, args["n_folds"] - 1),
                    reason="Reduce folds due to insufficient data",
                ))
            if "controls_ratio" in args and isinstance(args["controls_ratio"], int) and args["controls_ratio"] > 0:
                corrections.append(Correction(
                    param="controls_ratio",
                    old_value=args["controls_ratio"],
                    new_value=0,
                    reason="Use all eligible controls instead of downsampling the control cohort",
                ))
            if corrections:
                return ReflectionResult(
                    root_cause="Insufficient data for requested analysis parameters",
                    error_category="data",
                    retry_recommended=True,
                    corrections=corrections,
                    confidence=0.8,
                    summary="Fast-path: reduce parameters for small datasets",
                )

        # Memory/timeout errors — reduce computational knobs, but do not reduce
        # data volume automatically. Biobank analyses default to full eligible
        # data; sampling is only applied when the user explicitly requests it.
        if error_type in ("MemoryError", "TimeoutError"):
            corrections = []
            for key in ("top_n", "n_folds"):
                if key in args and isinstance(args[key], int):
                    corrections.append(Correction(
                        param=key,
                        old_value=args[key],
                        new_value=max(2, args[key] // 2),
                        reason=f"Reduce {key} due to {error_type}",
                    ))
            if corrections:
                return ReflectionResult(
                    root_cause=f"Resource limit: {error_type}",
                    error_category="external",
                    retry_recommended=True,
                    corrections=corrections,
                    confidence=0.7,
                    summary=f"Fast-path: halve scope for {error_type}",
                )

        return None  # No fast-path available

    def _heuristic_fallback(
        self,
        skill_name: str,
        args: dict,
        error_type: str,
        error_msg: str,
    ) -> ReflectionResult:
        """Last resort: conservative heuristic corrections."""
        corrections = []
        for key in ("n_folds", "top_n"):
            if key in args and isinstance(args[key], int) and args[key] > 2:
                corrections.append(Correction(
                    param=key,
                    old_value=args[key],
                    new_value=max(2, args[key] // 2),
                    reason=f"Heuristic: reduce {key} on error",
                ))

        return ReflectionResult(
            root_cause=f"{error_type}: {error_msg[:200]}",
            error_category="unknown",
            retry_recommended=bool(corrections),
            corrections=corrections,
            confidence=0.3,
            summary="Heuristic fallback — low confidence",
        )
