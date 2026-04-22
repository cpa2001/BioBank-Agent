"""Tool learning engine — tracks skill performance and suggests improvements.

Learns from execution history:
1. Which parameter ranges produce successful runs
2. Which skill sequences are most productive
3. Which skills fail on which data patterns
4. Suggests parameter adjustments for known skill+data combos

Integrates with Tier 3 (long-term memory) for persistence and
Tier 4 (error catalog) for failure pattern recognition.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import LongTermMemory
    from .state import AnalysisRecord

logger = logging.getLogger(__name__)


@dataclass
class SkillStats:
    """Aggregated statistics for a single skill."""
    skill_name: str
    total_calls: int = 0
    successes: int = 0
    failures: int = 0
    avg_elapsed_s: float = 0.0
    common_args: dict[str, list] = field(default_factory=dict)
    common_errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.total_calls if self.total_calls > 0 else 0.0


@dataclass
class ParamSuggestion:
    """A suggested parameter value with justification."""
    param: str
    suggested_value: Any
    reason: str
    confidence: float = 0.0


class ToolLearner:
    """Track skill performance and suggest improvements.

    Usage::

        learner = ToolLearner(memory)
        learner.record(skill, args, result, elapsed)
        suggestions = learner.suggest_params("train_model", {"icd10_code": "E11"})
    """

    def __init__(self, memory: Optional[LongTermMemory] = None) -> None:
        self.memory = memory
        self._history: list[dict] = []    # in-memory execution log
        self._stats: dict[str, SkillStats] = {}

    def record(
        self,
        skill_name: str,
        args: dict,
        result: dict,
        elapsed_s: float = 0.0,
    ) -> None:
        """Record a skill execution for learning."""
        is_error = isinstance(result, dict) and "error" in result

        # Update stats
        if skill_name not in self._stats:
            self._stats[skill_name] = SkillStats(skill_name=skill_name)
        stats = self._stats[skill_name]
        stats.total_calls += 1
        if is_error:
            stats.failures += 1
            error_msg = str(result.get("error", ""))[:100]
            if error_msg and error_msg not in stats.common_errors:
                stats.common_errors.append(error_msg)
                stats.common_errors = stats.common_errors[-5:]
        else:
            stats.successes += 1

        # Track parameter values for successful runs
        if not is_error:
            for k, v in args.items():
                if k == "ctx":
                    continue
                if k not in stats.common_args:
                    stats.common_args[k] = []
                stats.common_args[k].append(v)
                stats.common_args[k] = stats.common_args[k][-20:]  # keep last 20

        # Update running average elapsed time
        n = stats.total_calls
        stats.avg_elapsed_s = (stats.avg_elapsed_s * (n - 1) + elapsed_s) / n

        # Keep in-memory history
        self._history.append({
            "skill": skill_name,
            "args": {k: v for k, v in args.items() if k != "ctx"},
            "success": not is_error,
            "elapsed_s": elapsed_s,
            "timestamp": datetime.now().isoformat(),
        })

        # Trim history
        if len(self._history) > 200:
            self._history = self._history[-200:]

    def suggest_params(
        self,
        skill_name: str,
        context: dict | None = None,
    ) -> list[ParamSuggestion]:
        """Suggest parameter values based on successful execution history."""
        stats = self._stats.get(skill_name)
        if not stats or stats.successes < 3:
            return []  # Not enough data to suggest

        suggestions = []
        for param, values in stats.common_args.items():
            if not values:
                continue

            # For numeric params: suggest the median of successful values
            numeric_vals = [v for v in values if isinstance(v, (int, float))]
            if numeric_vals:
                from statistics import median
                med = median(numeric_vals)
                suggestions.append(ParamSuggestion(
                    param=param,
                    suggested_value=int(med) if isinstance(values[0], int) else round(med, 4),
                    reason=f"Median of {len(numeric_vals)} successful runs",
                    confidence=min(0.9, 0.5 + len(numeric_vals) * 0.05),
                ))

            # For string params: suggest the most common value
            str_vals = [v for v in values if isinstance(v, str)]
            if str_vals:
                most_common = Counter(str_vals).most_common(1)[0]
                if most_common[1] >= 2:  # At least 2 occurrences
                    suggestions.append(ParamSuggestion(
                        param=param,
                        suggested_value=most_common[0],
                        reason=f"Most common value ({most_common[1]} of {len(str_vals)} runs)",
                        confidence=min(0.9, most_common[1] / len(str_vals)),
                    ))

        return suggestions

    def get_stats(self, skill_name: str) -> Optional[SkillStats]:
        """Get stats for a specific skill."""
        return self._stats.get(skill_name)

    def all_stats(self) -> list[SkillStats]:
        """Get stats for all tracked skills, sorted by usage."""
        return sorted(
            self._stats.values(),
            key=lambda s: s.total_calls,
            reverse=True,
        )

    def identify_skill_gap(self, failed_queries: list[str]) -> Optional[str]:
        """Analyze failed queries to identify missing capabilities.

        Returns a description of what skill might be needed, or None.
        """
        if not failed_queries:
            return None

        # Count common keywords in failed queries
        from collections import Counter
        import re
        words = Counter()
        for q in failed_queries[-10:]:
            tokens = re.findall(r"\b[a-zA-Z]{4,}\b", q.lower())
            words.update(tokens)

        # Remove already-known skill names
        known_skills = set(self._stats.keys())
        interesting = [
            (w, c) for w, c in words.most_common(10)
            if w not in known_skills and c >= 2
        ]

        if interesting:
            top_words = ", ".join(w for w, _ in interesting[:5])
            return f"Repeated query terms not served by existing skills: {top_words}"

        return None

    def summary(self) -> str:
        """Compact summary for system prompt injection."""
        if not self._stats:
            return ""

        parts = ["Tool learning:"]
        for stats in self.all_stats()[:5]:
            rate = f"{stats.success_rate:.0%}"
            parts.append(f"  {stats.skill_name}: {stats.total_calls} calls, {rate} success")

        # Identify underperforming skills
        low_perf = [
            s for s in self._stats.values()
            if s.total_calls >= 3 and s.success_rate < 0.5
        ]
        if low_perf:
            names = ", ".join(s.skill_name for s in low_perf)
            parts.append(f"  Low performers: {names}")

        return "\n".join(parts)
