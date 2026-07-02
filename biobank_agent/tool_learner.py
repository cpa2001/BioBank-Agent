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
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from .core.evolution.patch_classifier import classify

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


@dataclass
class FailurePattern:
    """Repeated failure cluster mined from recent tool executions."""
    skill_name: str
    error_signature: str
    count: int
    examples: list[dict] = field(default_factory=list)
    suggested_action: str = ""


@dataclass
class SequencePattern:
    """A frequently-repeated run of consecutive SUCCESSFUL skill calls.

    Raw material for auto-capturing a reusable wrapper skill: ``sequence`` is the
    ordered tuple of skill names, ``count`` how often it recurred, ``length`` its
    n-gram size, and ``examples`` a few observed arg lists for parameterisation.
    """
    sequence: tuple[str, ...]
    count: int
    length: int
    examples: list[dict] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return "_then_".join(self.sequence)


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
            "error": str(result.get("error", ""))[:300] if is_error else "",
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

    def mine_failure_patterns(self, min_count: int = 3) -> list[FailurePattern]:
        """Mine repeated ``(skill, error_signature)`` failures from recent history.

        This is intentionally deterministic and local: it gives `/evolve`
        something auditable to show before any LLM-generated patch or
        auto-merge path is considered.
        """
        clusters: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for item in self._history:
            if item.get("success"):
                continue
            skill = str(item.get("skill", ""))
            signature = self._normalise_error_signature(str(item.get("error", "")))
            if not skill or not signature:
                continue
            clusters[(skill, signature)].append(item)

        patterns: list[FailurePattern] = []
        for (skill, signature), examples in clusters.items():
            if len(examples) < min_count:
                continue
            patterns.append(FailurePattern(
                skill_name=skill,
                error_signature=signature,
                count=len(examples),
                examples=examples[-3:],
                suggested_action=self._suggest_failure_action(skill, signature),
            ))
        patterns.sort(key=lambda p: (-p.count, p.skill_name, p.error_signature))
        return patterns

    def mine_success_sequences(
        self,
        min_count: int = 3,
        n_values: tuple[int, ...] = (2, 3),
    ) -> list["SequencePattern"]:
        """Mine repeated runs of consecutive SUCCESSFUL skill calls from recent history.

        A failure breaks a run, so only genuinely end-to-end successful sub-pipelines are counted.
        Deterministic and local (like :meth:`mine_failure_patterns`): it gives the self-evolution
        pipeline an auditable set of candidate sequences to wrap into a skill before any code is written.
        Note: ``_history`` is a flat cross-task log, so adjacency is best-effort until a per-session
        learner (see ``learner_from_trajectory``) supplies cleaner boundaries.
        """
        runs: list[list[dict]] = []
        current: list[dict] = []
        for item in self._history:
            if item.get("success"):
                current.append(item)
            else:
                if len(current) >= min(n_values):
                    runs.append(current)
                current = []
        if len(current) >= min(n_values):
            runs.append(current)

        counters: dict[int, Counter] = {n: Counter() for n in n_values}
        examples: dict[tuple[str, ...], list[dict]] = defaultdict(list)
        for run in runs:
            names = [str(i.get("skill", "")) for i in run]
            for n in n_values:
                for idx in range(len(names) - n + 1):
                    gram = tuple(names[idx:idx + n])
                    if "" in gram:
                        continue
                    counters[n][gram] += 1
                    if len(examples[gram]) < 3:
                        examples[gram].append(
                            {"skills": list(gram),
                             "args": [run[idx + j].get("args", {}) for j in range(n)]}
                        )

        patterns: list[SequencePattern] = []
        for n in n_values:
            for gram, count in counters[n].items():
                if count < min_count:
                    continue
                patterns.append(SequencePattern(
                    sequence=gram, count=count, length=n, examples=examples[gram][:3],
                ))
        # Most frequent first, then longer sequences (a better wrapper), then name for determinism.
        patterns.sort(key=lambda p: (-p.count, -p.length, p.sequence))
        return patterns

    def suggest_workflow(self, query: str) -> list[str]:
        """Suggest a reusable workflow skeleton for a broad user query."""
        q = (query or "").lower()
        steps: list[str] = []
        if any(t in q for t in ("literature", "paper", "related work", "research status")):
            steps.extend(["deep_research", "literature_qa"])
        if any(t in q for t in ("field", "biomarker", "phenotype", "trajectory", "longitudinal")):
            steps.extend(["field_search", "cohort_card"])
        if any(t in q for t in ("cohort", "icd", "case", "control", "prevalence")):
            steps.extend(["cohort_summary", "missing_data"])
        if any(t in q for t in ("model", "predict", "auc", "calibration", "risk")):
            steps.extend([
                "train_model",
                "evaluate_model",
                "calibration",
                "feature_importance",
            ])
        if any(t in q for t in ("trajectory", "longitudinal", "forecast")):
            steps.extend(["trajectory_tokenize", "world_model_audit"])
        steps.extend(["statistical_review", "safety_check", "generate_report"])

        seen: set[str] = set()
        return [s for s in steps if not (s in seen or seen.add(s))]

    def auto_propose_skill_improvement(self, min_count: int = 3) -> list[dict]:
        """Return auditable improvement proposals for repeated failures.

        This does not edit code. It creates deterministic candidate patch
        plans that can be written under ``reports/generated_skills`` and
        reviewed by a human before any custom skill or code change exists.
        """
        proposals = []
        for pattern in self.mine_failure_patterns(min_count=min_count):
            target_path, patch = self._candidate_patch_plan(pattern)
            assessment = classify(patch)
            proposals.append({
                "skill": pattern.skill_name,
                "failure_count": pattern.count,
                "error_signature": pattern.error_signature,
                "suggested_action": pattern.suggested_action,
                "risk": assessment.risk.value,
                "risk_reason": assessment.reason,
                "target_path": target_path,
                "candidate_patch": patch,
                "candidate_patch_required": True,
                "apply_mode": "manual_review",
            })
        return proposals

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

    @staticmethod
    def _normalise_error_signature(error: str) -> str:
        text = " ".join((error or "").strip().split())
        if not text:
            return ""
        lowered = text.lower()
        for token in ("traceback", "file \"", "/users/", "line "):
            if token in lowered:
                lowered = lowered.split(token, 1)[0].strip() or lowered
        return lowered[:120]

    @staticmethod
    def _suggest_failure_action(skill: str, signature: str) -> str:
        sig = signature.lower()
        if "unexpected keyword" in sig or "schema" in sig:
            return "tighten schema validation and planner argument aliases"
        if "field" in sig or "not found" in sig:
            return "insert field_search or semantic field resolution before this skill"
        if "memory" in sig or "oom" in sig or "timeout" in sig:
            return "try a smaller explicit sample only after recording full-dataset infeasibility"
        if skill == "generate_report":
            return "verify report prerequisites and artifact paths before marking plan done"
        return "review repeated failure and consider a targeted custom skill only after tests"

    @staticmethod
    def _candidate_patch_plan(pattern: FailurePattern) -> tuple[str, str]:
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", pattern.skill_name).strip("_").lower() or "skill"
        target_path = f"reports/generated_skills/{slug}_repair_plan.md"
        body = (
            f"# Evolution proposal: {pattern.skill_name}\n\n"
            f"- Failure count: {pattern.count}\n"
            f"- Error signature: `{pattern.error_signature}`\n"
            f"- Suggested action: {pattern.suggested_action}\n"
            "- Apply mode: manual review only\n\n"
            "## Required Review\n\n"
            "1. Confirm this failure is still reproducible with the current code.\n"
            "2. Prefer planner/schema fixes before creating a new skill.\n"
            "3. If a custom skill is justified, add tests that fail before the change.\n"
            "4. Move any approved custom skill manually after review.\n"
        )
        diff_lines = [
            "--- /dev/null",
            f"+++ b/{target_path}",
            "@@",
            *[f"+{line}" for line in body.splitlines()],
            "",
        ]
        return target_path, "\n".join(diff_lines)


def learner_from_trajectory(path: "str | Path", *, memory: "Optional[LongTermMemory]" = None) -> ToolLearner:
    """Reconstruct a :class:`ToolLearner` from a runtime ``trajectory.jsonl`` (the durable v3 log).

    The runtime engine records tool calls to ``trajectory.jsonl`` rather than feeding ``ToolLearner``
    directly; this bridges the two so success-sequence mining can run over a persisted session. Each
    ``tool_call_completed`` event contributes one recorded call in order, with success taken from the
    event ``state`` (``failed``/``error`` — or a non-empty result error — count as a failure).
    """
    learner = ToolLearner(memory=memory)
    p = Path(path)
    if not p.exists():
        return learner
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = record.get("event") if isinstance(record, dict) else None
        event = event if isinstance(event, dict) else record
        if str(event.get("type", "")) != "tool_call_completed":
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        tool = str(payload.get("tool") or result.get("name") or "")
        if not tool:
            continue
        state = str(payload.get("state") or "").lower()
        error = str(result.get("error") or "")
        # A non-success terminal state (failure, cancellation, or timeout) must break a "successful"
        # run, so a cancelled/timed-out call never slips into a mined success sequence.
        failed = state in {"failed", "error", "cancelled", "canceled", "timeout", "timed_out"} or bool(error)
        learner.record(tool, {}, {"error": error or "failed"} if failed else {"summary": "ok"})
    return learner
