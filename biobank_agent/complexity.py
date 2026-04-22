"""Task complexity classification — decides single-model vs multi-model routing.

Factors considered:
  - Query length and structural complexity
  - Number of distinct analytical steps implied
  - Presence of comparison / review / verification language
  - Session history (repeated failures suggest harder problem)
  - Explicit user requests for multi-model collaboration

The classifier runs heuristics first (zero API cost for 90%+ of queries).
Only ambiguous queries trigger LLM-based classification.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import AnalysisRecord

logger = logging.getLogger(__name__)


class Strategy(Enum):
    """Multi-model routing strategy."""
    SINGLE = "single"          # Default: one model (fast path)
    DEBATE = "debate"          # N models propose → cross-critique → judge selects
    SUPERVISOR = "supervisor"  # Planner decomposes → specialists execute → merge
    ENSEMBLE = "ensemble"      # Parallel answers → vote / merge


@dataclass
class ComplexityScore:
    """Result of complexity analysis."""
    score: float               # 0.0 (trivial) → 1.0 (very complex)
    strategy: Strategy         # recommended routing strategy
    factors: dict = field(default_factory=dict)
    reason: str = ""


# ── Heuristic patterns ────────────────────────────────────────

_MULTI_STEP_KEYWORDS = re.compile(
    r"(?:"
    r"\bcompare\b|\bcontrast\b|\bversus\b|\bvs\.?\b|\bdebate\b|\breview\b|"
    r"\bverify\b|\bcritique\b|\bvalidate\b|\bcross.?check\b|\bdouble.?check\b|"
    r"\bmultiple\b|\bfirst.+then\b|\bstep\s*\d|"
    r"综合|对比|讨论|多模型|验证|审查|比较|评估.*对比"
    r")",
    re.IGNORECASE,
)

_COMPLEX_TASK_KEYWORDS = re.compile(
    r"(?:"
    r"\bpipeline\b|\bworkflow\b|\bend.?to.?end\b|\bcomprehensive\b|\bsystematic\b|"
    r"\bcomplete analysis\b|\bdiscovery\b|\bhypothesis\b|\bresearch plan\b|"
    r"\bmeta.?analysis\b|\bliterature\b|\bsurvey\b|\bbatch\b|"
    r"全面|研究方案|多个疾病|流程|所有|系统性"
    r")",
    re.IGNORECASE,
)

_SIMPLE_TASK_KEYWORDS = re.compile(
    r"(?:"
    r"\bhello\b|\bhi\b|\bhelp\b|\bwhat is\b|\bshow me\b|\blist\b|\bcount\b|"
    r"\bhow many\b|\bstatus\b|\btop \d+\b|\bprevalence\b|\bsingle\b|\bone\b|"
    r"简单|查看|显示|你好"
    r")",
    re.IGNORECASE,
)


def classify_complexity(
    query: str,
    records: list[AnalysisRecord] | None = None,
    threshold: float = 0.7,
) -> ComplexityScore:
    """Classify a query's complexity using heuristics.

    Parameters
    ----------
    query : str
        The user's natural-language query.
    records : list[AnalysisRecord], optional
        Recent analysis history (for failure detection).
    threshold : float
        Score above this triggers multi-model strategy (default 0.7).

    Returns
    -------
    ComplexityScore
    """
    score = 0.0
    factors = {}

    # ── Factor 1: Query length ────────────────
    n_chars = len(query)
    n_words = len(query.split())
    if n_chars > 500 or n_words > 80:
        score += 0.25
        factors["long_query"] = True
    elif n_chars < 50:
        score -= 0.15
        factors["short_query"] = True

    # ── Factor 2: Multi-step language ─────────
    multi_matches = _MULTI_STEP_KEYWORDS.findall(query)
    if multi_matches:
        # Scale by number of matches (more keywords = more complex)
        bonus = min(0.5, 0.15 * len(multi_matches))
        score += bonus
        factors["multi_step_keywords"] = list(set(multi_matches))[:5]

    # ── Factor 3: Complex task language ───────
    complex_matches = _COMPLEX_TASK_KEYWORDS.findall(query)
    if complex_matches:
        score += 0.25
        factors["complex_keywords"] = list(set(complex_matches))[:5]

    # ── Factor 4: Simple task language ────────
    simple_matches = _SIMPLE_TASK_KEYWORDS.findall(query)
    if simple_matches and not multi_matches and not complex_matches:
        score -= 0.25
        factors["simple_keywords"] = list(set(simple_matches))[:5]

    # ── Factor 5: Question marks / clauses ────
    n_questions = query.count("?") + query.count("？")
    n_clauses = len(re.split(r"[;；。\n]", query))
    if n_questions > 2 or n_clauses > 3:
        score += 0.15
        factors["multiple_questions"] = n_questions
        factors["multiple_clauses"] = n_clauses

    # ── Factor 6: Recent failure history ──────
    if records:
        recent_errors = sum(
            1 for r in records[-5:]
            if isinstance(r.key_results, dict) and "error" in r.key_results
        )
        if recent_errors >= 2:
            score += 0.2
            factors["recent_failures"] = recent_errors

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    # ── Route to strategy ─────────────────────
    # Explicit requests override threshold
    query_lower = query.lower()
    explicit_debate = any(kw in query_lower for kw in ("debate", "讨论", "多模型", "对比", "比较"))
    explicit_pipeline = any(kw in query_lower for kw in ("pipeline", "workflow", "end-to-end", "步骤", "阶段", "流程"))

    if explicit_debate and score >= 0.4:
        strategy = Strategy.DEBATE
        reason = "Explicit debate/comparison request detected"
    elif explicit_pipeline and score >= 0.2:
        strategy = Strategy.SUPERVISOR
        reason = "Multi-step pipeline detected — supervisor decomposition"
    elif score < threshold:
        strategy = Strategy.SINGLE
        reason = "Low complexity — single model sufficient"
    elif score >= 0.85:
        strategy = Strategy.DEBATE
        reason = "Very high complexity — debate for thorough analysis"
    else:
        strategy = Strategy.ENSEMBLE
        reason = "Moderately complex — ensemble for robustness"

    return ComplexityScore(
        score=round(score, 3),
        strategy=strategy,
        factors=factors,
        reason=reason,
    )
