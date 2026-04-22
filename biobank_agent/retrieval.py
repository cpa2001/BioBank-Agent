"""Agentic RAG — autonomous retrieval decision engine.

Monitors agent reasoning for knowledge gaps and autonomously
triggers retrieval when uncertainty is detected. Retrieved context
is integrated into domain memory (Tier 5) for long-term use.

Research basis: Self-RAG (Asai et al., 2023), Agentic RAG patterns (2025).

Uncertainty signals:
  - Explicit: "I'm not sure", "need to check", "unclear"
  - Temporal: references to recent events, "2024", "2025", "latest"
  - Academic: "reference", "citation", "paper", "study"
  - Domain: unknown ICD10 codes, unfamiliar biomarker names
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMClient
    from .memory import LongTermMemory
    from .registry import SkillRegistry

logger = logging.getLogger(__name__)


@dataclass
class RetrievalDecision:
    """Result of retrieval analysis."""
    should_retrieve: bool = False
    query: str = ""
    source: str = ""          # "web_search" | "fetch_paper" | "recall_session"
    reason: str = ""
    confidence: float = 0.0


# ── Uncertainty signal patterns ────────────────────────────

_UNCERTAINTY_PATTERNS = [
    (re.compile(r"I('m| am) not (sure|certain)", re.I), "explicit_uncertainty", 0.9),
    (re.compile(r"I (don't|do not) (know|have)", re.I), "explicit_gap", 0.85),
    (re.compile(r"need(s?) to (check|verify|look up|search|confirm)", re.I), "verification_need", 0.8),
    (re.compile(r"\b(unclear|uncertain|ambiguous|unknown)\b", re.I), "ambiguity", 0.7),
    (re.compile(r"\b(recent|latest|current|up.?to.?date)\b", re.I), "temporal_need", 0.6),
    (re.compile(r"\b20(2[4-6]|3\d)\b"), "year_reference", 0.5),
    (re.compile(r"\b(reference|citation|paper|study|published|literature)\b", re.I), "academic_need", 0.7),
    (re.compile(r"\b(guideline|protocol|standard of care|clinical practice)\b", re.I), "clinical_ref", 0.65),
]

_DOMAIN_UNKNOWN_PATTERNS = [
    # Unknown ICD10 pattern (4+ char code not in common set)
    (re.compile(r"\b[A-Z]\d{2,3}(?:\.\d{1,2})?\b"), "icd10_reference", 0.3),
    # Biomarker names that might need lookup
    (re.compile(r"\b(biomarker|marker|assay|metabolite|protein|gene)\b", re.I), "biomarker_ref", 0.2),
]


class AgenticRAG:
    """Autonomous retrieval decision engine.

    Usage::

        rag = AgenticRAG(llm, memory)
        decision = rag.analyze(thought, context)
        if decision.should_retrieve:
            # Auto-trigger web_search or fetch_paper skill
            result = registry.execute(decision.source, {"query": decision.query}, ctx)
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        memory: Optional[LongTermMemory] = None,
        threshold: float = 0.6,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.threshold = threshold

    def analyze(
        self,
        thought: str,
        context: str = "",
    ) -> RetrievalDecision:
        """Analyze text for knowledge gaps and decide whether to retrieve.

        Parameters
        ----------
        thought : str
            Agent's internal reasoning (from think tool) or response text.
        context : str
            Current session context for disambiguation.

        Returns
        -------
        RetrievalDecision with should_retrieve, query, and source.
        """
        if not thought or len(thought) < 10:
            return RetrievalDecision()

        # ── Heuristic signal detection ────────
        max_confidence = 0.0
        triggered_signals = []

        for pattern, signal_name, confidence in _UNCERTAINTY_PATTERNS:
            if pattern.search(thought):
                triggered_signals.append((signal_name, confidence))
                max_confidence = max(max_confidence, confidence)

        for pattern, signal_name, confidence in _DOMAIN_UNKNOWN_PATTERNS:
            if pattern.search(thought):
                triggered_signals.append((signal_name, confidence))
                # Domain patterns have lower base confidence
                max_confidence = max(max_confidence, confidence)

        if max_confidence < self.threshold:
            return RetrievalDecision(
                should_retrieve=False,
                reason=f"Below threshold ({max_confidence:.2f} < {self.threshold})",
                confidence=max_confidence,
            )

        # ── Determine source and query ────────
        source, query = self._route_retrieval(thought, triggered_signals)

        return RetrievalDecision(
            should_retrieve=True,
            query=query,
            source=source,
            reason=f"Signals: {', '.join(s[0] for s in triggered_signals)}",
            confidence=max_confidence,
        )

    def _route_retrieval(
        self,
        thought: str,
        signals: list[tuple[str, float]],
    ) -> tuple[str, str]:
        """Determine the best retrieval source and construct a query."""
        signal_names = {s[0] for s in signals}

        # Academic references → fetch papers
        if signal_names & {"academic_need", "clinical_ref"}:
            query = self._extract_academic_query(thought)
            return "web_search", query

        # Temporal needs → web search
        if signal_names & {"temporal_need", "year_reference"}:
            query = self._extract_temporal_query(thought)
            return "web_search", query

        # Session recall
        if "explicit_gap" in signal_names and self.memory:
            query = self._extract_key_terms(thought)
            return "recall_session", query

        # Default: web search with extracted terms
        query = self._extract_key_terms(thought)
        return "web_search", query

    def _extract_academic_query(self, thought: str) -> str:
        """Extract a search query for academic literature."""
        # Remove common filler words, keep domain terms
        terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[a-z]+){0,2}\b", thought)
        if terms:
            return " ".join(terms[:5]) + " biomedical study"
        return thought[:100] + " research paper"

    def _extract_temporal_query(self, thought: str) -> str:
        """Extract a search query for recent information."""
        terms = self._extract_key_terms(thought)
        return terms + " 2025 2026"

    def _extract_key_terms(self, thought: str) -> str:
        """Extract key terms from thought for search."""
        # Remove stopwords and common agent phrases
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "i", "we", "you", "it",
            "this", "that", "these", "those", "of", "in", "to", "for",
            "with", "on", "at", "by", "from", "not", "but", "and", "or",
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", thought.lower())
        key_words = [w for w in words if w not in stopwords][:8]
        return " ".join(key_words) if key_words else thought[:80]
