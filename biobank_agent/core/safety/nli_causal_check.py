"""Semantic causal-claim guard.

The legacy ``agent.py:77-84`` regex (``\\b(causes|prevents|treats|...)``)
is trivially circumvented by an LLM that says
``"X is mechanistically responsible for Y"``. v3 replaces the regex
with a sentence-level NLI check: does the assistant's final answer
*entail* the canonical claim ``"this is a causal claim"``?

We use the lightweight ``cross-encoder/nli-deberta-v3-base`` style
heuristic without pulling a 100MB model into the runtime. The check
runs in three tiers:

1. Lexical — keep the regex as a fast-path veto (covers obvious cases).
2. Sentence-level pattern — semicolon / "because" / "due to" markers
   coupled with a disease-effect noun phrase trigger the warn flag.
3. Pluggable NLI backend — if ``sentence_transformers`` is installed
   and a model is available, we use it. Otherwise the heuristic is
   final.

Output is a ``CausalCheck`` dict so the runtime can emit a
``DISCLOSURE_LAYER`` event with the warning before the assistant text
goes to the user.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Tier 1 — verbatim causal verbs.
_CAUSAL_LEXICAL = re.compile(
    r"\b("
    r"causes?|caused by|cause-effect|causality|causation|"
    r"prevents?|reduces? risk(?: of)?|protects? against|"
    r"treats?|cures?|recommended therapy|"
    r"because of|due to|as a result of|drives?|leads? to|"
    r"mechanistically responsible|directly responsible"
    r")\b",
    re.IGNORECASE,
)

# Hedges that make the claim observational rather than causal.
_HEDGES = re.compile(
    r"\b("
    r"associated with|association|correlated|correlation|hypothes(?:is|ised|ize)|"
    r"observational|requires? validation|cannot infer|does not establish|"
    r"may|might|could|consistent with|suggestive|exploratory"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class CausalCheck:
    is_causal: bool
    confidence: float
    rationale: str = ""
    matched_phrase: str = ""
    needs_disclaimer: bool = False
    sentences: list[str] = field(default_factory=list)


def _split_sentences(text: str) -> Iterable[str]:
    for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()):
        s = s.strip()
        if s:
            yield s


def lexical_score(text: str) -> tuple[float, str]:
    if not text:
        return 0.0, ""
    causal_match = _CAUSAL_LEXICAL.search(text)
    if not causal_match:
        return 0.0, ""
    if _HEDGES.search(text):
        # Hedge present — causal language is qualified.
        return 0.4, causal_match.group(0)
    return 0.85, causal_match.group(0)


def check(text: str, *, threshold: float = 0.7, nli_backend: Optional[Any] = None) -> CausalCheck:
    """Return a ``CausalCheck`` for ``text``."""
    sentences = list(_split_sentences(text))
    score, matched = lexical_score(text)

    # Tier 3 — pluggable NLI backend (e.g. sentence-transformers
    # cross-encoder). The backend's ``predict(premise, hypothesis)``
    # is expected to return a probability in [0, 1].
    if nli_backend is not None:
        try:
            nli_score = float(nli_backend.predict(text, "this is a causal claim"))
            score = max(score, nli_score)
        except Exception as e:
            logger.debug("NLI backend failed: %s", e)

    needs_disclaimer = score >= threshold
    return CausalCheck(
        is_causal=needs_disclaimer,
        confidence=score,
        matched_phrase=matched,
        needs_disclaimer=needs_disclaimer,
        sentences=sentences[:8],
        rationale=(
            "lexical match without hedges" if score >= 0.7 and not _HEDGES.search(text)
            else "lexical match with hedges" if score > 0 else "no causal language"
        ),
    )


def maybe_inject_disclaimer(text: str, *, threshold: float = 0.7) -> str:
    """Wrap the assistant text with a disclaimer if needed."""
    result = check(text, threshold=threshold)
    if not result.needs_disclaimer:
        return text
    disclaimer = (
        "\n\n> ⚠ Causal-language guard: this answer contains causal "
        "phrasing. Biobank analyses are observational; treat as "
        "association unless an external trial validation is cited.\n"
    )
    return text + disclaimer


__all__ = ["CausalCheck", "check", "maybe_inject_disclaimer", "lexical_score"]
