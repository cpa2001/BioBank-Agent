"""Tree-of-Thought reasoning — explore multiple analytical paths.

When the agent faces a question with multiple valid approaches (e.g.,
"What's the best way to predict Type 2 Diabetes?"), this module generates
and evaluates N candidate reasoning paths before committing to one.

Architecture:
  1. BRANCH: Generate N candidate approaches (diverse, not redundant)
  2. EVALUATE: Score each branch on feasibility, data availability, rigor
  3. SELECT: Return ranked paths for the agent to execute

Integrated into the `think` skill — activated when the thought contains
branching decisions. Also used by enhanced PlanMode for task decomposition.

Reference: Yao et al. "Tree of Thoughts: Deliberate Problem Solving with
Large Language Models" (NeurIPS 2023).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ThoughtPath:
    """A single reasoning path through the thought tree."""
    approach: str              # e.g. "Use blood biochemistry markers only"
    steps: list[str]           # ordered list of concrete steps
    strengths: list[str]       # why this approach is good
    risks: list[str]           # potential issues
    score: float = 0.0        # 0.0-1.0 evaluation score
    rationale: str = ""        # why this score

    def summary(self) -> str:
        steps_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self.steps))
        return (
            f"**{self.approach}** (score: {self.score:.2f})\n"
            f"{steps_str}\n"
            f"Strengths: {', '.join(self.strengths)}\n"
            f"Risks: {', '.join(self.risks)}"
        )


@dataclass
class ThoughtTreeResult:
    """Complete result of tree-of-thought exploration."""
    question: str
    paths: list[ThoughtPath]
    recommended: ThoughtPath | None = None
    consensus: str = ""        # what all paths agree on

    def summary(self) -> str:
        parts = [f"## Tree-of-Thought: {self.question}\n"]
        if self.consensus:
            parts.append(f"**Consensus:** {self.consensus}\n")
        for i, p in enumerate(self.paths, 1):
            parts.append(f"\n### Path {i}: {p.summary()}")
        if self.recommended:
            parts.append(f"\n**Recommended:** {self.recommended.approach}")
        return "\n".join(parts)


class ThoughtTree:
    """Explore multiple reasoning paths for complex analytical decisions.

    Usage::

        tree = ThoughtTree(llm_client)
        result = tree.explore("Best approach to predict E11?", context="...")
        print(result.recommended.approach)
    """

    def __init__(
        self,
        llm: LLMClient,
        max_branches: int = 3,
        max_depth: int = 2,
    ):
        self.llm = llm
        self.max_branches = max_branches
        self.max_depth = max_depth

    def explore(
        self,
        question: str,
        context: str = "",
        available_skills: list[str] | None = None,
    ) -> ThoughtTreeResult:
        """Generate and evaluate multiple reasoning paths.

        Parameters
        ----------
        question : str
            The analytical question to explore.
        context : str
            Current session context (available data, past analyses).
        available_skills : list[str], optional
            Names of available skills to use in steps.

        Returns
        -------
        ThoughtTreeResult with ranked paths.
        """
        # Step 1: Generate diverse branches
        paths = self._generate_branches(question, context, available_skills)

        if not paths:
            return ThoughtTreeResult(question=question, paths=[])

        # Step 2: Evaluate and score each branch
        scored_paths = self._evaluate_branches(paths, question, context)

        # Step 3: Find consensus (what all paths agree on)
        consensus = self._find_consensus(scored_paths)

        # Step 4: Sort by score
        scored_paths.sort(key=lambda p: p.score, reverse=True)
        recommended = scored_paths[0] if scored_paths else None

        return ThoughtTreeResult(
            question=question,
            paths=scored_paths,
            recommended=recommended,
            consensus=consensus,
        )

    def _generate_branches(
        self,
        question: str,
        context: str,
        available_skills: list[str] | None,
    ) -> list[ThoughtPath]:
        """Generate N diverse analytical approaches."""
        skills_str = ""
        if available_skills:
            skills_str = f"\nAvailable analysis tools: {', '.join(available_skills[:20])}"

        prompt = f"""You are a biobank research strategist. Given this analytical question,
generate {self.max_branches} DIVERSE approaches to solve it. Each approach should be
meaningfully different (not just parameter variations).

**Question:** {question}
{f"**Context:** {context[:500]}" if context else ""}
{skills_str}

Output a JSON array where each element has:
- "approach": short name for this approach
- "steps": array of concrete steps (use tool names if available)
- "strengths": array of advantages
- "risks": array of potential issues

Example:
[
  {{
    "approach": "Metabolic biomarker panel",
    "steps": ["Check prevalence of E11", "Build case-control cohort", "Train XGBoost with metabolic markers", "Evaluate with cross-validation"],
    "strengths": ["Well-established biomarkers", "Large literature support"],
    "risks": ["May miss novel markers", "Confounding with medication use"]
  }}
]

Respond with ONLY the JSON array."""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "You are a biobank research strategist. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
            )

            text = response.text.strip()
            # Extract JSON from possible markdown code blocks
            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    clean = part.strip().removeprefix("json").strip()
                    if clean.startswith("["):
                        text = clean
                        break

            data = json.loads(text)
            if not isinstance(data, list):
                return []

            paths = []
            for item in data[:self.max_branches]:
                paths.append(ThoughtPath(
                    approach=item.get("approach", "Unknown"),
                    steps=item.get("steps", []),
                    strengths=item.get("strengths", []),
                    risks=item.get("risks", []),
                ))
            return paths

        except Exception as e:
            logger.warning("Branch generation failed: %s", e)
            return []

    def _evaluate_branches(
        self,
        paths: list[ThoughtPath],
        question: str,
        context: str,
    ) -> list[ThoughtPath]:
        """Score each branch on feasibility, rigor, and novelty."""
        paths_summary = "\n".join(
            f"Path {i+1}: {p.approach}\n  Steps: {', '.join(p.steps[:5])}"
            for i, p in enumerate(paths)
        )

        prompt = f"""Evaluate these analytical approaches for the question: {question}
{f"Context: {context[:300]}" if context else ""}

{paths_summary}

Score each path on:
1. Feasibility (can it be done with available data/tools?)
2. Scientific rigor (is the methodology sound?)
3. Novelty/insight (will it produce interesting findings?)

Output a JSON array with one object per path:
[{{"path": 1, "score": 0.85, "rationale": "Strong approach because..."}}]

Respond with ONLY the JSON array."""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "You are a scientific reviewer. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
            )

            text = response.text.strip()
            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    clean = part.strip().removeprefix("json").strip()
                    if clean.startswith("["):
                        text = clean
                        break

            scores = json.loads(text)
            for item in scores:
                idx = item.get("path", 0) - 1
                if 0 <= idx < len(paths):
                    paths[idx].score = float(item.get("score", 0.5))
                    paths[idx].rationale = item.get("rationale", "")

        except Exception as e:
            logger.warning("Branch evaluation failed: %s. Assigning equal scores.", e)
            for p in paths:
                p.score = 0.5

        return paths

    def _find_consensus(self, paths: list[ThoughtPath]) -> str:
        """Identify what all paths agree on (common steps/elements)."""
        if len(paths) < 2:
            return ""

        # Find common steps (simplified: check for overlapping keywords)
        all_steps_flat = [step.lower() for p in paths for step in p.steps]
        if not all_steps_flat:
            return ""

        # Count word frequency across all steps
        from collections import Counter
        words = Counter()
        for step in all_steps_flat:
            words.update(step.split())

        # Find words that appear in steps from ALL paths
        n_paths = len(paths)
        common = []
        for p in paths:
            for step in p.steps:
                step_words = set(step.lower().split())
                # Check if this step's key concepts appear across paths
                high_freq = [w for w in step_words if words[w] >= n_paths and len(w) > 3]
                if high_freq:
                    common.append(step)
                    break

        if common:
            return f"All paths include: {'; '.join(common[:3])}"
        return ""


def is_branching_question(thought: str) -> bool:
    """Detect whether a thought contains branching decisions.

    Returns True if the thought asks about choosing between approaches,
    comparing methods, or exploring alternatives.
    """
    import re
    text = thought.lower()
    patterns = [
        r"(which|what|how).{0,30}(best|better|optimal|approach|method|strategy|model|use)",
        r"(compare|versus|vs\.?).{0,30}(model|approach|method|feature|marker|algorithm)",
        r"(should I|should we|could we|could I).{0,20}(use|try|choose|select|pick)",
        r"(multiple|several|different|alternative).{0,20}(approach|way|method|option|path)",
        r"(trade.?off|pros? and cons?|advantage|disadvantage|strengths? and weakness)",
        r"(explore|consider|evaluate).{0,20}(option|approach|alternative|possibility)",
        r"(compare|对比|比较).{0,30}(approach|method|方法|模型|策略)",
    ]
    return any(re.search(pat, text) for pat in patterns)
