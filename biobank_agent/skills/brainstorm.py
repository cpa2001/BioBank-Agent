"""Brainstorm skill — structured research ideation for biomedical topics.

Generates candidate research directions with hypotheses, approaches,
data requirements, and UK Biobank feasibility assessments.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_DIRECTION_TEMPLATE_QUICK = """\
### Direction {n}: {title}

**Hypothesis**: {hypothesis}
**Approach**: {approach}
**Data Requirements**: {data_requirements}
**UKB Feasibility**: {feasibility}
"""

_DIRECTION_TEMPLATE_DEEP = """\
### Direction {n}: {title}

**Hypothesis**: {hypothesis}

**Approach**: {approach}

**Data Requirements**: {data_requirements}

**UKB Feasibility**: {feasibility}

**Evidence Support**:
- What existing literature supports this direction?
- What preliminary signals (if any) exist in UK Biobank?
- Strength of prior evidence: [weak / moderate / strong]

**Potential Pitfalls**:
- Confounders to address:
- Sample size concerns:
- Measurement issues:
- Multiple testing burden:
- Healthy volunteer bias considerations:

**Estimated Effort**: [low / medium / high]

---
"""


def _build_brainstorm_document(
    topic: str,
    n_ideas: int,
    depth: str,
    context_summary: str,
) -> str:
    """Build the structured brainstorm markdown document."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    template = _DIRECTION_TEMPLATE_DEEP if depth == "deep" else _DIRECTION_TEMPLATE_QUICK

    sections = [
        f"# Research Brainstorm: {topic}",
        f"**Date**: {now}",
        f"**Depth**: {depth}",
        f"**Requested directions**: {n_ideas}",
        "",
        "---",
        "",
        "## 1. Topic & Scope",
        "",
        f"**Topic**: {topic}",
        "",
        "**Scope refinement**: Define the precise research question, target population, "
        "and outcome of interest. Consider both primary and exploratory endpoints.",
        "",
        "---",
        "",
        "## 2. Current State of Knowledge",
        "",
        "Summarise what is already known about this topic, particularly:",
        "- Key findings from large cohort studies",
        "- Established risk factors and biomarkers",
        "- Known genetic associations",
        "- Gaps in current understanding",
        "",
    ]

    if context_summary and context_summary != "No analyses performed yet.":
        sections.extend([
            "### Session Context",
            "",
            f"```\n{context_summary}\n```",
            "",
        ])

    sections.extend([
        "---",
        "",
        "## 3. Research Directions",
        "",
        "For each direction, evaluate feasibility using UK Biobank data.",
        "",
    ])

    for i in range(1, n_ideas + 1):
        sections.append(
            template.format(
                n=i,
                title=f"[Direction {i} title]",
                hypothesis=f"[Specific, testable hypothesis for direction {i}]",
                approach="[Study design, statistical methods, analysis pipeline]",
                data_requirements="[UKB fields needed, sample size estimates, inclusion/exclusion criteria]",
                feasibility="[Assessment of whether UK Biobank has the required data and sufficient power]",
            )
        )

    if depth == "deep":
        sections.extend([
            "",
            "## 4. Cross-cutting Considerations",
            "",
            "- **Common confounders** across all directions:",
            "- **Shared data requirements** (fields used by multiple directions):",
            "- **Synergies** (directions that could feed into each other):",
            "- **Priority ranking** (which direction to pursue first and why):",
            "",
        ])

    sections.extend([
        "",
        "## Summary & Recommended Next Steps",
        "",
        "1. [Most promising direction and rationale]",
        "2. [Immediate next action (data pull, literature search, pilot analysis)]",
        "3. [Key risk to monitor]",
    ])

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------

@skill(
    name="brainstorm",
    description="Generate research ideas and hypotheses for a biomedical topic. "
                "Uses structured ideation with evidence grounding and critical evaluation.",
    parameters={
        "topic": {
            "type": "string",
            "description": "Research topic or question to brainstorm about",
        },
        "n_ideas": {
            "type": "integer",
            "description": "Number of research directions to generate",
            "default": 5,
        },
        "depth": {
            "type": "string",
            "description": "'quick' (ideas only) or 'deep' (ideas + evidence + critique)",
            "default": "quick",
        },
    },
    required=["topic"],
)
def brainstorm(topic: str, n_ideas: int = 5, depth: str = "quick", *, ctx=None) -> dict:
    """Generate a structured brainstorm workspace for a research topic.

    Creates a markdown document in ``{ctx.report_dir}/brainstorm/`` with
    *n_ideas* candidate research directions.  When *depth* is ``"deep"``,
    each direction includes evidence support analysis and pitfall assessment.

    Returns
    -------
    dict
        ``{"topic": ..., "directions": [...], "workspace": str, "summary": str}``
    """
    # Validate inputs
    depth = depth.lower().strip()
    if depth not in ("quick", "deep"):
        depth = "quick"
    n_ideas = max(1, min(n_ideas, 20))  # clamp

    # Gather session context
    context_summary = ""
    if ctx is not None and hasattr(ctx, "state"):
        try:
            context_summary = ctx.state.context_summary()
        except Exception:
            pass

    # Build the brainstorm document
    document = _build_brainstorm_document(topic, n_ideas, depth, context_summary)

    # Save to workspace
    workspace_path: str | None = None
    try:
        if ctx is not None and hasattr(ctx, "report_dir"):
            brainstorm_dir = Path(ctx.report_dir) / "brainstorm"
        else:
            brainstorm_dir = Path("./brainstorm")
        brainstorm_dir.mkdir(parents=True, exist_ok=True)

        # Sanitise topic for filename
        safe_topic = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)
        safe_topic = safe_topic.strip().replace(" ", "_")[:60]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"brainstorm_{safe_topic}_{timestamp}.md"

        file_path = brainstorm_dir / filename
        file_path.write_text(document, encoding="utf-8")
        workspace_path = str(file_path)
    except Exception as exc:
        logger.warning("Could not save brainstorm document: %s", exc)

    # Build direction stubs for the return value
    directions = []
    for i in range(1, n_ideas + 1):
        directions.append({
            "number": i,
            "title": f"[Direction {i} — to be filled by analysis]",
            "depth": depth,
        })

    summary = (
        f"Brainstorm workspace created for '{topic}' with {n_ideas} direction slots "
        f"({'deep' if depth == 'deep' else 'quick'} mode). "
        f"The agent should now fill in each direction with specific hypotheses, "
        f"methods, and UK Biobank feasibility assessments."
    )

    return {
        "topic": topic,
        "n_ideas": n_ideas,
        "depth": depth,
        "directions": directions,
        "workspace": workspace_path,
        "document": document,
        "summary": summary,
    }
