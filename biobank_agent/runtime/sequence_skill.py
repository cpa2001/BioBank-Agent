"""Auto-capture a frequent successful skill sequence into a review-only wrapper skill.

Self-evolution, safe by construction: given a :class:`~biobank_agent.tool_learner.SequencePattern`
(a run of consecutive successful skill calls that recurred), synthesize a DETERMINISTIC, self-contained
recipe skill — no LLM, no network. The generated file passes the same AST safety validation the runtime
enforces at author and load time (``SkillGenerator.validate_code``), targets the allow-listed
``custom_skills/`` directory, and is meant to be applied through
``self_evolve.apply_proposal(..., force_review_branch=True)`` so even a verified skill lands on a review
branch, never on ``main`` and never in core.

Skills cannot invoke other skills in-process, so the recipe does not execute the sequence; it is a
discoverable, self-documenting capability that returns the ordered steps (and representative arguments)
for replay via the existing ``replay_pipeline`` skill.
"""

from __future__ import annotations

import re
from typing import Any

from biobank_agent.skills.generator import SkillGenerator

from .evolution import EvolutionProposal

# Allow-listed landing dir (in self_evolve.DEFAULT_APPLY_ALLOW_PATHS; not counted by the README skill total).
AUTOCAPTURE_SKILLS_DIR = "custom_skills"
_MAX_NAME_LEN = 80


def sequence_skill_name(sequence: "tuple[str, ...] | list[str]") -> str:
    """Deterministic, identifier-safe skill name for a sequence (``pipeline_a_then_b``)."""
    parts = [re.sub(r"[^a-z0-9_]+", "", str(s).strip().lower()) for s in sequence]
    parts = [p for p in parts if p]
    if not parts:
        return "pipeline_captured"
    name = "pipeline_" + "_then_".join(parts)
    return name[:_MAX_NAME_LEN].rstrip("_") or "pipeline_captured"


def render_sequence_skill(
    sequence: "tuple[str, ...] | list[str]", *, count: int = 0, examples: list[dict] | None = None
) -> str:
    """Render a self-contained, validated recipe-skill file (skill + inline test) for a sequence."""
    name = sequence_skill_name(sequence)
    steps = [str(s) for s in sequence]
    example_args: list[Any] = list((examples or [{}])[0].get("args", []) or []) if examples else []
    desc = f"Auto-captured pipeline recipe: {' then '.join(steps)} (observed {int(count)}x)."
    return f'''"""Auto-generated pipeline recipe skill (self-evolution, review-only).

Captured from {int(count)} repeated successful runs of the sequence below. Calling it returns the
ordered steps (and representative arguments) so the workflow can be replayed via replay_pipeline; it
executes no sub-skills in-process."""

from biobank_agent.registry import skill


@skill(
    name="{name}",
    description="{desc}",
    parameters={{}},
    required=[],
)
def {name}(*, ctx=None) -> dict:
    steps = {steps!r}
    example_args = {example_args!r}
    return {{
        "status": "success",
        "pipeline": steps,
        "observed_count": {int(count)},
        "example_args": example_args,
        "note": "Auto-captured from repeated successful runs; replay via replay_pipeline.",
    }}


def test_{name}():
    result = {name}()
    assert result["status"] == "success"
    assert result["pipeline"] == {steps!r}
'''


def synthesize_sequence_skill(
    sequence: "tuple[str, ...] | list[str]",
    *,
    count: int = 0,
    examples: list[dict] | None = None,
    skills_dir: str = AUTOCAPTURE_SKILLS_DIR,
) -> dict[str, Any]:
    """Render + safety-validate a recipe skill; return its name, target path, code, and test command."""
    name = sequence_skill_name(sequence)
    code = render_sequence_skill(sequence, count=count, examples=examples)
    ok, msg = SkillGenerator.validate_code(code)
    if not ok:
        raise ValueError(f"generated sequence skill failed safety validation: {msg}")
    target_path = f"{skills_dir.rstrip('/')}/{name}.py"
    return {
        "name": name,
        "target_path": target_path,
        "code": code,
        "test_commands": [f"pytest {target_path} -q"],
    }


def propose_skill_from_sequence(pattern: Any, *, skills_dir: str = AUTOCAPTURE_SKILLS_DIR) -> EvolutionProposal:
    """Build a review-only :class:`EvolutionProposal` that captures one sequence as a skill.

    ``diff`` carries the full new-file content (``apply_patch_transactionally`` writes it directly for a
    new file); route it through ``apply_proposal(..., force_review_branch=True)`` to land it on a review
    branch under the allow-listed skills dir.
    """
    seq = tuple(getattr(pattern, "sequence", ()) or ())
    count = int(getattr(pattern, "count", 0) or 0)
    examples = list(getattr(pattern, "examples", []) or [])
    built = synthesize_sequence_skill(seq, count=count, examples=examples, skills_dir=skills_dir)
    return EvolutionProposal(
        proposal_id=f"seqskill:{built['name']}",
        category="skill_from_sequence",
        summary=f"Auto-capture pipeline '{' -> '.join(seq)}' (observed {count}x) as skill {built['name']}",
        evidence=[{"sequence": list(seq), "count": count, "examples": examples[:3]}],
        risk="review_required",
        approval_required=True,
        apply_mode="review_only",
        target_path=built["target_path"],
        diff=built["code"],
        test_commands=built["test_commands"],
    )


def propose_skills_from_sequences(
    patterns: list[Any], *, top_k: int = 3, skills_dir: str = AUTOCAPTURE_SKILLS_DIR
) -> list[EvolutionProposal]:
    """Turn the top mined sequences into review-only proposals, collapsing rotations/duplicates.

    ``_history`` is a flat cross-task log, so a repeated pipeline shows up as several rotated n-grams of
    the same skill set; keying dedup on the skill SET keeps the highest-count representative of each.
    """
    proposals: list[EvolutionProposal] = []
    seen: set[frozenset[str]] = set()
    for pattern in patterns or []:
        seq = tuple(getattr(pattern, "sequence", ()) or ())
        if not seq:
            continue
        key = frozenset(seq)
        if key in seen:
            continue
        seen.add(key)
        proposals.append(propose_skill_from_sequence(pattern, skills_dir=skills_dir))
        if len(proposals) >= top_k:
            break
    return proposals


__all__ = [
    "AUTOCAPTURE_SKILLS_DIR",
    "sequence_skill_name",
    "render_sequence_skill",
    "synthesize_sequence_skill",
    "propose_skill_from_sequence",
    "propose_skills_from_sequences",
]
