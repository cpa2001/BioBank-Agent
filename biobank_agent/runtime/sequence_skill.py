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

import hashlib
import json
import re
from typing import Any

from biobank_agent.skills.generator import SkillGenerator

from .evolution import EvolutionProposal

# Allow-listed landing dir (in self_evolve.DEFAULT_APPLY_ALLOW_PATHS; not counted by the README skill total).
AUTOCAPTURE_SKILLS_DIR = "custom_skills"
# Auto-captured skills may target ONLY these non-core, allow-listed dirs. Anything else (e.g. a caller
# passing skills_dir="biobank_agent/core") is refused up front so a generated skill can never reach core,
# even onto a review branch.
ALLOWED_AUTOCAPTURE_DIRS = ("custom_skills", "biobank_agent/skills/custom")
_MAX_NAME_LEN = 80


def _validated_skills_dir(skills_dir: str) -> str:
    norm = str(skills_dir or "").strip().replace("\\", "/").rstrip("/")
    if norm not in ALLOWED_AUTOCAPTURE_DIRS:
        raise ValueError(f"auto-captured skills may only target {ALLOWED_AUTOCAPTURE_DIRS}, not {skills_dir!r}")
    return norm


def _json_safe(value: Any) -> Any:
    """Coerce to JSON-native values so a later ``repr`` embeds only inert primitives (a crafted object
    whose ``__repr__`` returns code can otherwise slip through ``{...!r}``)."""
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def sequence_skill_name(sequence: "tuple[str, ...] | list[str]") -> str:
    """Deterministic, identifier-safe skill name for a sequence (``pipeline_a_then_b``)."""
    parts = [re.sub(r"[^a-z0-9_]+", "", str(s).strip().lower()) for s in sequence]
    parts = [p for p in parts if p]
    if not parts:
        return "pipeline_captured"
    name = "pipeline_" + "_then_".join(parts)
    if len(name) <= _MAX_NAME_LEN:
        return name
    # Truncate long names but keep them collision-free with a short stable hash of the full sequence,
    # so two sequences sharing an 80-char prefix don't produce the same name/path/proposal id.
    digest = hashlib.sha1("\x1f".join(str(s) for s in sequence).encode("utf-8", "replace")).hexdigest()[:8]
    return name[: _MAX_NAME_LEN - 9].rstrip("_") + "_" + digest


def render_sequence_skill(
    sequence: "tuple[str, ...] | list[str]", *, count: int = 0, examples: list[dict] | None = None
) -> str:
    """Render a self-contained, validated recipe-skill file (skill + inline test) for a sequence."""
    name = sequence_skill_name(sequence)
    steps = [str(s) for s in sequence]
    example_args: list[Any] = _json_safe(list((examples or [{}])[0].get("args", []) or []) if examples else [])
    # The executable payload embeds steps/args via repr (safe); the human-facing description is the only
    # place a name is interpolated into a string literal, so strip it to alnum/underscore to close any
    # quote/newline break-out (validate_code is the backstop, but this fails safe up front).
    safe_steps = [re.sub(r"[^A-Za-z0-9_]+", "", s) or "step" for s in steps]
    desc = f"Auto-captured pipeline recipe: {' then '.join(safe_steps)} (observed {int(count)}x)."
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
    safe_dir = _validated_skills_dir(skills_dir)
    name = sequence_skill_name(sequence)
    code = render_sequence_skill(sequence, count=count, examples=examples)
    ok, msg = SkillGenerator.validate_code(code)
    if not ok:
        raise ValueError(f"generated sequence skill failed safety validation: {msg}")
    target_path = f"{safe_dir}/{name}.py"
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


def _rotation_key(seq: tuple[str, ...]) -> tuple[str, ...]:
    """Canonical key that collapses CYCLIC rotations of a sequence but keeps genuinely different
    orderings distinct (the flat ``_history`` log emits rotated n-grams of one repeated pipeline)."""
    if not seq:
        return ()
    return min(tuple(seq[i:] + seq[:i]) for i in range(len(seq)))


def propose_skills_from_sequences(
    patterns: list[Any], *, top_k: int = 3, skills_dir: str = AUTOCAPTURE_SKILLS_DIR
) -> list[EvolutionProposal]:
    """Turn the top mined sequences into review-only proposals, collapsing rotated duplicates only.

    ``_history`` is a flat cross-task log, so a repeated pipeline shows up as several rotated n-grams;
    keying dedup on the canonical rotation keeps the highest-count representative of each while leaving
    genuinely different orderings (e.g. ``a→b→c`` vs ``b→a→c``) as separate proposals.
    """
    proposals: list[EvolutionProposal] = []
    seen: set[tuple[str, ...]] = set()
    for pattern in patterns or []:
        seq = tuple(getattr(pattern, "sequence", ()) or ())
        if not seq:
            continue
        key = _rotation_key(seq)
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
