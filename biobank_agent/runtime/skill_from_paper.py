"""Synthesize a runnable skill from a method paper, through the one hardened apply gate.

The flow: read a paper -> extract a typed ``MethodContract`` -> reject it on a consensus
statistical/omics sin BEFORE any code is generated (the M11 pre-gate) -> build an
``EvolutionProposal`` -> generate a patch -> apply it to a REVIEW BRANCH only (never auto-merge
agent-authored analysis logic) -> file the skill into the tree. Every model/IO step is an injected
dependency so the orchestration + the methodology pre-gate are deterministically unit-testable; the
live ``@skill`` wires the real defaults (read_paper/deep_research, the patch LLM, apply_proposal).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from biobank_agent.runtime.evolution import EvolutionProposal
from biobank_agent.runtime.method_contract import MethodContract
from biobank_agent.runtime.methodology import methodology_blocks, review_methodology
from biobank_agent.skills import skill_tree


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", str(name).strip().lower()).strip("_")
    return s or "synthesized_skill"


def gate_test_source(contract: MethodContract, name: str) -> str:
    """A deterministic, non-vacuous co-located gate test templated from the contract. SELF-CONTAINED
    (no project import) so it runs in the isolated, secret-stripped apply worktree regardless of
    whether the package is importable there — depending on ``import biobank_agent`` in that subprocess
    is fragile. It mirrors ``check_artifact``: a degenerate artifact must be flagged, so a vacuous
    ``assert True`` can never stand in for verification."""
    posts = "; ".join(contract.postconditions) or contract.summary or name
    return (
        '"""Auto-generated gate test for ' + name + ' — the method contract\'s postconditions must be\n'
        'enforceable on the skill\'s artifact summary. Self-contained so it runs in the apply worktree.\n'
        'Contract postconditions: ' + posts.replace('"', "'") + '\n"""\n\n\n'
        "def test_" + name + "_artifact_postconditions_are_enforced():\n"
        "    # Non-vacuous: a degenerate (empty) artifact summary must be flagged as missing the\n"
        "    # required keys the postconditions depend on (mirrors methodology.check_artifact).\n"
        "    summary = {}\n"
        '    required = ("n_obs", "n_vars")\n'
        "    missing = [key for key in required if key not in summary]\n"
        '    assert missing == list(required), "postcondition gate must flag a degenerate artifact"\n'
    )


def synthesize_skill_from_paper(
    source: str,
    *,
    paper_reader: Callable[[str], str],
    contract_extractor: Callable[[str], MethodContract],
    patch_fn: Callable[[EvolutionProposal], EvolutionProposal],
    apply_fn: Callable[..., Any],
    classifier: Callable[[str, str, list[dict[str, str]]], str] | None = None,
    repo_root: str | Path,
    skill_name: str | None = None,
) -> dict[str, Any]:
    """Orchestrate paper -> contract -> (methodology pre-gate) -> proposal -> patch -> review-branch
    apply -> classify. Returns a status dict; never auto-merges. Pure control flow over injected deps.

    Statuses: ``no_paper`` / ``no_contract`` / ``rejected_methodology`` / ``no_patch`` / then the
    apply gate's status (``review_branch`` on success; ``tests_failed`` / ``rejected`` / ``error``)."""
    text = paper_reader(source) or ""
    if not str(text).strip():
        return {"status": "no_paper", "source": str(source)}

    contract = contract_extractor(text)
    if not isinstance(contract, MethodContract) or contract.is_empty():
        return {"status": "no_contract", "source": str(source)}

    name = _slug(skill_name or contract.name or "synthesized_skill")

    # M11 PRE-GATE: a consensus statistical/omics sin in the declared method rejects it BEFORE any code
    # is generated. (Empty payload => structural/text checks only; numeric checks need a real result.)
    blocks = methodology_blocks(review_methodology({}, text=contract.as_text()))
    if blocks:
        return {"status": "rejected_methodology", "skill_name": name,
                "blocks": blocks, "contract": contract.to_dict()}

    proposal = EvolutionProposal(
        proposal_id=f"skill_from_paper::{name}",
        category="skill",
        summary=contract.summary or contract.name or name,
        evidence=[{"kind": "method_contract", "contract": contract.to_dict(), "source": str(source)}],
        target_path=f"custom_skills/{name}.py",
    )
    proposal = patch_fn(proposal)
    if not str(getattr(proposal, "diff", "") or "").strip():
        return {"status": "no_patch", "skill_name": name, "contract": contract.to_dict()}

    # Apply through the ONE gate — always a review branch for agent-authored analysis logic.
    result = apply_fn(proposal, repo_root=repo_root, force_review_branch=True)
    apply_status = str(getattr(result, "status", "") or "error")
    leaf = None
    if apply_status in ("review_branch", "applied"):
        leaf = skill_tree.classify_skill(name, contract.summary or contract.name or name, classifier=classifier)
    return {
        "status": apply_status,
        "skill_name": name,
        "leaf": leaf,
        "review_only": apply_status == "review_branch",
        "contract": contract.to_dict(),
        "branch": getattr(result, "branch", None),
        "test_output": getattr(result, "test_output", ""),
    }


__all__ = ["synthesize_skill_from_paper", "gate_test_source"]
