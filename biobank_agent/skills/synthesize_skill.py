"""Live paper → skill synthesis, through the safety gates.

The agent reads a method paper (read_paper / deep_research / read_pdf), extracts a typed method
contract and drafts the skill module, then calls this. The skill does NOT trust that input blindly:
it runs the methodology pre-gate (reject consensus statistical/omics sins before any code lands),
the load-safety validator (reject unsafe imports / import-time side effects), then applies the
code to a REVIEW BRANCH only (never auto-merged) with a non-vacuous co-located gate test, and files
the new skill into the tree. Default OFF (``skill_synthesis_enabled``).
"""

from __future__ import annotations

import os
import re

from biobank_agent.registry import skill


def _new_file_patch(path: str, content: str) -> str:
    """A git-apply unified-diff hunk that creates ``path`` with ``content`` (a new file)."""
    body = content if content.endswith("\n") else content + "\n"
    lines = body.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    hunk = "".join("+" + line + "\n" for line in lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        f"new file mode 100644\n"
        f"--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{hunk}"
    )


@skill(
    name="synthesize_skill_from_paper",
    description=(
        "Turn a method paper into a tree-filed skill THROUGH THE SAFETY GATES. You supply the "
        "extracted method contract and the drafted skill module source (after reading the paper "
        "with read_paper/deep_research); this runs the methodology pre-gate (rejects consensus "
        "statistical/omics sins), the load-safety validator, then applies the code to a REVIEW "
        "BRANCH ONLY (never auto-merged) with a gate test, and files it into the skill tree. "
        "Off unless skill_synthesis_enabled is set."
    ),
    parameters={
        "skill_name": {"type": "string", "description": "snake_case name for the new skill"},
        "skill_code": {"type": "string", "description": "The drafted skill module source (a @skill function)"},
        "contract": {
            "type": "object",
            "description": ("MethodContract fields: name, summary, inputs, outputs, preconditions, "
                            "postconditions, statistical_assumptions, citation"),
        },
        "summary": {"type": "string", "description": "One-line method summary", "default": ""},
        "paper_source": {"type": "string", "description": "Paper path / URL / DOI for provenance", "default": ""},
    },
    required=["skill_name", "skill_code", "contract"],
)
def synthesize_skill_from_paper(skill_name, skill_code, contract=None, summary="", paper_source="", *, ctx=None) -> dict:
    from biobank_agent.runtime.method_contract import MethodContract
    from biobank_agent.runtime.self_evolve import apply_proposal
    from biobank_agent.runtime.skill_from_paper import gate_test_source
    from biobank_agent.runtime.skill_from_paper import synthesize_skill_from_paper as _orchestrate
    from biobank_agent.skills.generator import SkillGenerator

    settings = getattr(ctx, "settings", None)
    if not getattr(settings, "skill_synthesis_enabled", False):
        return {"status": "disabled",
                "message": "Set skill_synthesis_enabled to enable paper -> skill synthesis."}

    code = str(skill_code or "").strip()
    if not code:
        return {"status": "no_code", "message": "Provide the drafted skill module source as skill_code."}

    # Load-safety gate BEFORE any apply: refuse unsafe imports / import-time side effects.
    ok_code, code_reason = SkillGenerator.validate_code(code)
    ok_load, load_reason = SkillGenerator.validate_load_safety(code)
    if not (ok_code and ok_load):
        errors = SkillGenerator.get_validation_errors(code) or [r for r in (code_reason, load_reason) if r]
        return {"status": "unsafe_code", "errors": errors}

    contract_obj = MethodContract.from_dict(contract if isinstance(contract, dict) else {})
    name = re.sub(r"[^a-z0-9_]+", "_", str(skill_name).strip().lower()).strip("_") or "synthesized_skill"
    repo_root = getattr(ctx, "workspace_root", None) or os.getcwd()

    def _patch_fn(proposal):
        gate_path = f"custom_skills/{name}_gate_test.py"
        proposal.diff = (_new_file_patch(proposal.target_path, code)
                         + _new_file_patch(gate_path, gate_test_source(contract_obj, name)))
        proposal.test_commands = [f"pytest {gate_path} -q"]
        return proposal

    # Reuse the tested orchestration: paper_reader/contract_extractor are trivial here because the
    # agent already did the reading/extraction; this skill is the methodology + apply safety gate.
    return _orchestrate(
        paper_source or skill_name,
        paper_reader=lambda _src: (summary or contract_obj.as_text() or str(skill_name)),
        contract_extractor=lambda _text: contract_obj,
        patch_fn=_patch_fn,
        apply_fn=apply_proposal,
        repo_root=repo_root,
        skill_name=name,
    )
