"""Phase 4 self-evolution: auto-capture a frequent successful sequence into a review-only skill.

Pins the safety contract: the generated skill passes AST validation, targets the allow-listed
``custom_skills/`` dir, and — routed through ``apply_proposal(force_review_branch=True)`` — lands on a
review branch (verified by its own inline test) and NEVER on the live tree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from biobank_agent.runtime.self_evolve import apply_proposal
from biobank_agent.runtime.sequence_skill import (
    propose_skill_from_sequence,
    propose_skills_from_sequences,
    sequence_skill_name,
    synthesize_sequence_skill,
)
from biobank_agent.skills.generator import SkillGenerator
from biobank_agent.tool_learner import SequencePattern, ToolLearner, learner_from_trajectory


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "tester")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "custom_skills").mkdir()
    (repo / "custom_skills" / "__keep__").write_text("")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _branches(repo: Path) -> str:
    return _git(repo, "branch", "--list")


def test_sequence_skill_name_is_identifier_safe():
    assert sequence_skill_name(("cohort_summary", "train_model")) == "pipeline_cohort_summary_then_train_model"
    # non-identifier characters are stripped, never injected
    assert sequence_skill_name(("a; import os", "b")) == "pipeline_aimportos_then_b"
    assert sequence_skill_name(()) == "pipeline_captured"


def test_synthesized_skill_passes_safety_validation_and_lands_in_custom_skills():
    built = synthesize_sequence_skill(("cohort_summary", "train_model"), count=4)
    assert built["target_path"].startswith("custom_skills/")
    ok, msg = SkillGenerator.validate_code(built["code"])
    assert ok, msg
    assert built["test_commands"] == [f"pytest {built['target_path']} -q"]
    # The generated recipe executes no sub-skills and shells out to nothing.
    assert "subprocess" not in built["code"] and "shell_exec" not in built["code"]


def test_crafted_skill_name_cannot_break_out_of_the_generated_file():
    # A step name with quotes/newlines/code must not escape into executable code: it may appear only as
    # inert data inside a repr'd string literal, the name is reduced to an identifier, and nothing runs.
    built = synthesize_sequence_skill(('a"""\nimport os\nos.system("x")', "b"), count=1)
    assert built["name"].startswith("pipeline_") and built["name"].replace("_", "").isalnum()
    ok, msg = SkillGenerator.validate_code(built["code"])
    assert ok, msg  # the AST safety gate accepts it — the payload is safe
    ns: dict = {}
    exec(compile(built["code"], built["target_path"], "exec"), ns)  # execs cleanly, no injected import runs
    assert "os" not in ns  # the crafted `import os` stayed data, never executed
    assert ns[built["name"]]()["status"] == "success"


def test_proposals_collapse_rotations_and_respect_top_k():
    patterns = [
        SequencePattern(sequence=("a", "b", "c"), count=6, length=3),
        SequencePattern(sequence=("b", "c", "a"), count=6, length=3),  # rotation of the same set
        SequencePattern(sequence=("d", "e"), count=5, length=2),
        SequencePattern(sequence=("f", "g"), count=4, length=2),
    ]
    props = propose_skills_from_sequences(patterns, top_k=2)
    assert len(props) == 2
    assert all(p.category == "skill_from_sequence" and p.apply_mode == "review_only" for p in props)
    assert all(p.diff and p.test_commands and p.target_path.startswith("custom_skills/") for p in props)


def test_autocaptured_skill_applies_to_review_branch_never_live_tree(tmp_path):
    repo = _init_repo(tmp_path)
    proposal = propose_skill_from_sequence(SequencePattern(sequence=("cohort_summary", "train_model"), count=4, length=2))

    result = apply_proposal(proposal, repo_root=repo, force_review_branch=True)

    assert result.status == "review_branch", result.error
    assert result.tests_passed  # the generated skill's inline test ran and passed in the worktree
    assert "evolve/" in _branches(repo)  # parked for human review
    assert not (repo / proposal.target_path).exists()  # never auto-merged onto main


def test_learner_from_trajectory_rebuilds_successful_sequences(tmp_path):
    # A synthetic trajectory.jsonl with three successful A→B→C runs and one failure of B.
    import json

    events = []
    for _ in range(3):
        for name in ("cohort_summary", "train_model", "evaluate_model"):
            events.append({"event": {"type": "tool_call_completed",
                                     "payload": {"tool": name, "state": "completed", "result": {"summary": "ok"}}}})
    events.append({"event": {"type": "tool_call_completed",
                             "payload": {"tool": "train_model", "state": "failed", "result": {"error": "boom"}}}})
    traj = tmp_path / "trajectory.jsonl"
    traj.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

    learner = learner_from_trajectory(traj)
    seqs = {p.sequence: p.count for p in learner.mine_success_sequences(min_count=3)}
    assert seqs.get(("cohort_summary", "train_model", "evaluate_model")) == 3
    # missing file yields an empty learner, not an error
    assert isinstance(learner_from_trajectory(tmp_path / "nope.jsonl"), ToolLearner)
