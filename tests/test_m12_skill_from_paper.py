"""Tests for M12: synthesize_skill_from_paper (orchestration + methodology pre-gate + review-branch)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from biobank_agent.runtime.method_contract import MethodContract
from biobank_agent.runtime.self_evolve import apply_proposal
from biobank_agent.runtime.skill_from_paper import gate_test_source, synthesize_skill_from_paper


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)


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


def _clean_patch_fn(proposal):
    # Full-file payload (the apply gate accepts this form); a self-passing test stands in for the skill.
    proposal.diff = "def test_ok():\n    assert 1 + 1 == 2\n"
    proposal.test_commands = [f"pytest {proposal.target_path} -q"]
    return proposal


def test_end_to_end_lands_on_review_branch_and_classifies(tmp_path):
    repo = _init_repo(tmp_path)
    contract = MethodContract(name="sc_qc", summary="single-cell QC and normalization",
                              outputs=["filtered h5ad"], postconditions=["obs has n_genes_by_counts"])
    out = synthesize_skill_from_paper(
        "paper.pdf",
        paper_reader=lambda s: "scanpy single-cell QC workflow",
        contract_extractor=lambda t: contract,
        patch_fn=_clean_patch_fn,
        apply_fn=apply_proposal,
        repo_root=repo,
    )
    assert out["status"] == "review_branch", out
    assert out["review_only"] is True
    assert out["leaf"]  # filed into the tree (a node id or pending_classification)
    assert not (repo / "custom_skills" / "sc_qc.py").exists()  # never auto-merged onto the live tree


def test_methodology_pregate_rejects_before_generation(tmp_path):
    called = []

    def patch_fn(p):
        called.append(1)
        return p

    # RNA velocity with no spliced/unspliced -> velocity_without_splicing (block) -> reject pre-generation.
    contract = MethodContract(name="velo", summary="RNA velocity on the single-cell UMAP", outputs=["velocity"])
    out = synthesize_skill_from_paper(
        "p", paper_reader=lambda s: "x", contract_extractor=lambda t: contract,
        patch_fn=patch_fn, apply_fn=apply_proposal, repo_root=tmp_path,
    )
    assert out["status"] == "rejected_methodology"
    assert any(b["issue"] == "velocity_without_splicing" for b in out["blocks"])
    assert not called, "patch generation must not run when the method is rejected"


def test_no_contract_aborts_without_proposal(tmp_path):
    out = synthesize_skill_from_paper(
        "p", paper_reader=lambda s: "some text", contract_extractor=lambda t: MethodContract(name="x"),
        patch_fn=_clean_patch_fn, apply_fn=apply_proposal, repo_root=tmp_path,
    )
    assert out["status"] == "no_contract"


def test_no_paper_aborts():
    out = synthesize_skill_from_paper(
        "missing", paper_reader=lambda s: "", contract_extractor=lambda t: MethodContract(name="x", outputs=["y"]),
        patch_fn=_clean_patch_fn, apply_fn=apply_proposal, repo_root=".",
    )
    assert out["status"] == "no_paper"


def test_gate_test_source_is_deterministic_and_non_vacuous():
    src = gate_test_source(MethodContract(name="sc_qc", postconditions=["obs has leiden"]), "sc_qc")
    assert "check_artifact" in src and "assert any(" in src
    assert "assert True" not in src
    compile(src, "<gate_test>", "exec")  # valid Python
