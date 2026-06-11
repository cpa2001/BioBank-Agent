"""里程碑5/M12: the live @skill synthesize_skill_from_paper — gates + review-branch apply."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from biobank_agent.skills.synthesize_skill import _new_file_patch, synthesize_skill_from_paper


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)


def _init_repo(tmp_path):
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


_CLEAN_CODE = (
    "from biobank_agent.registry import skill\n\n\n"
    '@skill(name="sc_qc_demo", description="demo single-cell QC", parameters={})\n'
    "def sc_qc_demo(*, ctx=None) -> dict:\n"
    '    return {"status": "ok"}\n'
)
_CLEAN_CONTRACT = {
    "name": "sc_qc", "summary": "single-cell QC and normalization",
    "inputs": ["h5ad"], "outputs": ["qc_summary"], "postconditions": ["n_obs > 0"],
}


def _ctx(repo=None, enabled=True):
    return SimpleNamespace(settings=SimpleNamespace(skill_synthesis_enabled=enabled),
                           workspace_root=str(repo) if repo else None)


def test_new_file_patch_applies_cleanly_in_git(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    patch = _new_file_patch("custom_skills/a.py", "x = 1\ny = 2\n")
    (repo / "p.diff").write_text(patch)
    res = subprocess.run(["git", "-C", str(repo), "apply", "p.diff"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (repo / "custom_skills" / "a.py").read_text() == "x = 1\ny = 2\n"


def test_disabled_by_default():
    out = synthesize_skill_from_paper("sc_qc", _CLEAN_CODE, _CLEAN_CONTRACT, ctx=_ctx(enabled=False))
    assert out["status"] == "disabled"


def test_unsafe_code_is_rejected_before_apply():
    out = synthesize_skill_from_paper("evil", "import os\nos.system('rm -rf /tmp/x')\n",
                                      _CLEAN_CONTRACT, ctx=_ctx(enabled=True))
    assert out["status"] == "unsafe_code"
    assert out["errors"]


def test_methodology_sin_rejects_before_apply():
    sinful = {"name": "vel", "summary": "single-cell RNA velocity from the neighbor graph",
              "outputs": ["velocity"], "postconditions": ["a velocity field"]}
    out = synthesize_skill_from_paper("vel", _CLEAN_CODE, sinful, ctx=_ctx(enabled=True))
    assert out["status"] == "rejected_methodology"
    assert out["blocks"]


def test_happy_path_lands_on_review_branch_and_classifies(tmp_path):
    repo = _init_repo(tmp_path)
    out = synthesize_skill_from_paper("sc_qc", _CLEAN_CODE, _CLEAN_CONTRACT,
                                      summary="single-cell QC", ctx=_ctx(repo))
    assert out["status"] == "review_branch", out
    assert out["review_only"] is True
    assert out["skill_name"] == "sc_qc"
    assert out["leaf"]
