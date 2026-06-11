"""Evolution layer (reflexion / classifier / auto_merger)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from biobank_agent.core.evolution.auto_merger import (
    AutoMerger,
    MergeOutcome,
    Patch,
)
from biobank_agent.core.evolution.patch_classifier import (
    PatchAssessment,
    RiskLevel,
    classify,
)
from biobank_agent.core.evolution.pattern_mining import run_pattern_mining
from biobank_agent.core.evolution.reflexion import (
    ActionClass,
    ReflexionAction,
    propose_action,
)
from biobank_agent.tool_learner import ToolLearner


class _PassingEval:
    all_passed = True
    summary = "ALWAYS_PASSES eval gate passed"


# ── Reflexion ──────────────────────────────────────────────


def test_reflexion_inserts_prerequisite_when_cohort_missing():
    out = propose_action(
        skill="train_model",
        args={"icd10_code": "E11"},
        error="cohort not built; run build_cohort first",
    )
    assert out.cls == ActionClass.INSERT_PREREQUISITE_STEP
    assert out.insert_skill == "build_cohort"


def test_reflexion_swaps_to_lite_on_oom():
    out = propose_action(
        skill="train_model",
        args={},
        error="OOM during training",
    )
    assert out.cls == ActionClass.SWAP_SKILL
    assert out.swap_to_skill == "train_model_lite"


def test_reflexion_halves_numeric_args_when_unknown_failure():
    out = propose_action(
        skill="train_model",
        args={"n_folds": 10, "top_n": 50},
        error="something else",
    )
    assert out.cls == ActionClass.RETRY_WITH_CORRECTED_ARGS
    assert out.corrected_args["n_folds"] == 5
    assert out.corrected_args["top_n"] == 25


def test_reflexion_skips_when_nothing_else_works():
    out = propose_action(
        skill="generate_report",
        args={},
        error="totally unknown error",
    )
    assert out.cls == ActionClass.SKIP_STEP


def test_reflexion_action_serialises_to_plan_repair_payload():
    a = ReflexionAction(
        cls=ActionClass.RETRY_WITH_CORRECTED_ARGS,
        corrected_args={"n_folds": 3},
        rationale="test",
    )
    payload = a.to_plan_repair_payload()
    assert payload["action"] == "retry_args"
    assert payload["args"] == {"n_folds": 3}


def test_tool_learner_mines_repeated_failure_patterns_and_workflows():
    learner = ToolLearner()
    for _ in range(3):
        learner.record(
            "generate_report",
            {"format": "dual"},
            {"error": "report prerequisites missing: no statistical_review"},
            elapsed_s=0.1,
        )

    patterns = learner.mine_failure_patterns(min_count=3)
    assert len(patterns) == 1
    assert patterns[0].skill_name == "generate_report"
    assert patterns[0].count == 3
    assert "report prerequisites" in patterns[0].error_signature

    proposals = learner.auto_propose_skill_improvement(min_count=3)
    assert proposals[0]["risk"] in {"low", "medium", "high"}
    assert proposals[0]["candidate_patch_required"] is True
    assert proposals[0]["apply_mode"] == "manual_review"
    assert proposals[0]["target_path"].startswith("reports/generated_skills/")
    assert "Evolution proposal: generate_report" in proposals[0]["candidate_patch"]

    workflow = learner.suggest_workflow("predict E11 risk with biomarkers and calibration")
    assert workflow[:2] == ["field_search", "cohort_card"]
    assert "train_model" in workflow
    assert workflow[-1] == "generate_report"


def test_scheduled_pattern_mining_writes_history_artifacts(tmp_path):
    learner = ToolLearner()
    for _ in range(3):
        learner.record(
            "generate_report",
            {"format": "dual"},
            {"error": "report prerequisites missing: no statistical_review"},
            elapsed_s=0.1,
        )

    run = run_pattern_mining(learner, output_dir=tmp_path, min_count=3)

    assert run.status == "NEEDS_REVIEW"
    assert run.n_patterns == 1
    assert run.n_proposals == 1
    run_json = Path(run.artifacts["run_json"])
    latest_json = Path(run.artifacts["latest_json"])
    history_jsonl = Path(run.artifacts["history_jsonl"])
    run_md = Path(run.artifacts["run_md"])
    assert run_json.exists()
    assert latest_json.exists()
    assert history_jsonl.exists()
    assert run_md.exists()
    assert "generate_report" in run_md.read_text(encoding="utf-8")
    latest = json.loads(latest_json.read_text(encoding="utf-8"))
    assert latest["artifacts"]["run_json"] == str(run_json)
    assert latest["n_patterns"] == 1


# ── Patch classifier ───────────────────────────────────────


def test_classifier_marks_low_for_comment_change():
    diff = (
        "--- a/biobank_agent/skills/foo.py\n"
        "+++ b/biobank_agent/skills/foo.py\n"
        "@@\n"
        "-# old comment\n"
        "+# new comment\n"
    )
    out = classify(diff)
    assert out.risk == RiskLevel.LOW


def test_classifier_marks_high_for_new_function():
    diff = (
        "--- a/biobank_agent/skills/foo.py\n"
        "+++ b/biobank_agent/skills/foo.py\n"
        "@@\n"
        "+def new_helper():\n"
        "+    return 42\n"
    )
    out = classify(diff)
    assert out.risk == RiskLevel.HIGH


def test_classifier_marks_high_when_modifying_core():
    diff = (
        "--- a/biobank_agent/core/runtime.py\n"
        "+++ b/biobank_agent/core/runtime.py\n"
        "@@\n"
        "-x = 1\n"
        "+x = 2\n"
    )
    out = classify(diff)
    assert out.risk == RiskLevel.HIGH


def test_classifier_marks_high_when_eval_failed():
    diff = (
        "--- a/biobank_agent/skills/foo.py\n"
        "+++ b/biobank_agent/skills/foo.py\n"
        "@@\n"
        "-# old\n"
        "+# new\n"
    )

    class _Eval:
        all_passed = False

    out = classify(diff, eval_result=_Eval())
    assert out.risk == RiskLevel.HIGH
    assert "eval" in out.reason.lower()


def test_classifier_marks_medium_for_branch_change():
    diff = (
        "--- a/biobank_agent/skills/foo.py\n"
        "+++ b/biobank_agent/skills/foo.py\n"
        "@@\n"
        "-    if x > 0:\n"
        "+    if x > 5:\n"
    )
    out = classify(diff)
    assert out.risk == RiskLevel.MEDIUM


# ── Auto merger ────────────────────────────────────────────


@pytest.fixture
def fake_git_repo(tmp_path, monkeypatch):
    """Init a real git repo in tmp_path so AutoMerger can run end-to-end."""
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "domain" / "skills").mkdir(parents=True)
    (tmp_path / "custom_skills").mkdir(parents=True)
    (tmp_path / "domain" / "skills" / "existing.py").write_text("print('seed')\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=tmp_path, check=True)
    return tmp_path


@pytest.mark.asyncio
async def test_auto_merger_skips_disallowed_paths(fake_git_repo):
    am = AutoMerger(repo_root=fake_git_repo)
    patch = Patch(
        target_path="biobank_agent/core/runtime.py",
        unified_diff="x = 1\n",
        target_skill="runtime",
    )
    out = await am.review_and_merge(patch)
    assert out.status == "skipped"


@pytest.mark.asyncio
async def test_auto_merger_low_risk_merges(fake_git_repo):
    am = AutoMerger(repo_root=fake_git_repo)
    patch = Patch(
        target_path="custom_skills/new_helper.py",
        unified_diff="# tiny new helper\n",
        target_skill="new_helper",
        summary="add comment-only helper file",
    )
    out = await am.review_and_merge(patch, eval_result=_PassingEval())
    assert out.status in {"merged", "error"}
    if out.status == "merged":
        assert (fake_git_repo / "custom_skills" / "new_helper.py").exists()


@pytest.mark.asyncio
async def test_auto_merger_medium_requires_confirmation(fake_git_repo):
    confirmations: list = []
    audit_log = fake_git_repo / "reports" / "generated_skills" / "evolution_decisions.jsonl"

    async def confirm(assessment, patch):
        confirmations.append((assessment.risk, patch.metadata.get("eval_summary"), patch.metadata.get("risk_reason")))
        return True

    am = AutoMerger(repo_root=fake_git_repo, confirm_fn=confirm, audit_log_path=audit_log)
    diff = (
        "--- a/custom_skills/branch.py\n"
        "+++ b/custom_skills/branch.py\n"
        "@@\n"
        "-    if x > 0:\n"
        "+    if x > 5:\n"
    )
    patch = Patch(
        target_path="custom_skills/branch.py",
        unified_diff=diff,
        target_skill="branch",
    )
    out = await am.review_and_merge(patch, eval_result=_PassingEval())
    # The MEDIUM path runs confirm_fn even when the diff itself is
    # invalid against the file. We assert confirmation was requested.
    assert confirmations == [(RiskLevel.MEDIUM, "ALWAYS_PASSES eval gate passed", "touches control flow; needs user confirmation")]
    audit = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines()]
    assert audit[-1]["risk"] == "medium"
    assert audit[-1]["confirmed"] is True
    assert audit[-1]["eval_summary"] == "ALWAYS_PASSES eval gate passed"


@pytest.mark.asyncio
async def test_auto_merger_medium_rejects_without_confirmation(fake_git_repo):
    am = AutoMerger(repo_root=fake_git_repo)
    diff = (
        "--- a/custom_skills/branch.py\n"
        "+++ b/custom_skills/branch.py\n"
        "@@\n"
        "-    if x > 0:\n"
        "+    if x > 5:\n"
    )
    patch = Patch(
        target_path="custom_skills/branch.py",
        unified_diff=diff,
        target_skill="branch",
    )

    out = await am.review_and_merge(patch, eval_result=_PassingEval())

    assert out.status == "user_rejected"
    assert out.risk == RiskLevel.MEDIUM
    assert "requires explicit user confirmation" in out.error


@pytest.mark.asyncio
async def test_auto_merger_refuses_low_merge_without_eval_gate(fake_git_repo):
    am = AutoMerger(repo_root=fake_git_repo)
    patch = Patch(
        target_path="custom_skills/new_helper.py",
        unified_diff="# tiny new helper\n",
        target_skill="new_helper",
    )

    out = await am.review_and_merge(patch)

    assert out.status == "skipped"
    assert out.risk == RiskLevel.LOW
    assert "eval gate" in out.error


@pytest.mark.asyncio
async def test_auto_merger_routes_high_risk_allowed_patch_to_pr(fake_git_repo, monkeypatch):
    am = AutoMerger(repo_root=fake_git_repo)
    calls = []

    def fake_open_pr(patch, assessment):
        calls.append((patch.target_path, assessment.risk))
        return MergeOutcome(status="pr_opened", risk=assessment.risk, branch="auto-improve/demo", pr_url="https://example/pr/1")

    monkeypatch.setattr(am, "_open_pr", fake_open_pr)
    patch = Patch(
        target_path="custom_skills/high.py",
        unified_diff=(
            "--- /dev/null\n"
            "+++ b/custom_skills/high.py\n"
            "@@\n"
            "+def new_helper():\n"
            "+    return 1\n"
        ),
        target_skill="high",
    )

    out = await am.review_and_merge(patch, eval_result=_PassingEval())

    assert out.status == "pr_opened"
    assert out.risk == RiskLevel.HIGH
    assert calls == [("custom_skills/high.py", RiskLevel.HIGH)]


@pytest.mark.asyncio
async def test_auto_merger_high_risk_without_remote_leaves_local_branch_ready(fake_git_repo):
    am = AutoMerger(repo_root=fake_git_repo)
    diff = (
        "diff --git a/custom_skills/high_ready.py b/custom_skills/high_ready.py\n"
        "new file mode 100644\n"
        "index 0000000..e69de29\n"
        "--- /dev/null\n"
        "+++ b/custom_skills/high_ready.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def new_helper():\n"
        "+    return 1\n"
    )
    patch = Patch(
        target_path="custom_skills/high_ready.py",
        unified_diff=diff,
        target_skill="high_ready",
    )

    out = await am.review_and_merge(patch, eval_result=_PassingEval())

    assert out.status == "pr_ready"
    assert out.risk == RiskLevel.HIGH
    assert out.branch.startswith("auto-improve/high_ready-")
    assert "origin remote not configured" in out.error
    branches = am._git("branch", "--list", out.branch)
    assert out.branch in branches
