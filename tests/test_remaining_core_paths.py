"""Tests for remaining isolated skill and data-layer paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biobank_agent.data.catalog import FieldCatalog
from biobank_agent.data.cohort import build_cohort
from biobank_agent.skills import git_clean_push as git_mod
from biobank_agent.skills import min_sample as min_sample_mod
from biobank_agent.skills import recall_session as recall_mod
from biobank_agent.skills import smart_plot as smart_mod


class PlotState:
    def __init__(self):
        self.cohorts = {}
        self.model_metadata = {}
        self.figures = []
        self.records = []


def plot_ctx(tmp_path):
    return SimpleNamespace(report_dir=tmp_path, state=PlotState())


def fake_save_figure(fig, name, report_dir, formats=("png", "svg")):
    return [report_dir / f"{name}.{fmt}" for fmt in formats]


def plot_cohort():
    return pd.DataFrame(
        {
            "eid": np.arange(12),
            "label": [0, 1] * 6,
            "glucose": np.linspace(1, 12, 12),
            "hba1c": np.linspace(12, 1, 12),
            "bmi": [25, 28, 24, 30, 23, 31, 26, 29, 22, 32, 24, 33],
        }
    )


def test_smart_plot_covers_supported_plot_types_and_references(tmp_path, monkeypatch):
    ctx = plot_ctx(tmp_path)
    ctx.state.cohorts["E11"] = plot_cohort()
    ctx.state.model_metadata["E11_xgb"] = {"auc": 0.81234}
    monkeypatch.setattr("biobank_agent.utils.plotting.save_figure", fake_save_figure)
    monkeypatch.setattr(
        "biobank_agent.skills.web_search.web_search",
        lambda query, max_results=3, ctx=None: {"results": [{"title": "ref"}, {"title": "ref2"}, {"title": "ref3"}, {"title": "ref4"}]},
    )

    bar = smart_mod.smart_plot("bar", "cohort:E11", reference_search="Nature PheWAS", ctx=ctx)
    violin = smart_mod.smart_plot("violin", "cohort:E11", style="icml", title="Distribution", ctx=ctx)
    heatmap = smart_mod.smart_plot("heatmap", "cohort:E11", ctx=ctx)
    scatter = smart_mod.smart_plot("scatter", "cohort:E11", ctx=ctx)
    roc = smart_mod.smart_plot("roc", "model:E11_xgb", ctx=ctx)
    km = smart_mod.smart_plot("km", "cohort:E11", ctx=ctx)
    ctx.state.records = [
        SimpleNamespace(skill="cohort_summary", key_results={"n_cases": 6, "n_controls": 6}),
        SimpleNamespace(skill="trajectory_tokenize", key_results={"n_tokens": 42, "n_participants": 12}),
        SimpleNamespace(skill="train_model", key_results={"auc_mean": 0.81, "model_type": "lgbm", "incident_risk_supported": False}),
        SimpleNamespace(skill="evaluate_model", key_results={"mean_auc": 0.80}),
        SimpleNamespace(skill="calibration", key_results={"ece": 0.05}),
        SimpleNamespace(skill="feature_importance", key_results={"top_features": [{"feature": "HbA1c", "importance": 10.0}]}),
        SimpleNamespace(skill="statistical_review", key_results={"overall_assessment": "WARNING"}),
        SimpleNamespace(skill="safety_check", key_results={"overall": "PASS"}),
        SimpleNamespace(skill="world_model_audit", key_results={"safety_status": "PARTIAL", "allowed_claim_type": "association_conditioned_forecast"}),
    ]
    summary = smart_mod.smart_plot("summary", "session", title="Session diagnostics", ctx=ctx)
    fallback = smart_mod.smart_plot("waterfall", "raw:anything", ctx=ctx)

    assert bar["style"] == "nature"
    assert bar["reference_hints"]["search_results"] == [{"title": "ref"}, {"title": "ref2"}, {"title": "ref3"}]
    assert bar["selection_metadata"]["selected_style"] == "nature"
    assert bar["selection_metadata"]["palette"] == {
        "control": "#0072B2",
        "case": "#D55E00",
    }
    assert violin["style"] == "icml"
    assert violin["selection_metadata"]["style_reason"] == "explicit style 'icml' requested"
    assert heatmap["figures"][0].endswith("smart_heatmap_E11.png")
    assert heatmap["selection_metadata"]["figure_width"] == "double"
    assert scatter["plot_type"] == "scatter"
    assert roc["figures"][0].endswith("smart_roc_E11_xgb.png")
    assert km["plot_type"] == "km"
    assert summary["figures"][0].endswith("smart_summary_session.png")
    assert summary["selection_metadata"]["render_path"] == "inline"
    assert summary["selection_metadata"]["figure_width"] == "double"
    assert fallback["figures"][0].endswith("smart_waterfall_anything.png")
    assert fallback["selection_metadata"]["render_path"] == "placeholder"
    assert len(ctx.state.figures) == 16
    plt.close("all")


def test_smart_plot_parse_missing_data_and_error_paths(tmp_path, monkeypatch):
    ctx = plot_ctx(tmp_path)
    monkeypatch.setattr("biobank_agent.utils.plotting.save_figure", fake_save_figure)
    monkeypatch.setattr(
        "biobank_agent.skills.web_search.web_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    assert smart_mod._parse_data_source("field:30750") == ("field", "30750")
    assert smart_mod._parse_data_source("plain") == ("raw", "plain")

    missing = smart_mod.smart_plot("bar", "cohort:missing", reference_search="paper", ctx=ctx)
    error_plot = smart_mod.smart_plot("bar", "cohort:E11:bad/name", ctx=ctx)
    violin_missing = smart_mod.smart_plot("violin", "cohort:missing", ctx=ctx)
    heatmap_missing = smart_mod.smart_plot("heatmap", "cohort:missing", ctx=ctx)
    scatter_missing = smart_mod.smart_plot("scatter", "cohort:missing", ctx=ctx)
    roc_missing = smart_mod.smart_plot("roc", "model:missing", ctx=ctx)
    ctx.state.cohorts["nolabel"] = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    bar_nolabel = smart_mod.smart_plot("bar", "cohort:nolabel", ctx=ctx)
    violin_nolabel = smart_mod.smart_plot("violin", "cohort:nolabel", ctx=ctx)
    heatmap_small = smart_mod.smart_plot("heatmap", "cohort:nolabel", ctx=ctx)
    scatter_nolabel = smart_mod.smart_plot("scatter", "cohort:nolabel", ctx=ctx)

    assert missing["reference_hints"]["search_error"] == "offline"
    assert missing["figures"][0].endswith("smart_bar_missing.png")
    assert "E11_bad/name" in error_plot["figures"][0]
    assert violin_missing["plot_type"] == "violin"
    assert heatmap_missing["plot_type"] == "heatmap"
    assert scatter_missing["plot_type"] == "scatter"
    assert roc_missing["plot_type"] == "roc"
    assert bar_nolabel["plot_type"] == "bar"
    assert violin_nolabel["plot_type"] == "violin"
    assert heatmap_small["plot_type"] == "heatmap"
    assert scatter_nolabel["plot_type"] == "scatter"
    plt.close("all")


def test_smart_plot_handles_axes_arrays_and_plot_exceptions(tmp_path, monkeypatch):
    ctx = plot_ctx(tmp_path)
    ctx.state.cohorts["E11"] = plot_cohort()
    monkeypatch.setattr("biobank_agent.utils.plotting.save_figure", fake_save_figure)

    def array_nature_figure(*args, **kwargs):
        fig, ax = plt.subplots()
        return fig, np.array([ax])

    monkeypatch.setattr("biobank_agent.utils.plotting.nature_figure", array_nature_figure)
    monkeypatch.setattr(smart_mod, "_plot_bar", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad plot")))

    result = smart_mod.smart_plot("bar", "cohort:E11", ctx=ctx)

    assert result["figures"][0].endswith("smart_bar_E11.png")
    plt.close("all")


def min_sample_cohort(n_cases=30, n_controls=120):
    labels = np.array([1] * n_cases + [0] * n_controls)
    n = len(labels)
    return pd.DataFrame(
        {
            "eid": np.arange(n),
            "label": labels,
            "30740-0.0": np.r_[np.linspace(7, 9, n_cases), np.linspace(4, 6, n_controls)],
            "30750-0.0": np.r_[np.linspace(60, 80, n_cases), np.linspace(35, 48, n_controls)],
            "text": ["x"] * n,
        }
    )


def test_min_sample_reuses_cohort_skips_oversized_counts_and_plots(tmp_path, monkeypatch):
    state = SimpleNamespace(cohorts={}, figures=[])
    ctx = SimpleNamespace(
        report_dir=tmp_path,
        state=state,
        dm=SimpleNamespace(),
        settings=SimpleNamespace(subject_id_col="eid"),
    )
    calls = []

    def fake_build_cohort(dm, icd10_code, controls_ratio=0):
        calls.append((icd10_code, controls_ratio))
        return min_sample_cohort()

    monkeypatch.setattr(min_sample_mod, "build_cohort", fake_build_cohort)
    monkeypatch.setattr(min_sample_mod, "save_figure", fake_save_figure)
    monkeypatch.setattr(
        min_sample_mod,
        "cross_val_score",
        lambda *args, **kwargs: np.array([0.7, 0.8]),
    )

    class FakeXGBClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(__import__("sys").modules, "xgboost", SimpleNamespace(XGBClassifier=FakeXGBClassifier))

    result = min_sample_mod.min_sample("E11", case_counts="10,20,50", n_repeats=2, ctx=ctx)
    reused = min_sample_mod.min_sample("E11", case_counts="10", n_repeats=1, ctx=ctx)

    assert calls == [("E11", 0)]
    assert result["total_available_cases"] == 30
    assert result["sample_sizes_tested"] == [10, 20]
    assert result["results"]["10"]["mean"] == 0.75
    assert result["figure"].endswith("min_sample_E11.png")
    assert reused["sample_sizes_tested"] == [10]
    assert len(ctx.state.figures) == 4
    plt.close("all")


def test_min_sample_skips_cross_validation_failures(tmp_path, monkeypatch):
    state = SimpleNamespace(cohorts={"E11_1:all": min_sample_cohort(n_cases=12, n_controls=48)}, figures=[])
    ctx = SimpleNamespace(
        report_dir=tmp_path,
        state=state,
        dm=SimpleNamespace(),
        settings=SimpleNamespace(subject_id_col="eid"),
    )

    def fail_cross_validation(*args, **kwargs):
        raise RuntimeError("fold failed")

    class FakeXGBClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(min_sample_mod, "save_figure", fake_save_figure)
    monkeypatch.setattr(min_sample_mod, "cross_val_score", fail_cross_validation)
    monkeypatch.setitem(__import__("sys").modules, "xgboost", SimpleNamespace(XGBClassifier=FakeXGBClassifier))

    result = min_sample_mod.min_sample("E11", case_counts="10", n_repeats=1, ctx=ctx)

    assert result["sample_sizes_tested"] == []
    assert result["results"] == {}
    assert result["figure"].endswith("min_sample_E11.png")
    plt.close("all")


def test_field_catalog_load_search_and_missing_files(tmp_path):
    category_txt = tmp_path / "category.txt"
    category_txt.write_text("category_id\ttitle\tavailability\nbadrow\n100\tBiochemistry\tpublic\n", encoding="utf-8")

    parts = ["30740", "Glucose"] + [""] * 16
    parts[5] = "31"
    parts[11] = "mmol/L"
    parts[12] = "100"
    parts[17] = "Blood glucose biomarker"
    field_txt = tmp_path / "field.txt"
    field_txt.write_text("field_id\ttitle\nnot-a-field\tHeader\n" + "\t".join(parts) + "\n1\n", encoding="utf-8")

    catalog = FieldCatalog(field_txt, category_txt)

    assert catalog.field_name("30740") == "Glucose"
    assert catalog.field_name("99999") == "Field 99999"
    assert catalog.field_info("30740")["units"] == "mmol/L"
    assert catalog.category_name("100") == "Biochemistry"
    assert catalog.category_name("nope") == "Category nope"
    assert catalog.search("glucose", limit=1)[0]["field_id"] == "30740"
    assert catalog.fields_by_category("100")[0]["title"] == "Glucose"

    missing = FieldCatalog(tmp_path / "missing_fields.txt", tmp_path / "missing_categories.txt")
    assert missing.fields == {}
    assert missing.categories == {}

    assert [f["field_id"] for f in catalog.search("blood", limit=10)] == ["30740"]
    assert catalog.search("not present", limit=10) == []


def test_build_cohort_parameterized_queries_death_cases_and_explicit_fields():
    conn = duckdb.connect(":memory:")
    biomarkers = pd.DataFrame(
        {
            "eid": [1, 2, 3, 4, 5],
            "30740-0.0": [8.0, 9.0, 7.5, 5.0, 4.5],
            "41270-0.0": [1, 1, 1, 0, 0],
        }
    )
    diagnoses = pd.DataFrame({"eid": [1, 2], "diag_icd10": ["E11", "E11.9"]})
    deaths = pd.DataFrame({"eid": [3], "cause_icd10": ["E11"]})
    conn.execute("CREATE TABLE biomarkers AS SELECT * FROM biomarkers")
    conn.execute("CREATE TABLE diagnoses AS SELECT * FROM diagnoses")
    conn.execute("CREATE TABLE deaths AS SELECT * FROM deaths")

    dm = SimpleNamespace(
        conn=conn,
        settings=SimpleNamespace(subject_id_col="eid", diagnoses_code_col="diag_icd10", deaths_code_col="cause_icd10"),
        count_subjects=lambda: 5,
        list_parquet_columns=lambda: ["eid", "30740-0.0", "41270-0.0"],
    )

    default = build_cohort(dm, "E11", controls_ratio=1, random_state=1)
    full_controls = build_cohort(dm, "E11", random_state=1)
    explicit = build_cohort(dm, "E11", controls_ratio=1, biomarker_fields=["30740"], random_state=1)

    assert default["label"].sum() == 3
    assert full_controls["label"].sum() == 3
    assert len(full_controls) == 5
    assert "30740-0.0" in default.columns
    assert "41270-0.0" not in default.columns
    assert explicit.columns.tolist() == ["eid", "30740-0.0", "label"]

    with pytest.raises(ValueError, match="No cases found"):
        build_cohort(dm, "I25", random_state=1)


def test_build_cohort_falls_back_without_deaths_or_subject_count():
    conn = duckdb.connect(":memory:")
    biomarkers = pd.DataFrame({"eid": [1, 2, 3, 4], "30740-0.0": [8.0, 9.0, 5.0, 4.5]})
    diagnoses = pd.DataFrame({"eid": [1, 2], "diag_icd10": ["E11", "E11.9"]})
    conn.execute("CREATE TABLE biomarkers AS SELECT * FROM biomarkers")
    conn.execute("CREATE TABLE diagnoses AS SELECT * FROM diagnoses")

    dm = SimpleNamespace(
        conn=conn,
        settings=SimpleNamespace(subject_id_col="eid", diagnoses_code_col="diag_icd10", deaths_code_col="cause_icd10"),
        count_subjects=lambda: (_ for _ in ()).throw(RuntimeError("count unavailable")),
        list_parquet_columns=lambda: ["eid", "30740-0.0"],
    )

    cohort = build_cohort(dm, "E11", controls_ratio=1, random_state=1)

    assert cohort["label"].sum() == 2
    assert len(cohort) == 4


def test_recall_session_available_empty_and_unavailable():
    no_memory = recall_mod.recall_session("diabetes", ctx=SimpleNamespace())
    assert no_memory == {"results": [], "message": "Session search not available."}

    sessions = SimpleNamespace(search=lambda query, limit=5: [])
    empty = recall_mod.recall_session("diabetes", ctx=SimpleNamespace(memory=SimpleNamespace(sessions=sessions)))
    assert empty["message"] == "No past sessions found matching 'diabetes'."

    sessions = SimpleNamespace(search=lambda query, limit=5: [{"id": "s1", "summary": query}])
    found = recall_mod.recall_session("diabetes", limit=1, ctx=SimpleNamespace(memory=SimpleNamespace(sessions=sessions)))
    assert found["n_results"] == 1
    assert found["results"][0]["id"] == "s1"


def test_git_clean_push_helpers_and_dry_run(monkeypatch):
    def fake_run_for_commits(cmd, cwd=None, timeout=120):
        if cmd.startswith("git log --format=\"%H|"):
            return 0, "\n".join(
                [
                    "abcdef123|Codex|bot@openai.com|CHEN Pengan|chen@example.com",
                    "123456789|Human|human@example.com|Human|human@example.com",
                    "malformed",
                ]
            )
        if cmd == 'git log --format="%b" --all':
            return 0, "Co-Authored-By: Claude <noreply@anthropic.com>\nbody"
        raise AssertionError(cmd)

    monkeypatch.setattr(git_mod, "_run", fake_run_for_commits)
    ai_commits = git_mod._find_ai_commits("/repo")
    assert ai_commits == [{"sha": "abcdef12", "author": "Codex <bot@openai.com>", "committer": "CHEN Pengan <chen@example.com>"}]
    assert git_mod._find_coauthored_commits("/repo") == 1

    def fake_run(cmd, cwd=None, timeout=120):
        if cmd == "git rev-parse --show-toplevel":
            return 0, "/repo"
        if cmd == "git rev-list --count --all":
            return 0, "2"
        if cmd == "git status --porcelain":
            return 0, " M file.py"
        return 0, ""

    monkeypatch.setattr(git_mod, "_run", fake_run)
    monkeypatch.setattr(git_mod, "_find_ai_commits", lambda cwd: ai_commits)
    monkeypatch.setattr(git_mod, "_find_coauthored_commits", lambda cwd: 1)

    result = git_mod.git_clean_push(dry_run=True)

    assert result["status"] == "dry_run"
    assert result["ai_commit_count"] == 1
    assert result["coauthored_count"] == 1
    assert result["has_uncommitted_changes"] is True


def test_git_clean_push_run_routes_through_shell_exec(monkeypatch):
    calls = []

    def fake_shell_exec(command, cwd="", timeout_s=0, write_policy="read_only", confirmed=False, **kwargs):
        calls.append({"command": command, "cwd": cwd, "timeout_s": timeout_s,
                      "write_policy": write_policy, "confirmed": confirmed})
        return {"status": "success", "returncode": 7, "stdout": "out", "stderr": "err"}

    monkeypatch.setattr(git_mod, "shell_exec", fake_shell_exec)

    rc, out = git_mod._run("git status", cwd="/repo", timeout=3)

    assert rc == 7
    assert out == "out\nerr"
    assert calls[0]["command"] == "git status"
    assert calls[0]["cwd"] == "/repo"
    assert calls[0]["timeout_s"] == 3
    assert calls[0]["confirmed"] is True  # git mutations run through the gate with confirmation pre-granted

    monkeypatch.setattr(git_mod, "_run", lambda *args, **kwargs: (1, "git failed"))
    assert git_mod._find_ai_commits("/repo") == []
    assert git_mod._find_coauthored_commits("/repo") == 0


def test_git_clean_push_success_and_error_paths(monkeypatch):
    monkeypatch.setattr(git_mod, "_find_ai_commits", lambda cwd: [])
    monkeypatch.setattr(git_mod, "_find_coauthored_commits", lambda cwd: 0)

    commands = []

    def fake_run_success(cmd, cwd=None, timeout=120):
        commands.append(cmd)
        if cmd == "git rev-parse --show-toplevel":
            return 0, "/repo"
        if cmd == "git rev-list --count --all":
            return 0, "4"
        if cmd == "git status --porcelain":
            return 0, ""
        if cmd.startswith("git push --force"):
            return 0, "pushed"
        return 0, ""

    monkeypatch.setattr(git_mod, "_run", fake_run_success)
    success = git_mod.git_clean_push(remote="origin", branch="main")

    assert success["status"] == "success"
    assert any("No AI traces found" in step for step in success["steps"])
    assert any(cmd.startswith("git push --force origin main") for cmd in commands)

    monkeypatch.setattr(git_mod, "_run", lambda cmd, cwd=None, timeout=120: (1, "not git") if cmd == "git rev-parse --show-toplevel" else (0, ""))
    assert git_mod.git_clean_push()["error"] == "Not inside a git repository"

    def fake_run_push_error(cmd, cwd=None, timeout=120):
        if cmd == "git rev-parse --show-toplevel":
            return 0, "/repo"
        if cmd == "git rev-list --count --all":
            return 0, "4"
        if cmd == "git status --porcelain":
            return 0, ""
        if cmd.startswith("git push --force"):
            return 1, "remote rejected"
        return 0, ""

    monkeypatch.setattr(git_mod, "_run", fake_run_push_error)
    push_error = git_mod.git_clean_push()
    assert push_error["status"] == "error"
    assert "Push failed" in push_error["error"]


def test_git_clean_push_commit_rewrite_and_cleanup_paths(monkeypatch):
    monkeypatch.setattr(
        git_mod,
        "_find_ai_commits",
        lambda cwd: [{"sha": "abc12345", "author": "Codex <bot@openai.com>", "committer": "Codex <bot@openai.com>"}],
    )
    monkeypatch.setattr(git_mod, "_find_coauthored_commits", lambda cwd: 2)
    commands = []

    def fake_run(cmd, cwd=None, timeout=120):
        commands.append(cmd)
        if cmd == "git rev-parse --show-toplevel":
            return 0, "/repo"
        if cmd == "git rev-list --count --all":
            return 0, "3"
        if cmd == "git status --porcelain":
            return (0, " M file.py") if commands.count("git status --porcelain") == 1 else (0, "")
        if cmd == "git add -A":
            return 0, ""
        if "git commit -m" in cmd:
            return 0, "committed"
        if cmd.startswith("FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch"):
            return 0, "rewritten"
        if cmd.startswith("git push --force"):
            return 0, "pushed"
        return 0, ""

    monkeypatch.setattr(git_mod, "_run", fake_run)

    result = git_mod.git_clean_push(commit_message="save changes")

    assert result["status"] == "success"
    assert result["ai_commits_rewritten"] == 1
    assert any("Committed pending changes" in step for step in result["steps"])
    assert any("Rewrote 1 AI commit" in step for step in result["steps"])
    assert "git stash clear" in commands
    assert "git gc --prune=now" in commands


def test_git_clean_push_commit_failure_stash_and_filter_error(monkeypatch):
    monkeypatch.setattr(
        git_mod,
        "_find_ai_commits",
        lambda cwd: [{"sha": "abc12345", "author": "Claude <bot@anthropic.com>", "committer": "Claude <bot@anthropic.com>"}],
    )
    monkeypatch.setattr(git_mod, "_find_coauthored_commits", lambda cwd: 0)
    commands = []

    def fake_run(cmd, cwd=None, timeout=120):
        commands.append(cmd)
        if cmd == "git rev-parse --show-toplevel":
            return 0, "/repo"
        if cmd == "git rev-list --count --all":
            return 1, "count failed"
        if cmd == "git status --porcelain":
            return 0, " M file.py"
        if cmd == "git add -A":
            return 0, ""
        if "git commit -m" in cmd:
            return 1, "nothing committed"
        if cmd == "git stash":
            return 0, "stashed"
        if cmd.startswith("FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch"):
            return 1, "filter failed"
        if cmd == "git stash pop":
            return 0, "restored"
        return 0, ""

    monkeypatch.setattr(git_mod, "_run", fake_run)

    result = git_mod.git_clean_push(commit_message="try save")

    assert result["status"] == "error"
    assert result["error"].startswith("filter-branch failed")
    assert any("Commit attempt returned" in step for step in result["steps"])
    assert "git stash" in commands
    assert "git stash pop" in commands

    commands.clear()

    def fake_run_clean_filter_error(cmd, cwd=None, timeout=120):
        commands.append(cmd)
        if cmd == "git rev-parse --show-toplevel":
            return 0, "/repo"
        if cmd == "git rev-list --count --all":
            return 0, "5"
        if cmd == "git status --porcelain":
            return 0, ""
        if cmd.startswith("FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch"):
            return 1, "filter failed"
        return 0, ""

    monkeypatch.setattr(git_mod, "_run", fake_run_clean_filter_error)
    clean_filter_error = git_mod.git_clean_push()
    assert clean_filter_error["status"] == "error"
    assert "git stash pop" not in commands


def test_git_clean_push_restores_stash_after_successful_rewrite(monkeypatch):
    monkeypatch.setattr(
        git_mod,
        "_find_ai_commits",
        lambda cwd: [{"sha": "abc12345", "author": "Codex <bot@openai.com>", "committer": "Codex <bot@openai.com>"}],
    )
    monkeypatch.setattr(git_mod, "_find_coauthored_commits", lambda cwd: 0)
    commands = []

    def fake_run(cmd, cwd=None, timeout=120):
        commands.append(cmd)
        if cmd == "git rev-parse --show-toplevel":
            return 0, "/repo"
        if cmd == "git rev-list --count --all":
            return 0, "5"
        if cmd == "git status --porcelain":
            return 0, " M file.py"
        if cmd == "git stash":
            return 0, "stashed"
        if cmd.startswith("FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch"):
            return 0, "rewritten"
        if cmd == "git stash pop":
            return 0, "restored"
        if cmd.startswith("git push --force"):
            return 0, "pushed"
        return 0, ""

    monkeypatch.setattr(git_mod, "_run", fake_run)

    result = git_mod.git_clean_push()

    assert result["status"] == "success"
    assert any("Restored stashed changes" in step for step in result["steps"])
    assert commands.count("git stash pop") == 1
