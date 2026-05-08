"""Tests for writing helpers, brainstorm workspaces, and hypothesis tracking."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from biobank_agent.skills.brainstorm import _build_brainstorm_document, brainstorm
from biobank_agent.skills.hypothesis import _generate_id, hypothesis
from biobank_agent.skills.nature_writer import nature_writer


class FakeState:
    def __init__(self, summary="No analyses performed yet."):
        self.custom_data = {}
        self._summary = summary

    def context_summary(self):
        if isinstance(self._summary, Exception):
            raise self._summary
        return self._summary


def fake_ctx(tmp_path, summary="No analyses performed yet."):
    return SimpleNamespace(
        report_dir=tmp_path,
        settings=SimpleNamespace(
            biobank_name="UK Biobank",
            biobank_abbreviation="UKB",
            biobank_caveats="healthy volunteer bias and linkage lag",
        ),
        state=FakeState(summary),
    )


def test_brainstorm_clamps_inputs_writes_workspace_and_includes_context(tmp_path):
    ctx = fake_ctx(tmp_path, summary="cohort: E11 cases=1,200")

    result = brainstorm("HbA1c: diabetes risk?", n_ideas=99, depth="DEEP", ctx=ctx)

    assert result["n_ideas"] == 20
    assert result["depth"] == "deep"
    assert len(result["directions"]) == 20
    assert "Evidence Support" in result["document"]
    assert "cohort: E11 cases=1,200" in result["document"]
    assert "UK Biobank" in result["summary"]
    assert result["workspace"] is not None
    workspace = tmp_path / "brainstorm" / result["workspace"].split("/")[-1]
    assert workspace.exists()
    assert ":" not in workspace.name


def test_brainstorm_invalid_depth_low_count_and_context_failure(tmp_path):
    ctx = fake_ctx(tmp_path, summary=RuntimeError("state unavailable"))

    result = brainstorm("lipids", n_ideas=-5, depth="unknown", ctx=ctx)

    assert result["n_ideas"] == 1
    assert result["depth"] == "quick"
    assert len(result["directions"]) == 1
    assert "Session Context" not in result["document"]


def test_brainstorm_default_workspace_and_save_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    default_workspace = brainstorm("blood pressure", n_ideas=1, ctx=None)
    assert default_workspace["workspace"] is not None
    assert (tmp_path / "brainstorm").is_dir()

    def bad_write_text(self, *args, **kwargs):
        raise RuntimeError("read-only")

    monkeypatch.setattr("biobank_agent.skills.brainstorm.Path.write_text", bad_write_text)
    unsaved = brainstorm("cholesterol", n_ideas=1, ctx=fake_ctx(tmp_path))
    assert unsaved["workspace"] is None


def test_build_brainstorm_document_quick_template_defaults():
    document = _build_brainstorm_document(
        topic="kidney disease",
        n_ideas=2,
        depth="quick",
        context_summary="No analyses performed yet.",
    )

    assert document.count("### Direction") == 2
    assert "Potential Pitfalls" not in document
    assert "Summary & Recommended Next Steps" in document


def test_nature_writer_validates_section_and_style():
    bad_section = nature_writer("cover_letter", "finding", style="nature")
    bad_style = nature_writer("abstract", "finding", style="unknown")

    assert "Unknown section" in bad_section["error"]
    assert "Unknown style" in bad_style["error"]


def test_nature_writer_normalizes_style_and_builds_context_prompt(tmp_path):
    ctx = fake_ctx(tmp_path)

    result = nature_writer(
        "methods",
        "We analysed 12,000 participants with incident diabetes outcomes.",
        style="Nature Medicine",
        ctx=ctx,
    )

    assert result["section"] == "methods"
    assert result["style"] == "nature_medicine"
    assert result["journal"] == "Nature Medicine"
    assert result["input_word_count"] == 8
    assert "UK Biobank" in result["writing_prompt"]
    assert "clinical relevance" in result["writing_prompt"]
    assert "software versions" in result["writing_prompt"]


def test_nature_writer_non_context_section_uses_title_rules():
    result = nature_writer("title", "High HbA1c predicts incident diabetes", style="nature_methods")

    assert result["journal"] == "Nature Methods"
    assert "Maximum 10-12 words" in result["writing_prompt"]
    assert "method enables" in result["writing_prompt"]


def test_nature_writer_all_section_templates_without_context():
    abstract = nature_writer("abstract", "HbA1c predicted E11 with OR 1.4.", style="nature")
    introduction = nature_writer("introduction", "Diabetes prevention remains incomplete.", style="nature")
    results = nature_writer("results", "AUC was 0.82 with 10,000 participants.", style="nature")
    discussion = nature_writer("discussion", "Residual confounding remains possible.", style="nature")

    assert "Write an abstract" in abstract["writing_prompt"]
    assert "dataset (Biobank)" in introduction["writing_prompt"]
    assert "Finding first" in results["writing_prompt"]
    assert "healthy volunteer bias" in discussion["writing_prompt"]


def test_hypothesis_requires_session_context():
    result = hypothesis("list", ctx=None)

    assert result["error"] == "A session context with state.custom_data is required."


def test_hypothesis_lifecycle_duplicate_update_and_evaluate(tmp_path):
    ctx = fake_ctx(tmp_path)
    statement = "Higher HbA1c is associated with incident type 2 diabetes."
    expected_id = _generate_id(statement)

    missing = hypothesis("propose", ctx=ctx)
    proposed = hypothesis("propose", statement=statement, ctx=ctx)
    duplicate = hypothesis("propose", statement=statement, ctx=ctx)

    assert missing["error"] == "Provide a hypothesis statement."
    assert proposed["hypothesis"]["id"] == expected_id
    assert proposed["hypothesis"]["status"] == "PROPOSED"
    assert duplicate["error"] == f"Hypothesis already exists: {expected_id}"

    listed = hypothesis("list", ctx=ctx)
    assert listed["n_hypotheses"] == 1
    assert listed["summary"]["PROPOSED"] == 1

    updated = hypothesis(
        "update",
        statement="HbA1c",
        status="testing",
        evidence="Pilot model AUC was 0.78.",
        confidence=0.82,
        ctx=ctx,
    )
    assert updated["hypothesis"]["status"] == "TESTING"
    assert updated["hypothesis"]["evidence"][0]["confidence"] == 0.82

    evaluation = hypothesis("evaluate", ctx=ctx)
    assert evaluation["n_hypotheses"] == 1
    assert "SUPPORTED" in evaluation["evaluations"][0]["recommendation"]


def test_hypothesis_unknown_actions_statuses_and_missing_ids(tmp_path):
    ctx = fake_ctx(tmp_path)
    hypothesis("propose", statement="A first claim", ctx=ctx)
    hypothesis("propose", statement="A second claim", ctx=ctx)

    bad_status = hypothesis("update", statement="first", status="causal", ctx=ctx)
    missing_id = hypothesis("update", statement="not present", ctx=ctx)
    ambiguous = hypothesis("update", statement="claim", ctx=ctx)
    unknown = hypothesis("archive", ctx=ctx)

    assert "Unknown status" in bad_status["error"]
    assert "not found" in missing_id["error"]
    assert "not found" in ambiguous["error"]
    assert "Unknown action" in unknown["error"]


def test_hypothesis_empty_list_and_empty_evaluation(tmp_path):
    ctx = fake_ctx(tmp_path)

    listed = hypothesis("list", ctx=ctx)
    evaluated = hypothesis("evaluate", ctx=ctx)

    assert listed["hypotheses"] == []
    assert listed["message"] == "No hypotheses tracked yet."
    assert evaluated["message"] == "No hypotheses to evaluate."


def test_hypothesis_evaluation_recommendation_branches(tmp_path):
    ctx = fake_ctx(tmp_path)
    proposed = hypothesis("propose", statement="Untested biomarker claim", ctx=ctx)["hypothesis"]["id"]
    weak = hypothesis("propose", statement="Weak evidence claim", ctx=ctx)["hypothesis"]["id"]
    mixed = hypothesis("propose", statement="Mixed evidence claim", ctx=ctx)["hypothesis"]["id"]
    status_only = hypothesis("propose", statement="Status only claim", ctx=ctx)["hypothesis"]["id"]

    hypothesis("update", statement=weak, status="testing", evidence="weak result", confidence=0.2, ctx=ctx)
    hypothesis("update", statement=mixed, status="testing", evidence="mixed result", confidence=0.5, ctx=ctx)
    hypothesis("update", statement=mixed, evidence="more mixed result", confidence=0.5, ctx=ctx)
    hypothesis("update", statement=status_only, status="testing", ctx=ctx)

    evaluation = hypothesis("evaluate", ctx=ctx)
    by_id = {item["id"]: item["recommendation"] for item in evaluation["evaluations"] if "recommendation" in item}

    assert "Not yet tested" in by_id[proposed]
    assert "REFUTED" in by_id[weak]
    assert "mixed" in by_id[mixed]
    assert status_only not in by_id
