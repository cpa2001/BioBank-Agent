"""Tests for the M3 domain layer (banks / compliance / NLI / paper replicator)."""

from __future__ import annotations

from biobank_agent.core.safety.nli_causal_check import check, maybe_inject_disclaimer
from biobank_agent.domain.banks import (
    CKBAdapter,
    CohortCriteria,
    HPPAdapter,
    UKBAdapter,
    canonical_bank_id,
    get_adapter,
)
from biobank_agent.domain.banks.rap_adapter import RAPAdapter
from biobank_agent.domain.compliance import enforce_disclosure, filter_pii
from biobank_agent.domain.compliance.policy_engine import DisclosurePolicy
from biobank_agent.domain.reproducibility import PaperReplicator, extract_study_design


# ── Bank adapters ──────────────────────────────────────────


def test_bank_registry_has_three_local_adapters_plus_rap():
    assert isinstance(get_adapter("ukb"), UKBAdapter)
    assert isinstance(get_adapter("hpp"), HPPAdapter)
    assert isinstance(get_adapter("ckb"), CKBAdapter)
    assert isinstance(get_adapter("ukb_rap"), RAPAdapter)
    assert isinstance(get_adapter("rap"), RAPAdapter)
    assert isinstance(get_adapter("ukb-rap"), RAPAdapter)
    assert canonical_bank_id("UKB-RAP") == "ukb_rap"


def test_ukb_adapter_field_id_handles_semantic_and_numeric():
    a = UKBAdapter()
    assert a.field_id("hba1c") == "30750"
    assert a.field_id("21001") == "21001"


def test_ukb_cohort_query_filters_icd_age_sex():
    a = UKBAdapter()
    sql = a.cohort_query(CohortCriteria(
        icd_codes=["E11"], age_range=(40, 70), sex="F"
    ))
    assert "E11" in sql
    assert "BETWEEN 40" in sql
    assert "31-0.0" in sql  # sex column
    assert "diagnoses" in sql.lower()


def test_hpp_normalizes_icd9_to_icd10():
    a = HPPAdapter()
    assert a.normalize_icd("250.0") == "E11"  # ICD-9 250 -> T2D
    assert a.normalize_icd("E11") == "E11"


def test_ckb_query_uses_study_id_join():
    a = CKBAdapter()
    sql = a.cohort_query(CohortCriteria(icd_codes=["I10"], sex="M"))
    assert "study_id" in sql
    assert "is_male = 1" in sql


def test_rap_query_uses_sparksql_dialect():
    a = RAPAdapter()
    sql = a.cohort_query(CohortCriteria(icd_codes=["E11"], age_range=(40, 60)))
    assert "`participant`" in sql or "participant" in sql
    assert "BETWEEN 40" in sql


# ── Compliance ─────────────────────────────────────────────


def test_disclosure_blocks_below_min_cell_count():
    out = enforce_disclosure(payload={"n_cases": 3, "auc": 0.8})
    assert out.passed is False
    assert any(v.code == "MIN_CELL" for v in out.violations)


def test_disclosure_rounds_counts_to_nearest_5():
    out = enforce_disclosure(payload={"n_cases": 1234, "n_controls": 9876, "auc": 0.7})
    assert out.passed is True
    assert out.sanitised["n_cases"] == 1235
    assert out.sanitised["n_controls"] == 9875


def test_disclosure_strips_phi_fields():
    payload = {"n_cases": 1000, "eid": "1234567", "auc": 0.83}
    out = enforce_disclosure(payload=payload)
    assert out.passed is True
    assert out.sanitised["eid"] == "<REDACTED>"


def test_strict_policy_uses_higher_thresholds():
    out = enforce_disclosure(payload={"n_cases": 60}, policy=DisclosurePolicy.strict())
    # n_cases=60 < strict min_cohort=200
    assert any(v.code == "MIN_COHORT" for v in out.violations)


def test_filter_pii_facade():
    p = filter_pii({"name": "Jane Doe", "score": 0.5})
    assert p["name"] == "<REDACTED>"
    assert p["score"] == 0.5


# ── NLI causal check ───────────────────────────────────────


def test_nli_flags_unhedged_causal_phrase():
    out = check("E11 causes severe complications.")
    assert out.is_causal is True
    assert out.matched_phrase.lower().startswith("cause")
    assert out.needs_disclaimer is True


def test_nli_does_not_flag_hedged_association():
    out = check("E11 is associated with cardiovascular outcomes; further validation is needed.")
    assert out.is_causal is False


def test_maybe_inject_disclaimer_appends_warning():
    text = "We conclude that high LDL causes coronary disease."
    wrapped = maybe_inject_disclaimer(text)
    assert "Causal-language guard" in wrapped


# ── Paper replicator ───────────────────────────────────────


def test_extract_study_design_picks_design_and_n():
    text = (
        "We performed a prospective cohort study of 12345 participants "
        "aged 40-69 years. The primary outcome was incident type 2 "
        "diabetes."
    )
    out = extract_study_design(text)
    assert out["design"] == "cohort"
    assert out["n"] == 12345
    assert out["age_range"] == (40.0, 69.0)
    assert any("type 2" in o.lower() or "diabetes" in o.lower() for o in out["outcomes"])
    assert out["icd10_code"] == "E11"


def test_paper_replicator_proposes_executable_ukb_plan():
    text = (
        "DOI 10.1038/s41588-024-01898-1. We performed a prospective cohort study "
        "of 12345 participants aged 40-69 years. The primary outcome was incident "
        "type 2 diabetes. Biomarkers included HbA1c, glucose, BMI, blood pressure "
        "and cholesterol. Table 1: Baseline characteristics. Figure 2: Model "
        "calibration in the test set."
    )
    outcome = PaperReplicator().from_text(text, paper_path="10.1038/s41588-024-01898-1")
    skills = [step["skill"] for step in outcome.plan]

    assert skills[:4] == ["read_paper", "deep_research", "field_search", "cohort_summary"]
    assert "train_model" in skills
    assert "evaluate_model" in skills
    assert "calibration" in skills
    assert "feature_importance" in skills
    assert "paper_replication_compare" in skills
    assert skills[-4:] == ["statistical_review", "safety_check", "world_model_audit", "generate_report"]
    train_step = next(step for step in outcome.plan if step["skill"] == "train_model")
    assert train_step["args"]["icd10_code"] == "E11"
    assert train_step["args"]["model_type"] == "auto"
    assert outcome.diff["status"] == "not_run"
    assert outcome.comparison_checklist
    assert [row["target_type"] for row in outcome.table_figure_diff] == ["table", "figure"]
    assert outcome.table_figure_diff[1]["comparison_method"] == "visual/pixel diff after execution"


def test_extract_study_design_infers_prediction_outcome_from_prompt():
    text = (
        "Reproduce a disease prediction paper for E11 Type 2 Diabetes "
        "using HbA1c, glucose, BMI and cholesterol biomarkers."
    )
    out = extract_study_design(text)

    assert out["design"] == "prediction_modeling"
    assert out["icd10_code"] == "E11"
    assert out["outcomes"] == ["Local UKB ICD-10 E11 approximation for Type 2 Diabetes"]


def test_replicate_paper_skill_writes_review_artifact(tmp_path):
    from types import SimpleNamespace

    from biobank_agent.skills.replicate_paper import replicate_paper

    ctx = SimpleNamespace(report_dir=tmp_path)
    result = replicate_paper(
        "We performed a prospective cohort study of 12345 participants aged 40-69 years. "
        "The primary outcome was incident type 2 diabetes.",
        ctx=ctx,
    )

    assert result["status"] == "AWAITING_USER_APPROVAL"
    assert result["proposed_spec"]["design"] == "cohort"
    assert result["awaiting_user_approval"] is True
    assert (tmp_path / "paper_replication_spec.json").exists()
    assert (tmp_path / "paper_replication_review.md").exists()
    assert (tmp_path / "paper_replication_plan.json").exists()
    assert any(step["skill"] == "generate_report" for step in result["plan"])
    review = (tmp_path / "paper_replication_review.md").read_text(encoding="utf-8")
    assert "Table/Figure Diff Targets" in review


def test_paper_replication_compare_writes_matrix_artifacts(tmp_path):
    from types import SimpleNamespace

    from biobank_agent.skills.paper_replication_compare import paper_replication_compare
    from biobank_agent.state import AnalysisRecord

    roc = tmp_path / "roc_pr_E11_lgbm.svg"
    roc.write_text("<svg/>", encoding="utf-8")
    calibration_fig = tmp_path / "calibration_E11_lgbm.svg"
    calibration_fig.write_text("<svg/>", encoding="utf-8")
    records = [
        AnalysisRecord(
            timestamp="t",
            skill="replicate_paper",
            args={"source": "10.1038/example"},
            key_results={
                "proposed_spec": {
                    "source": "10.1038/example",
                    "icd10_code": "E11",
                    "outcomes": ["paper T2D outcome"],
                    "biomarkers": ["HbA1c", "BMI"],
                },
                "table_figure_diff": [
                    {
                        "target_type": "figure",
                        "paper_caption": "Figure 1: ROC curve",
                        "local_artifact": "",
                        "comparison_method": "visual/pixel diff after execution",
                        "status": "awaiting_execution",
                        "limitation": "local figure pending",
                    }
                ],
            },
            figure_paths=[],
        ),
        AnalysisRecord("t", "read_paper", {"paper_path_or_doi": "10.1038/example"}, {"paper_text": "text", "text_truncated": False, "summary": {"doi": "10.1038/example"}}, []),
        AnalysisRecord("t", "cohort_summary", {"icd10_code": "E11"}, {"n_cases": 123, "n_controls": 456}, []),
        AnalysisRecord("t", "missing_data", {}, {"overall_missing_pct": 8.5}, []),
        AnalysisRecord("t", "train_model", {"icd10_code": "E11", "model_type": "auto"}, {"selected_model_type": "lgbm", "auc": 0.82, "n_features": 20}, []),
        AnalysisRecord("t", "evaluate_model", {}, {"mean_auc": 0.82}, [str(roc)]),
        AnalysisRecord("t", "calibration", {}, {"ece": 0.04}, [str(calibration_fig)]),
        AnalysisRecord("t", "feature_importance", {}, {"top_features": [{"feature": "HbA1c", "importance": 0.3}]}, []),
    ]
    ctx = SimpleNamespace(state=SimpleNamespace(records=records, figures=[str(roc), str(calibration_fig)]), report_dir=tmp_path)

    result = paper_replication_compare(ctx=ctx)

    assert result["overall_status"] == "partial_replication"
    assert result["local_endpoint"] == "UKB ICD-10 E11 diagnosis-centered cohort"
    assert any(row["dimension"] == "Model evaluation" for row in result["comparison_rows"])
    assert result["table_figure_diff"][0]["target_type"] == "figure"
    assert result["table_figure_diff"][0]["status"] == "local_artifact_available"
    assert result["table_figure_diff"][0]["local_artifact"] == str(roc)
    assert result["n_local_artifacts_matched"] == 1
    assert result["acceptance_summary"]["verdict"] == "PASS_WITH_LIMITATIONS"
    assert result["acceptance_summary"]["failed"] == 0
    gate_status = {gate["gate"]: gate["status"] for gate in result["acceptance_gates"]}
    assert gate_status["cohort_count"] == "PASS"
    assert gate_status["model_auc"] == "PASS"
    assert gate_status["calibration_ece"] == "PASS"
    assert gate_status["figure_artifacts"] == "PASS"
    assert (tmp_path / "paper_replication_comparison.json").exists()
    assert (tmp_path / "paper_replication_comparison.md").exists()
    comparison_md = (tmp_path / "paper_replication_comparison.md").read_text(encoding="utf-8")
    assert "Acceptance Gates" in comparison_md
    assert "Table/Figure Diff Targets" in comparison_md
    assert "roc_pr_E11_lgbm.svg" in comparison_md


def test_paper_replication_compare_fails_missing_numeric_gates(tmp_path):
    from types import SimpleNamespace

    from biobank_agent.skills.paper_replication_compare import paper_replication_compare
    from biobank_agent.state import AnalysisRecord

    records = [
        AnalysisRecord(
            "t",
            "replicate_paper",
            {"source": "10.1038/example"},
            {
                "proposed_spec": {"source": "10.1038/example", "icd10_code": "E11"},
                "table_figure_diff": [{"target_type": "figure", "paper_caption": "ROC curve"}],
            },
            [],
        ),
        AnalysisRecord("t", "read_paper", {}, {"error": "paywalled"}, []),
        AnalysisRecord("t", "cohort_summary", {}, {"n_cases": 12, "n_controls": 50}, []),
        AnalysisRecord("t", "train_model", {}, {"auc": 0.51}, []),
        AnalysisRecord("t", "calibration", {}, {"ece": 0.4}, []),
    ]
    ctx = SimpleNamespace(state=SimpleNamespace(records=records, figures=[]), report_dir=tmp_path)

    result = paper_replication_compare(ctx=ctx)

    assert result["acceptance_summary"]["verdict"] == "FAIL"
    failed = {gate["gate"] for gate in result["acceptance_gates"] if gate["status"] == "FAIL"}
    assert {"paper_access", "cohort_count", "model_auc", "calibration_ece", "feature_importance", "figure_artifacts"}.issubset(failed)
