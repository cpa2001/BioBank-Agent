"""Tests for UKB+HPP+CKB agentic cohort discovery primitives."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from biobank_agent.banks import list_banks, load_bank
from biobank_agent.data.cohort_cards import build_cohort_card, infer_cohort_type, record_cohort_card_to_action_graph
from biobank_agent.data.phenotype import (
    harmonize_phenotype,
    record_phenotype_to_action_graph,
)
from biobank_agent.data.trajectory import TrajectoryTokenizer
from biobank_agent.memory import LongTermMemory
from biobank_agent.registry import autodiscover_skills, get_registry
from biobank_agent.skills.trajectory_tokenize import trajectory_tokenize
from biobank_agent.skills.phenotype_harmonize import phenotype_harmonize
from biobank_agent.skills.world_model_audit import world_model_audit
from biobank_agent.skills.report import _interpret_skill
from biobank_agent.state import AnalysisRecord, SessionState
from biobank_agent.world_model import audit_world_model_prediction, record_world_model_card_to_action_graph


def test_hpp_and_ckb_bank_configs_load():
    banks = list_banks()
    assert {"ukb", "hpp", "ckb"}.issubset(set(banks))

    hpp = load_bank("hpp")
    ckb = load_bank("ckb")

    assert hpp.patient_id_column == "participant_id"
    assert "longitudinal_deep_phenotyping" in hpp.feature_groups
    assert ckb.display_name == "China Kadoorie Biobank"
    assert "exposure_lifestyle" in ckb.feature_groups


def test_bank_registry_error_paths(tmp_path: Path):
    assert list_banks(tmp_path / "missing") == []

    with pytest.raises(FileNotFoundError, match="No config for bank"):
        load_bank("missing", configs_dir=tmp_path)

    (tmp_path / "bad.yaml").write_text("- not-a-dict\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid config format"):
        load_bank("bad", configs_dir=tmp_path)


def test_cross_cohort_phenotype_harmonisation_records_action_graph(tmp_path: Path):
    phenotype = harmonize_phenotype("I21", ["ukb", "hpp", "ckb"])

    assert phenotype.concept_id == "acute_myocardial_infarction"
    assert phenotype.harmonisation_status == "READY"
    assert {m.bank_id for m in phenotype.mappings} == {"ukb", "hpp", "ckb"}
    assert "case_ascertainment_shift" in phenotype.drift_risks

    mem = LongTermMemory(tmp_path / "memory")
    record_phenotype_to_action_graph(mem, phenotype)

    evidence = mem.action_graph.search_nodes("acute_myocardial_infarction", node_types=["phenotype"], limit=5)
    assert evidence
    linked = mem.retrieve_claim_evidence("nonexistent")
    assert linked == []
    mappings = mem.action_graph.search_nodes("ukb", node_types=["mapping"], limit=5)
    assert mappings


def test_biomarker_harmonisation_has_cross_cohort_fields():
    phenotype = harmonize_phenotype("LDL", ["ukb", "hpp", "ckb"], phenotype_type="biomarker")
    mapping = {m.bank_id: m.field_ids for m in phenotype.mappings}

    assert mapping["ukb"] == ["30780"]
    assert mapping["hpp"] == ["hpp_ldl"]
    assert mapping["ckb"] == ["ckb_ldl"]
    assert phenotype.harmonisation_status == "READY"

    partial = harmonize_phenotype("LDL", ["ukb"], phenotype_type="biomarker")
    assert partial.harmonisation_status == "READY"

    with pytest.raises(ValueError, match="Unknown biomarker concept"):
        harmonize_phenotype("not-a-biomarker", ["ukb"], phenotype_type="biomarker")


def test_unknown_disease_harmonisation_and_missing_memory_are_partial_safe():
    phenotype = harmonize_phenotype("Z99", ["ukb"])

    assert phenotype.concept_id == "z99"
    assert phenotype.label == "ICD-coded phenotype Z99"
    assert phenotype.mappings[0].codes == ["Z99"]
    assert record_phenotype_to_action_graph(None, phenotype) is None
    assert record_phenotype_to_action_graph(SimpleNamespace(), phenotype) is None
    assert phenotype_harmonize("Z99", banks="ukb", ctx=None)["concept_id"] == "z99"


def test_healthformer_style_trajectory_tokenizer_handles_missing_and_future_query():
    df = pd.DataFrame([
        {"participant_id": "p1", "timestamp": "2026-01-01T08:00:00", "modality": "glucose", "value": 5.1, "value_type": "continuous", "unit": "mmol/L"},
        {"participant_id": "p1", "timestamp": "2026-01-01T08:05:00", "modality": "exercise", "value": "walking", "value_type": "categorical"},
        {"participant_id": "p1", "timestamp": "2026-01-02T08:00:00", "modality": "glucose", "value": None, "value_type": "continuous"},
        {"participant_id": "p2", "timestamp": "2026-01-01T09:00:00", "modality": "glucose", "value": 7.2, "value_type": "continuous", "unit": "mmol/L"},
        {"participant_id": "p2", "timestamp": "2026-01-01T09:10:00", "modality": "exercise", "value": "cycling", "value_type": "categorical"},
    ])

    tokenizer = TrajectoryTokenizer(max_bins=5)
    dataset = tokenizer.fit_transform(df)
    query = tokenizer.build_future_query("glucose", "2026-06-01T08:00:00")

    assert dataset.n_participants == 2
    assert dataset.n_tokens == 4
    assert set(dataset.vocab) == {"exercise", "glucose"}
    assert len(dataset.sequences[0].time_features[0]) == 7
    assert query["target_modality"] == "glucose"
    assert query["token_range"][0] <= query["token_range"][1]


def test_trajectory_tokenizer_edge_cases_and_serialization():
    tokenizer = TrajectoryTokenizer(max_bins=4, min_bins=2)

    with pytest.raises(ValueError, match="missing required columns"):
        tokenizer.fit(pd.DataFrame({"participant_id": ["p1"]}))

    with pytest.raises(ValueError, match="must be fitted"):
        tokenizer.transform(pd.DataFrame())

    fit_df = pd.DataFrame([
        {"participant_id": "p1", "timestamp": "2026-01-01T08:00:00", "modality": "constant", "value": 5.0, "value_type": "continuous"},
        {"participant_id": "p2", "timestamp": "2026-01-01T09:00:00", "modality": "constant", "value": 5.0, "value_type": "continuous"},
        {"participant_id": "p1", "timestamp": "2026-01-01T08:05:00", "modality": "nonnumeric", "value": "bad", "value_type": "continuous"},
        {"participant_id": "p1", "timestamp": "2026-01-01T08:10:00", "modality": "empty_category", "value": None, "value_type": "categorical"},
        {"participant_id": "p1", "timestamp": "2026-01-01T08:15:00", "modality": "activity", "value": "walking", "value_type": "categorical"},
    ])
    tokenizer.fit(fit_df)

    assert tokenizer.vocab["constant"].bins == [4.5, 5.5]
    assert tokenizer.vocab["nonnumeric"].bins == [0.0, 1.0]
    assert tokenizer.vocab["empty_category"].categories == ["__missing_category__"]
    assert tokenizer._encode_value(tokenizer.vocab["activity"], float("nan")) is None

    transform_df = pd.DataFrame([
        {"participant_id": "p1", "timestamp": "2026-01-01T08:00:00", "modality": "constant", "value": 5.0, "sleep": 1},
        {"participant_id": "p1", "timestamp": "2026-01-01T08:05:00", "modality": "nonnumeric", "value": "still bad", "sleep": 0},
        {"participant_id": "p1", "timestamp": "2026-01-01T08:10:00", "modality": "unknown", "value": "ignored", "sleep": 0},
        {"participant_id": "p2", "timestamp": "2026-01-01T09:00:00", "modality": "activity", "value": "cycling", "sleep": 0},
        {"participant_id": "p3", "timestamp": "not-a-date", "modality": "activity", "value": "walking", "sleep": 0},
    ])
    dataset = tokenizer.transform(transform_df)
    payload = dataset.to_dict()

    assert dataset.n_participants == 2
    assert dataset.n_tokens == 2
    assert payload["sequences"][0]["time_features"][0][-1] == 1
    assert payload["vocab"]["activity"]["categories"] == ["walking"]

    with pytest.raises(ValueError, match="Unknown target modality"):
        tokenizer.build_future_query("missing", "2026-01-01")


def test_cohort_card_infers_target_trial_and_flags_cross_cohort_review():
    df = pd.DataFrame({"label": [1] * 120 + [0] * 240, "ldl": [1.0] * 360})
    card = build_cohort_card(
        query="Emulate a target trial of LDL lowering intervention for myocardial infarction",
        endpoint="I21",
        banks=["ukb", "hpp", "ckb"],
        cohort_df=df,
    )

    assert card.cohort_type == "target_trial_emulation"
    assert card.n_cases == 120
    assert card.status == "PASS"
    assert any("cross-cohort phenotype drift" in flag for flag in card.bias_flags)


def test_cohort_card_defaults_types_missingness_and_graph_recording():
    assert infer_cohort_type("forecast HealthFormer trajectory") == "trajectory_prediction"
    assert infer_cohort_type("incident survival time-to-event") == "survival_cohort"
    assert infer_cohort_type("prevalent case-control analysis") == "prevalent_case_control"
    assert infer_cohort_type("") == "incident_cohort"

    df = pd.DataFrame({"label": [1] * 10 + [0] * 20, "bmi": [None] * 15 + [25.0] * 15})
    card = build_cohort_card(query="prevalent case-control", endpoint="E11", banks=["ukb"], cohort_df=df)
    no_counts = build_cohort_card(query="incident", endpoint="I21", banks=["ukb"], cohort_df=pd.DataFrame())
    nonnumeric = build_cohort_card(
        query="incident",
        endpoint="I10",
        banks=["ukb"],
        cohort_df=pd.DataFrame({"status": ["case", "control"]}),
    )
    trajectory = build_cohort_card(query="trajectory forecast", endpoint="BMI", banks=["ukb"], cohort_df=df)

    assert card.cohort_type == "prevalent_case_control"
    assert card.status == "PARTIAL"
    assert "critical: n_cases < 100 guardrail" in card.bias_flags
    assert "high missingness in one or more covariates" in card.bias_flags
    assert no_counts.n_cases is None
    assert "case count unavailable" in no_counts.bias_flags[0]
    assert nonnumeric.missingness == {}
    assert trajectory.cohort_type == "trajectory_prediction"
    assert trajectory.follow_up_window == "future query horizon defined by target timestamp"
    explicit_unknown = build_cohort_card(
        query="custom eligibility",
        endpoint="I21",
        banks=["ukb"],
        cohort_type="bespoke_design",
        cohort_df=df,
    )
    assert explicit_unknown.cohort_type == "bespoke_design"
    assert explicit_unknown.inclusion[-1] == "linked longitudinal outcome records"
    assert explicit_unknown.follow_up_window == "incident endpoint follow-up window"

    calls = []
    memory = SimpleNamespace(
        upsert_node=lambda *args, **kwargs: calls.append(("upsert", args, kwargs)),
        link_nodes=lambda *args, **kwargs: calls.append(("link", args, kwargs)),
    )

    record_cohort_card_to_action_graph(None, card)
    record_cohort_card_to_action_graph(SimpleNamespace(), card)
    record_cohort_card_to_action_graph(memory, card)

    assert any(call[0] == "upsert" and call[1][0] == "cohort_card" for call in calls)
    assert any(call[0] == "link" and call[2]["relation"] == "defines_endpoint" for call in calls)


def test_cohort_card_skill_uses_session_cohort_and_optional_memory():
    from biobank_agent.skills.cohort_card import cohort_card as cohort_card_skill

    df = pd.DataFrame({"label": [1, 0, 0], "bmi": [25.0, None, 31.0]})
    calls = []
    memory = SimpleNamespace(
        upsert_node=lambda *args, **kwargs: calls.append(("upsert", args, kwargs)),
        link_nodes=lambda *args, **kwargs: calls.append(("link", args, kwargs)),
    )
    ctx = SimpleNamespace(state=SimpleNamespace(cohorts={"active": df}), memory=memory)

    result = cohort_card_skill(
        query="prevalent case-control analysis",
        endpoint="E11",
        banks=" ukb, hpp ,,",
        cohort_key="active",
        ctx=ctx,
    )
    no_ctx = cohort_card_skill(query="incident cohort", endpoint="I21", banks="ukb")

    assert result["n_cases"] == 1
    assert result["banks"] == ["ukb", "hpp"]
    assert any(call[0] == "upsert" for call in calls)
    assert no_ctx["cohort_type"] == "survival_cohort"


def test_world_model_audit_blocks_unsupported_causal_claims():
    card = audit_world_model_prediction(
        task="simulate semaglutide effect on HbA1c",
        simulation_type="causal_effect_estimate",
        input_modalities=["blood", "bmi"],
        available_tokens=12,
        training_distribution_coverage=0.3,
        calibration_status="unknown",
        external_validation_status="not_validated",
    )

    assert card.safety_status == "FAIL"
    assert card.allowed_claim_type == "association_conditioned_forecast"
    assert "sparse_context" in card.ood_flags


def test_world_model_pass_causal_target_trial_and_graph_recording():
    clean = audit_world_model_prediction(
        task="forecast HbA1c",
        available_tokens=100,
        training_distribution_coverage=0.9,
        calibration_status="PASS",
        external_validation_status="replicated",
    )
    causal = audit_world_model_prediction(
        task="estimate intervention hypothesis",
        simulation_type="causal_effect_estimate",
        available_tokens=100,
        training_distribution_coverage=0.9,
        calibration_status="calibrated",
        external_validation_status="trial_validated",
    )
    target_trial = audit_world_model_prediction(
        task="emulate trial",
        simulation_type="target_trial_emulation",
        available_tokens=100,
        training_distribution_coverage=0.9,
        calibration_status="validated",
        external_validation_status="not_validated",
    )
    validated_trial = audit_world_model_prediction(
        task="validated emulation",
        simulation_type="target_trial_emulation",
        available_tokens=100,
        training_distribution_coverage=0.9,
        calibration_status="validated",
        external_validation_status="externally_validated",
    )

    assert clean.safety_status == "PASS"
    assert clean.to_dict()["task"] == "forecast HbA1c"
    assert causal.allowed_claim_type == "causal_hypothesis"
    assert causal.safety_status == "PARTIAL"
    assert target_trial.allowed_claim_type == "target_trial_emulation_estimate"
    assert target_trial.safety_status == "PARTIAL"
    assert validated_trial.safety_status == "PASS"

    calls = []
    memory = SimpleNamespace(
        upsert_node=lambda *args, **kwargs: calls.append(("upsert", args, kwargs)),
        link_nodes=lambda *args, **kwargs: calls.append(("link", args, kwargs)),
    )
    record_world_model_card_to_action_graph(None, clean)
    record_world_model_card_to_action_graph(SimpleNamespace(), clean)
    record_world_model_card_to_action_graph(memory, clean)

    assert any(call[0] == "upsert" and call[1][0] == "world_model_audit" for call in calls)
    assert any(call[0] == "link" and call[2]["relation"] == "audits_prediction" for call in calls)
    assert world_model_audit("forecast HbA1c", available_tokens=5, ctx=None)["task"] == "forecast HbA1c"


def test_trajectory_tokenize_empty_rows_json_and_memory_recording():
    missing = trajectory_tokenize(ctx=SimpleNamespace(state=SimpleNamespace(custom_data={})))
    missing_no_ctx = trajectory_tokenize()
    assert missing["error"].startswith("No trajectory rows supplied")
    assert missing_no_ctx["error"].startswith("No trajectory rows supplied")

    calls = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(custom_data={}),
        memory=SimpleNamespace(upsert_node=lambda *args, **kwargs: calls.append((args, kwargs))),
    )
    rows = [
        {"participant_id": "p1", "timestamp": "2026-01-01", "modality": "bmi", "value": 25.0},
    ]

    result = trajectory_tokenize(rows_json=pd.Series(rows).to_json(orient="values"), max_bins=3, ctx=ctx)
    ctx_rows = trajectory_tokenize(ctx=SimpleNamespace(state=SimpleNamespace(custom_data={"trajectory_rows": rows})))

    assert result["status"] == "READY"
    assert result["future_query"] is None
    assert calls[0][0][0] == "trajectory_dataset"
    assert ctx_rows["status"] == "READY"


def test_new_skills_autodiscover_and_execute(tmp_path: Path):
    autodiscover_skills()
    reg = get_registry()

    class Ctx:
        pass

    ctx = Ctx()
    ctx.memory = LongTermMemory(tmp_path / "memory")
    ctx.state = SessionState()
    ctx.state.custom_data["trajectory_rows"] = [
        {"participant_id": "p1", "timestamp": "2026-01-01", "modality": "bmi", "value": 24.1, "value_type": "continuous"},
        {"participant_id": "p1", "timestamp": "2026-01-02", "modality": "exercise", "value": "walking", "value_type": "categorical"},
    ]

    pheno = reg.execute("phenotype_harmonize", {"concept": "HbA1c", "phenotype_type": "biomarker"}, ctx=ctx)
    traj = reg.execute("trajectory_tokenize", {"target_modality": "bmi", "target_timestamp": "2026-06-01"}, ctx=ctx)
    audit = reg.execute("world_model_audit", {"task": "forecast HbA1c", "available_tokens": 50}, ctx=ctx)
    cohort = reg.execute("cohort_card", {"query": "survival analysis for I21", "endpoint": "I21"}, ctx=ctx)

    assert pheno["concept_id"] == "hba1c"
    assert traj["n_tokens"] == 2
    assert audit["allowed_claim_type"] == "association_conditioned_forecast"
    assert cohort["cohort_type"] == "survival_cohort"


def test_reliability_layers_understand_new_records():
    autodiscover_skills()
    reg = get_registry()

    class Ctx:
        pass

    ctx = Ctx()
    ctx.state = SessionState()
    ctx.state.records = [
        AnalysisRecord(
            timestamp="2026-05-02T00:00:00",
            skill="world_model_audit",
            args={"task": "simulate intervention"},
            key_results={
                "safety_status": "FAIL",
                "allowed_claim_type": "association_conditioned_forecast",
                "reasons": ["causal effect estimate requested without sufficient trial/target-trial support"],
            },
            figure_paths=[],
        ),
        AnalysisRecord(
            timestamp="2026-05-02T00:01:00",
            skill="cohort_card",
            args={"endpoint": "I21"},
            key_results={
                "endpoint": "I21",
                "cohort_type": "target_trial_emulation",
                "status": "PARTIAL",
                "n_cases": 80,
                "banks": ["ukb", "hpp", "ckb"],
                "bias_flags": ["requires explicit confounding, adherence, censoring and positivity checks"],
            },
            figure_paths=[],
        ),
    ]

    stats = reg.execute("statistical_review", {"scope": "session"}, ctx=ctx)
    safety = reg.execute("safety_check", {"scope": "session"}, ctx=ctx)

    assert stats["n_critical"] >= 1
    assert any(i["type"] == "unsupported_world_model_claim" for i in stats["issues"])
    assert safety["overall"].startswith("BLOCK")
    assert any(i["type"] == "blocked_world_model_claim" for i in safety["issues"])


def test_report_interpretation_for_new_records():
    rec = AnalysisRecord(
        timestamp="2026-05-02T00:00:00",
        skill="trajectory_tokenize",
        args={},
        key_results={"n_tokens": 42, "n_participants": 3, "modalities": ["bmi", "glucose"]},
        figure_paths=[],
    )

    text = _interpret_skill(rec)
    assert "Trajectory tokenization" in text
    assert "HealthFormer-style" in text
