"""Tests for UKB+HPP+CKB agentic cohort discovery primitives."""

from pathlib import Path

import pandas as pd

from biobank_agent.banks import list_banks, load_bank
from biobank_agent.data.cohort_cards import build_cohort_card
from biobank_agent.data.phenotype import (
    harmonize_phenotype,
    record_phenotype_to_action_graph,
)
from biobank_agent.data.trajectory import TrajectoryTokenizer
from biobank_agent.memory import LongTermMemory
from biobank_agent.registry import autodiscover_skills, get_registry
from biobank_agent.skills.report import _interpret_skill
from biobank_agent.state import AnalysisRecord, SessionState
from biobank_agent.world_model import audit_world_model_prediction


def test_hpp_and_ckb_bank_configs_load():
    banks = list_banks()
    assert {"ukb", "hpp", "ckb"}.issubset(set(banks))

    hpp = load_bank("hpp")
    ckb = load_bank("ckb")

    assert hpp.patient_id_column == "participant_id"
    assert "longitudinal_deep_phenotyping" in hpp.feature_groups
    assert ckb.display_name == "China Kadoorie Biobank"
    assert "exposure_lifestyle" in ckb.feature_groups


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
