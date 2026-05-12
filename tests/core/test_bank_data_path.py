"""BankAdapter data-path probes from DataManager through model training."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from biobank_agent.config import Settings
from biobank_agent.data.cohort import build_cohort
from biobank_agent.data.loader import DataManager
from biobank_agent.skills.bank_data_probe import bank_data_probe
from biobank_agent.skills.bank_data_readiness import bank_data_readiness
from biobank_agent.skills.train_model import train_model


class _State:
    def __init__(self) -> None:
        self.cohorts = {}
        self.models = {}
        self.model_metadata = {}
        self.feature_matrix = None
        self.labels = None
        self.figures = []


def _settings(tmp_path, bank_id: str) -> Settings:
    return Settings(
        data_dir=tmp_path,
        raw_dir=tmp_path,
        reports_dir=tmp_path / "reports",
        memory_dir=tmp_path / "memory",
        plans_dir=tmp_path / "plans",
        bank_id=bank_id,
        max_train_rows_default=0,
        default_analysis_sample_size=0,
    )


def _ctx(tmp_path, bank_id: str):
    settings = _settings(tmp_path, bank_id)
    dm = DataManager(settings)
    return SimpleNamespace(
        dm=dm,
        settings=settings,
        state=_State(),
        report_dir=tmp_path / "reports",
        emit_progress=lambda *args, **kwargs: None,
    )


def _base_features(ids: np.ndarray, case_mask: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {
        "hba1c": np.where(case_mask, 65.0, 39.0) + rng.normal(0, 2.0, len(ids)),
        "bmi": np.where(case_mask, 32.0, 25.0) + rng.normal(0, 1.0, len(ids)),
        "glucose": np.where(case_mask, 7.6, 4.9) + rng.normal(0, 0.2, len(ids)),
        "hdl_cholesterol": np.where(case_mask, 1.0, 1.4) + rng.normal(0, 0.05, len(ids)),
    }


def _write_hpp_fixture(tmp_path, n_cases: int = 120, n_controls: int = 180) -> None:
    ids = np.arange(1, n_cases + n_controls + 1)
    case_mask = ids <= n_cases
    biomarker = pd.DataFrame({"participant_id": ids, **_base_features(ids, case_mask)})
    biomarker["age_at_baseline"] = np.where(case_mask, 62, 54)
    biomarker["sex"] = np.where(ids % 2 == 0, 1, 0)
    biomarker.to_parquet(tmp_path / "hpp_biomarkers.parquet")

    diag_rows = []
    for pid in ids:
        if pid <= n_cases // 2:
            diag_rows.append({"participant_id": pid, "icd10": "E11", "icd9": ""})
        elif pid <= n_cases:
            diag_rows.append({"participant_id": pid, "icd10": "", "icd9": "250"})
        else:
            diag_rows.append({"participant_id": pid, "icd10": "I10", "icd9": "401"})
    pd.DataFrame(diag_rows).to_parquet(tmp_path / "hpp_diagnoses.parquet")


def _write_ckb_fixture(tmp_path, n_cases: int = 120, n_controls: int = 180) -> None:
    ids = np.arange(10_001, 10_001 + n_cases + n_controls)
    case_mask = np.arange(len(ids)) < n_cases
    biomarker = pd.DataFrame({"study_id": ids, **_base_features(ids, case_mask)})
    biomarker["age_at_baseline"] = np.where(case_mask, 61, 53)
    biomarker["is_male"] = np.where(ids % 2 == 0, 1, 0)
    biomarker.to_parquet(tmp_path / "ckb_biomarkers.parquet")

    diagnoses = pd.DataFrame({
        "study_id": ids,
        "icd10_code": np.where(case_mask, "E11", "I10"),
    })
    diagnoses.to_parquet(tmp_path / "ckb_diagnoses.parquet")


@pytest.mark.parametrize(
    ("bank_id", "writer", "id_col"),
    [
        ("hpp", _write_hpp_fixture, "participant_id"),
        ("ckb", _write_ckb_fixture, "study_id"),
    ],
)
def test_bank_adapter_data_path_builds_cohort_and_trains_model(tmp_path, bank_id, writer, id_col):
    writer(tmp_path)
    ctx = _ctx(tmp_path, bank_id)

    probe = bank_data_probe(ctx=ctx)
    assert probe["status"] == "READY"
    assert probe["bank_id"] == bank_id
    assert probe["subject_id_col"] == id_col
    assert probe["n_subjects"] == 300
    assert probe["diagnosis_probe"]["n_case_subjects"] == 120
    assert {item["query"]: item["status"] for item in probe["field_probe"]} == {
        "hba1c": "READY",
        "bmi": "READY",
        "glucose": "READY",
    }

    cohort = build_cohort(ctx.dm, "E11")
    assert len(cohort) == 300
    assert int(cohort["label"].sum()) == 120
    assert int((cohort["label"] == 0).sum()) == 180
    assert id_col in cohort.columns
    assert {"hba1c", "bmi", "glucose"}.issubset(cohort.columns)

    result = train_model("E11", model_type="logistic", n_folds=2, ctx=ctx)
    assert "error" not in result
    assert result["model_key"] == "E11_logistic"
    assert result["n_cases"] == 120
    assert result["n_controls"] == 180
    assert result["training_sample"]["applied"] is False
    assert len(ctx.state.feature_matrix) == 300


def test_rap_alias_resolves_to_remote_adapter_without_local_data(tmp_path):
    ctx = _ctx(tmp_path, "UKB-RAP")

    probe = bank_data_probe(ctx=ctx)

    assert probe["bank_id"] == "ukb_rap"
    assert probe["is_remote"] is True
    assert probe["status"] == "REMOTE_READY"
    assert probe["subject_id_col"] == "eid"
    assert probe["views"] == {"biomarkers": False, "diagnoses": False, "deaths": False}


def test_bank_data_readiness_skips_missing_external_configs_and_writes_artifacts(tmp_path, monkeypatch):
    for name in [
        "BIOBANK_HPP_DATA_DIR",
        "HPP_DATA_DIR",
        "BIOBANK_CKB_DATA_DIR",
        "CKB_DATA_DIR",
        "BIOBANK_RAP_ENABLED",
        "DX_PROJECT_CONTEXT_ID",
        "DX_WORKSPACE_ID",
        "DNANEXUS_PROJECT_ID",
    ]:
        monkeypatch.delenv(name, raising=False)
    ctx = _ctx(tmp_path / "ctx", "ukb")

    result = bank_data_readiness(
        banks="hpp,ckb,ukb_rap",
        output_dir=str(tmp_path / "readiness"),
        ctx=ctx,
    )

    assert result["status"] == "SKIPPED"
    assert {item["bank_id"]: item["status"] for item in result["banks"]} == {
        "hpp": "SKIPPED_NO_CONFIG",
        "ckb": "SKIPPED_NO_CONFIG",
        "ukb_rap": "SKIPPED_NO_CONFIG",
    }
    assert Path(result["artifact_json"]).exists()
    assert Path(result["artifact_markdown"]).exists()
    assert "raw_rows" not in Path(result["artifact_json"]).read_text()


def test_bank_data_readiness_runs_configured_hpp_ckb_fixture_probes(tmp_path, monkeypatch):
    hpp_dir = tmp_path / "hpp"
    ckb_dir = tmp_path / "ckb"
    hpp_dir.mkdir()
    ckb_dir.mkdir()
    _write_hpp_fixture(hpp_dir)
    _write_ckb_fixture(ckb_dir)
    monkeypatch.setenv("BIOBANK_HPP_DATA_DIR", str(hpp_dir))
    monkeypatch.setenv("BIOBANK_CKB_DATA_DIR", str(ckb_dir))
    ctx = _ctx(tmp_path / "ctx", "ukb")

    result = bank_data_readiness(
        banks="hpp,ckb",
        probe_fields="hba1c,bmi,glucose",
        output_dir=str(tmp_path / "readiness"),
        ctx=ctx,
    )

    assert result["status"] == "READY"
    by_bank = {item["bank_id"]: item for item in result["banks"]}
    assert by_bank["hpp"]["status"] == "READY"
    assert by_bank["ckb"]["status"] == "READY"
    assert by_bank["hpp"]["diagnosis_probe"]["n_case_subjects"] == 120
    assert by_bank["ckb"]["diagnosis_probe"]["n_case_subjects"] == 120
    assert by_bank["hpp"]["n_subjects"] == 300
    assert by_bank["ckb"]["n_subjects"] == 300
    artifact = Path(result["artifact_markdown"]).read_text()
    assert "| hpp | READY | 300 | 120 | env:BIOBANK_HPP_DATA_DIR |  |" in artifact
    assert "| ckb | READY | 300 | 120 | env:BIOBANK_CKB_DATA_DIR |  |" in artifact
