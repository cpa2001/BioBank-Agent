"""Edge coverage for routing heuristics and shared configuration objects."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from biobank_agent.banks.base import BankConfig, FeatureGroup
from biobank_agent.complexity import Strategy, classify_complexity
from biobank_agent.config import Settings, get_settings
from biobank_agent.constants import (
    BOUNDED_01_METRICS,
    POSITIVE_ONLY_METRICS,
    SAMPLE_SIZE_KEYS,
    UKB_ASSESSMENT_CENTRES,
    UKB_MAX_AGE_RECRUITMENT,
    UKB_MIN_AGE_RECRUITMENT,
    UKB_TOTAL_PARTICIPANTS,
    UKB_VALID_CHROMOSOMES,
)


def test_shared_constants_expose_expected_domain_bounds():
    assert UKB_TOTAL_PARTICIPANTS == 502_411
    assert (UKB_MIN_AGE_RECRUITMENT, UKB_MAX_AGE_RECRUITMENT) == (37, 73)
    assert UKB_ASSESSMENT_CENTRES == 22
    assert min(UKB_VALID_CHROMOSOMES) == 1
    assert max(UKB_VALID_CHROMOSOMES) == 23
    assert {"auc", "r2"}.issubset(BOUNDED_01_METRICS)
    assert {"hazard_ratio", "or_value"}.issubset(POSITIVE_ONLY_METRICS)
    assert {"n_cases", "sample_size"}.issubset(SAMPLE_SIZE_KEYS)


def test_bank_config_formats_columns_and_flattens_feature_groups():
    config = BankConfig(
        bank_id="demo",
        display_name="Demo Biobank",
        description="Synthetic biobank",
        patient_id_column="eid",
        diagnoses_code_column="diag_icd10",
        deaths_code_column="cause_icd10",
        field_column_pattern="{field_id}-{instance}.{array}",
        default_instance=1,
        default_array=2,
        feature_groups={
            "blood": FeatureGroup("blood", "Blood", {"30740": "Glucose"}),
            "liver": FeatureGroup("liver", "Liver", {"30620": "ALT"}),
        },
    )

    assert config.format_column("30740") == "30740-1.2"
    assert config.format_column("30740", instance=0, array=0) == "30740-0.0"
    assert config.get_feature_group("blood").display_name == "Blood"
    assert config.get_feature_group("missing") is None
    assert config.all_feature_ids() == {"30740": "Glucose", "30620": "ALT"}


def test_settings_derived_paths_and_aliases(tmp_path):
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw"
    settings = Settings(
        data_dir=data_dir,
        raw_dir=raw_dir,
        reports_dir=tmp_path / "reports",
        memory_dir=tmp_path / "memory",
    )

    assert settings.ukb_parquet_dir == data_dir
    assert settings.ukb_raw_dir == raw_dir
    assert settings.biomarker_parquet == data_dir / "ukb.parquet"
    assert settings.diagnoses_parquet == data_dir / "hesin_diag.parquet"
    assert settings.deaths_parquet == data_dir / "death_cause.parquet"
    assert settings.field_txt == data_dir / "field.txt"
    assert settings.category_txt == data_dir / "category.txt"
    assert settings.encoding_txt == data_dir / "esimpint.txt"
    assert settings.category_parquet_dir == data_dir / "categories"
    assert settings.raw_csv_dir == raw_dir / "UKB_info"
    assert settings.main_csv == raw_dir / "UKB" / "ukb672073.csv"
    assert settings.data_dict_csv == raw_dir / "UKB" / "Data_Dictionary_Showcase.csv"

    settings.ensure_dirs()
    assert settings.reports_dir.is_dir()
    assert settings.memory_dir.is_dir()


def test_get_settings_walks_up_for_env_file_and_supports_overrides(tmp_path, monkeypatch):
    env_data = tmp_path / "env-data"
    env_raw = tmp_path / "env-raw"
    (tmp_path / ".env").write_text(
        f"DATA_DIR={env_data}\nRAW_DIR={env_raw}\nBIOBANK_NAME=Env Bank\n",
        encoding="utf-8",
    )
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    settings = get_settings(biobank_abbreviation="ENV")

    assert settings.data_dir == env_data
    assert settings.raw_dir == env_raw
    assert settings.biobank_name == "Env Bank"
    assert settings.biobank_abbreviation == "ENV"


def test_get_settings_without_env_uses_direct_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    settings = get_settings(data_dir=Path("direct-data"), raw_dir=Path("direct-raw"))

    assert settings.data_dir == Path("direct-data")
    assert settings.raw_dir == Path("direct-raw")


def test_complexity_explicit_debate_and_pipeline_routes():
    debate = classify_complexity("Compare diabetes and I21 in a debate with systematic review")
    pipeline = classify_complexity("Run multiple workflow pipeline stages for phenotype QC")

    assert debate.strategy is Strategy.DEBATE
    assert debate.reason == "Explicit debate/comparison request detected"
    assert pipeline.strategy is Strategy.SUPERVISOR
    assert pipeline.reason == "Multi-step pipeline detected — supervisor decomposition"


def test_complexity_high_score_ensemble_and_failure_factors():
    records = [
        SimpleNamespace(key_results={"error": "failed"}),
        SimpleNamespace(key_results={"ok": True}),
        SimpleNamespace(key_results={"error": "failed again"}),
    ]

    result = classify_complexity(
        "verify biomarkers? validate cohort? double check report? extra; clause; final",
        records=records,
    )

    assert result.strategy is Strategy.ENSEMBLE
    assert result.factors["multiple_questions"] == 3
    assert result.factors["multiple_clauses"] == 3
    assert result.factors["recent_failures"] == 2


def test_complexity_history_without_repeated_failures_does_not_add_factor():
    records = [
        SimpleNamespace(key_results={"error": "failed"}),
        SimpleNamespace(key_results={"ok": True}),
        SimpleNamespace(key_results="not a dict"),
    ]

    result = classify_complexity("show me the diabetes status", records=records)

    assert result.strategy is Strategy.SINGLE
    assert "recent_failures" not in result.factors


def test_complexity_long_very_high_score_routes_to_debate_without_explicit_request():
    query = (
        "verify validate critique cross-check double-check multiple first then step 1 "
        "comprehensive systematic discovery hypothesis research plan meta-analysis literature survey batch "
        + ("biomarker " * 70)
    )

    result = classify_complexity(query)

    assert result.strategy is Strategy.DEBATE
    assert result.reason == "Very high complexity — debate for thorough analysis"
    assert result.factors["long_query"] is True
