"""Configuration loaded from .env via pydantic-settings.

All biobank-specific settings have sensible UK Biobank defaults but are
fully overridable for other biobanks (FinnGen, CKB, HPP, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────
    llm_base_url: str = "http://api.shubiaobiao.cn"
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"

    # ── Data Paths ───────────────────────────────────────────────
    data_dir: Path = Path("./milton_data")
    raw_dir: Path = Path("./UKB")

    # Backward compatibility aliases
    @property
    def ukb_parquet_dir(self) -> Path:
        return self.data_dir

    @property
    def ukb_raw_dir(self) -> Path:
        return self.raw_dir

    # ── Biobank Identity ─────────────────────────────────────────
    # ── Biobank Identity ─────────────────────────────────────────
    bank_id: str = "ukb"
    biobank_name: str = "UK Biobank"
    biobank_abbreviation: str = "UKB"
    biobank_description: str = (
        "a large-scale prospective cohort study comprising over 500,000 "
        "participants aged 40-69 at recruitment"
    )
    biobank_caveats: str = (
        "healthy volunteer cohort with known selection biases"
    )

    # ── Column / Schema Identity ─────────────────────────────────
    subject_id_col: str = "eid"
    diagnoses_code_col: str = "diag_icd10"
    deaths_code_col: str = "cause_icd10"
    field_column_pattern: str = "{field_id}-{instance}.{array}"

    # ── Dataset File Names ───────────────────────────────────────
    biomarker_parquet_name: str = "ukb.parquet"
    diagnoses_parquet_name: str = "hesin_diag.parquet"
    deaths_parquet_name: str = "death_cause.parquet"
    catalog_fields_file: str = "field.txt"
    catalog_categories_file: str = "category.txt"
    catalog_encoding_file: str = "esimpint.txt"

    # ── Output ───────────────────────────────────────────────────
    reports_dir: Path = Path("./reports")
    memory_dir: Path = Path.home() / ".biobank_agent"

    # ── Web Search ───────────────────────────────────────────────
    search_provider: str = "duckduckgo"
    search_api_key: str = ""

    # ── Plan Mode ────────────────────────────────────────────────
    plans_dir: Path = Path("./plans")

    # ── Custom Skills ────────────────────────────────────────────
    custom_skills_dir: Path = Path("./custom_skills")

    # ── Agent ────────────────────────────────────────────────────
    max_tool_rounds: int = 30
    context_window: int = 180_000

    # ── Derived Paths ────────────────────────────────────────────

    @property
    def biomarker_parquet(self) -> Path:
        return self.data_dir / self.biomarker_parquet_name

    @property
    def diagnoses_parquet(self) -> Path:
        return self.data_dir / self.diagnoses_parquet_name

    @property
    def deaths_parquet(self) -> Path:
        return self.data_dir / self.deaths_parquet_name

    @property
    def field_txt(self) -> Path:
        return self.data_dir / self.catalog_fields_file

    @property
    def category_txt(self) -> Path:
        return self.data_dir / self.catalog_categories_file

    @property
    def encoding_txt(self) -> Path:
        return self.data_dir / self.catalog_encoding_file

    @property
    def category_parquet_dir(self) -> Path:
        return self.data_dir / "categories"

    @property
    def raw_csv_dir(self) -> Path:
        return self.raw_dir / "UKB_info"

    @property
    def main_csv(self) -> Path:
        return self.raw_dir / "UKB" / "ukb672073.csv"

    @property
    def data_dict_csv(self) -> Path:
        return self.raw_dir / "UKB" / "Data_Dictionary_Showcase.csv"

    def ensure_dirs(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)


def get_settings(**overrides) -> Settings:
    """Factory that finds .env walking up from cwd."""
    cwd = Path.cwd()
    for d in [cwd, *cwd.parents]:
        env = d / ".env"
        if env.exists():
            return Settings(_env_file=str(env), **overrides)
    return Settings(**overrides)
