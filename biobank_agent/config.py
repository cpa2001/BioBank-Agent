"""Configuration loaded from .env via pydantic-settings."""

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
    ukb_parquet_dir: Path = Path("./milton_data")
    ukb_raw_dir: Path = Path("./UKB")

    # ── Output ───────────────────────────────────────────────────
    reports_dir: Path = Path("./reports")
    memory_dir: Path = Path.home() / ".biobank_agent"

    # ── Agent ────────────────────────────────────────────────────
    max_tool_rounds: int = 30
    context_window: int = 180_000

    # ── Derived paths ────────────────────────────────────────────
    @property
    def biomarker_parquet(self) -> Path:
        return self.ukb_parquet_dir / "ukb.parquet"

    @property
    def diagnoses_parquet(self) -> Path:
        return self.ukb_parquet_dir / "hesin_diag.parquet"

    @property
    def deaths_parquet(self) -> Path:
        return self.ukb_parquet_dir / "death_cause.parquet"

    @property
    def field_txt(self) -> Path:
        return self.ukb_parquet_dir / "field.txt"

    @property
    def category_txt(self) -> Path:
        return self.ukb_parquet_dir / "category.txt"

    @property
    def encoding_txt(self) -> Path:
        return self.ukb_parquet_dir / "esimpint.txt"

    @property
    def category_parquet_dir(self) -> Path:
        return self.ukb_parquet_dir / "categories"

    @property
    def raw_csv_dir(self) -> Path:
        return self.ukb_raw_dir / "UKB_info"

    @property
    def main_csv(self) -> Path:
        return self.ukb_raw_dir / "UKB" / "ukb672073.csv"

    @property
    def data_dict_csv(self) -> Path:
        return self.ukb_raw_dir / "UKB" / "Data_Dictionary_Showcase.csv"

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
