"""Biobank config registry — load BankConfig from YAML files.

Configs live in ``biobank_agent/banks/configs/``. Each YAML file
defines one biobank (e.g. ``ukb.yaml``, ``finngen.yaml``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml  # PyYAML — already a transitive dep of pydantic

from .base import BankConfig, FeatureGroup

logger = logging.getLogger(__name__)

# Directory containing YAML config files
_CONFIGS_DIR = Path(__file__).parent / "configs"


def _parse_feature_groups(raw: dict) -> dict[str, FeatureGroup]:
    """Parse the feature_groups section of a YAML config."""
    groups = {}
    for name, fields in raw.items():
        display = name.replace("_", " ").title()
        groups[name] = FeatureGroup(
            name=name,
            display_name=display,
            fields={str(k): str(v) for k, v in fields.items()},
        )
    return groups


def load_bank(bank_id: str, configs_dir: Optional[Path] = None) -> BankConfig:
    """Load a BankConfig from a YAML file.

    Parameters
    ----------
    bank_id : str
        Bank identifier (e.g. "ukb", "finngen"). Must match a YAML filename.
    configs_dir : Path, optional
        Override the default configs directory.

    Returns
    -------
    BankConfig — frozen dataclass instance.

    Raises
    ------
    FileNotFoundError if the YAML file doesn't exist.
    """
    search_dir = configs_dir or _CONFIGS_DIR
    yaml_path = search_dir / f"{bank_id}.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"No config for bank '{bank_id}'. "
            f"Expected: {yaml_path}. "
            f"Available: {', '.join(list_banks(search_dir))}"
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {yaml_path}")

    feature_groups = _parse_feature_groups(data.get("feature_groups", {}))

    return BankConfig(
        bank_id=data["bank_id"],
        display_name=data["display_name"],
        description=data.get("description", ""),
        patient_id_column=data["patient_id_column"],
        diagnoses_code_column=data["diagnoses_code_column"],
        deaths_code_column=data["deaths_code_column"],
        field_column_pattern=data.get("field_column_pattern", "{field_id}-{instance}.{array}"),
        feature_groups=feature_groups,
        caveats=data.get("caveats", ""),
        coding_system=data.get("coding_system", "ICD10"),
        default_instance=data.get("default_instance", 0),
        default_array=data.get("default_array", 0),
        biomarker_parquet_name=data.get("biomarker_parquet_name", "biomarkers.parquet"),
        diagnoses_parquet_name=data.get("diagnoses_parquet_name", "diagnoses.parquet"),
        deaths_parquet_name=data.get("deaths_parquet_name", "deaths.parquet"),
        catalog_fields_file=data.get("catalog_fields_file", "field.txt"),
        catalog_categories_file=data.get("catalog_categories_file", "category.txt"),
    )


def list_banks(configs_dir: Optional[Path] = None) -> list[str]:
    """List available bank IDs (YAML files without extension)."""
    search_dir = configs_dir or _CONFIGS_DIR
    if not search_dir.exists():
        return []
    return sorted(p.stem for p in search_dir.glob("*.yaml"))
