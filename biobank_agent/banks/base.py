"""Biobank configuration dataclasses — frozen, validated, serializable.

Each biobank (UKB, FinnGen, CKB, HPP) is described by a BankConfig
instance loaded from YAML. This provides schema validation and a
clear contract for what each biobank must define.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class FeatureGroup:
    """A named group of biomarker/phenotype fields.

    Example: FeatureGroup(name="blood_biochemistry", display_name="Blood Biochemistry",
                          fields={"30600": "Albumin", "30610": "ALT", ...})
    """
    name: str
    display_name: str
    fields: dict[str, str]  # field_id → human-readable name


@dataclass(frozen=True)
class BankConfig:
    """Complete configuration for a population biobank.

    Immutable once loaded — pass it around freely. Every biobank-specific
    value that was previously hardcoded now lives here.
    """

    # ── Identity ──────────────────────────────────────
    bank_id: str                    # "ukb", "finngen", "ckb", "hpp"
    display_name: str               # "UK Biobank"
    description: str                # one-liner for prompts and reports

    # ── Schema ────────────────────────────────────────
    patient_id_column: str          # "eid" for UKB, "FINNGENID" for FinnGen
    diagnoses_code_column: str      # "diag_icd10"
    deaths_code_column: str         # "cause_icd10"
    field_column_pattern: str       # "{field_id}-{instance}.{array}" for UKB

    # ── Feature groups ────────────────────────────────
    feature_groups: dict[str, FeatureGroup]

    # ── Optional metadata ─────────────────────────────
    caveats: str = ""               # known biases, caveats for reports
    coding_system: str = "ICD10"    # "ICD10", "ICD9", "ICD10-FI"
    default_instance: int = 0       # which instance index to use by default
    default_array: int = 0

    # ── File naming defaults ──────────────────────────
    biomarker_parquet_name: str = "biomarkers.parquet"
    diagnoses_parquet_name: str = "diagnoses.parquet"
    deaths_parquet_name: str = "deaths.parquet"
    catalog_fields_file: str = "field.txt"
    catalog_categories_file: str = "category.txt"

    def format_column(self, field_id: str, instance: int = -1, array: int = -1) -> str:
        """Format a field ID into the biobank's column naming convention."""
        inst = instance if instance >= 0 else self.default_instance
        arr = array if array >= 0 else self.default_array
        return self.field_column_pattern.format(
            field_id=field_id, instance=inst, array=arr,
        )

    def get_feature_group(self, group_name: str) -> Optional[FeatureGroup]:
        """Retrieve a feature group by name."""
        return self.feature_groups.get(group_name)

    def all_feature_ids(self) -> dict[str, str]:
        """Flatten all feature groups into a single field_id → name mapping."""
        result = {}
        for group in self.feature_groups.values():
            result.update(group.fields)
        return result
