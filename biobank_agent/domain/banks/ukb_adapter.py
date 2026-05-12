"""UK Biobank adapter — the canonical ICD-10 + DuckDB local store."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import BankAdapter, CohortCriteria, Modality, register_adapter

# Subset of common semantic→field-id mappings for the M3 reference
# implementation. Full coverage lives in the field catalogue
# (``biobank_agent/data/catalog.py``) — this dict is only a fallback
# when callers have already coerced semantic shorthands.
_UKB_SEMANTIC_TO_FIELD = {
    "hba1c": "30750",
    "glucose": "30740",
    "ldl_cholesterol": "30780",
    "hdl_cholesterol": "30760",
    "bmi": "21001",
    "systolic_bp": "4080",
    "diastolic_bp": "4079",
    "smoking_status": "20116",
    "townsend_index": "189",
    "ethnicity": "21000",
    "sex": "31",
    "year_of_birth": "34",
    "age_at_recruitment": "21022",
}


class UKBAdapter:
    bank_id: str = "ukb"
    display_name: str = "UK Biobank"
    is_remote: bool = False

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else Path("./milton_data")

    def field_id(self, semantic: str) -> str:
        norm = semantic.lower().strip().replace(" ", "_")
        if norm in _UKB_SEMANTIC_TO_FIELD:
            return _UKB_SEMANTIC_TO_FIELD[norm]
        if norm.isdigit():
            return norm
        return semantic

    def cohort_query(self, criteria: CohortCriteria) -> str:
        clauses: list[str] = []
        if criteria.icd_codes:
            codes = ",".join(f"'{self.normalize_icd(c)}'" for c in criteria.icd_codes)
            clauses.append(
                "EXISTS (SELECT 1 FROM diagnoses d WHERE d.eid = b.eid "
                f"AND d.diag_icd10 IN ({codes}))"
            )
        if criteria.age_range:
            lo, hi = criteria.age_range
            clauses.append(f"b.\"21022-0.0\" BETWEEN {lo} AND {hi}")
        if criteria.sex:
            sex_map = {"M": 1, "F": 0}
            clauses.append(f"b.\"31-0.0\" = {sex_map.get(criteria.sex.upper(), 1)}")
        for biomarker in criteria.require_biomarkers:
            fid = self.field_id(biomarker)
            clauses.append(f"b.\"{fid}-0.0\" IS NOT NULL")
        sql = "SELECT b.eid FROM biomarkers b"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return sql

    def parquet_path(self, modality: Modality) -> Optional[Path]:
        names = {
            Modality.BIOMARKER: "ukb.parquet",
            Modality.DIAGNOSIS: "hesin_diag.parquet",
            Modality.DEATH: "death_cause.parquet",
        }
        name = names.get(modality)
        return (self.data_dir / name) if name else None

    def to_phecode(self, icd_code: str) -> Optional[str]:
        # ICD-10 → phecode mapping is a multi-thousand-row table; we
        # leave the actual lookup to ``data/phecode.csv`` and only
        # provide a hook here. Returns None until the real mapping
        # file lives in the repo.
        return None

    def normalize_icd(self, code: str) -> str:
        c = (code or "").upper().strip()
        c = c.replace(".", "")  # UKB diagnoses table stores codes without dots
        return c


register_adapter(UKBAdapter())


__all__ = ["UKBAdapter"]
