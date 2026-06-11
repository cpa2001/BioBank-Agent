"""China Kadoorie Biobank (CKB) adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import BankAdapter, CohortCriteria, Modality, register_adapter


# Common CKB-specific column conventions. CKB stores phenotypes on a
# wide table keyed by ``study_id`` rather than UKB's ``eid``. Many
# diagnoses are coded in ICD-10 but with leading zeros stripped.
class CKBAdapter:
    bank_id: str = "ckb"
    display_name: str = "China Kadoorie Biobank"
    is_remote: bool = False

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else Path("./ckb_data")

    def field_id(self, semantic: str) -> str:
        return semantic.lower().strip().replace(" ", "_")

    def cohort_query(self, criteria: CohortCriteria) -> str:
        clauses: list[str] = []
        if criteria.icd_codes:
            codes = ",".join(f"'{self.normalize_icd(c)}'" for c in criteria.icd_codes)
            clauses.append(
                "EXISTS (SELECT 1 FROM diagnoses d "
                "WHERE d.study_id = b.study_id "
                f"AND d.icd10_code IN ({codes}))"
            )
        if criteria.age_range:
            lo, hi = criteria.age_range
            clauses.append(f"b.age_at_baseline BETWEEN {lo} AND {hi}")
        if criteria.sex:
            sex_map = {"M": 1, "F": 0}
            clauses.append(f"b.is_male = {sex_map.get(criteria.sex.upper(), 1)}")
        sql = "SELECT b.study_id FROM biomarkers b"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return sql

    def parquet_path(self, modality: Modality) -> Optional[Path]:
        names = {
            Modality.BIOMARKER: "ckb_biomarkers.parquet",
            Modality.DIAGNOSIS: "ckb_diagnoses.parquet",
            Modality.DEATH: "ckb_deaths.parquet",
        }
        name = names.get(modality)
        return (self.data_dir / name) if name else None

    def to_phecode(self, icd_code: str) -> Optional[str]:
        return None

    def normalize_icd(self, code: str) -> str:
        c = (code or "").upper().strip().replace(".", "")
        # CKB sometimes drops leading zeros; restore the canonical form
        # for ICD-10 codes like "I0" → "I10".
        if len(c) == 2 and c[0].isalpha() and c[1].isdigit():
            return c[0] + "1" + c[1]
        return c


register_adapter(CKBAdapter())


__all__ = ["CKBAdapter"]
