"""Helsinki Population Project (HPP) adapter.

HPP uses ICD-9 and ICD-10 mixed in legacy diagnosis tables. The
adapter normalises both into a unified ICD-10 view at query time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import BankAdapter, CohortCriteria, Modality, register_adapter


# Curated ICD-9 → ICD-10 mappings for the most common Finnish HPP
# disease codes. Full crosswalk lives in clinical-classifications
# data; this dict is a small expedient for the reference impl.
_ICD9_TO_ICD10 = {
    "250": "E11",   # T2D
    "401": "I10",   # essential hypertension
    "414": "I25",   # ischaemic heart disease
    "490": "J44",   # COPD-ish
    "493": "J45",   # asthma
    "496": "J44",
}
_ICD10_TO_ICD9 = {
    icd10: sorted({icd9 for icd9, mapped in _ICD9_TO_ICD10.items() if mapped == icd10})
    for icd10 in set(_ICD9_TO_ICD10.values())
}


class HPPAdapter:
    bank_id: str = "hpp"
    display_name: str = "Helsinki Population Project"
    is_remote: bool = False

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else Path("./hpp_data")

    def field_id(self, semantic: str) -> str:
        return semantic.lower().strip().replace(" ", "_")

    def cohort_query(self, criteria: CohortCriteria) -> str:
        # HPP uses snake_case columns and a join on participant_id.
        clauses: list[str] = []
        if criteria.icd_codes:
            normed = sorted({self.normalize_icd(c) for c in criteria.icd_codes})
            icd10_terms = [f"d.icd10 LIKE '{c}%'" for c in normed]
            icd9_prefixes = sorted({
                prefix
                for code in normed
                for prefix in self.legacy_icd9_prefixes(code)
            })
            icd9_terms = [f"d.icd9 LIKE '{prefix}%'" for prefix in icd9_prefixes]
            code_clause = " OR ".join([*icd10_terms, *icd9_terms]) or "1=0"
            clauses.append(
                "EXISTS (SELECT 1 FROM diagnoses d "
                "WHERE d.participant_id = b.participant_id "
                f"AND ({code_clause}))"
            )
        if criteria.age_range:
            lo, hi = criteria.age_range
            clauses.append(f"b.age_at_baseline BETWEEN {lo} AND {hi}")
        if criteria.sex:
            clauses.append(f"b.sex = '{criteria.sex.upper()}'")
        sql = "SELECT b.participant_id FROM biomarkers b"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return sql

    def parquet_path(self, modality: Modality) -> Optional[Path]:
        names = {
            Modality.BIOMARKER: "hpp_biomarkers.parquet",
            Modality.DIAGNOSIS: "hpp_diagnoses.parquet",
            Modality.DEATH: "hpp_deaths.parquet",
        }
        name = names.get(modality)
        return (self.data_dir / name) if name else None

    def to_phecode(self, icd_code: str) -> Optional[str]:
        return None

    def normalize_icd(self, code: str) -> str:
        c = (code or "").upper().strip().replace(".", "")
        # If it looks like an ICD-9 code (3 digits, optional letter
        # extension), map to ICD-10 if known.
        head3 = c[:3]
        if head3.isdigit() and head3 in _ICD9_TO_ICD10:
            return _ICD9_TO_ICD10[head3]
        return c

    def legacy_icd9_prefixes(self, icd10_code: str) -> list[str]:
        """Return ICD-9 prefixes that map to a normalized ICD-10 code."""
        normalized = self.normalize_icd(icd10_code)
        prefixes = list(_ICD10_TO_ICD9.get(normalized, []))
        raw = (icd10_code or "").upper().strip().replace(".", "")
        if raw[:3].isdigit() and raw[:3] not in prefixes:
            prefixes.append(raw[:3])
        return sorted(prefixes)


register_adapter(HPPAdapter())


__all__ = ["HPPAdapter"]
