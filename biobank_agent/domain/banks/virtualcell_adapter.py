"""VirtualCell/BWhair multimodal cohort adapter.

28-sample whole genome sequencing cohort plus BWhair Stereo-seq/scRNA/scATAC
h5ad resources. Donor links WGS to single-cell/spatial modalities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import BankAdapter, CohortCriteria, Modality, register_adapter

_VC_SEMANTIC_MAP = {
    "age": "age",
    "sex": "sex",
    "is_male": "is_male",
    "sample_id": "sample_id",
    "source_sample_id": "source_sample_id",
    "donor": "donor",
    "part": "part",
    "phenotype": "phenotype",
    "phenotype_group": "phenotype_group",
    "vcf_path": "vcf_path",
    "h5ad_path": "file_path",
    "file_path": "file_path",
    "modality": "modality",
    "technology": "technology",
    "hair_state": "hair_state",
    "hair_state_label": "hair_state_label",
    "cells": "cells",
}


class VirtualCellAdapter:
    bank_id: str = "virtualcell"
    display_name: str = "VirtualCell/BWhair Multimodal Cohort"
    is_remote: bool = False

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else Path("./data/virtualcell")

    def field_id(self, semantic: str) -> str:
        norm = semantic.lower().strip().replace(" ", "_")
        return _VC_SEMANTIC_MAP.get(norm, norm)

    def cohort_query(self, criteria: CohortCriteria) -> str:
        clauses: list[str] = []
        if criteria.age_range:
            lo, hi = criteria.age_range
            clauses.append(f"b.age BETWEEN {lo} AND {hi}")
        if criteria.sex:
            sex_map = {"M": 1, "F": 0}
            clauses.append(f"b.is_male = {sex_map.get(criteria.sex.upper(), 1)}")
        for key, value in criteria.extra_filters.items():
            if key == "phenotype":
                clauses.append(f"b.phenotype = '{value}'")
            elif key == "phenotype_group":
                clauses.append(f"b.phenotype_group = '{value}'")
        sql = "SELECT b.sample_id FROM biomarkers b"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return sql

    def parquet_path(self, modality: Modality) -> Optional[Path]:
        if modality == Modality.BIOMARKER:
            return self.data_dir / "vc_samples.parquet"
        return None

    def to_phecode(self, icd_code: str) -> Optional[str]:
        return None

    def normalize_icd(self, code: str) -> str:
        return (code or "").upper().strip().replace(".", "")


register_adapter(VirtualCellAdapter())

__all__ = ["VirtualCellAdapter"]
