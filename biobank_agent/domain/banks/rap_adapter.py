"""UKB-RAP adapter — DNAnexus remote SparkSQL.

Read-only for v3 (no remote training). The adapter rewrites cohort
queries to use SparkSQL dialect and points the parquet hooks at a
local cache under ``~/.biobank_agent/rap_cache``.

Heavy SDK dependencies (``dxpy``, ``pyspark``) are imported lazily so
``import biobank_agent.domain.banks`` doesn't fail on machines that
only run local UKB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import BankAdapter, CohortCriteria, Modality, register_adapter


class RAPAdapter:
    bank_id: str = "ukb_rap"
    display_name: str = "UK Biobank Research Analysis Platform"
    is_remote: bool = True

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else (
            Path.home() / ".biobank_agent" / "rap_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def field_id(self, semantic: str) -> str:
        return semantic

    def cohort_query(self, criteria: CohortCriteria) -> str:
        # SparkSQL on the RAP cohort browser uses backtick-quoted
        # column names and the ``participant`` table. The semantics
        # remain similar enough that the SQL is portable for the few
        # cases we care about in v3.
        clauses: list[str] = []
        if criteria.icd_codes:
            codes = ",".join(f"'{self.normalize_icd(c)}'" for c in criteria.icd_codes)
            clauses.append(
                "ARRAY_LENGTH(ARRAY_INTERSECT("
                "FLATTEN(`p41202_arrAll`), "
                f"ARRAY({codes})"
                ")) > 0"
            )
        if criteria.age_range:
            lo, hi = criteria.age_range
            clauses.append(f"`p21022` BETWEEN {lo} AND {hi}")
        if criteria.sex:
            sex_map = {"M": 1, "F": 0}
            clauses.append(f"`p31` = {sex_map.get(criteria.sex.upper(), 1)}")
        sql = "SELECT `eid` FROM `participant`"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return sql

    def parquet_path(self, modality: Modality) -> Optional[Path]:
        # Cached parquet snapshots keyed by the upstream query hash.
        # Returning None is the canonical "remote, fetch on demand"
        # signal — DataManager treats that as "open via dxpy".
        return None

    def to_phecode(self, icd_code: str) -> Optional[str]:
        return None

    def normalize_icd(self, code: str) -> str:
        return (code or "").upper().strip().replace(".", "")


register_adapter(RAPAdapter())


__all__ = ["RAPAdapter"]
