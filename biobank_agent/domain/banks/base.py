"""Bank adapter Protocol.

Replaces the placeholder ``biobank_agent/banks/*`` with a real
``BankAdapter`` Protocol that captures the parts of biobank operation
that genuinely differ between UKB / HPP / CKB / UKB-RAP:

* Field id mapping (ICD-10 fields, biomarker codes, dialect)
* Cohort SQL (DuckDB local vs RAP SparkSQL)
* Phecode / ICD-9/10 conversion
* Parquet path resolution

Each adapter is registered under its ``bank_id``; the ``DataManager``
resolves ``settings.bank_id`` to an adapter at agent startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


class Modality(str, Enum):
    BIOMARKER = "biomarker"
    DIAGNOSIS = "diagnosis"
    DEATH = "death"
    GENOMICS = "genomics"
    METABOLOMICS = "metabolomics"
    PROTEOMICS = "proteomics"
    IMAGING = "imaging"
    LIFESTYLE = "lifestyle"


@dataclass
class CohortCriteria:
    """Inputs to ``BankAdapter.cohort_query``."""

    icd_codes: list[str] = field(default_factory=list)
    age_range: Optional[tuple[float, float]] = None
    sex: Optional[str] = None  # "M" | "F"
    follow_up_years: Optional[float] = None
    require_biomarkers: list[str] = field(default_factory=list)
    extra_filters: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BankAdapter(Protocol):
    """The contract every biobank backend implements."""

    @property
    def bank_id(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def is_remote(self) -> bool: ...

    def field_id(self, semantic: str) -> str: ...

    def cohort_query(self, criteria: CohortCriteria) -> str: ...

    def parquet_path(self, modality: Modality) -> Optional[Path]: ...

    def to_phecode(self, icd_code: str) -> Optional[str]: ...

    def normalize_icd(self, code: str) -> str: ...


# ── Adapter registry ────────────────────────────────────────


_REGISTRY: dict[str, BankAdapter] = {}
_ALIASES = {
    "rap": "ukb_rap",
    "ukb-rap": "ukb_rap",
    "ukbrap": "ukb_rap",
    "ukb rap": "ukb_rap",
}


def canonical_bank_id(bank_id: str | None) -> str:
    """Return the stable registry key for a user-facing bank identifier."""
    raw = str(bank_id or "ukb").strip().lower()
    normalized = raw.replace("_", "-")
    return _ALIASES.get(raw, _ALIASES.get(normalized, raw.replace("-", "_")))


def register_adapter(adapter: BankAdapter) -> None:
    _REGISTRY[canonical_bank_id(adapter.bank_id)] = adapter


def get_adapter(bank_id: str) -> Optional[BankAdapter]:
    return _REGISTRY.get(canonical_bank_id(bank_id))


__all__ = [
    "Modality",
    "CohortCriteria",
    "BankAdapter",
    "canonical_bank_id",
    "register_adapter",
    "get_adapter",
]
