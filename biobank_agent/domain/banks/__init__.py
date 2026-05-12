"""Bank adapters — UKB / HPP / CKB / UKB-RAP."""
from .base import BankAdapter, CohortCriteria, Modality, canonical_bank_id, register_adapter, get_adapter
from .ukb_adapter import UKBAdapter
from .hpp_adapter import HPPAdapter
from .ckb_adapter import CKBAdapter
from .rap_adapter import RAPAdapter

__all__ = [
    "BankAdapter",
    "CohortCriteria",
    "Modality",
    "canonical_bank_id",
    "register_adapter",
    "get_adapter",
    "UKBAdapter",
    "HPPAdapter",
    "CKBAdapter",
    "RAPAdapter",
]
