"""Multi-biobank configuration system.

Provides BankConfig dataclasses and a YAML-based registry for
supporting multiple biobanks (UK Biobank, FinnGen, CKB, HPP, etc.).
"""

from .base import BankConfig, FeatureGroup
from .registry import load_bank, list_banks

__all__ = ["BankConfig", "FeatureGroup", "load_bank", "list_banks"]
