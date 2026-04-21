"""Abstract interface for multimodal data fusion.

Stub for combining tabular biomarkers with FM embeddings,
imaging features, genomic variants, etc.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class MultimodalFusionInterface(ABC):
    """Abstract base for fusing multiple data modalities."""

    @abstractmethod
    def fuse(self, modalities: dict[str, np.ndarray],
             method: str = "concat") -> np.ndarray:
        """Fuse embeddings from multiple modalities.

        Parameters
        ----------
        modalities : dict mapping modality name → embedding matrix (n_samples, dim)
        method : fusion strategy ('concat', 'attention', 'gated')

        Returns
        -------
        np.ndarray of shape (n_samples, fused_dim)
        """

    @abstractmethod
    def supported_methods(self) -> list[str]:
        """List available fusion methods."""
