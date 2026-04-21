"""Abstract interface for Foundation Model embedding extraction.

Stub for future integration with Evo2, ESM-2, BrainLM, etc.
Implementations will call remote servers via HTTP/gRPC.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class FMEmbeddingInterface(ABC):
    """Abstract base for calling remote foundation models."""

    @abstractmethod
    def encode(self, data: pd.DataFrame, modality: str,
               model_name: Optional[str] = None) -> np.ndarray:
        """Extract embeddings from data using a remote FM.

        Parameters
        ----------
        data : pd.DataFrame — input data (rows = samples)
        modality : str — data modality (e.g. 'genomic', 'protein', 'imaging')
        model_name : optional model name override

        Returns
        -------
        np.ndarray of shape (n_samples, embedding_dim)
        """

    @abstractmethod
    def available_models(self) -> list[dict]:
        """List available FM models with metadata.

        Returns list of dicts with keys: name, modality, embedding_dim, description.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the remote FM server is reachable."""


class RemoteFMClient(FMEmbeddingInterface):
    """Placeholder implementation — to be completed when FM servers are deployed."""

    def __init__(self, base_url: str = "", api_key: str = ""):
        self.base_url = base_url
        self.api_key = api_key

    def encode(self, data, modality, model_name=None):
        raise NotImplementedError(
            "FM embedding server not yet configured. "
            "Set FM_SERVER_URL in .env and deploy the FM service."
        )

    def available_models(self):
        return [
            {"name": "evo2", "modality": "genomic", "embedding_dim": 1024,
             "description": "Evo2 DNA foundation model (not yet deployed)"},
            {"name": "esm2", "modality": "protein", "embedding_dim": 1280,
             "description": "ESM-2 protein language model (not yet deployed)"},
            {"name": "brainlm", "modality": "imaging", "embedding_dim": 768,
             "description": "BrainLM neuroimaging model (not yet deployed)"},
        ]

    def health_check(self):
        return False
