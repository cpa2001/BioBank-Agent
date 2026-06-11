"""Abstract interface for multimodal data fusion.

Stub for combining tabular biomarkers with FM embeddings,
imaging features, genomic variants, etc.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
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


@dataclass
class StructuralSignal:
    """Minimal cross-modal structural anomaly signal."""

    modality: str
    signal_type: str
    value: float
    detail: str = ""


class SimpleMultimodalGrounder(MultimodalFusionInterface):
    """MVP multimodal grounder with graceful degradation.

    This class intentionally avoids heavyweight training and instead extracts
    robust structural signals from available modalities (tabular, figure tensor).
    """

    _SUPPORTED = ("concat", "zscore_concat")

    def fuse(self, modalities: dict[str, np.ndarray], method: str = "concat") -> np.ndarray:
        if not modalities:
            return np.empty((0, 0))
        arrays = [np.asarray(v) for v in modalities.values() if v is not None and np.asarray(v).size > 0]
        if not arrays:
            return np.empty((0, 0))

        if method == "concat":
            return np.concatenate(arrays, axis=1)
        if method == "zscore_concat":
            normalized = []
            for arr in arrays:
                mu = arr.mean(axis=0, keepdims=True)
                sigma = arr.std(axis=0, keepdims=True)
                sigma[sigma == 0] = 1.0
                normalized.append((arr - mu) / sigma)
            return np.concatenate(normalized, axis=1)
        raise ValueError(f"Unsupported fusion method: {method}")

    def supported_methods(self) -> list[str]:
        return list(self._SUPPORTED)

    def extract_structural_signals(
        self,
        table: Optional[pd.DataFrame] = None,
        modality_tensors: Optional[dict[str, np.ndarray]] = None,
    ) -> list[StructuralSignal]:
        """Extract lightweight anomaly cues from available modalities."""
        signals: list[StructuralSignal] = []
        if table is not None and not table.empty:
            na_ratio = float(table.isna().mean().mean())
            signals.append(StructuralSignal(
                modality="table",
                signal_type="missingness_ratio",
                value=na_ratio,
                detail="fraction of missing entries",
            ))
            numeric = table.select_dtypes(include="number")
            if not numeric.empty:
                z = np.abs((numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1))
                outlier_ratio = float((z > 3.0).mean().mean())
                signals.append(StructuralSignal(
                    modality="table",
                    signal_type="outlier_ratio",
                    value=outlier_ratio,
                    detail="fraction of z-score > 3",
                ))

        modality_tensors = modality_tensors or {}
        for name, tensor in modality_tensors.items():
            arr = np.asarray(tensor)
            if arr.size == 0:
                continue
            entropy_proxy = float(np.std(arr))
            signals.append(StructuralSignal(
                modality=name,
                signal_type="variance_proxy",
                value=entropy_proxy,
                detail="std-based structural complexity",
            ))

        return signals

    def figure_to_tensor(self, figure_path: str) -> np.ndarray:
        """Convert figure to a simple tensor representation.

        Degrades gracefully when Pillow is unavailable.
        """
        p = Path(figure_path)
        if not p.exists():
            return np.empty((0, 0))
        try:
            from PIL import Image

            img = Image.open(p).convert("L").resize((128, 128))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return arr.reshape(1, -1)
        except Exception:
            # Fallback: filesystem metadata embedding.
            return np.array([[float(p.stat().st_size), float(p.stat().st_mtime % 100000)]], dtype=np.float32)
