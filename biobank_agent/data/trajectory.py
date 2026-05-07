"""HealthFormer-style longitudinal trajectory tokenization.

This v1 does not train a foundation model. It creates auditable participant-level
token sequences that can be used for cross-cohort alignment, evaluation datasets,
or a future HealthFormer-like inference service.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModalityVocabulary:
    """Token range for a single measurement modality."""

    modality: str
    modality_id: int
    value_type: str
    token_offset: int
    token_count: int
    bins: list[float] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    unit: str = ""

    @property
    def token_range(self) -> tuple[int, int]:
        return (self.token_offset, self.token_offset + self.token_count - 1)


@dataclass(frozen=True)
class ParticipantSequence:
    """One participant's ordered physiological trajectory."""

    participant_id: str
    token_ids: list[int]
    modality_ids: list[int]
    time_features: list[list[int]]
    continuous_values: list[float]
    timestamps: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryDataset:
    """Tokenized cohort-level trajectory dataset."""

    vocab: dict[str, ModalityVocabulary]
    sequences: list[ParticipantSequence]
    vocab_size: int
    n_participants: int
    n_tokens: int
    missingness_by_modality: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab": {k: asdict(v) for k, v in self.vocab.items()},
            "sequences": [s.to_dict() for s in self.sequences],
            "vocab_size": self.vocab_size,
            "n_participants": self.n_participants,
            "n_tokens": self.n_tokens,
            "missingness_by_modality": dict(self.missingness_by_modality),
        }


class TrajectoryTokenizer:
    """Quantile / categorical tokenizer for longitudinal multimodal tables."""

    def __init__(self, max_bins: int = 20, min_bins: int = 5) -> None:
        self.max_bins = max_bins
        self.min_bins = min_bins
        self.vocab: dict[str, ModalityVocabulary] = {}

    def fit(
        self,
        df: pd.DataFrame,
        *,
        participant_col: str = "participant_id",
        time_col: str = "timestamp",
        modality_col: str = "modality",
        value_col: str = "value",
        value_type_col: str = "value_type",
        unit_col: str = "unit",
    ) -> "TrajectoryTokenizer":
        required = {participant_col, time_col, modality_col, value_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Trajectory table missing required columns: {sorted(missing)}")

        offset = 0
        vocab = {}
        modalities = sorted(str(m) for m in df[modality_col].dropna().unique())
        for modality_id, modality in enumerate(modalities):
            sub = df[df[modality_col].astype(str) == modality]
            explicit_type = (
                str(sub[value_type_col].dropna().iloc[0]).lower()
                if value_type_col in sub.columns and not sub[value_type_col].dropna().empty
                else ""
            )
            values = sub[value_col].dropna()
            numeric = pd.to_numeric(values, errors="coerce")
            is_continuous = explicit_type == "continuous" or (
                explicit_type != "categorical" and numeric.notna().mean() >= 0.9
            )
            unit = (
                str(sub[unit_col].dropna().iloc[0])
                if unit_col in sub.columns and not sub[unit_col].dropna().empty
                else ""
            )

            if is_continuous:
                clean = numeric.dropna().astype(float)
                if clean.empty:
                    bins = [0.0, 1.0]
                else:
                    unique_n = clean.nunique()
                    n_bins = int(min(self.max_bins, max(self.min_bins, min(unique_n, len(clean) // 10 or self.min_bins))))
                    quantiles = np.linspace(0.0, 1.0, max(2, n_bins + 1))
                    bins = sorted(set(float(x) for x in np.quantile(clean, quantiles)))
                    if len(bins) < 2:
                        center = float(clean.iloc[0])
                        bins = [center - 0.5, center + 0.5]
                token_count = max(1, len(bins) - 1)
                vocab[modality] = ModalityVocabulary(
                    modality=modality,
                    modality_id=modality_id,
                    value_type="continuous",
                    token_offset=offset,
                    token_count=token_count,
                    bins=bins,
                    unit=unit,
                )
            else:
                categories = sorted(str(v) for v in values.astype(str).unique())
                if not categories:
                    categories = ["__missing_category__"]
                vocab[modality] = ModalityVocabulary(
                    modality=modality,
                    modality_id=modality_id,
                    value_type="categorical",
                    token_offset=offset,
                    token_count=len(categories),
                    categories=categories,
                    unit=unit,
                )
            offset += vocab[modality].token_count
        self.vocab = vocab
        return self

    def transform(
        self,
        df: pd.DataFrame,
        *,
        participant_col: str = "participant_id",
        time_col: str = "timestamp",
        modality_col: str = "modality",
        value_col: str = "value",
        sleep_col: str = "sleep",
    ) -> TrajectoryDataset:
        if not self.vocab:
            raise ValueError("TrajectoryTokenizer must be fitted before transform().")

        work = df.copy()
        work = work.dropna(subset=[participant_col, time_col, modality_col, value_col])
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.dropna(subset=[time_col]).sort_values([participant_col, time_col, modality_col])

        sequences = []
        for participant_id, part in work.groupby(participant_col, sort=True):
            token_ids: list[int] = []
            modality_ids: list[int] = []
            time_features: list[list[int]] = []
            continuous_values: list[float] = []
            timestamps: list[str] = []
            for _, row in part.iterrows():
                modality = str(row[modality_col])
                spec = self.vocab.get(modality)
                if spec is None:
                    continue
                value = row[value_col]
                token = self._encode_value(spec, value)
                if token is None:
                    continue
                ts = pd.Timestamp(row[time_col])
                sleep = int(row.get(sleep_col, 0) or 0) if sleep_col in row else 0
                token_ids.append(token)
                modality_ids.append(spec.modality_id)
                time_features.append(_time_features(ts, sleep=sleep))
                continuous_values.append(float(pd.to_numeric(value, errors="coerce")) if spec.value_type == "continuous" else 0.0)
                timestamps.append(ts.isoformat())
            sequences.append(ParticipantSequence(
                participant_id=str(participant_id),
                token_ids=token_ids,
                modality_ids=modality_ids,
                time_features=time_features,
                continuous_values=continuous_values,
                timestamps=timestamps,
            ))

        observed = set(str(m) for m in work[modality_col].dropna().unique())
        participants = max(1, work[participant_col].nunique())
        missingness = {}
        for modality in self.vocab:
            with_mod = work.loc[work[modality_col].astype(str) == modality, participant_col].nunique()
            missingness[modality] = round(1.0 - (with_mod / participants), 6)

        return TrajectoryDataset(
            vocab=self.vocab,
            sequences=sequences,
            vocab_size=sum(v.token_count for v in self.vocab.values()),
            n_participants=len(sequences),
            n_tokens=sum(len(s.token_ids) for s in sequences),
            missingness_by_modality={m: missingness[m] for m in self.vocab if m in observed or m in missingness},
        )

    def fit_transform(self, df: pd.DataFrame, **kwargs: Any) -> TrajectoryDataset:
        fit_keys = {
            "participant_col",
            "time_col",
            "modality_col",
            "value_col",
            "value_type_col",
            "unit_col",
        }
        transform_keys = {"participant_col", "time_col", "modality_col", "value_col", "sleep_col"}
        self.fit(df, **{k: v for k, v in kwargs.items() if k in fit_keys})
        return self.transform(df, **{k: v for k, v in kwargs.items() if k in transform_keys})

    def build_future_query(self, target_modality: str, timestamp: Any, sleep: int = 0) -> dict[str, Any]:
        """Construct a HealthFormer-style modality/time query for a future value."""
        if target_modality not in self.vocab:
            raise ValueError(f"Unknown target modality: {target_modality}")
        ts = pd.Timestamp(timestamp)
        spec = self.vocab[target_modality]
        return {
            "target_modality": target_modality,
            "modality_id": spec.modality_id,
            "time_features": _time_features(ts, sleep=sleep),
            "token_range": spec.token_range,
            "timestamp": ts.isoformat(),
        }

    def _encode_value(self, spec: ModalityVocabulary, value: Any) -> int | None:
        if pd.isna(value):
            return None
        if spec.value_type == "continuous":
            x = float(pd.to_numeric(value, errors="coerce"))
            if np.isnan(x):
                return None
            bins = spec.bins
            idx = int(np.searchsorted(bins, x, side="right") - 1)
            idx = max(0, min(idx, spec.token_count - 1))
            return spec.token_offset + idx
        value_s = str(value)
        try:
            idx = spec.categories.index(value_s)
        except ValueError:
            idx = 0
        return spec.token_offset + idx


def _time_features(ts: pd.Timestamp, sleep: int = 0) -> list[int]:
    return [
        int(ts.dayofweek),
        int(ts.hour),
        int(ts.minute),
        int(ts.month),
        int(ts.year),
        int(ts.day),
        int(sleep),
    ]
