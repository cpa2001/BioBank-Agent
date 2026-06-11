"""Session state shared across all skills."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass
class TokenUsage:
    """Cumulative token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @staticmethod
    def _coerce_count(value: Any) -> int:
        """Convert relay-provided token count to a safe non-negative int."""
        if value is None:
            return 0
        try:
            count = int(value)
        except (TypeError, ValueError):
            return 0
        return max(0, count)

    def update(self, usage: dict) -> None:
        self.prompt_tokens += self._coerce_count(usage.get("prompt_tokens", 0))
        self.completion_tokens += self._coerce_count(usage.get("completion_tokens", 0))


@dataclass
class Provenance:
    """Reproducibility record for a single analysis step.

    Links each result to its inputs, parameters, and parent results,
    enabling full provenance chains for scientific reproducibility.
    """
    provenance_id: str          # short hash of skill+args+timestamp
    skill: str
    args: dict
    timestamp: str
    bank_id: str                # which biobank config was active
    result_hash: str            # hash of key_results for integrity
    parent_ids: list[str] = field(default_factory=list)

    @staticmethod
    def compute_hash(result: dict) -> str:
        """Deterministic hash of a result dict for integrity verification."""
        serialized = json.dumps(result, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:12]

    @staticmethod
    def make_id(skill: str, args: dict, timestamp: str) -> str:
        """Generate a short provenance ID."""
        key = f"{skill}:{json.dumps(args, sort_keys=True, default=str)}:{timestamp}"
        return hashlib.md5(key.encode()).hexdigest()[:8]


@dataclass
class AnalysisRecord:
    """Immutable record of one skill execution — exact numbers, never summarised."""
    timestamp: str
    skill: str
    args: dict
    key_results: dict          # {"auc": 0.93, "n_cases": 4821, ...}
    figure_paths: list[str]
    interpretation: str = ""   # LLM-generated, lossy is OK


@dataclass
class SessionState:
    """Mutable bag of state passed to every skill via ``ctx``."""

    # ── DuckDB ────────────────────────────────────────────
    duckdb_conn: Any = None        # duckdb.DuckDBPyConnection

    # ── Cohorts & Models ──────────────────────────────────
    cohorts: dict[str, pd.DataFrame] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    model_metadata: dict[str, dict] = field(default_factory=dict)

    # ── Figures & Reports ─────────────────────────────────
    figures: list[Path] = field(default_factory=list)
    report_sections: list[str] = field(default_factory=list)
    current_report_dir: Optional[Path] = None
    executive_findings: list[dict[str, Any]] = field(default_factory=list)

    # ── Analysis History ──────────────────────────────────
    records: list[AnalysisRecord] = field(default_factory=list)
    execution_log: list[dict[str, Any]] = field(default_factory=list)

    # ── Feature cache ─────────────────────────────────────
    feature_matrix: Optional[pd.DataFrame] = None
    labels: Optional[pd.Series] = None

    # ── Interruption ──────────────────────────────────────
    interrupted: bool = False

    # ── Extension slots (future multimodal / FM) ──────────
    embeddings: dict[str, Any] = field(default_factory=dict)
    custom_data: dict[str, Any] = field(default_factory=dict)

    # ── Token tracking ────────────────────────────────────
    token_usage: TokenUsage = field(default_factory=TokenUsage)

    # ── Provenance chain ─────────────────────────────────
    provenances: list[Provenance] = field(default_factory=list)

    # ── Orchestration diagnostics ───────────────────────
    last_orchestration: dict[str, Any] = field(default_factory=dict)

    def add_record(self, record: AnalysisRecord) -> None:
        self.records.append(record)

    def context_summary(self) -> str:
        """Inject into system prompt: structured history with exact numbers."""
        if not self.records:
            return "No analyses performed yet."
        lines = []
        for r in self.records[-20:]:
            args_str = ", ".join(f"{k}={v!r}" for k, v in r.args.items())
            results_str = ", ".join(f"{k}={v}" for k, v in r.key_results.items())
            lines.append(f"  - {r.skill}({args_str}) → {results_str}")
        header = f"Session history ({len(self.records)} steps, showing last {min(20, len(self.records))}):"
        cohort_info = ""
        if self.cohorts:
            parts = []
            for name, df in self.cohorts.items():
                n_cases = int(df["label"].sum()) if "label" in df.columns else "?"
                parts.append(f"{name}: {len(df)} subjects ({n_cases} cases)")
            cohort_info = "\nActive cohorts: " + "; ".join(parts)
        model_info = ""
        if self.models:
            model_info = "\nTrained models: " + ", ".join(self.models.keys())
        return header + "\n" + "\n".join(lines) + cohort_info + model_info
