"""Session state shared across all skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd


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

    # ── Analysis History ──────────────────────────────────
    records: list[AnalysisRecord] = field(default_factory=list)

    # ── Feature cache ─────────────────────────────────────
    feature_matrix: Optional[pd.DataFrame] = None
    labels: Optional[pd.Series] = None

    # ── Interruption ──────────────────────────────────────
    interrupted: bool = False

    # ── Extension slots (future multimodal / FM) ──────────
    embeddings: dict[str, Any] = field(default_factory=dict)
    custom_data: dict[str, Any] = field(default_factory=dict)

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
