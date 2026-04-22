"""Cohort builder — construct case/control datasets from diagnosis codes.

Uses DuckDB SQL on the diagnoses view for efficient filtering with
parameterized queries to prevent SQL injection.

Biobank-agnostic: column names (subject ID, diagnosis code, death code)
are driven by DataManager.settings. Default targets UK Biobank.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..state import SessionState
from .loader import DataManager

logger = logging.getLogger(__name__)

# UKB-specific wide-field IDs to exclude from default feature selection.
# These are self-reported ICD10 arrays that inflate the feature space.
_DEFAULT_EXCLUDED_WIDE_FIELDS = {"41270", "41271", "41280", "41281"}


def build_cohort(
    dm: DataManager,
    icd_code: str,
    controls_ratio: int = 4,
    biomarker_fields: Optional[list[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a labelled cohort DataFrame with biomarker features.

    Parameters
    ----------
    dm : DataManager
    icd_code : str — diagnosis code prefix (e.g. "E11" for Type 2 Diabetes)
    controls_ratio : int — controls per case
    biomarker_fields : optional list of field IDs; None = all biomarkers
    random_state : int

    Returns
    -------
    pd.DataFrame with columns: <subject_id>, label (1=case, 0=control), + biomarker cols
    """
    conn = dm.conn
    id_col = dm.settings.subject_id_col
    diag_col = dm.settings.diagnoses_code_col
    death_col = dm.settings.deaths_code_col

    # 1. Identify cases from diagnoses (parameterized query)
    cases_df = conn.execute(f"""
        SELECT DISTINCT {id_col} FROM diagnoses
        WHERE {diag_col} LIKE ?
    """, [f"{icd_code}%"]).df()
    n_cases = len(cases_df)
    logger.info("Code %s: %d cases from diagnoses", icd_code, n_cases)

    if n_cases == 0:
        raise ValueError(f"No cases found for diagnosis code '{icd_code}'")

    # 2. Also check death causes (parameterized query)
    try:
        death_cases = conn.execute(f"""
            SELECT DISTINCT {id_col} FROM deaths
            WHERE {death_col} LIKE ?
        """, [f"{icd_code}%"]).df()
        all_case_ids = set(cases_df[id_col].tolist()) | set(death_cases[id_col].tolist())
    except Exception:
        all_case_ids = set(cases_df[id_col].tolist())

    n_cases = len(all_case_ids)
    logger.info("Total cases (diagnoses + deaths): %d", n_cases)

    # 3. Sample controls (not in cases)
    try:
        total_subjects = dm.count_subjects()
    except Exception:
        total_subjects = 500_000  # fallback estimate
    n_controls = min(n_cases * controls_ratio, total_subjects - n_cases)

    conn.execute(
        f"CREATE OR REPLACE TEMP TABLE case_ids AS SELECT UNNEST(?) AS {id_col}",
        [list(all_case_ids)],
    )

    controls_df = conn.execute(f"""
        SELECT {id_col} FROM (
            SELECT DISTINCT {id_col} FROM biomarkers
            WHERE {id_col} NOT IN (SELECT {id_col} FROM case_ids)
        ) sub
        USING SAMPLE {n_controls} (reservoir, {random_state})
    """).df()
    logger.info("Sampled %d controls", len(controls_df))

    # 4. Build feature matrix
    all_ids = list(all_case_ids) + controls_df[id_col].tolist()

    conn.execute(
        f"CREATE OR REPLACE TEMP TABLE cohort_ids AS SELECT UNNEST(?) AS {id_col}",
        [all_ids],
    )

    if biomarker_fields:
        cols = ", ".join(f'"{fid}-0.0"' for fid in biomarker_fields)
    else:
        # Select all numeric-looking columns (field-instance.array pattern)
        all_cols = dm.list_parquet_columns()
        excluded = _DEFAULT_EXCLUDED_WIDE_FIELDS
        numeric_cols = [
            c for c in all_cols
            if c != id_col
            and c.split("-")[0].isdigit()
            and not any(c.startswith(ex) for ex in excluded)
            and "-0.0" in c
        ]
        cols = ", ".join(f'"{c}"' for c in numeric_cols)

    df = conn.execute(f"""
        SELECT b.{id_col}, {cols}
        FROM biomarkers b
        INNER JOIN cohort_ids c ON b.{id_col} = c.{id_col}
    """).df()

    # 5. Add labels
    df["label"] = df[id_col].isin(all_case_ids).astype(int)
    logger.info(
        "Cohort built: %d subjects (%d cases, %d controls), %d features",
        len(df), df["label"].sum(), (df["label"] == 0).sum(), df.shape[1] - 2,
    )

    # Clean up
    conn.execute("DROP TABLE IF EXISTS case_ids")
    conn.execute("DROP TABLE IF EXISTS cohort_ids")

    return df
