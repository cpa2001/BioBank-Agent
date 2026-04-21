"""Cohort builder — construct case/control datasets from ICD10 codes.

Uses DuckDB SQL on the diagnoses view for efficient filtering.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..state import SessionState
from .loader import DataManager

logger = logging.getLogger(__name__)


def build_cohort(
    dm: DataManager,
    icd10_code: str,
    controls_ratio: int = 4,
    biomarker_fields: Optional[list[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a labelled cohort DataFrame with biomarker features.

    Parameters
    ----------
    dm : DataManager
    icd10_code : str — ICD10 code prefix (e.g. "E11" for Type 2 Diabetes)
    controls_ratio : int — controls per case
    biomarker_fields : optional list of field IDs; None = all biomarkers
    random_state : int

    Returns
    -------
    pd.DataFrame with columns: eid, label (1=case, 0=control), + biomarker cols
    """
    conn = dm.conn

    # 1. Identify cases
    cases_df = conn.execute(f"""
        SELECT DISTINCT eid FROM diagnoses
        WHERE diag_icd10 LIKE '{icd10_code}%'
    """).df()
    n_cases = len(cases_df)
    logger.info("ICD10 %s: %d cases found", icd10_code, n_cases)

    if n_cases == 0:
        raise ValueError(f"No cases found for ICD10 code '{icd10_code}'")

    # 2. Also check death causes
    death_cases = conn.execute(f"""
        SELECT DISTINCT eid FROM deaths
        WHERE cause_icd10 LIKE '{icd10_code}%'
    """).df()
    all_case_eids = set(cases_df["eid"].tolist()) | set(death_cases["eid"].tolist())
    n_cases = len(all_case_eids)
    logger.info("Total cases (diagnoses + deaths): %d", n_cases)

    # 3. Sample controls (not in cases)
    n_controls = min(n_cases * controls_ratio, 502_370 - n_cases)

    # For large case sets, use a temp table
    conn.execute("CREATE OR REPLACE TEMP TABLE case_eids AS SELECT UNNEST(?) AS eid",
                 [list(all_case_eids)])

    controls_df = conn.execute(f"""
        SELECT eid FROM (
            SELECT DISTINCT eid FROM biomarkers
            WHERE eid NOT IN (SELECT eid FROM case_eids)
        ) sub
        USING SAMPLE {n_controls} (reservoir, {random_state})
    """).df()
    logger.info("Sampled %d controls", len(controls_df))

    # 4. Build feature matrix
    all_eids = list(all_case_eids) + controls_df["eid"].tolist()

    # Create temp table of all eids for efficient join
    conn.execute("CREATE OR REPLACE TEMP TABLE cohort_eids AS SELECT UNNEST(?) AS eid",
                 [all_eids])

    if biomarker_fields:
        cols = ", ".join(f'"{fid}-0.0"' for fid in biomarker_fields)
    else:
        # Select all numeric-looking columns (field-instance.array pattern)
        all_cols = dm.list_parquet_columns()
        numeric_cols = [c for c in all_cols
                        if c != "eid" and c.split("-")[0].isdigit()
                        and not c.startswith("41270")  # skip wide ICD10
                        and not c.startswith("41271")
                        and not c.startswith("41280")
                        and not c.startswith("41281")
                        and "-0.0" in c]  # only instance 0
        cols = ", ".join(f'"{c}"' for c in numeric_cols)

    df = conn.execute(f"""
        SELECT b.eid, {cols}
        FROM biomarkers b
        INNER JOIN cohort_eids c ON b.eid = c.eid
    """).df()

    # 5. Add labels
    df["label"] = df["eid"].isin(all_case_eids).astype(int)
    logger.info("Cohort built: %d subjects (%d cases, %d controls), %d features",
                len(df), df["label"].sum(), (df["label"] == 0).sum(), df.shape[1] - 2)

    # Clean up temp tables
    conn.execute("DROP TABLE IF EXISTS case_eids")
    conn.execute("DROP TABLE IF EXISTS cohort_eids")

    return df
