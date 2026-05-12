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
from .loader import DataManager, _quote_ident

logger = logging.getLogger(__name__)

# UKB-specific wide-field IDs to exclude from default feature selection.
# These are self-reported ICD10 arrays that inflate the feature space.
_DEFAULT_EXCLUDED_WIDE_FIELDS = {"41270", "41271", "41280", "41281"}


def _default_feature_columns(dm: DataManager, id_col: str) -> list[str]:
    """Select full default feature columns for the active bank.

    UKB's very wide table benefits from the historical ``field-0.0`` filter,
    but HPP/CKB-style fixtures may expose native numeric columns like
    ``hba1c`` or ``bmi``. In that case use every numeric analytical column.
    """
    all_cols = dm.list_parquet_columns()
    excluded = _DEFAULT_EXCLUDED_WIDE_FIELDS
    ukb_instance_cols = [
        c for c in all_cols
        if c != id_col
        and c.split("-")[0].isdigit()
        and not any(c.startswith(ex) for ex in excluded)
        and "-0.0" in c
    ]
    if ukb_instance_cols:
        return ukb_instance_cols
    return [
        c for c in dm.list_numeric_biomarker_columns()
        if c != id_col and not any(c.startswith(ex) for ex in excluded)
    ]


def build_cohort(
    dm: DataManager,
    icd_code: str,
    controls_ratio: int = 0,
    biomarker_fields: Optional[list[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a labelled cohort DataFrame with biomarker features.

    Parameters
    ----------
    dm : DataManager
    icd_code : str — diagnosis code prefix (e.g. "E11" for Type 2 Diabetes)
    controls_ratio : int — controls per case; <=0 uses all eligible controls
    biomarker_fields : optional list of field IDs; None = all biomarkers
    random_state : int

    Returns
    -------
    pd.DataFrame with columns: <subject_id>, label (1=case, 0=control), + biomarker cols
    """
    conn = dm.conn
    id_col = getattr(dm, "subject_id_col", dm.settings.subject_id_col)
    diag_col = getattr(dm, "diagnoses_code_col", dm.settings.diagnoses_code_col)
    death_col = getattr(dm, "deaths_code_col", dm.settings.deaths_code_col)
    if hasattr(dm, "normalize_icd_code"):
        icd_code = dm.normalize_icd_code(icd_code)

    # 1. Identify cases from diagnoses (parameterized query)
    if hasattr(dm, "code_prefix_filter"):
        case_where, case_params = dm.code_prefix_filter("diagnoses", icd_code, diag_col)
    else:
        case_where, case_params = f"{_quote_ident(diag_col)} LIKE ?", [f"{icd_code}%"]
    if not case_where:
        raise ValueError(f"No diagnosis code column found for active bank while searching '{icd_code}'")
    cases_df = conn.execute(f"""
        SELECT DISTINCT {_quote_ident(id_col)} FROM diagnoses
        WHERE {case_where}
    """, case_params).df()
    n_cases = len(cases_df)
    logger.info("Code %s: %d cases from diagnoses", icd_code, n_cases)

    if n_cases == 0:
        raise ValueError(f"No cases found for diagnosis code '{icd_code}'")

    # 2. Also check death causes (parameterized query)
    try:
        if hasattr(dm, "code_prefix_filter"):
            death_where, death_params = dm.code_prefix_filter("deaths", icd_code, death_col)
        else:
            death_where, death_params = f"{_quote_ident(death_col)} LIKE ?", [f"{icd_code}%"]
        if not death_where:
            raise ValueError("No death code column for active bank")
        death_cases = conn.execute(f"""
            SELECT DISTINCT {_quote_ident(id_col)} FROM deaths
            WHERE {death_where}
        """, death_params).df()
        all_case_ids = set(cases_df[id_col].tolist()) | set(death_cases[id_col].tolist())
    except Exception:
        all_case_ids = set(cases_df[id_col].tolist())

    n_cases = len(all_case_ids)
    logger.info("Total cases (diagnoses + deaths): %d", n_cases)

    # 3. Select controls (not in cases). The local data is already
    # de-identified, so default to all eligible controls; sampling is an
    # explicit smoke-test/performance option.
    try:
        total_subjects = dm.count_subjects()
    except Exception:
        total_subjects = 500_000  # fallback estimate
    total_controls = max(0, total_subjects - n_cases)
    if controls_ratio and controls_ratio > 0:
        n_controls = min(n_cases * controls_ratio, total_controls)
    else:
        n_controls = total_controls

    conn.execute(
        f"CREATE OR REPLACE TEMP TABLE case_ids AS SELECT UNNEST(?) AS {_quote_ident(id_col)}",
        [list(all_case_ids)],
    )

    control_base_sql = f"""
        SELECT DISTINCT {_quote_ident(id_col)} FROM biomarkers
        WHERE {_quote_ident(id_col)} NOT IN (SELECT {_quote_ident(id_col)} FROM case_ids)
    """
    if controls_ratio and controls_ratio > 0:
        controls_df = conn.execute(f"""
            SELECT {_quote_ident(id_col)} FROM ({control_base_sql}) sub
            USING SAMPLE {n_controls} (reservoir, {random_state})
        """).df()
        logger.info("Sampled %d controls", len(controls_df))
    else:
        controls_df = conn.execute(control_base_sql).df()
        logger.info("Using all %d eligible controls", len(controls_df))

    # 4. Build feature matrix
    all_ids = list(all_case_ids) + controls_df[id_col].tolist()

    conn.execute(
        f"CREATE OR REPLACE TEMP TABLE cohort_ids AS SELECT UNNEST(?) AS {_quote_ident(id_col)}",
        [all_ids],
    )

    if biomarker_fields:
        feature_cols = []
        for fid in biomarker_fields:
            if hasattr(dm, "field_column"):
                physical = dm.field_column(fid, source="biomarkers")
            else:
                resolved = dm.resolve_field_id(fid) if hasattr(dm, "resolve_field_id") else str(fid)
                physical = f"{resolved}-0.0"
            if physical not in feature_cols:
                feature_cols.append(physical)
    else:
        feature_cols = _default_feature_columns(dm, id_col)

    if not feature_cols:
        raise ValueError(
            "No numeric biomarker feature columns found for the active bank. "
            "Provide biomarker_fields or check the biomarker parquet schema."
        )
    cols = ", ".join(f"b.{_quote_ident(c)}" for c in feature_cols)

    df = conn.execute(f"""
        SELECT b.{_quote_ident(id_col)}, {cols}
        FROM biomarkers b
        INNER JOIN cohort_ids c ON b.{_quote_ident(id_col)} = c.{_quote_ident(id_col)}
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
