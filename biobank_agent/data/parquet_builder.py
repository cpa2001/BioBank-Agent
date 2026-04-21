"""Incremental parquet builder — convert CSV fields to parquet on demand.

When the agent needs a field that's only in raw CSV (not in the existing
parquet), this module can extract those columns and write a new parquet
file for faster future access.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def build_field_parquet(
    csv_path: Path,
    field_ids: list[str],
    output_dir: Path,
    output_name: str = "extended_fields.parquet",
    sample_size: Optional[int] = None,
) -> Path:
    """Extract specific fields from a CSV and write to parquet.

    Parameters
    ----------
    csv_path : Path to the source CSV
    field_ids : list of UKB field IDs to extract (e.g. ['22000', '22001'])
    output_dir : directory to write the parquet file
    output_name : output filename
    sample_size : if set, only read this many rows (for testing)

    Returns
    -------
    Path to the written parquet file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name

    # Build column selection for DuckDB
    # First, read the CSV header to find matching columns
    conn = duckdb.connect(":memory:")

    # Get all column names from CSV header
    header_df = conn.execute(
        f"SELECT column_name FROM (DESCRIBE SELECT * FROM "
        f"read_csv_auto('{csv_path}', header=true, sample_size=1))"
    ).df()
    all_cols = header_df["column_name"].tolist()

    # Find columns matching requested field IDs
    selected = ["eid"]
    for col in all_cols:
        if col == "eid":
            continue
        fid = col.split("-")[0].strip('"')
        if fid in field_ids:
            selected.append(col)

    if len(selected) <= 1:
        raise ValueError(
            f"No columns found for field IDs {field_ids} in {csv_path}. "
            f"Available field prefixes: {sorted(set(c.split('-')[0] for c in all_cols[:20]))}"
        )

    logger.info("Extracting %d columns for %d fields from %s",
                len(selected), len(field_ids), csv_path.name)

    # Read and write via DuckDB (memory-efficient streaming)
    col_str = ", ".join(f'"{c}"' for c in selected)
    limit_clause = f"LIMIT {sample_size}" if sample_size else ""

    conn.execute(
        f"COPY (SELECT {col_str} FROM read_csv_auto('{csv_path}', header=true) "
        f"{limit_clause}) TO '{out_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)"
    )

    # Verify
    meta = conn.execute(f"SELECT COUNT(*) AS n FROM read_parquet('{out_path}')").fetchone()
    logger.info("Wrote %s: %d rows, %d columns", out_path, meta[0], len(selected))

    conn.close()
    return out_path


def register_extended_parquet(
    dm,
    parquet_path: Path,
    view_name: str = "extended",
) -> None:
    """Register an extended parquet file as a DuckDB view in the DataManager."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    dm.conn.execute(
        f"CREATE VIEW IF NOT EXISTS {view_name} AS "
        f"SELECT * FROM read_parquet('{parquet_path}')"
    )
    logger.info("Registered extended view '%s' from %s", view_name, parquet_path)
