"""Incremental parquet builder — convert CSV fields to parquet on demand.

When the agent needs a field that's only in raw CSV (not in the existing
parquet), this module can extract those columns and write a new parquet
file for faster future access.

Also provides batch rebuild: convert ALL category CSVs to parquet partitions
for comprehensive field coverage.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)


def _escape_path(path: Path) -> str:
    """Escape a file path for safe use in DuckDB SQL strings.

    Prevents SQL injection via crafted path names containing single quotes.
    """
    return str(path).replace("'", "''")


def _quote_identifier(name: str) -> str:
    """Quote a DuckDB identifier such as a view name."""
    return '"' + str(name).replace('"', '""') + '"'


# Map category name → CSV filename
CATEGORY_CSVS = {
    "Population_Characteristics": "ukb672073_Population_Characteristics.csv",
    "Biological_Samples": "ukb672073_Biological_Samples.csv",
    "Additional_Exposures": "ukb672073_Additional_Exposures.csv",
    "Genomics": "ukb672073_Genomics.csv",
    "Health_Related_Outcomes": "ukb672073_Health_Related_Outcomes.csv",
    "Online_Follow_up": "ukb672073_Online_Follow_up.csv",
}

FULL_UKB_SOURCES = {
    "main_672073": "UKB/ukb672073.csv",
    "main_671626": "UKB/ukb671626.csv",
    "category_population_characteristics": "UKB_info/ukb672073_Population_Characteristics.csv",
    "category_biological_samples": "UKB_info/ukb672073_Biological_Samples.csv",
    "category_additional_exposures": "UKB_info/ukb672073_Additional_Exposures.csv",
    "category_genomics": "UKB_info/ukb672073_Genomics.csv",
    "category_health_related_outcomes": "UKB_info/ukb672073_Health_Related_Outcomes.csv",
    "category_online_follow_up": "UKB_info/ukb672073_Online_Follow_up.csv",
}


def _get_csv_columns(conn: duckdb.DuckDBPyConnection, csv_path: Path) -> list[str]:
    """Read column names from a CSV header without loading data."""
    safe_path = _escape_path(csv_path)
    header_df = conn.execute(
        f"SELECT column_name FROM (DESCRIBE SELECT * FROM "
        f"read_csv_auto('{safe_path}', header=true, sample_size=1))"
    ).df()
    return header_df["column_name"].tolist()


def _get_csv_header(csv_path: Path) -> list[str]:
    """Read only the CSV header with Python's csv module.

    DuckDB's DESCRIBE path is convenient but can still inspect enough content
    to be slow on 50GB UKB main CSVs. Header parsing is deterministic and
    avoids touching participant rows during dry-runs and planning.
    """
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as handle:
        return next(csv.reader(handle))


def _extract_field_ids(columns: list[str]) -> set[str]:
    """Extract unique field ID prefixes from UKB column names."""
    field_ids = set()
    for col in columns:
        if col == "eid":
            continue
        fid = col.split("-")[0].strip('"')
        if fid.isdigit():
            field_ids.add(fid)
    return field_ids


def _field_id_for_column(column: str) -> str:
    return str(column).split("-")[0].strip('"')


def _select_field_columns(columns: list[str], field_ids: set[str] | None = None) -> list[str]:
    selected = ["eid"] if "eid" in columns else []
    for col in columns:
        if col == "eid":
            continue
        fid = _field_id_for_column(col)
        if not fid.isdigit():
            continue
        if field_ids is not None and fid not in field_ids:
            continue
        selected.append(col)
    return selected


def _chunk_columns(selected: list[str], chunk_cols: int) -> list[list[str]]:
    id_col = "eid" if "eid" in selected else selected[0]
    payload = [c for c in selected if c != id_col]
    size = max(1, int(chunk_cols))
    return [[id_col, *payload[i:i + size]] for i in range(0, len(payload), size)]


def _write_manifest(output_dir: Path, manifest: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _source_path(raw_dir: Path, source_key: str) -> Path:
    try:
        rel = FULL_UKB_SOURCES[source_key]
    except KeyError as exc:
        raise ValueError(f"Unknown UKB source: {source_key}") from exc
    return raw_dir / rel


def full_ukb_inventory(
    raw_dir: Path,
    *,
    sources: Optional[list[str]] = None,
    field_ids: Optional[list[str]] = None,
    chunk_cols: int = 500,
) -> dict:
    """Dry-run inventory for a full UKB feature-store build.

    This only reads CSV headers and file metadata. It is safe to run before a
    large conversion and is also used by the agent to understand which fields
    are available outside the existing Milton parquet subset.
    """
    source_keys = sources or list(FULL_UKB_SOURCES)
    wanted = {str(fid).strip() for fid in field_ids or [] if str(fid).strip()}
    field_filter = wanted or None
    rows: list[dict] = []
    all_fields: set[str] = set()
    total_cols = 0
    total_selected = 0
    for source_key in source_keys:
        csv_path = _source_path(raw_dir, source_key)
        if not csv_path.exists():
            rows.append({
                "source_key": source_key,
                "csv_path": str(csv_path),
                "status": "MISSING",
                "n_cols": 0,
                "n_fields": 0,
                "n_selected_cols": 0,
                "n_chunks": 0,
            })
            continue
        columns = _get_csv_header(csv_path)
        fields = _extract_field_ids(columns)
        selected = _select_field_columns(columns, field_filter)
        selected_fields = _extract_field_ids(selected)
        all_fields.update(fields)
        total_cols += len(columns)
        total_selected += max(0, len(selected) - (1 if "eid" in selected else 0))
        rows.append({
            "source_key": source_key,
            "csv_path": str(csv_path),
            "status": "READY",
            "size_bytes": csv_path.stat().st_size,
            "n_cols": len(columns),
            "n_fields": len(fields),
            "n_selected_cols": len(selected),
            "n_selected_fields": len(selected_fields),
            "n_chunks": len(_chunk_columns(selected, chunk_cols)) if len(selected) > 1 else 0,
        })
    return {
        "status": "READY" if any(r["status"] == "READY" for r in rows) else "MISSING",
        "raw_dir": str(raw_dir),
        "sources": rows,
        "n_sources": len(rows),
        "n_ready_sources": sum(1 for r in rows if r["status"] == "READY"),
        "n_total_columns": total_cols,
        "n_total_fields": len(all_fields),
        "n_selected_feature_columns": total_selected,
        "chunk_cols": int(chunk_cols),
        "field_filter": sorted(wanted),
    }


def build_full_ukb_feature_store(
    raw_dir: Path,
    output_dir: Path,
    *,
    sources: Optional[list[str]] = None,
    field_ids: Optional[list[str]] = None,
    chunk_cols: int = 500,
    resume: bool = True,
    sample_size: Optional[int] = None,
    dry_run: bool = False,
    callback=None,
) -> dict:
    """Build a partitioned parquet feature store from the full UKB CSV export.

    The output is deliberately split by source file and column chunk instead of
    one huge wide table. Skills can materialize only the columns they need,
    while the manifest still proves full-field availability.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    inventory = full_ukb_inventory(
        raw_dir,
        sources=sources,
        field_ids=field_ids,
        chunk_cols=chunk_cols,
    )
    if dry_run:
        return {**inventory, "dry_run": True, "output_dir": str(output_dir), "manifest": None}

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_filter = {str(fid).strip() for fid in field_ids or [] if str(fid).strip()} or None
    manifest = {
        "schema_version": 1,
        "kind": "ukb_full_feature_store",
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "chunk_cols": int(chunk_cols),
        "sample_size": int(sample_size) if sample_size else 0,
        "field_filter": sorted(feature_filter or []),
        "sources": {},
    }

    for source in inventory["sources"]:
        source_key = source["source_key"]
        if source["status"] != "READY":
            manifest["sources"][source_key] = source
            continue
        csv_path = Path(source["csv_path"])
        columns = _get_csv_header(csv_path)
        selected = _select_field_columns(columns, feature_filter)
        chunks = _chunk_columns(selected, chunk_cols) if len(selected) > 1 else []
        source_dir = output_dir / "sources" / source_key
        source_dir.mkdir(parents=True, exist_ok=True)
        source_payload = {
            **source,
            "chunks": [],
        }
        manifest["sources"][source_key] = source_payload
        safe_csv = _escape_path(csv_path)
        conn = duckdb.connect(":memory:")
        for idx, chunk in enumerate(chunks):
            out_path = source_dir / f"chunk-{idx:04d}.parquet"
            if resume and out_path.exists() and out_path.stat().st_size > 0:
                status = "SKIPPED_EXISTS"
                n_rows = conn.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{_escape_path(out_path)}')"
                ).fetchone()[0]
            else:
                if callback:
                    callback(source_key, f"writing chunk {idx + 1}/{len(chunks)} ({len(chunk)} columns)")
                col_str = ", ".join(f'"{c}"' for c in chunk)
                limit_clause = f"LIMIT {int(sample_size)}" if sample_size else ""
                tmp_path = out_path.with_name(out_path.name + ".tmp")
                if tmp_path.exists():
                    tmp_path.unlink()
                conn.execute(
                    f"COPY (SELECT {col_str} FROM read_csv_auto('{safe_csv}', header=true, "
                    f"all_varchar=true) {limit_clause}) "
                    f"TO '{_escape_path(tmp_path)}' (FORMAT PARQUET, COMPRESSION SNAPPY)"
                )
                tmp_path.replace(out_path)
                n_rows = conn.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{_escape_path(out_path)}')"
                ).fetchone()[0]
                status = "WRITTEN"
            chunk_fields = sorted(_extract_field_ids(chunk))
            source_payload["chunks"].append({
                "chunk_index": idx,
                "path": str(out_path),
                "status": status,
                "n_rows": int(n_rows),
                "n_cols": len(chunk),
                "n_fields": len(chunk_fields),
                "fields": chunk_fields,
                "columns": chunk,
            })
            source_payload["status"] = "IN_PROGRESS"
            source_payload["n_chunks_written"] = len(source_payload["chunks"])
            _write_manifest(output_dir, manifest)
        conn.close()
        source_payload["status"] = "READY" if chunks else "NO_SELECTED_COLUMNS"
        source_payload["n_chunks_written"] = len(source_payload["chunks"])
        _write_manifest(output_dir, manifest)

    total_chunks = sum(len(src.get("chunks", [])) for src in manifest["sources"].values())
    manifest["status"] = "READY" if total_chunks else "EMPTY"
    manifest["n_chunks"] = total_chunks
    manifest["n_selected_feature_columns"] = sum(
        max(0, int(src.get("n_selected_cols", 0) or 0) - 1)
        for src in manifest["sources"].values()
    )
    manifest["manifest_path"] = str(_write_manifest(output_dir, manifest))
    return manifest


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

    conn = duckdb.connect(":memory:")
    all_cols = _get_csv_columns(conn, csv_path)

    # Find columns matching requested field IDs
    field_set = set(field_ids)
    selected = ["eid"]
    for col in all_cols:
        if col == "eid":
            continue
        fid = col.split("-")[0].strip('"')
        if fid in field_set:
            selected.append(col)

    if len(selected) <= 1:
        raise ValueError(
            f"No columns found for field IDs {field_ids} in {csv_path}. "
            f"Available field prefixes: {sorted(set(c.split('-')[0] for c in all_cols[:20]))}"
        )

    logger.info("Extracting %d columns for %d fields from %s",
                len(selected), len(field_ids), csv_path.name)

    col_str = ", ".join(f'"{c}"' for c in selected)
    limit_clause = f"LIMIT {sample_size}" if sample_size else ""
    safe_csv = _escape_path(csv_path)
    safe_out = _escape_path(out_path)

    conn.execute(
        f"COPY (SELECT {col_str} FROM read_csv_auto('{safe_csv}', header=true) "
        f"{limit_clause}) TO '{safe_out}' (FORMAT PARQUET, COMPRESSION SNAPPY)"
    )

    meta = conn.execute(f"SELECT COUNT(*) AS n FROM read_parquet('{safe_out}')").fetchone()
    logger.info("Wrote %s: %d rows, %d columns", out_path, meta[0], len(selected))

    conn.close()
    return out_path


def build_category_parquet(
    csv_path: Path,
    output_path: Path,
    exclude_fields: Optional[set[str]] = None,
) -> dict:
    """Convert a single category CSV to parquet, optionally excluding fields already in parquet.

    Parameters
    ----------
    csv_path : Path to the category CSV
    output_path : Path to write the output parquet file
    exclude_fields : set of field ID prefixes to skip (already in existing parquet)

    Returns
    -------
    dict with keys: path, n_rows, n_cols, n_fields, new_fields
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(":memory:")

    all_cols = _get_csv_columns(conn, csv_path)
    all_field_ids = _extract_field_ids(all_cols)

    # Filter out fields already in parquet
    if exclude_fields:
        new_field_ids = all_field_ids - exclude_fields
    else:
        new_field_ids = all_field_ids

    if not new_field_ids:
        conn.close()
        return {"path": None, "n_rows": 0, "n_cols": 0, "n_fields": 0, "new_fields": []}

    # Select only columns for new fields + eid
    selected = ["eid"]
    for col in all_cols:
        if col == "eid":
            continue
        fid = col.split("-")[0].strip('"')
        if fid in new_field_ids:
            selected.append(col)

    logger.info("Category %s: %d new fields (%d columns) to extract",
                csv_path.stem, len(new_field_ids), len(selected) - 1)

    col_str = ", ".join(f'"{c}"' for c in selected)
    t0 = time.time()

    # Use all_varchar=true to avoid type detection errors on UKB's mixed-type columns
    # (e.g. columns like 41209-0.0 contain both integers and ICD10 codes like "A41")
    # DuckDB can auto-cast VARCHAR to numeric types at query time.
    safe_csv = _escape_path(csv_path)
    safe_out = _escape_path(output_path)
    conn.execute(
        f"COPY (SELECT {col_str} FROM read_csv_auto('{safe_csv}', header=true, "
        f"all_varchar=true)) TO '{safe_out}' (FORMAT PARQUET, COMPRESSION SNAPPY)"
    )

    meta = conn.execute(f"SELECT COUNT(*) AS n FROM read_parquet('{safe_out}')").fetchone()
    elapsed = time.time() - t0
    n_rows = meta[0]

    logger.info("Wrote %s: %d rows, %d columns in %.1fs",
                output_path.name, n_rows, len(selected), elapsed)

    conn.close()
    return {
        "path": str(output_path),
        "n_rows": n_rows,
        "n_cols": len(selected),
        "n_fields": len(new_field_ids),
        "new_fields": sorted(new_field_ids),
        "elapsed_s": round(elapsed, 1),
    }


def batch_rebuild(
    raw_csv_dir: Path,
    output_dir: Path,
    existing_parquet_dir: Optional[Path] = None,
    categories: Optional[list[str]] = None,
    callback=None,
) -> dict:
    """Rebuild parquet from ALL category CSVs in one batch.

    Writes each category as a separate parquet file in output_dir/
    (NOT into the row-partitioned ukb.parquet/ directory, to avoid row duplication).

    Parameters
    ----------
    raw_csv_dir : Path to UKB_info directory containing category CSVs
    output_dir : Path to write category parquet files (e.g. milton_data/categories/)
    existing_parquet_dir : Path to existing ukb.parquet/ to detect already-covered fields
    categories : Optional list of category names to process (default: all)
    callback : Optional callable(category, status_msg) for progress reporting

    Returns
    -------
    dict with overall stats and per-category results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover existing field IDs from current parquet
    exclude_fields: set[str] = set()
    if existing_parquet_dir and existing_parquet_dir.exists():
        try:
            conn = duckdb.connect(":memory:")
            safe_pattern = _escape_path(existing_parquet_dir) + "/*.parquet"
            cols = conn.execute(
                f"SELECT column_name FROM (DESCRIBE SELECT * FROM "
                f"read_parquet('{safe_pattern}', union_by_name=true))"
            ).df()
            for col_name in cols["column_name"].tolist():
                if col_name != "eid":
                    fid = col_name.split("-")[0].strip('"')
                    if fid.isdigit():
                        exclude_fields.add(fid)
            conn.close()
            logger.info("Found %d existing field IDs in parquet", len(exclude_fields))
        except Exception as e:
            # Fail fast: if we can't read existing parquet, dedup is impossible
            raise RuntimeError(
                f"Cannot read existing parquet at {existing_parquet_dir} for field deduplication: {e}"
            ) from e

    # Determine which categories to process
    cats = categories or list(CATEGORY_CSVS.keys())

    results = {}
    total_new_fields = 0
    total_new_cols = 0

    for i, cat in enumerate(cats):
        csv_name = CATEGORY_CSVS.get(cat)
        if not csv_name:
            logger.warning("Unknown category: %s", cat)
            continue

        csv_path = raw_csv_dir / csv_name
        if not csv_path.exists():
            logger.warning("CSV not found: %s", csv_path)
            continue

        if callback:
            callback(cat, f"Processing {cat} ({i+1}/{len(cats)})...")

        # Write each category as a separate parquet file
        output_path = output_dir / f"{cat}.parquet"

        try:
            result = build_category_parquet(
                csv_path=csv_path,
                output_path=output_path,
                exclude_fields=exclude_fields,
            )
        except (duckdb.Error, OSError) as e:
            logger.error("Failed to process %s (%s): %s", cat, csv_path, e)
            results[cat] = {"error": str(e), "csv_path": str(csv_path)}
            if callback:
                callback(cat, f"{cat}: FAILED ({e})")
            continue

        if result["path"] is not None:
            results[cat] = result
            total_new_fields += result["n_fields"]
            total_new_cols += result["n_cols"] - 1  # exclude eid
            # Add new fields to exclude set for subsequent categories
            exclude_fields.update(result["new_fields"])
        else:
            results[cat] = {"skipped": True, "reason": "all fields already in parquet"}

        if callback:
            callback(cat, f"{cat}: {result.get('n_fields', 0)} new fields")

    return {
        "total_new_fields": total_new_fields,
        "total_new_columns": total_new_cols,
        "total_existing_fields": len(exclude_fields) - total_new_fields,
        "categories": results,
        "output_dir": str(output_dir),
    }


def register_extended_parquet(
    dm,
    parquet_path: Path,
    view_name: str = "extended",
) -> None:
    """Register an extended parquet file as a DuckDB view in the DataManager."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    safe_path = _escape_path(parquet_path)
    safe_view = _quote_identifier(view_name)
    dm.conn.execute(
        f"CREATE VIEW IF NOT EXISTS {safe_view} AS "
        f"SELECT * FROM read_parquet('{safe_path}')"
    )
    logger.info("Registered extended view '%s' from %s", view_name, parquet_path)
