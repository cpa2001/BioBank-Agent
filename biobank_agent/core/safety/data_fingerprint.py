"""Data-source fingerprinting for reproducible scientific replay.

Codex review of the M1 plan flagged that the legacy ``Provenance``
record (``state.py:42``) hashes only the *result*, not the data sources
that produced it. Without a source fingerprint, ``/replay`` cannot
detect that a DuckDB view was rebuilt or a parquet snapshot replaced —
it would report "successful replay" while the underlying study has
silently changed.

This module captures a deterministic snapshot of:

1. Active biobank configuration (bank id + parquet roots + version)
2. DuckDB schema (table/view names, column dtypes, row counts)
3. Cohort identity (deterministic IDs derived from query criteria)
4. Skill input arguments

The output is a stable SHA-256 hash plus the underlying components,
which can be persisted alongside ``ReproducibilityHarness.checkpoint``
and surfaced by ``/replay`` to either warn or refuse on data drift.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Tables/views that materially affect cohort definitions or model outputs.
# Adding a name here means the row-count + schema becomes part of the
# fingerprint. Keep this list explicit so additions are auditable.
_DEFAULT_INDEXED_TABLES: tuple[str, ...] = (
    "biomarkers",
    "diagnoses",
    "deaths",
    "subjects",
    "field_lookup",
)


@dataclass
class TableSnapshot:
    """One DuckDB table/view's structural identity."""

    name: str
    column_signature: str  # "name:dtype,..." sorted
    row_count: int = -1   # -1 => not measured (network/large view)


@dataclass
class DataFingerprint:
    """Deterministic snapshot of the data inputs to a skill execution."""

    bank_id: str
    bank_config_hash: str
    duckdb_schema_hash: str
    tables: list[TableSnapshot] = field(default_factory=list)
    cohort_ids: list[str] = field(default_factory=list)
    inputs_hash: str = ""
    fingerprint_hash: str = ""
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "bank_config_hash": self.bank_config_hash,
            "duckdb_schema_hash": self.duckdb_schema_hash,
            "tables": [asdict(t) for t in self.tables],
            "cohort_ids": list(self.cohort_ids),
            "inputs_hash": self.inputs_hash,
            "fingerprint_hash": self.fingerprint_hash,
            "schema_version": self.schema_version,
        }


def _sha(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bank_config_payload(settings: Any) -> dict[str, Any]:
    """Pull the data-relevant fields off the agent settings.

    We deliberately enumerate the keys instead of dumping ``settings.__dict__``
    so non-deterministic fields (e.g. timestamps, API keys) don't influence
    the fingerprint.
    """

    candidates = (
        "biobank_name",
        "bank_id",
        "data_dir",
        "field_txt",
        "category_txt",
        "biomarkers_path",
        "diagnoses_path",
        "deaths_path",
        "memory_dir",
    )
    payload: dict[str, Any] = {}
    for key in candidates:
        try:
            value = getattr(settings, key)
        except AttributeError:
            continue
        # Stringify Paths so they hash deterministically across instances.
        payload[key] = str(value) if value is not None else None
    return payload


def snapshot_duckdb(
    conn: Any,
    indexed_tables: tuple[str, ...] = _DEFAULT_INDEXED_TABLES,
    measure_row_counts: bool = True,
) -> tuple[list[TableSnapshot], str]:
    """Capture column signatures (and optional row counts) for indexed tables.

    Returns (snapshots, schema_hash). Failures on individual tables are
    swallowed (logged at debug) so a missing optional view does not break
    the fingerprint of the rest.
    """
    snapshots: list[TableSnapshot] = []
    if conn is None:
        return snapshots, _sha([])

    for name in indexed_tables:
        cols: list[tuple[str, str]] = []
        try:
            rows = conn.execute(
                f"SELECT column_name, data_type FROM information_schema.columns "
                f"WHERE LOWER(table_name) = LOWER('{name}') "
                f"ORDER BY ordinal_position"
            ).fetchall()
            cols = [(str(r[0]), str(r[1])) for r in rows]
        except Exception as e:
            logger.debug("duckdb schema read failed for %s: %s", name, e)
            continue
        if not cols:
            continue
        signature = ",".join(f"{c}:{t}" for c, t in cols)
        row_count = -1
        if measure_row_counts:
            try:
                # COUNT(*) over a parquet view is cheap (metadata-only) but
                # we still cap with a safety guard for unexpected views.
                row_count_value = conn.execute(
                    f"SELECT COUNT(*) FROM {name}"
                ).fetchone()
                row_count = int(row_count_value[0]) if row_count_value else -1
            except Exception as e:
                logger.debug("duckdb row-count failed for %s: %s", name, e)
                row_count = -1
        snapshots.append(TableSnapshot(name=name, column_signature=signature, row_count=row_count))

    schema_payload = [
        {"name": s.name, "sig": s.column_signature, "rows": s.row_count}
        for s in snapshots
    ]
    return snapshots, _sha(schema_payload)


def derive_cohort_ids(state: Any) -> list[str]:
    """Pull deterministic cohort IDs off the SessionState.

    Uses ``state.cohorts`` keys (the cohort names) and a quick
    ``len/cases`` hash so two SessionStates with the same cohort
    definitions but different DataFrame instances produce the same id.
    """
    out: list[str] = []
    cohorts = getattr(state, "cohorts", None)
    if not cohorts:
        return out
    try:
        for name, df in cohorts.items():
            n = int(getattr(df, "shape", (0, 0))[0])
            cases = 0
            try:
                if "label" in df.columns:
                    cases = int(df["label"].sum())
            except Exception:
                cases = 0
            out.append(f"{name}:n={n}:cases={cases}")
    except Exception:
        return out
    return sorted(out)


def fingerprint(
    *,
    settings: Any,
    duckdb_conn: Any,
    state: Any,
    inputs: dict[str, Any],
    indexed_tables: tuple[str, ...] = _DEFAULT_INDEXED_TABLES,
    measure_row_counts: bool = True,
) -> DataFingerprint:
    """Compute a ``DataFingerprint`` for a single skill invocation."""

    bank_id = str(
        getattr(settings, "bank_id", None)
        or getattr(settings, "biobank_name", "")
        or "default"
    )
    bank_payload = _bank_config_payload(settings)
    bank_hash = _sha(bank_payload)

    snapshots, schema_hash = snapshot_duckdb(
        duckdb_conn,
        indexed_tables=indexed_tables,
        measure_row_counts=measure_row_counts,
    )

    cohort_ids = derive_cohort_ids(state)
    inputs_hash = _sha(inputs or {})

    composite_hash = _sha(
        {
            "bank": bank_hash,
            "schema": schema_hash,
            "cohorts": cohort_ids,
            "inputs": inputs_hash,
        }
    )
    return DataFingerprint(
        bank_id=bank_id,
        bank_config_hash=bank_hash,
        duckdb_schema_hash=schema_hash,
        tables=snapshots,
        cohort_ids=cohort_ids,
        inputs_hash=inputs_hash,
        fingerprint_hash=composite_hash,
    )


def diff_fingerprints(
    saved: DataFingerprint, current: DataFingerprint
) -> dict[str, Any]:
    """Compare two fingerprints and report drift sources.

    Used by ``/replay`` to refuse or warn when data has shifted under a
    saved provenance ID.
    """
    diff: dict[str, Any] = {}
    if saved.bank_id != current.bank_id:
        diff["bank_id"] = (saved.bank_id, current.bank_id)
    if saved.bank_config_hash != current.bank_config_hash:
        diff["bank_config_hash"] = (saved.bank_config_hash[:12], current.bank_config_hash[:12])
    if saved.duckdb_schema_hash != current.duckdb_schema_hash:
        # Granular per-table diff.
        saved_by_name = {t.name: t for t in saved.tables}
        cur_by_name = {t.name: t for t in current.tables}
        table_diffs: dict[str, Any] = {}
        for name in set(saved_by_name) | set(cur_by_name):
            s = saved_by_name.get(name)
            c = cur_by_name.get(name)
            if s is None:
                table_diffs[name] = "added"
            elif c is None:
                table_diffs[name] = "removed"
            elif s.column_signature != c.column_signature:
                table_diffs[name] = "schema_changed"
            elif s.row_count != c.row_count and s.row_count >= 0 and c.row_count >= 0:
                table_diffs[name] = f"row_count_changed:{s.row_count}->{c.row_count}"
        if table_diffs:
            diff["tables"] = table_diffs
    if saved.cohort_ids != current.cohort_ids:
        diff["cohorts"] = (saved.cohort_ids, current.cohort_ids)
    if saved.inputs_hash != current.inputs_hash:
        diff["inputs_hash"] = (saved.inputs_hash[:12], current.inputs_hash[:12])
    return diff


__all__ = [
    "DataFingerprint",
    "TableSnapshot",
    "fingerprint",
    "diff_fingerprints",
    "snapshot_duckdb",
    "derive_cohort_ids",
]
