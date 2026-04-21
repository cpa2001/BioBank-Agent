"""DuckDB-based data loader — unified access to parquet + CSV.

Design:
  - Parquet files registered as VIEWs (zero-copy, scanned on demand)
  - CSV files registered lazily as VIEWs when a field is requested that
    isn't in parquet
  - field_route() determines fastest source for any field ID
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from ..config import Settings

logger = logging.getLogger(__name__)

# Map category CSV filename stem → file
_CATEGORY_CSVS = {
    "Population_Characteristics": "ukb672073_Population_Characteristics.csv",
    "Biological_Samples": "ukb672073_Biological_Samples.csv",
    "Health_Related_Outcomes": "ukb672073_Health_Related_Outcomes.csv",
    "Additional_Exposures": "ukb672073_Additional_Exposures.csv",
    "Online_Follow_up": "ukb672073_Online_Follow_up.csv",
    "Genomics": "ukb672073_Genomics.csv",
}


class DataManager:
    """Singleton data access layer backed by DuckDB."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.conn = duckdb.connect(":memory:")
        self._parquet_fields: set[str] | None = None
        self._csv_views_registered: set[str] = set()
        self._category_views: list[str] = []
        self._category_field_map: dict[str, str] = {}  # field_id → category view name
        self._init_parquet_views()

    # ── Initialisation ──────────────────────────────────────

    def _init_parquet_views(self) -> None:
        """Register parquet directories as zero-copy views."""
        for attr, view_name in [
            ("biomarker_parquet", "biomarkers"),
            ("diagnoses_parquet", "diagnoses"),
            ("deaths_parquet", "deaths"),
        ]:
            path = getattr(self.settings, attr)
            if not path.exists():
                continue
            safe = str(path).replace("'", "''")
            # Single file OR directory of parquets
            if path.is_file() and path.suffix == ".parquet":
                self.conn.execute(
                    f"CREATE OR REPLACE VIEW {view_name} AS "
                    f"SELECT * FROM read_parquet('{safe}')"
                )
            else:
                pattern = f"{safe}/*.parquet"
                self.conn.execute(
                    f"CREATE OR REPLACE VIEW {view_name} AS "
                    f"SELECT * FROM read_parquet('{pattern}', union_by_name=true)"
                )
            logger.info("Registered %s view from %s", view_name, path)

        # Register category parquets if they exist
        self._register_category_parquets()

    def _register_category_parquets(self) -> None:
        """Register category parquet files (from batch_rebuild) as individual DuckDB views."""
        cat_dir = self.settings.category_parquet_dir
        if not cat_dir.exists():
            return

        for pq_file in sorted(cat_dir.glob("*.parquet")):
            view_name = f"cat_{pq_file.stem.lower()}"
            safe = str(pq_file).replace("'", "''")
            try:
                self.conn.execute(
                    f"CREATE OR REPLACE VIEW {view_name} AS "
                    f"SELECT * FROM read_parquet('{safe}')"
                )
                self._category_views.append(view_name)

                # Map field IDs in this category to the view name
                cat_cols = self.conn.execute(
                    f"SELECT column_name FROM information_schema.columns "
                    f"WHERE table_name = '{view_name}'"
                ).fetchall()
                for (col,) in cat_cols:
                    if col != "eid":
                        fid = col.split("-")[0]
                        if fid.isdigit():
                            self._category_field_map[fid] = view_name

                logger.info("Registered category view %s from %s (%d columns)",
                            view_name, pq_file.name, len(cat_cols))
            except Exception as e:
                logger.warning("Failed to register %s: %s", pq_file.name, e)

    def _get_parquet_fields(self) -> set[str]:
        """Lazily cache the set of field IDs available in biomarkers view."""
        if self._parquet_fields is None:
            try:
                cols = self.conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'biomarkers'"
                ).fetchall()
                self._parquet_fields = set()
                for (col,) in cols:
                    if col != "eid":
                        fid = col.split("-")[0]
                        self._parquet_fields.add(fid)
            except Exception:
                self._parquet_fields = set()
            # Also include fields from category parquets
            for view_name in self._category_views:
                try:
                    cat_cols = self.conn.execute(
                        f"SELECT column_name FROM information_schema.columns "
                        f"WHERE table_name = '{view_name}'"
                    ).fetchall()
                    for (col,) in cat_cols:
                        if col != "eid":
                            fid = col.split("-")[0]
                            self._parquet_fields.add(fid)
                except Exception:
                    pass
        return self._parquet_fields

    def _register_csv_view(self, category: str) -> None:
        """Lazily register a category CSV as a DuckDB view."""
        if category in self._csv_views_registered:
            return
        fname = _CATEGORY_CSVS.get(category)
        if fname is None:
            return
        csv_path = self.settings.raw_csv_dir / fname
        if not csv_path.exists():
            logger.warning("CSV not found: %s", csv_path)
            return
        view_name = f"csv_{category.lower()}"
        self.conn.execute(
            f"CREATE VIEW IF NOT EXISTS {view_name} AS "
            f"SELECT * FROM read_csv_auto('{csv_path}', header=true, "
            f"sample_size=1000, all_varchar=false)"
        )
        self._csv_views_registered.add(category)
        logger.info("Registered CSV view %s from %s", view_name, csv_path)

    # ── Field routing ───────────────────────────────────────

    # Pre-computed mapping: field ID prefix → category CSV
    _FIELD_TO_CSV: dict[str, str] | None = None

    def _build_field_csv_map(self) -> dict[str, str]:
        """Build mapping from field ID → category by reading .txt files."""
        if DataManager._FIELD_TO_CSV is not None:
            return DataManager._FIELD_TO_CSV
        mapping: dict[str, str] = {}
        for category in _CATEGORY_CSVS:
            txt_path = self.settings.raw_csv_dir / f"{category}.txt"
            if txt_path.exists():
                for line in txt_path.read_text().strip().splitlines():
                    fid = line.strip()
                    if fid:
                        mapping[fid] = category
        DataManager._FIELD_TO_CSV = mapping
        return mapping

    def field_source(self, field_id: str) -> str:
        """Return 'parquet', a category view name, or a category CSV name for the best source."""
        if field_id in self._get_parquet_fields():
            # Check if it's specifically in a category parquet
            if field_id in self._category_field_map:
                return self._category_field_map[field_id]
            return "parquet"
        csv_map = self._build_field_csv_map()
        return csv_map.get(field_id, "unknown")

    # ── High-level queries ──────────────────────────────────

    def query(self, sql: str, params: list | tuple | None = None) -> pd.DataFrame:
        """Execute arbitrary DuckDB SQL and return DataFrame.
        
        Args:
            sql: SQL query string. Use ? placeholders for parameters.
            params: Optional list/tuple of parameters to bind (prevents SQL injection).
        
        Example:
            df = dm.query("SELECT * FROM diagnoses WHERE diag_icd10 LIKE ?", ["E11%"])
        """
        if params:
            return self.conn.execute(sql, params).df()
        return self.conn.execute(sql).df()

    def get_field(self, field_id: str, instance: int = 0, array: int = 0,
                  eids: Optional[list[int]] = None) -> pd.DataFrame:
        """Get a single field's values for all (or specified) subjects."""
        col_name = f'"{field_id}-{instance}.{array}"'
        source = self.field_source(field_id)

        if source == "parquet":
            sql = f"SELECT eid, {col_name} AS value FROM biomarkers"
        elif source.startswith("cat_"):
            # Category parquet view
            sql = f"SELECT eid, {col_name} AS value FROM {source}"
        elif source != "unknown":
            self._register_csv_view(source)
            view = f"csv_{source.lower()}"
            sql = f"SELECT eid, {col_name} AS value FROM {view}"
        else:
            raise ValueError(f"Field {field_id} not found in any data source")

        if eids:
            eid_list = ",".join(str(e) for e in eids)
            sql += f" WHERE eid IN ({eid_list})"
        return self.conn.execute(sql).df()

    def get_biomarker_matrix(self, field_ids: list[str],
                             eids: Optional[list[int]] = None) -> pd.DataFrame:
        """Get a matrix of biomarker values (columns = field names)."""
        cols = []
        for fid in field_ids:
            col = f'"{fid}-0.0"'
            cols.append(f"{col} AS \"{fid}\"")
        col_str = ", ".join(cols)
        sql = f"SELECT eid, {col_str} FROM biomarkers"
        if eids:
            eid_list = ",".join(str(e) for e in eids)
            sql += f" WHERE eid IN ({eid_list})"
        return self.conn.execute(sql).df()

    def get_diagnoses(self, icd10_prefix: Optional[str] = None) -> pd.DataFrame:
        """Get diagnosis records, optionally filtered by ICD10 prefix.
        
        Uses parameterized queries to prevent SQL injection.
        """
        sql = "SELECT * FROM diagnoses"
        if icd10_prefix:
            sql += " WHERE diag_icd10 LIKE ?"
            return self.query(sql, [f"{icd10_prefix}%"])
        return self.query(sql)

    def get_deaths(self, icd10_prefix: Optional[str] = None) -> pd.DataFrame:
        """Get death cause records.

        Uses parameterized queries to prevent SQL injection.
        """
        sql = "SELECT * FROM deaths"
        if icd10_prefix:
            sql += " WHERE cause_icd10 LIKE ?"
            return self.query(sql, [f"{icd10_prefix}%"])
        return self.query(sql)

    def count_subjects(self) -> int:
        """Total subjects in biomarkers table."""
        return self.conn.execute("SELECT COUNT(DISTINCT eid) FROM biomarkers").fetchone()[0]

    def list_parquet_columns(self) -> list[str]:
        """List all column names in the biomarkers view."""
        rows = self.conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'biomarkers' ORDER BY ordinal_position"
        ).fetchall()
        return [r[0] for r in rows]

    def refresh_parquet_views(self) -> None:
        """Re-register parquet views after parquet rebuild. Invalidates field cache."""
        self._parquet_fields = None
        self._init_parquet_views()
