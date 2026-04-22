"""DuckDB-based data loader — unified access to parquet + CSV.

Design:
  - Parquet files registered as VIEWs (zero-copy, scanned on demand)
  - CSV files registered lazily as VIEWs when a field is requested that
    isn't in parquet
  - field_route() determines fastest source for any field ID

Biobank-agnostic: all file names, column names, and table structures
are driven by Settings. Default configuration targets UK Biobank.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from ..config import Settings

logger = logging.getLogger(__name__)

# Default category CSV map (UKB application 672073). Overridable via Settings.
_DEFAULT_CATEGORY_CSVS = {
    "Population_Characteristics": "ukb672073_Population_Characteristics.csv",
    "Biological_Samples": "ukb672073_Biological_Samples.csv",
    "Health_Related_Outcomes": "ukb672073_Health_Related_Outcomes.csv",
    "Additional_Exposures": "ukb672073_Additional_Exposures.csv",
    "Online_Follow_up": "ukb672073_Online_Follow_up.csv",
    "Genomics": "ukb672073_Genomics.csv",
}


def _escape_path(p: Path) -> str:
    """Escape path for DuckDB SQL literal."""
    return str(p).replace("'", "''")


class DataManager:
    """Singleton data access layer backed by DuckDB.

    All biobank-specific names (subject ID column, diagnosis code column,
    parquet file names) are read from ``settings`` — nothing is hardcoded.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.conn = duckdb.connect(":memory:")
        self._parquet_fields: set[str] | None = None
        self._csv_views_registered: set[str] = set()
        self._category_views: list[str] = []
        self._category_field_map: dict[str, str] = {}
        self._id_col = settings.subject_id_col
        self._diag_col = settings.diagnoses_code_col
        self._death_col = settings.deaths_code_col
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
            safe = _escape_path(path)
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

        self._register_category_parquets()

    def _register_category_parquets(self) -> None:
        """Register category parquet files as individual DuckDB views."""
        cat_dir = self.settings.category_parquet_dir
        if not cat_dir.exists():
            return

        id_col = self._id_col
        for pq_file in sorted(cat_dir.glob("*.parquet")):
            view_name = f"cat_{pq_file.stem.lower()}"
            safe = _escape_path(pq_file)
            try:
                self.conn.execute(
                    f"CREATE OR REPLACE VIEW {view_name} AS "
                    f"SELECT * FROM read_parquet('{safe}')"
                )
                self._category_views.append(view_name)

                cat_cols = self.conn.execute(
                    f"SELECT column_name FROM information_schema.columns "
                    f"WHERE table_name = '{view_name}'"
                ).fetchall()
                for (col,) in cat_cols:
                    if col != id_col:
                        fid = col.split("-")[0]
                        if fid.isdigit():
                            self._category_field_map[fid] = view_name

                logger.info("Registered category view %s (%d columns)",
                            view_name, len(cat_cols))
            except Exception as e:
                logger.warning("Failed to register %s: %s", pq_file.name, e)

    def _get_parquet_fields(self) -> set[str]:
        """Lazily cache the set of field IDs available in biomarkers + category views."""
        if self._parquet_fields is None:
            id_col = self._id_col
            self._parquet_fields = set()
            try:
                cols = self.conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'biomarkers'"
                ).fetchall()
                for (col,) in cols:
                    if col != id_col:
                        fid = col.split("-")[0]
                        self._parquet_fields.add(fid)
            except Exception:
                pass
            for view_name in self._category_views:
                try:
                    cat_cols = self.conn.execute(
                        f"SELECT column_name FROM information_schema.columns "
                        f"WHERE table_name = '{view_name}'"
                    ).fetchall()
                    for (col,) in cat_cols:
                        if col != id_col:
                            fid = col.split("-")[0]
                            self._parquet_fields.add(fid)
                except Exception:
                    pass
        return self._parquet_fields

    def _register_csv_view(self, category: str) -> None:
        """Lazily register a category CSV as a DuckDB view."""
        if category in self._csv_views_registered:
            return
        fname = _DEFAULT_CATEGORY_CSVS.get(category)
        if fname is None:
            return
        csv_path = self.settings.raw_csv_dir / fname
        if not csv_path.exists():
            logger.warning("CSV not found: %s", csv_path)
            return
        view_name = f"csv_{category.lower()}"
        safe = _escape_path(csv_path)
        self.conn.execute(
            f"CREATE VIEW IF NOT EXISTS {view_name} AS "
            f"SELECT * FROM read_csv_auto('{safe}', header=true, "
            f"sample_size=1000, all_varchar=false)"
        )
        self._csv_views_registered.add(category)
        logger.info("Registered CSV view %s", view_name)

    # ── Field routing ───────────────────────────────────────

    _FIELD_TO_CSV: dict[str, str] | None = None

    def _build_field_csv_map(self) -> dict[str, str]:
        if DataManager._FIELD_TO_CSV is not None:
            return DataManager._FIELD_TO_CSV
        mapping: dict[str, str] = {}
        for category in _DEFAULT_CATEGORY_CSVS:
            txt_path = self.settings.raw_csv_dir / f"{category}.txt"
            if txt_path.exists():
                for line in txt_path.read_text().strip().splitlines():
                    fid = line.strip()
                    if fid:
                        mapping[fid] = category
        DataManager._FIELD_TO_CSV = mapping
        return mapping

    def field_source(self, field_id: str) -> str:
        """Return 'parquet', a category view name, or a category CSV name."""
        if field_id in self._get_parquet_fields():
            if field_id in self._category_field_map:
                return self._category_field_map[field_id]
            return "parquet"
        csv_map = self._build_field_csv_map()
        return csv_map.get(field_id, "unknown")

    # ── High-level queries ──────────────────────────────────

    def query(self, sql: str, params: list | tuple | None = None) -> pd.DataFrame:
        """Execute arbitrary DuckDB SQL and return DataFrame.

        Use ``?`` placeholders for parameters (prevents SQL injection).
        """
        if params:
            return self.conn.execute(sql, params).df()
        return self.conn.execute(sql).df()

    def get_field(self, field_id: str, instance: int = 0, array: int = 0,
                  eids: Optional[list[int]] = None) -> pd.DataFrame:
        """Get a single field's values for all (or specified) subjects."""
        id_col = self._id_col
        col_name = f'"{field_id}-{instance}.{array}"'
        source = self.field_source(field_id)

        if source == "parquet":
            sql = f"SELECT {id_col}, {col_name} AS value FROM biomarkers"
        elif source.startswith("cat_"):
            sql = f"SELECT {id_col}, {col_name} AS value FROM {source}"
        elif source != "unknown":
            self._register_csv_view(source)
            view = f"csv_{source.lower()}"
            sql = f"SELECT {id_col}, {col_name} AS value FROM {view}"
        else:
            raise ValueError(f"Field {field_id} not found in any data source")

        if eids:
            eid_list = ",".join(str(e) for e in eids)
            sql += f" WHERE {id_col} IN ({eid_list})"
        return self.conn.execute(sql).df()

    def get_biomarker_matrix(self, field_ids: list[str],
                             eids: Optional[list[int]] = None) -> pd.DataFrame:
        """Get a matrix of biomarker values (columns = field names)."""
        id_col = self._id_col
        cols = []
        for fid in field_ids:
            col = f'"{fid}-0.0"'
            cols.append(f"{col} AS \"{fid}\"")
        col_str = ", ".join(cols)
        sql = f"SELECT {id_col}, {col_str} FROM biomarkers"
        if eids:
            eid_list = ",".join(str(e) for e in eids)
            sql += f" WHERE {id_col} IN ({eid_list})"
        return self.conn.execute(sql).df()

    def get_diagnoses(self, code_prefix: Optional[str] = None) -> pd.DataFrame:
        """Get diagnosis records, optionally filtered by code prefix.

        Uses parameterized queries to prevent SQL injection.
        """
        sql = "SELECT * FROM diagnoses"
        if code_prefix:
            sql += f" WHERE {self._diag_col} LIKE ?"
            return self.query(sql, [f"{code_prefix}%"])
        return self.query(sql)

    def get_deaths(self, code_prefix: Optional[str] = None) -> pd.DataFrame:
        """Get death cause records.

        Uses parameterized queries to prevent SQL injection.
        """
        sql = "SELECT * FROM deaths"
        if code_prefix:
            sql += f" WHERE {self._death_col} LIKE ?"
            return self.query(sql, [f"{code_prefix}%"])
        return self.query(sql)

    def count_subjects(self) -> int:
        """Total subjects in biomarkers table."""
        id_col = self._id_col
        return self.conn.execute(f"SELECT COUNT(DISTINCT {id_col}) FROM biomarkers").fetchone()[0]

    def list_parquet_columns(self) -> list[str]:
        """List all column names in the biomarkers view."""
        rows = self.conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'biomarkers' ORDER BY ordinal_position"
        ).fetchall()
        return [r[0] for r in rows]

    def refresh_parquet_views(self) -> None:
        """Re-register parquet views after rebuild. Invalidates field cache."""
        self._parquet_fields = None
        self._init_parquet_views()
