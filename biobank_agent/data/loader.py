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

import csv
import json
import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from ..config import Settings
from ..domain.banks import (
    CKBAdapter,
    HPPAdapter,
    RAPAdapter,
    UKBAdapter,
    BankAdapter,
    Modality,
    canonical_bank_id,
    get_adapter,
)

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

_BANK_COLUMN_DEFAULTS = {
    "ukb": ("eid", "diag_icd10", "cause_icd10"),
    "ukb_rap": ("eid", "diag_icd10", "cause_icd10"),
    "hpp": ("participant_id", "icd10", "cause_icd10"),
    "ckb": ("study_id", "icd10_code", "cause_icd10"),
}

_BANK_ADAPTER_TYPES = {
    "ukb": UKBAdapter,
    "hpp": HPPAdapter,
    "ckb": CKBAdapter,
    "ukb_rap": RAPAdapter,
    "rap": RAPAdapter,
    "ukb-rap": RAPAdapter,
}


def _escape_path(p: Path) -> str:
    """Escape path for DuckDB SQL literal."""
    return str(p).replace("'", "''")


def _quote_ident(name: str) -> str:
    """Quote a DuckDB identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def _safe_view_token(value: str) -> str:
    """Return a conservative view-name token from an external source key."""
    return "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_") or "source"


def _resolve_adapter(settings: Settings) -> BankAdapter | None:
    """Create an adapter instance aligned to ``settings.data_dir`` when possible."""
    bank_id = canonical_bank_id(getattr(settings, "bank_id", "ukb"))
    data_dir = getattr(settings, "data_dir", None)
    adapter_type = _BANK_ADAPTER_TYPES.get(bank_id)
    if adapter_type is None:
        return get_adapter(bank_id)
    if bank_id == "ukb_rap":
        return adapter_type()
    return adapter_type(Path(data_dir)) if data_dir is not None else adapter_type()


def _resolve_column(settings: Settings, configured: str, position: int) -> str:
    """Use bank-specific columns unless the user explicitly overrode Settings."""
    bank_id = canonical_bank_id(getattr(settings, "bank_id", "ukb"))
    ukb_default = _BANK_COLUMN_DEFAULTS["ukb"][position]
    bank_default = _BANK_COLUMN_DEFAULTS.get(bank_id, _BANK_COLUMN_DEFAULTS["ukb"])[position]
    return bank_default if configured == ukb_default else configured


class DataManager:
    """Singleton data access layer backed by DuckDB.

    All biobank-specific names (subject ID column, diagnosis code column,
    parquet file names) are read from ``settings`` — nothing is hardcoded.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bank_adapter = _resolve_adapter(settings)
        self.conn = duckdb.connect(":memory:")
        self._parquet_fields: set[str] | None = None
        self._csv_views_registered: set[str] = set()
        self._category_views: list[str] = []
        self._category_field_map: dict[str, str] = {}
        self._raw_field_map: dict[str, str] | None = None
        self._raw_field_columns: dict[str, list[str]] = {}
        self._raw_source_paths: dict[str, Path] = {}
        self._full_feature_field_map: dict[str, str] = {}
        self._full_feature_column_map: dict[str, str] = {}
        self._id_col = _resolve_column(settings, settings.subject_id_col, 0)
        self._diag_col = _resolve_column(settings, settings.diagnoses_code_col, 1)
        self._death_col = _resolve_column(settings, settings.deaths_code_col, 2)
        self._init_parquet_views()

    @property
    def bank_id(self) -> str:
        return canonical_bank_id(getattr(self.settings, "bank_id", "ukb"))

    @property
    def subject_id_col(self) -> str:
        return self._id_col

    @property
    def diagnoses_code_col(self) -> str:
        return self._diag_col

    @property
    def deaths_code_col(self) -> str:
        return self._death_col

    def resolve_field_id(self, semantic_or_field_id: str) -> str:
        """Resolve a bank-specific semantic shorthand to a physical field id."""
        value = str(semantic_or_field_id or "").strip()
        if self.bank_adapter is None:
            return value
        return self.bank_adapter.field_id(value)

    def normalize_icd_code(self, code: str) -> str:
        """Normalise an ICD code according to the active bank adapter."""
        if self.bank_adapter is None:
            return str(code or "").upper().strip().replace(".", "")
        return self.bank_adapter.normalize_icd(code)

    def _configured_parquet_path(self, attr: str, modality: Modality) -> Path:
        """Resolve parquet path with user settings first, adapter fallback second."""
        configured = getattr(self.settings, attr)
        adapter_path = self.bank_adapter.parquet_path(modality) if self.bank_adapter else None
        if Path(configured).exists() or adapter_path is None:
            return Path(configured)
        return Path(adapter_path)

    # ── Initialisation ──────────────────────────────────────

    def _init_parquet_views(self) -> None:
        """Register parquet directories as zero-copy views."""
        for attr, view_name, modality in [
            ("biomarker_parquet", "biomarkers", Modality.BIOMARKER),
            ("diagnoses_parquet", "diagnoses", Modality.DIAGNOSIS),
            ("deaths_parquet", "deaths", Modality.DEATH),
        ]:
            path = self._configured_parquet_path(attr, modality)
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
        self._register_full_feature_store()

    def _register_category_parquets(self) -> None:
        """Register category parquet files as individual DuckDB views."""
        cat_dir = self.settings.category_parquet_dir
        if not cat_dir.exists():
            return

        self._category_views = []
        self._category_field_map = {}
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
                        self._category_field_map[fid] = view_name

                logger.info("Registered category view %s (%d columns)",
                            view_name, len(cat_cols))
            except Exception as e:
                logger.warning("Failed to register %s: %s", pq_file.name, e)

    def _register_full_feature_store(self) -> None:
        """Register optional full UKB feature-store chunks.

        The feature store is built from raw UKB CSV exports by
        ``build_full_ukb_feature_store``. It is intentionally chunked by source
        and columns; this method registers those chunks lazily as DuckDB views
        and maps field IDs to the first chunk containing them.
        """
        self._full_feature_field_map = {}
        self._full_feature_column_map = {}
        store_dir = getattr(self.settings, "full_ukb_feature_store", None)
        if store_dir is None:
            return
        manifest_path = Path(store_dir) / "manifest.json"
        if not manifest_path.exists():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read full UKB feature-store manifest %s: %s", manifest_path, exc)
            return
        sources = manifest.get("sources") if isinstance(manifest, dict) else {}
        if not isinstance(sources, dict):
            return
        for source_key, source in sources.items():
            chunks = source.get("chunks", []) if isinstance(source, dict) else []
            for chunk in chunks:
                path = Path(str(chunk.get("path", "")))
                if not path.exists():
                    continue
                view_name = f"full_{_safe_view_token(source_key)}_{int(chunk.get('chunk_index', 0)):04d}"
                try:
                    self.conn.execute(
                        f"CREATE OR REPLACE VIEW {_quote_ident(view_name)} AS "
                        f"SELECT * FROM read_parquet('{_escape_path(path)}')"
                    )
                except Exception as exc:
                    logger.warning("Failed to register full UKB chunk %s: %s", path, exc)
                    continue
                for column in chunk.get("columns", []) or []:
                    if column == self._id_col:
                        continue
                    fid = str(column).split("-")[0].strip('"')
                    if not fid:
                        continue
                    self._full_feature_column_map[str(column)] = view_name
                    self._full_feature_field_map.setdefault(fid, view_name)

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
            f"sample_size=-1, all_varchar=false)"
        )
        self._csv_views_registered.add(category)
        logger.info("Registered CSV view %s", view_name)

    # ── Field routing ───────────────────────────────────────

    def _build_field_csv_map(self) -> dict[str, str]:
        if self._raw_field_map is not None:
            return self._raw_field_map
        mapping: dict[str, str] = {}
        self._raw_field_columns = {}
        self._raw_source_paths = {}

        def add_source(source_key: str, csv_path: Path, columns: list[str]) -> None:
            self._raw_source_paths[source_key] = csv_path
            for col in columns:
                if col == self._id_col:
                    continue
                fid = col.split("-")[0].strip('"')
                if not fid.isdigit():
                    continue
                mapping.setdefault(fid, source_key)
                self._raw_field_columns.setdefault(fid, []).append(col)

        for category in _DEFAULT_CATEGORY_CSVS:
            txt_path = self.settings.raw_csv_dir / f"{category}.txt"
            if txt_path.exists():
                for line in txt_path.read_text().strip().splitlines():
                    fid = line.strip()
                    if fid:
                        mapping[fid] = category
            csv_path = self.settings.raw_csv_dir / _DEFAULT_CATEGORY_CSVS[category]
            if csv_path.exists():
                try:
                    with open(csv_path, newline="", encoding="utf-8", errors="replace") as handle:
                        add_source(category, csv_path, next(csv.reader(handle)))
                except Exception:
                    pass
        raw_dir = getattr(self.settings, "raw_dir", None)
        if raw_dir is not None:
            for source_key, csv_path in {
                "main_672073": Path(raw_dir) / "UKB" / "ukb672073.csv",
                "main_671626": Path(raw_dir) / "UKB" / "ukb671626.csv",
            }.items():
                if csv_path.exists():
                    try:
                        with open(csv_path, newline="", encoding="utf-8", errors="replace") as handle:
                            add_source(source_key, csv_path, next(csv.reader(handle)))
                    except Exception:
                        pass
        self._raw_field_map = mapping
        return mapping

    def _register_raw_csv_source(self, source_key: str) -> str:
        """Register a raw UKB CSV source as a DuckDB view and return view name."""
        if source_key in self._csv_views_registered:
            return f"csv_{_safe_view_token(source_key)}"
        csv_path = self._raw_source_paths.get(source_key)
        if csv_path is None:
            self._build_field_csv_map()
            csv_path = self._raw_source_paths.get(source_key)
        if csv_path is None or not csv_path.exists():
            raise ValueError(f"CSV source {source_key} is not available")
        view_name = f"csv_{_safe_view_token(source_key)}"
        self.conn.execute(
            f"CREATE VIEW IF NOT EXISTS {_quote_ident(view_name)} AS "
            f"SELECT * FROM read_csv_auto('{_escape_path(csv_path)}', header=true, "
            f"sample_size=1, all_varchar=true)"
        )
        self._csv_views_registered.add(source_key)
        logger.info("Registered raw CSV view %s from %s", view_name, csv_path)
        return view_name

    def field_source(self, field_id: str) -> str:
        """Return 'parquet', a category view name, or a category CSV name."""
        field_id = self.resolve_field_id(field_id)
        if field_id in self._get_parquet_fields():
            if field_id in self._category_field_map:
                return self._category_field_map[field_id]
            return "parquet"
        if field_id in self._full_feature_field_map:
            return self._full_feature_field_map[field_id]
        csv_map = self._build_field_csv_map()
        return csv_map.get(field_id, "unknown")

    # ── High-level queries ──────────────────────────────────

    def _view_columns(self, view_name: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [view_name],
        ).fetchall()
        return [r[0] for r in rows]

    def field_column(self, field_id: str, instance: int = 0, array: int = 0,
                     source: str = "biomarkers") -> str:
        """Return the physical column for a semantic/field id in a source view.

        UKB uses ``<field>-<instance>.<array>`` columns, while HPP/CKB style
        adapters often expose native snake_case columns such as ``hba1c``. This
        method resolves both shapes before query construction.
        """
        resolved = self.resolve_field_id(field_id)
        columns = set(self._view_columns(source))
        pattern = getattr(self.settings, "field_column_pattern", "{field_id}-{instance}.{array}")
        try:
            patterned = pattern.format(field_id=resolved, instance=instance, array=array)
        except Exception:
            patterned = f"{resolved}-{instance}.{array}"
        candidates = []
        for candidate in (resolved, patterned, f"{resolved}-{instance}.{array}"):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        for candidate in candidates:
            if candidate in columns:
                return candidate
        prefix = f"{resolved}-"
        suffix = f".{array}"
        for column in sorted(columns):
            if column.startswith(prefix) and column.endswith(suffix):
                return column
        raw_columns = self._raw_field_columns.get(resolved, [])
        for candidate in candidates:
            if candidate in raw_columns:
                return candidate
        for column in sorted(raw_columns):
            if column.startswith(prefix) and column.endswith(suffix):
                return column
        raise ValueError(f"Field {resolved} not found in source {source}")

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
        field_id = self.resolve_field_id(field_id)
        id_col = self._id_col
        source = self.field_source(field_id)

        if source == "parquet":
            source_view = "biomarkers"
        elif source.startswith("cat_"):
            source_view = source
        elif source.startswith("full_"):
            source_view = source
        elif source != "unknown":
            if source in _DEFAULT_CATEGORY_CSVS:
                self._register_csv_view(source)
                source_view = f"csv_{source.lower()}"
            else:
                source_view = self._register_raw_csv_source(source)
        else:
            raise ValueError(f"Field {field_id} not found in any data source")
        col_name = self.field_column(field_id, instance=instance, array=array, source=source_view)
        sql = f"SELECT {_quote_ident(id_col)}, {_quote_ident(col_name)} AS value FROM {source_view}"

        if eids:
            sql += f" WHERE {_quote_ident(id_col)} IN (SELECT UNNEST(?))"
            return self.conn.execute(sql, [list(eids)]).df()
        return self.conn.execute(sql).df()

    def get_biomarker_matrix(self, field_ids: list[str],
                             eids: Optional[list[int]] = None) -> pd.DataFrame:
        """Get a matrix of biomarker values (columns = field names)."""
        id_col = self._id_col
        cols = []
        for fid in field_ids:
            fid = self.resolve_field_id(fid)
            physical = self.field_column(fid, source="biomarkers")
            cols.append(f"{_quote_ident(physical)} AS {_quote_ident(fid)}")
        col_str = ", ".join(cols)
        sql = f"SELECT {_quote_ident(id_col)}, {col_str} FROM biomarkers"
        if eids:
            sql += f" WHERE {_quote_ident(id_col)} IN (SELECT UNNEST(?))"
            return self.conn.execute(sql, [list(eids)]).df()
        return self.conn.execute(sql).df()

    def get_diagnoses(self, code_prefix: Optional[str] = None) -> pd.DataFrame:
        """Get diagnosis records, optionally filtered by code prefix.

        Uses parameterized queries to prevent SQL injection.
        """
        sql = "SELECT * FROM diagnoses"
        if code_prefix:
            where_sql, params = self.code_prefix_filter("diagnoses", code_prefix, self._diag_col)
            if not where_sql:
                return pd.DataFrame()
            sql += f" WHERE {where_sql}"
            return self.query(sql, params)
        return self.query(sql)

    def get_deaths(self, code_prefix: Optional[str] = None) -> pd.DataFrame:
        """Get death cause records.

        Uses parameterized queries to prevent SQL injection.
        """
        sql = "SELECT * FROM deaths"
        if code_prefix:
            where_sql, params = self.code_prefix_filter("deaths", code_prefix, self._death_col)
            if not where_sql:
                return pd.DataFrame()
            sql += f" WHERE {where_sql}"
            return self.query(sql, params)
        return self.query(sql)

    def count_subjects(self) -> int:
        """Total subjects in biomarkers table."""
        id_col = self._id_col
        return self.conn.execute(f"SELECT COUNT(DISTINCT {_quote_ident(id_col)}) FROM biomarkers").fetchone()[0]

    def code_prefix_filter(
        self,
        view_name: str,
        code_prefix: str,
        default_col: str | None = None,
    ) -> tuple[str, list[str]]:
        """Return a bank-aware ICD prefix predicate for a diagnosis-like view.

        The result is ``(where_sql, params)`` and is safe to splice into a
        DuckDB query. HPP gets special handling because real exports may carry
        ICD-9 and ICD-10 columns side by side; a request for ``E11`` should
        also match legacy ICD-9 ``250*`` rows.
        """
        try:
            columns = set(self._view_columns(view_name))
        except Exception:
            return "", []

        normalized = self.normalize_icd_code(code_prefix)
        filters: list[str] = []
        params: list[str] = []

        def add_like(column: str, prefix: str) -> None:
            if not column or column not in columns:
                return
            clause = f"{_quote_ident(column)} LIKE ?"
            value = f"{prefix}%"
            if (clause, value) in zip(filters, params):
                return
            filters.append(clause)
            params.append(value)

        add_like(default_col or "", normalized)

        bank_id = self.bank_id
        if bank_id == "hpp":
            for column in ("icd10", "icd10_code", "diagnosis_icd10"):
                add_like(column, normalized)
            adapter = self.bank_adapter
            legacy_prefixes = []
            legacy_lookup = getattr(adapter, "legacy_icd9_prefixes", None)
            if callable(legacy_lookup):
                legacy_prefixes = list(legacy_lookup(normalized))
            raw = str(code_prefix or "").upper().strip().replace(".", "")
            if raw[:3].isdigit() and raw[:3] not in legacy_prefixes:
                legacy_prefixes.append(raw[:3])
            for column in ("icd9", "icd9_code", "diagnosis_icd9"):
                for prefix in legacy_prefixes:
                    add_like(column, prefix)
        elif bank_id == "ckb":
            for column in ("icd10_code", "icd10", "diagnosis_code"):
                add_like(column, normalized)
        else:
            for column in ("diag_icd10", "icd10", "icd10_code"):
                add_like(column, normalized)

        if not filters:
            return "", []
        return "(" + " OR ".join(filters) + ")", params

    @property
    def data_available(self) -> bool:
        """True if at least the biomarkers view is registered and has data."""
        try:
            self.conn.execute("SELECT 1 FROM biomarkers LIMIT 1")
            return True
        except Exception:
            return False

    def list_parquet_columns(self) -> list[str]:
        """List all column names in the biomarkers view."""
        rows = self.conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'biomarkers' ORDER BY ordinal_position"
        ).fetchall()
        return [r[0] for r in rows]

    def list_parquet_column_info(self) -> list[dict[str, str]]:
        """List biomarker view columns with DuckDB data types."""
        rows = self.conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'biomarkers' ORDER BY ordinal_position"
        ).fetchall()
        return [{"name": r[0], "data_type": r[1]} for r in rows]

    def list_numeric_biomarker_columns(self) -> list[str]:
        """Return numeric biomarker columns excluding the subject id."""
        numeric_tokens = (
            "INT", "DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC",
        )
        out: list[str] = []
        for info in self.list_parquet_column_info():
            name = info["name"]
            dtype = str(info["data_type"]).upper()
            if name == self._id_col:
                continue
            if any(token in dtype for token in numeric_tokens):
                out.append(name)
        return out

    def refresh_parquet_views(self) -> None:
        """Re-register parquet views after rebuild. Invalidates field cache."""
        self._parquet_fields = None
        self._init_parquet_views()
