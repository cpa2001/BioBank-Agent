"""Data-lake engine skills: index a large folder, infer schema, convert format, locate next-step data.

Thin, defensive wrappers over ``biobank_agent.data.indexer`` (DuckDB + PyArrow, bounded/out-of-core). They
let the agent self-index a data directory, sample-infer a file's schema, convert delimited files to
columnar parquet, and locate the cheapest files holding the columns the next analysis step needs — without
ever loading a whole file. Designed for 10TB-scale folders; validated on small synthetic fixtures.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from biobank_agent.data.indexer import (
    DataCatalog,
    convert_to_parquet,
    index_directory,
)
# Alias the indexer library functions so they do NOT shadow the same-named @skill wrappers in this
# module's namespace: the lazy registry resolves a skill via getattr(module, skill_name), so a skill
# function MUST be the module attribute of that exact name (see registry.SkillRegistry.execute).
from biobank_agent.data.indexer import infer_schema as _index_infer_schema
from biobank_agent.data.indexer import locate_data_for_step as _index_locate
from biobank_agent.registry import skill


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _index_dir(ctx: Any, output_dir: str = "") -> Path:
    if output_dir:
        return Path(output_dir).expanduser()
    settings = getattr(ctx, "settings", None)
    configured = str(getattr(settings, "data_index_dir", "") or "")
    if configured:
        return Path(configured).expanduser()
    return Path(getattr(ctx, "report_dir", getattr(settings, "reports_dir", "./reports")))


def _mem_cap(ctx: Any) -> int:
    return int(getattr(getattr(ctx, "settings", None), "data_engine_mem_cap_mb", 512) or 512)


def _sample_rows(ctx: Any) -> int:
    return int(getattr(getattr(ctx, "settings", None), "data_engine_sample_rows", 200) or 200)


@skill(
    name="index_data_lake",
    description=(
        "Walk a data directory and build a JSON catalog of every data file (CSV/TSV/Parquet/VCF/HDF5) "
        "with its format, size, and sampled schema. Bounded and out-of-core: never loads a whole file. "
        "Use before analysis to discover what data is available."
    ),
    parameters={
        "root": {"type": "string", "description": "Directory to index (walked recursively)."},
        "formats": {"type": "string", "description": "Comma-separated formats to keep (empty = all).", "default": ""},
        "max_files": {"type": "integer", "description": "Cap on files indexed.", "default": 100000},
        "output_dir": {"type": "string", "description": "Where to write the catalog JSON; defaults to the index dir.", "default": ""},
    },
    required=["root"],
)
def index_data_lake(root: str, formats: str = "", max_files: int = 100000, output_dir: str = "", *, ctx=None) -> dict:
    if ctx is None:
        return {"status": "ERROR", "error": "No agent context is available."}
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return {"status": "ERROR", "error": f"path not found: {root}"}
    out_dir = _index_dir(ctx, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"data_catalog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    catalog = index_directory(
        root_path, out_path=out_path, max_files=int(max_files),
        sample_rows=_sample_rows(ctx), mem_cap_mb=_mem_cap(ctx),
        formats=set(_split_csv(formats)) or None,
    )
    return {
        "status": "OK",
        "root": str(root_path),
        "n_files": catalog.n_files,
        "total_bytes": catalog.total_bytes,
        "formats": sorted({f.format for f in catalog.files}),
        "catalog_path": str(out_path),
    }


@skill(
    name="infer_schema",
    description=(
        "Infer one data file's schema (columns + dtypes) from a bounded sample — DuckDB for CSV/TSV/"
        "Parquet, header sniff for VCF/JSON, keys for HDF5. Never reads the whole file."
    ),
    parameters={"path": {"type": "string", "description": "Data file to inspect."}},
    required=["path"],
)
def infer_schema(path: str, *, ctx=None) -> dict:
    p = Path(path).expanduser()
    if not p.exists():
        return {"status": "ERROR", "error": f"path not found: {path}"}
    df = _index_infer_schema(p, sample_rows=_sample_rows(ctx) if ctx else 200, mem_cap_mb=_mem_cap(ctx) if ctx else 512)
    return {
        "status": "ERROR" if df.error else "OK",
        "path": df.path,
        "format": df.format,
        "n_columns": df.n_columns,
        "columns": df.columns,
        "schema_source": df.schema_source,
        "size_bytes": df.size_bytes,
        "error": df.error,
    }


@skill(
    name="convert_format",
    description=(
        "Convert a delimited file (CSV/TSV) to columnar SNAPPY Parquet via DuckDB, row-streamed and "
        "out-of-core (the whole file is never held in memory). Returns the row/column counts."
    ),
    parameters={
        "src": {"type": "string", "description": "Source CSV/TSV file."},
        "dst": {"type": "string", "description": "Destination .parquet path."},
        "overwrite": {"type": "boolean", "description": "Overwrite an existing destination.", "default": False},
    },
    required=["src", "dst"],
)
def convert_format(src: str, dst: str, overwrite: bool = False, *, ctx=None) -> dict:
    s = Path(src).expanduser()
    if not s.exists():
        return {"status": "ERROR", "error": f"path not found: {src}"}
    result = convert_to_parquet(s, Path(dst).expanduser(), mem_cap_mb=_mem_cap(ctx) if ctx else 512, overwrite=bool(overwrite))
    status = {"converted": "OK", "exists": "OK"}.get(result.get("status", ""), "ERROR")
    # keep the raw outcome under "outcome"; the skill-level status must win over result's own "status".
    return {**result, "outcome": result.get("status", ""), "status": status}


@skill(
    name="locate_data_for_step",
    description=(
        "Given a catalog (from index_data_lake) and the columns/keywords the next analysis step needs, "
        "return the cheapest files that hold them, ranked by coverage. Reads no data — only the catalog."
    ),
    parameters={
        "catalog_path": {"type": "string", "description": "Path to a catalog JSON from index_data_lake."},
        "columns": {"type": "string", "description": "Comma-separated column names the step needs.", "default": ""},
        "keywords": {"type": "string", "description": "Comma-separated keywords to match in paths/columns.", "default": ""},
        "formats": {"type": "string", "description": "Restrict to these formats (comma-separated).", "default": ""},
        "top_k": {"type": "integer", "description": "How many files to return.", "default": 5},
    },
    required=["catalog_path"],
)
def locate_data_for_step(catalog_path: str, columns: str = "", keywords: str = "", formats: str = "", top_k: int = 5, *, ctx=None) -> dict:
    cp = Path(catalog_path).expanduser()
    if not cp.exists():
        return {"status": "ERROR", "error": f"catalog not found: {catalog_path}"}
    try:
        catalog = DataCatalog.from_dict(json.loads(cp.read_text(encoding="utf-8")))
    except Exception as exc:
        return {"status": "ERROR", "error": f"unreadable catalog: {exc}"}
    hits = _index_locate(
        catalog,
        columns=_split_csv(columns) or None,
        keywords=_split_csv(keywords) or None,
        formats=set(_split_csv(formats)) or None,
        top_k=int(top_k),
    )
    return {
        "status": "OK",
        "matches": [
            {"path": h.file.path, "format": h.file.format, "size_bytes": h.file.size_bytes,
             "score": h.score, "matched_columns": h.matched_columns}
            for h in hits
        ],
        "n_matches": len(hits),
    }
