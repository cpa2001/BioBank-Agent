"""Self-indexing data-lake engine: walk a directory, sample-infer schema, persist a catalog, locate data.

Designed for 10TB-scale folders and validated on small synthetic fixtures. The scaling discipline is that
NOTHING here loads a whole file: inventory takes size from ``stat()``, schema is inferred from a bounded
sample (DuckDB ``read_csv_auto(sample_size=N)`` / ``read_parquet`` schema, header sniff for the rest) under
a DuckDB ``memory_limit``, and the catalog is persisted as JSON. ``locate_data_for_step`` then answers
"which cheapest files hold the columns the next step needs" straight from that catalog — no re-scan.

Out-of-core primitives are DuckDB + PyArrow (the repo's established spine); no whole-file reads, no polars.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

# Recognised data formats, longest suffix first so ".vcf.gz" wins over ".gz".
_FORMAT_BY_SUFFIX = (
    (".vcf.gz", "vcf"),
    (".vcf.bgz", "vcf"),
    (".csv.gz", "csv"),
    (".tsv.gz", "tsv"),
    (".parquet", "parquet"),
    (".pq", "parquet"),
    (".csv", "csv"),
    (".tsv", "tsv"),
    (".txt", "csv"),
    (".vcf", "vcf"),
    (".h5ad", "h5ad"),
    (".h5", "hdf5"),
    (".hdf5", "hdf5"),
    (".jsonl", "jsonl"),
    (".ndjson", "jsonl"),
    (".json", "json"),
)


def _escape_path(path: "str | Path") -> str:
    """Escape a path for safe interpolation into a DuckDB SQL string (crafted single quotes)."""
    return str(path).replace("'", "''")


def classify_format(path: "str | Path") -> str:
    """Data format for a path by suffix, or ``""`` if it is not a recognised data file."""
    name = str(path).lower()
    for suffix, fmt in _FORMAT_BY_SUFFIX:
        if name.endswith(suffix):
            return fmt
    return ""


@dataclass
class DataFile:
    path: str
    format: str
    size_bytes: int = 0
    n_columns: int = 0
    columns: list[dict[str, str]] = field(default_factory=list)  # [{"name":..., "dtype":...}]
    schema_source: str = "none"  # "sample" | "parquet" | "header" | "keys" | "none"
    error: str = ""

    @property
    def column_names(self) -> list[str]:
        return [str(c.get("name", "")) for c in self.columns]


@dataclass
class DataCatalog:
    root: str
    files: list[DataFile] = field(default_factory=list)

    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        return sum(int(f.size_bytes or 0) for f in self.files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "n_files": self.n_files,
            "total_bytes": self.total_bytes,
            "files": [asdict(f) for f in self.files],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataCatalog":
        files = [DataFile(**{k: v for k, v in f.items() if k in DataFile.__dataclass_fields__}) for f in data.get("files", [])]
        return cls(root=str(data.get("root", "")), files=files)

    def write(self, path: "str | Path") -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out


def _duck_describe(sql_from: str, *, mem_cap_mb: int) -> list[dict[str, str]]:
    """Run a bounded ``DESCRIBE`` under a memory cap and return ``[{name, dtype}]``. Never scans fully."""
    import duckdb  # local import: keep module import light and optional at call time

    con = duckdb.connect(":memory:")
    try:
        try:
            con.execute(f"SET memory_limit='{max(64, int(mem_cap_mb))}MB'")
            con.execute("SET threads=2")
        except Exception:
            pass  # memory_limit is a guard, not load-bearing
        rows = con.execute(f"DESCRIBE SELECT * FROM {sql_from}").fetchall()
        return [{"name": str(r[0]), "dtype": str(r[1])} for r in rows]
    finally:
        con.close()


def _sniff_header(path: Path, fmt: str) -> list[dict[str, str]]:
    """Header-only column names for a delimited/VCF file (opens gzip transparently)."""
    opener = gzip.open if str(path).lower().endswith((".gz", ".bgz")) else open
    delim = "\t" if fmt == "tsv" else ","
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        if fmt == "vcf":
            for line in handle:  # header lines start with ##; the column line starts with a single #
                if line.startswith("#CHROM") or (line.startswith("#") and not line.startswith("##")):
                    cols = line.lstrip("#").rstrip("\n").split("\t")
                    return [{"name": c, "dtype": ""} for c in cols if c]
                if not line.startswith("#"):
                    break
            return []
        try:
            header = next(csv.reader(handle, delimiter=delim))
        except StopIteration:
            return []
        return [{"name": c, "dtype": ""} for c in header]


def infer_schema(path: "str | Path", *, fmt: str = "", sample_rows: int = 200, mem_cap_mb: int = 512) -> DataFile:
    """Infer a file's schema from a BOUNDED sample (never a full read). Defensive: errors are captured."""
    p = Path(path)
    fmt = fmt or classify_format(p)
    try:
        size = p.stat().st_size
    except OSError:
        size = 0
    out = DataFile(path=str(p), format=fmt or "unknown", size_bytes=size)
    try:
        if fmt in ("csv", "tsv"):
            safe = _escape_path(p)
            sep = ", sep='\\t'" if fmt == "tsv" else ""
            out.columns = _duck_describe(
                f"read_csv_auto('{safe}', header=true, sample_size={max(1, int(sample_rows))}{sep})",
                mem_cap_mb=mem_cap_mb,
            )
            out.schema_source = "sample"
        elif fmt == "parquet":
            out.columns = _duck_describe(f"read_parquet('{_escape_path(p)}')", mem_cap_mb=mem_cap_mb)
            out.schema_source = "parquet"  # schema metadata only, zero-copy
        elif fmt == "vcf":
            out.columns = _sniff_header(p, "vcf")
            out.schema_source = "header"
        elif fmt in ("hdf5", "h5ad"):
            out.columns = _hdf5_keys(p)
            out.schema_source = "keys"
        elif fmt in ("json", "jsonl"):
            out.columns = _json_keys(p)
            out.schema_source = "sample"
    except Exception as exc:  # a broken/partial file must not abort indexing
        # fall back to a header sniff for delimited files before giving up
        if fmt in ("csv", "tsv"):
            try:
                out.columns = _sniff_header(p, fmt)
                out.schema_source = "header"
            except Exception:
                out.error = str(exc)[:300]
        else:
            out.error = str(exc)[:300]
    out.n_columns = len(out.columns)
    return out


def _hdf5_keys(path: Path) -> list[dict[str, str]]:
    try:
        import h5py
    except Exception:
        return []
    with h5py.File(path, "r") as handle:  # metadata only
        return [{"name": str(k), "dtype": ""} for k in handle.keys()]


def _json_keys(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        first = handle.readline().strip()
    try:
        obj = json.loads(first)
    except json.JSONDecodeError:
        return []
    if isinstance(obj, dict):
        return [{"name": str(k), "dtype": type(v).__name__} for k, v in obj.items()]
    return []


def iter_data_files(root: "str | Path", *, max_files: int = 100_000, follow_symlinks: bool = False) -> Iterable[Path]:
    """Yield recognised data files under ``root`` (bounded), skipping hidden dirs. Uses ``os.walk``."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))  # prune hidden, stable order
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            if not classify_format(name):
                continue
            yield Path(dirpath) / name
            count += 1
            if count >= max_files:
                return


def index_directory(
    root: "str | Path",
    *,
    out_path: "str | Path | None" = None,
    max_files: int = 100_000,
    sample_rows: int = 200,
    mem_cap_mb: int = 512,
    formats: "Optional[Iterable[str]]" = None,
) -> DataCatalog:
    """Walk ``root``, sample-infer each data file's schema, and return (optionally persist) a catalog.

    Bounded and defensive: capped at ``max_files``, header-only/bounded-sample schema, per-file errors
    captured rather than raised. ``formats`` optionally restricts to certain formats (e.g. ``{"parquet"}``).
    """
    keep = set(formats) if formats else None
    catalog = DataCatalog(root=str(root))
    for path in iter_data_files(root, max_files=max_files):
        fmt = classify_format(path)
        if keep is not None and fmt not in keep:
            continue
        catalog.files.append(infer_schema(path, fmt=fmt, sample_rows=sample_rows, mem_cap_mb=mem_cap_mb))
    if out_path is not None:
        catalog.write(out_path)
    return catalog


def convert_to_parquet(
    src: "str | Path",
    dst: "str | Path",
    *,
    fmt: str = "",
    sample_size: int = 0,
    mem_cap_mb: int = 512,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert a delimited file to columnar SNAPPY parquet via DuckDB ``COPY`` (row-streamed, out-of-core).

    DuckDB streams rows through the COPY under a ``memory_limit``, so the whole file is never held in
    Python. Writes atomically (``.tmp`` → ``replace``). ``sample_size`` limits rows (for tests). Returns a
    status dict; unsupported inputs and errors are reported, not raised.
    """
    import duckdb

    src, dst = Path(src), Path(dst)
    fmt = fmt or classify_format(src)
    if fmt not in ("csv", "tsv"):
        return {"status": "unsupported", "format": fmt or "unknown", "dst": str(dst)}
    if dst.exists() and not overwrite:
        return {"status": "exists", "dst": str(dst)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    con = duckdb.connect(":memory:")
    try:
        try:
            con.execute(f"SET memory_limit='{max(64, int(mem_cap_mb))}MB'")
        except Exception:
            pass
        sep = ", sep='\\t'" if fmt == "tsv" else ""
        limit = f" LIMIT {int(sample_size)}" if sample_size else ""
        con.execute(
            f"COPY (SELECT * FROM read_csv_auto('{_escape_path(src)}', header=true{sep}){limit}) "
            f"TO '{_escape_path(tmp)}' (FORMAT PARQUET, COMPRESSION SNAPPY)"
        )
        os.replace(tmp, dst)
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{_escape_path(dst)}')").fetchone()
        cols = _duck_describe(f"read_parquet('{_escape_path(dst)}')", mem_cap_mb=mem_cap_mb)
        return {
            "status": "converted",
            "rows": int(n[0]) if n else 0,
            "n_columns": len(cols),
            "columns": [c["name"] for c in cols],
            "dst": str(dst),
        }
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return {"status": "error", "error": str(exc)[:300], "dst": str(dst)}
    finally:
        con.close()


@dataclass
class LocatedFile:
    file: DataFile
    score: float
    matched_columns: list[str] = field(default_factory=list)


def locate_data_for_step(
    catalog: DataCatalog,
    *,
    columns: "Optional[Iterable[str]]" = None,
    keywords: "Optional[Iterable[str]]" = None,
    formats: "Optional[Iterable[str]]" = None,
    top_k: int = 5,
) -> list[LocatedFile]:
    """Rank catalog files for the next analysis step: prefer files that hold the needed columns/keywords,
    breaking ties toward the CHEAPEST (smallest) file. Pure ranking over the catalog — no data is read."""
    want_cols = {c.strip().lower() for c in (columns or []) if str(c).strip()}
    want_kw = {k.strip().lower() for k in (keywords or []) if str(k).strip()}
    fmt_filter = set(formats) if formats else None

    ranked: list[LocatedFile] = []
    for f in catalog.files:
        if fmt_filter is not None and f.format not in fmt_filter:
            continue
        names_lower = {n.lower() for n in f.column_names}
        matched = sorted(n for n in f.column_names if n.lower() in want_cols)
        col_hits = len(matched)
        kw_hits = 0
        if want_kw:
            hay = " ".join([f.path.lower(), *names_lower])
            kw_hits = sum(1 for kw in want_kw if kw in hay)
        if want_cols and col_hits == 0 and kw_hits == 0:
            continue  # asked for specific columns but this file has none of them
        # score: column coverage dominates, keyword hits next; cheaper files win ties.
        coverage = (col_hits / len(want_cols)) if want_cols else 0.0
        score = coverage * 100.0 + kw_hits * 10.0
        ranked.append(LocatedFile(file=f, score=round(score, 3), matched_columns=matched))
    ranked.sort(key=lambda r: (-r.score, int(r.file.size_bytes or 0), r.file.path))
    return ranked[: max(1, int(top_k))] if ranked else []


__all__ = [
    "DataFile",
    "DataCatalog",
    "LocatedFile",
    "classify_format",
    "iter_data_files",
    "infer_schema",
    "index_directory",
    "convert_to_parquet",
    "locate_data_for_step",
]
