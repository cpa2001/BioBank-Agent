"""Phase 6: self-indexing data-lake engine (walk + sampled schema + catalog + locate + convert).

All fixtures are tiny and built in ``tmp_path`` (mirrors tests/test_parquet_builder.py). Pins the
scaling discipline via behaviour: schema comes from a bounded sample, hidden/non-data files are skipped,
the catalog round-trips, locate ranks by coverage then cheapest, and conversion streams via DuckDB.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from biobank_agent.data.indexer import (
    DataCatalog,
    classify_format,
    convert_to_parquet,
    index_directory,
    infer_schema,
    locate_data_for_step,
)


def _write_parquet(path: Path, select_sql: str) -> None:
    duckdb.connect(":memory:").execute(f"COPY ({select_sql}) TO '{path}' (FORMAT PARQUET)")


def test_classify_format_longest_suffix_wins():
    assert classify_format("a.vcf.gz") == "vcf"
    assert classify_format("a.csv") == "csv"
    assert classify_format("a.tsv") == "tsv"
    assert classify_format("a.parquet") == "parquet"
    assert classify_format("readme.md") == ""


def test_infer_schema_samples_csv_and_reads_parquet_metadata(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("eid,age,sex\n1,55,M\n2,60,F\n")
    df = infer_schema(csv, sample_rows=50)
    assert df.format == "csv" and df.schema_source == "sample"
    assert df.column_names == ["eid", "age", "sex"]
    assert any(c["dtype"] for c in df.columns)  # dtypes inferred from the sample

    pq = tmp_path / "b.parquet"
    _write_parquet(pq, "SELECT 1 AS eid, 3.14 AS ldl")
    dfp = infer_schema(pq)
    assert dfp.format == "parquet" and dfp.schema_source == "parquet"
    assert dfp.column_names == ["eid", "ldl"]


def test_infer_schema_is_defensive_on_broken_file(tmp_path):
    bad = tmp_path / "bad.parquet"
    bad.write_text("this is not parquet")
    df = infer_schema(bad)
    assert df.format == "parquet" and df.error and df.n_columns == 0  # captured, not raised


def test_index_directory_skips_hidden_and_non_data_and_persists(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "cohort.csv").write_text("eid,age\n1,55\n")
    (tmp_path / "sub" / "labs.tsv").write_text("eid\tglucose\n1\t5.5\n")
    _write_parquet(tmp_path / "bio.parquet", "SELECT 1 AS eid, 2.0 AS ldl")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "skip.csv").write_text("a,b\n1,2\n")  # hidden dir
    (tmp_path / "notes.md").write_text("not data")  # non-data

    catalog = index_directory(tmp_path, out_path=tmp_path / "catalog.json", sample_rows=20)

    names = sorted(Path(f.path).name for f in catalog.files)
    assert names == ["bio.parquet", "cohort.csv", "labs.tsv"]  # hidden + md excluded
    assert catalog.n_files == 3 and catalog.total_bytes > 0

    loaded = DataCatalog.from_dict(json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8")))
    assert loaded.n_files == 3


def test_index_directory_respects_format_filter_and_max_files(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    _write_parquet(tmp_path / "b.parquet", "SELECT 1 AS x")
    only_pq = index_directory(tmp_path, formats={"parquet"})
    assert [f.format for f in only_pq.files] == ["parquet"]
    capped = index_directory(tmp_path, max_files=1)
    assert capped.n_files == 1


def test_locate_data_for_step_prefers_coverage_then_cheapest(tmp_path):
    (tmp_path / "cohort.csv").write_text("eid,age,sex\n1,55,M\n")
    (tmp_path / "labs.tsv").write_text("eid\tglucose\thba1c\n1\t5.5\t42\n")
    _write_parquet(tmp_path / "bio.parquet", "SELECT 1 AS eid, 2.0 AS ldl")
    catalog = index_directory(tmp_path)

    by_cols = locate_data_for_step(catalog, columns=["glucose", "hba1c"])
    assert Path(by_cols[0].file.path).name == "labs.tsv"
    assert by_cols[0].matched_columns == ["glucose", "hba1c"]

    by_kw = locate_data_for_step(catalog, keywords=["ldl"])
    assert Path(by_kw[0].file.path).name == "bio.parquet"

    assert locate_data_for_step(catalog, columns=["does_not_exist"]) == []  # no match → excluded


def test_convert_to_parquet_streams_and_reports(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("eid,age\n1,55\n2,60\n3,65\n")
    dst = tmp_path / "out" / "c.parquet"

    result = convert_to_parquet(csv, dst)
    assert result["status"] == "converted" and result["rows"] == 3
    assert result["columns"] == ["eid", "age"]
    assert dst.exists() and not dst.with_suffix(".parquet.tmp").exists()  # atomic, no temp left

    # existing file is not clobbered without overwrite
    assert convert_to_parquet(csv, dst)["status"] == "exists"
    # non-delimited input is refused, not crashed
    assert convert_to_parquet(tmp_path / "x.parquet", dst, overwrite=True)["status"] == "unsupported"
