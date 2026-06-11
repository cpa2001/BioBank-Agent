"""Tests for incremental parquet builder utilities."""

from types import SimpleNamespace

import duckdb
import pytest

from biobank_agent.data.parquet_builder import (
    _escape_path,
    _extract_field_ids,
    _get_csv_columns,
    _quote_identifier,
    batch_rebuild,
    build_full_ukb_feature_store,
    build_category_parquet,
    build_field_parquet,
    full_ukb_inventory,
    register_extended_parquet,
)


def _write_csv(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_escape_path_and_quote_identifier():
    assert _escape_path("x'y.csv") == "x''y.csv"
    assert _quote_identifier('view"name') == '"view""name"'


def test_extract_field_ids_ignores_eid_and_non_numeric():
    assert _extract_field_ids(["eid", "21001-0.0", '"30750-1.0"', "abc-0.0"]) == {"21001", "30750"}


def test_get_csv_columns_reads_header(tmp_path):
    csv = _write_csv(tmp_path / "input.csv", "eid,21001-0.0,abc\n1,25,x\n")
    conn = duckdb.connect(":memory:")
    try:
        assert _get_csv_columns(conn, csv) == ["eid", "21001-0.0", "abc"]
    finally:
        conn.close()


def test_build_field_parquet_extracts_requested_fields(tmp_path):
    csv = _write_csv(
        tmp_path / "fields.csv",
        "eid,21001-0.0,21001-1.0,30750-0.0\n1,25,26,4.5\n2,30,31,5.1\n",
    )

    out = build_field_parquet(csv, ["21001"], tmp_path / "out", sample_size=1)
    df = duckdb.connect(":memory:").execute(f"SELECT * FROM read_parquet('{out}')").df()

    assert list(df.columns) == ["eid", "21001-0.0", "21001-1.0"]
    assert len(df) == 1


def test_build_field_parquet_raises_when_no_fields_match(tmp_path):
    csv = _write_csv(tmp_path / "fields.csv", "eid,21001-0.0\n1,25\n")

    with pytest.raises(ValueError, match="No columns found"):
        build_field_parquet(csv, ["99999"], tmp_path / "out")


def test_build_category_parquet_skips_when_all_fields_excluded(tmp_path):
    csv = _write_csv(tmp_path / "category.csv", "eid,21001-0.0\n1,25\n")

    result = build_category_parquet(csv, tmp_path / "category.parquet", exclude_fields={"21001"})

    assert result == {"path": None, "n_rows": 0, "n_cols": 0, "n_fields": 0, "new_fields": []}


def test_build_category_parquet_writes_only_new_fields(tmp_path):
    csv = _write_csv(tmp_path / "category.csv", "eid,21001-0.0,30750-0.0\n1,25,4.5\n2,30,5.1\n")

    result = build_category_parquet(csv, tmp_path / "category.parquet", exclude_fields={"21001"})

    assert result["n_rows"] == 2
    assert result["n_cols"] == 2
    assert result["new_fields"] == ["30750"]


def test_full_ukb_inventory_and_feature_store_build_are_chunked(tmp_path):
    raw = tmp_path / "raw"
    _write_csv(
        raw / "UKB" / "ukb672073.csv",
        "eid,6153-0.0,6153-0.1,2443-0.0,21001-0.0\n1,1,2,1,25\n2,3,4,0,30\n",
    )

    inventory = full_ukb_inventory(
        raw,
        sources=["main_672073"],
        field_ids=["6153", "2443"],
        chunk_cols=2,
    )
    assert inventory["status"] == "READY"
    assert inventory["sources"][0]["n_selected_cols"] == 4
    assert inventory["sources"][0]["n_chunks"] == 2

    result = build_full_ukb_feature_store(
        raw,
        tmp_path / "store",
        sources=["main_672073"],
        field_ids=["6153", "2443"],
        chunk_cols=2,
    )
    assert result["status"] == "READY"
    assert result["n_chunks"] == 2
    assert (tmp_path / "store" / "manifest.json").exists()
    first_chunk = result["sources"]["main_672073"]["chunks"][0]
    df = duckdb.connect(":memory:").execute(
        f"SELECT * FROM read_parquet('{first_chunk['path']}')"
    ).df()
    assert len(df) == 2
    assert "eid" in df.columns


def test_batch_rebuild_handles_unknown_missing_success_and_callback(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_csv(
        raw / "ukb672073_Population_Characteristics.csv",
        "eid,21001-0.0\n1,25\n",
    )
    events = []

    result = batch_rebuild(
        raw_csv_dir=raw,
        output_dir=out,
        categories=["Unknown", "Biological_Samples", "Population_Characteristics"],
        callback=lambda cat, msg: events.append((cat, msg)),
    )

    assert result["total_new_fields"] == 1
    assert "Population_Characteristics" in result["categories"]
    assert "Unknown" not in result["categories"]
    assert "Biological_Samples" not in result["categories"]
    assert events


def test_batch_rebuild_reads_existing_parquet_and_skips_duplicates(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    existing = tmp_path / "existing"
    existing_csv = _write_csv(existing / "existing.csv", "eid,21001-0.0\n1,25\n")
    build_field_parquet(existing_csv, ["21001"], existing, output_name="existing.parquet")
    _write_csv(
        raw / "ukb672073_Population_Characteristics.csv",
        "eid,21001-0.0,30750-0.0\n1,25,4.5\n",
    )

    result = batch_rebuild(
        raw_csv_dir=raw,
        output_dir=out,
        existing_parquet_dir=existing,
        categories=["Population_Characteristics"],
    )

    assert result["total_existing_fields"] == 1
    assert result["total_new_fields"] == 1
    assert result["categories"]["Population_Characteristics"]["new_fields"] == ["30750"]

    skipped_raw = tmp_path / "skipped-raw"
    skipped_out = tmp_path / "skipped-out"
    _write_csv(
        skipped_raw / "ukb672073_Population_Characteristics.csv",
        "eid,21001-0.0\n1,25\n",
    )
    skipped = batch_rebuild(
        raw_csv_dir=skipped_raw,
        output_dir=skipped_out,
        existing_parquet_dir=existing,
        categories=["Population_Characteristics"],
    )
    assert skipped["categories"]["Population_Characteristics"] == {
        "skipped": True,
        "reason": "all fields already in parquet",
    }

    mixed_existing = tmp_path / "mixed-existing"
    mixed_existing.mkdir()
    mixed_source = _write_csv(mixed_existing / "mixed.csv", "eid,abc-0.0,21001-0.0\n1,x,25\n")
    conn = duckdb.connect(":memory:")
    try:
        conn.execute(
            f"COPY (SELECT * FROM read_csv_auto('{mixed_source}')) "
            f"TO '{mixed_existing / 'mixed.parquet'}' (FORMAT PARQUET)"
        )
    finally:
        conn.close()
    _write_csv(raw / "ukb672073_Disease_Outcomes.csv", "eid,30750-0.0\n1,4.5\n")
    mixed = batch_rebuild(
        raw_csv_dir=raw,
        output_dir=tmp_path / "mixed-out",
        existing_parquet_dir=mixed_existing,
        categories=["Disease_Outcomes"],
    )
    assert mixed["total_existing_fields"] == 1


def test_batch_rebuild_records_category_conversion_errors(tmp_path, monkeypatch):
    from biobank_agent.data import parquet_builder as parquet_mod

    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_csv(
        raw / "ukb672073_Population_Characteristics.csv",
        "eid,21001-0.0\n1,25\n",
    )
    events = []

    def broken_build(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(parquet_mod, "build_category_parquet", broken_build)

    result = batch_rebuild(
        raw_csv_dir=raw,
        output_dir=out,
        categories=["Population_Characteristics"],
        callback=lambda cat, msg: events.append((cat, msg)),
    )

    assert result["categories"]["Population_Characteristics"]["error"] == "disk full"
    assert result["categories"]["Population_Characteristics"]["csv_path"].endswith(
        "ukb672073_Population_Characteristics.csv"
    )
    assert events[-1][1] == "Population_Characteristics: FAILED (disk full)"

    no_callback = batch_rebuild(
        raw_csv_dir=raw,
        output_dir=out,
        categories=["Population_Characteristics"],
        callback=None,
    )
    assert no_callback["categories"]["Population_Characteristics"]["error"] == "disk full"


def test_batch_rebuild_raises_when_existing_parquet_unreadable(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "bad.parquet").write_text("not parquet")

    with pytest.raises(RuntimeError, match="Cannot read existing parquet"):
        batch_rebuild(tmp_path / "raw", tmp_path / "out", existing_parquet_dir=existing)


def test_register_extended_parquet_quotes_view_and_path(tmp_path):
    csv = _write_csv(tmp_path / "fields.csv", "eid,21001-0.0\n1,25\n")
    parquet_path = build_field_parquet(csv, ["21001"], tmp_path / "out")
    dm = SimpleNamespace(conn=duckdb.connect(":memory:"))
    try:
        register_extended_parquet(dm, parquet_path, view_name='extended"view')
        rows = dm.conn.execute('SELECT COUNT(*) FROM "extended""view"').fetchone()[0]
        assert rows == 1
    finally:
        dm.conn.close()


def test_register_extended_parquet_missing_file(tmp_path):
    dm = SimpleNamespace(conn=duckdb.connect(":memory:"))
    try:
        with pytest.raises(FileNotFoundError):
            register_extended_parquet(dm, tmp_path / "missing.parquet")
    finally:
        dm.conn.close()
