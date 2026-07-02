"""Phase 6: the four data-lake engine skills (index / infer_schema / convert_format / locate).

Thin wrappers over the indexer, exercised with a fake ctx and tiny tmp_path fixtures.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from biobank_agent.skills.data_engine import (
    convert_format,
    index_data_lake,
    infer_schema_skill,
    locate_step_data,
)


def _ctx(tmp_path: Path) -> SimpleNamespace:
    settings = SimpleNamespace(
        data_engine_mem_cap_mb=256,
        data_engine_sample_rows=50,
        data_index_dir="",
        reports_dir=str(tmp_path / "reports"),
    )
    return SimpleNamespace(settings=settings, report_dir=str(tmp_path / "reports"))


def test_index_data_lake_then_locate(tmp_path):
    (tmp_path / "cohort.csv").write_text("eid,age\n1,55\n")
    (tmp_path / "labs.tsv").write_text("eid\tglucose\thba1c\n1\t5.5\t42\n")
    ctx = _ctx(tmp_path)

    idx = index_data_lake(str(tmp_path), ctx=ctx)
    assert idx["status"] == "OK" and idx["n_files"] >= 2
    assert Path(idx["catalog_path"]).exists()

    loc = locate_step_data(idx["catalog_path"], columns="glucose,hba1c", ctx=ctx)
    assert loc["status"] == "OK" and loc["n_matches"] >= 1
    assert loc["matches"][0]["path"].endswith("labs.tsv")


def test_infer_schema_skill(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("a,b\n1,2\n")
    out = infer_schema_skill(str(csv), ctx=_ctx(tmp_path))
    assert out["status"] == "OK"
    assert [c["name"] for c in out["columns"]] == ["a", "b"]
    assert infer_schema_skill(str(tmp_path / "nope.csv"), ctx=_ctx(tmp_path))["status"] == "ERROR"


def test_convert_format_skill(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    dst = tmp_path / "out" / "c.parquet"
    out = convert_format(str(csv), str(dst), ctx=_ctx(tmp_path))
    assert out["status"] == "OK" and out["rows"] == 2 and dst.exists()


def test_skills_are_defensive(tmp_path):
    assert index_data_lake("/nonexistent/path", ctx=_ctx(tmp_path))["status"] == "ERROR"
    assert index_data_lake(str(tmp_path), ctx=None)["status"] == "ERROR"
    assert locate_step_data(str(tmp_path / "missing.json"), ctx=_ctx(tmp_path))["status"] == "ERROR"
