"""Tests for the planner inspection fast-path (issue #6: don't over-route to WGS).

A pure 'read first N rows / show columns / preview <file>' request must be
answered with a deterministic read-only preview, NOT routed through the council
(which over-plans a heavyweight genomics workflow for genomics-flavored files).
"""

from __future__ import annotations

import pytest

from biobank_agent.runtime.planner import (
    RuntimePlanner,
    extract_path,
    extract_row_count,
    is_pure_inspection_objective,
)


CSV_READ = (
    "Read the first 5 rows of /home/z/ckb_datal/all_traits_5e-11_gwas_results.csv "
    "and show the column names"
)
COMPLEX = (
    "file all_traits_5e-11_gwas_results.csv stores GWAS results for 243 traits. "
    "Read file, classify all the traits, and check the characteristic genes of each class"
)


@pytest.mark.parametrize("objective,expected", [
    (CSV_READ, True),
    ("Read the first 5 rows of /x/CKB_datal/results.csv and show the column names", True),
    ("show columns of data/gwas_analysis_results.csv", True),   # 'analysis' only in filename
    ("preview report.parquet", True),
    ("head /data/genotypes.vcf.gz", True),
    ("前 10 行 看一下 /data/x.tsv 的列名", True),
    (COMPLEX, False),                                            # classify + genes -> analysis
    ("Run a GWAS association analysis on cohort.vcf", False),
    ("train a model to predict E11 from biomarkers", False),
    ("what is the prevalence of diabetes", False),              # no file ref -> council
])
def test_is_pure_inspection_objective(objective, expected):
    assert is_pure_inspection_objective(objective) is expected


def test_extract_path_and_rows():
    assert extract_path(CSV_READ) == "/home/z/ckb_datal/all_traits_5e-11_gwas_results.csv"
    assert extract_path("show columns of data/gwas_analysis_results.csv") == "data/gwas_analysis_results.csv"
    assert extract_row_count(CSV_READ) == 5
    assert extract_row_count("preview report.parquet") is None


class _BoomRouter:
    """Any attribute access means the council ran — which must NOT happen for an
    inspection request."""

    def __getattr__(self, name):  # pragma: no cover - only hit on failure
        raise AssertionError(f"council ran for an inspection request (touched .{name})")


def test_build_plan_inspection_skips_council():
    planner = RuntimePlanner(_BoomRouter(), num_candidates=3)
    plan = planner.build_plan(CSV_READ, tool_names=["python_exec", "field_search", "vcf_qc"])
    # Deterministic single read-only step, council never invoked.
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.tool_scope == ["python_exec"]
    assert "/home/z/ckb_datal/all_traits_5e-11_gwas_results.csv" in step.file_scope
    purpose = step.purpose.lower()
    assert "read-only" in purpose and "do not" in purpose
    assert "inspection fast-path" in plan.audit_summary.lower()


def test_build_plan_complex_uses_council():
    # The complex classification task must fall through to the council, which will
    # touch the (boom) router — proving it is NOT treated as a trivial inspection.
    planner = RuntimePlanner(_BoomRouter(), num_candidates=1)
    with pytest.raises(AssertionError):
        planner.build_plan(COMPLEX, tool_names=["python_exec"])
