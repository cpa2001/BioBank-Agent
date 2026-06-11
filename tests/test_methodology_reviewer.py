"""Methodology reviewer + vcf_association PC-covariate fix."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from biobank_agent.runtime.methodology import review_methodology, methodology_blocks
from biobank_agent.skills.vcf_association import _write_plink2_inputs, _pc_columns


def test_vcf_association_consumes_pcs_as_covariates():
    pheno = pd.DataFrame({"sample_id": ["s1", "s2", "s3", "s4"], "age": [50, 60, 55, 45], "is_male": [1, 0, 1, 0]})
    pcs = pd.DataFrame({"sample_id": ["s1", "s2", "s3", "s4"],
                        "PC1": [0.1, -0.2, 0.3, -0.1], "PC2": [0.0, 0.1, -0.1, 0.2], "PC3": [0.5, 0.4, 0.3, 0.2]})
    d = Path(tempfile.mkdtemp())
    _ph, cov = _write_plink2_inputs(pheno, ["s1", "s2"], ["s3", "s4"], "sample_id", d, pcs_df=pcs, n_pcs=2)
    cols = list(pd.read_csv(cov, sep="\t").columns)
    assert "age" in cols and "is_male" in cols
    assert "PC1" in cols and "PC2" in cols   # PCs now consumed (was the bug)
    assert "PC3" not in cols                  # n_pcs=2 honored


def test_no_pcs_keeps_legacy_behavior():
    pheno = pd.DataFrame({"sample_id": ["s1", "s2", "s3", "s4"], "age": [50, 60, 55, 45], "is_male": [1, 0, 1, 0]})
    d = Path(tempfile.mkdtemp())
    _ph, cov = _write_plink2_inputs(pheno, ["s1", "s2"], ["s3", "s4"], "sample_id", d)
    cols = list(pd.read_csv(cov, sep="\t").columns)
    assert "age" in cols and "is_male" in cols and not any(c.startswith("PC") for c in cols)


def test_pc_columns_helper():
    df = pd.DataFrame(columns=["sample_id", "PC1", "PC2", "PC10", "other"])
    assert _pc_columns(df, 2) == ["PC1", "PC2"]
    assert _pc_columns(df, 99) == ["PC1", "PC2", "PC10"]


def test_uncorrected_multiple_testing_is_blocked():
    payload = {"results": [{"trait": f"t{i}", "p_value": 0.001} for i in range(50)]}
    flags = review_methodology(payload)
    assert "uncorrected_multiple_testing" in {f["issue"] for f in flags}
    assert methodology_blocks(flags)


def test_correction_present_clears_multiple_testing_flag():
    payload = {"results": [{"trait": f"t{i}", "p_value": 0.001, "fdr": 0.05} for i in range(50)]}
    flags = review_methodology(payload)
    assert "uncorrected_multiple_testing" not in {f["issue"] for f in flags}


def test_underpowered_group_flagged():
    flags = review_methodology({"n_cases": 3, "n_controls": 100})
    assert any(f["issue"] == "underpowered_group" and f["severity"] == "block" for f in flags)


def test_missing_ci_advisory():
    flags = review_methodology({"odds_ratio": 2.1})
    assert any(f["issue"] == "missing_confidence_interval" and f["severity"] == "advisory" for f in flags)
    flags2 = review_methodology({"odds_ratio": 2.1, "ci_95": [1.2, 3.4]})
    assert not any(f["issue"] == "missing_confidence_interval" for f in flags2)


def test_inflation_without_pcs_advisory():
    flags = review_methodology({"lambda_gc": 1.4})
    assert any(f["issue"] == "unaddressed_stratification" for f in flags)
    flags2 = review_methodology({"lambda_gc": 1.4, "covariates": ["PC1", "PC2"]})
    assert not any(f["issue"] == "unaddressed_stratification" for f in flags2)


def test_clean_result_has_no_flags():
    payload = {"n_cases": 500, "n_controls": 500, "odds_ratio": 1.3, "ci_95": [1.1, 1.5],
               "results": [{"p_value": 0.01, "fdr": 0.04}], "lambda_gc": 1.02}
    assert review_methodology(payload) == []
