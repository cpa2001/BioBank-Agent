"""Tests for M1: data-grounded planning probe (kills CSV->WGS misroute at root)."""

from __future__ import annotations

from biobank_agent.runtime.data_probe import extract_paths, probe_objective, format_data_context


def test_extract_paths_multi():
    obj = "Read /data/a.csv and compare with results/b.parquet then /proj/dir"
    paths = extract_paths(obj)
    assert "/data/a.csv" in paths
    assert "results/b.parquet" in paths
    assert "/proj/dir" in paths


def test_tabular_csv_probe(tmp_path):
    csv = tmp_path / "all_traits_5e-11_gwas_results.csv"
    csv.write_text("trait,beta,se,pval\nE11,0.12,0.01,5e-12\nI10,,0.02,3e-9\nJ45,0.08,0.03,1e-8\n")
    out = probe_objective(f"Read the first rows of {csv} and classify the traits", cwd=str(tmp_path))
    rec = out["files"][0]
    assert rec["exists"] and rec["kind"] == "tabular" and rec["confidence"] == "high"
    assert "trait" in rec["columns"] and "pval" in rec["columns"]
    assert rec["rowcount_approx"] >= 3
    # missingness detected on the empty beta cell
    assert any(c == "beta" for c in (rec.get("missingness_top") or {}))
    text = out["text"]
    assert "TABULAR" in text and "not as genotypes" in text


def test_missing_path_is_unknown_not_blocking(tmp_path):
    out = probe_objective("Run GWAS on /mnt/hpc/cluster/cohort.bgen", cwd=str(tmp_path))
    rec = out["files"][0]
    assert rec["exists"] is False
    assert rec["confidence"] == "unknown"
    assert "NOT found locally" in out["text"]


def test_vcf_flagged_as_genomic(tmp_path):
    vcf = tmp_path / "cohort.vcf"
    vcf.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\n")
    out = probe_objective(f"QC the variants in {vcf}", cwd=str(tmp_path))
    rec = out["files"][0]
    assert rec["kind"] == "genomic_variants" and rec["confidence"] == "medium"
    assert "GENOMIC" in out["text"]


def test_empty_when_no_paths():
    out = probe_objective("what is the prevalence of diabetes")
    assert out["files"] == []
    assert out["text"] == ""
    assert format_data_context([]) == ""
