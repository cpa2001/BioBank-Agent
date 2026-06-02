"""Tests for WGS sample-manifest loading and environment diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_load_sample_manifest_normalizes_user_sampleid_and_filepath(tmp_path):
    from biobank_agent.data.vcf_loader import load_sample_manifest

    vcf_dir = tmp_path / "vcfs"
    vcf_dir.mkdir()
    vcf = vcf_dir / "J1-41Y-F.genotyper.vcf.gz"
    vcf.touch()

    manifest = tmp_path / "WGS_Sample_info.tsv"
    manifest.write_text(
        "Donor\tPart\tsampleID\tAge\tSex\tPhenotype\tfilePath\n"
        f"J1\tTemple\tJ1-W-41Y-F\t41Y\tFemale\tJuvenile_White\t{vcf}\n",
        encoding="utf-8",
    )

    df = load_sample_manifest(manifest, vcf_dirs=[vcf_dir])

    assert df.loc[0, "sample_id"] == "J1-41Y-F"
    assert df.loc[0, "source_sample_id"] == "J1-W-41Y-F"
    assert df.loc[0, "age"] == 41
    assert df.loc[0, "sex"] == "F"
    assert df.loc[0, "is_male"] == 0
    assert df.loc[0, "phenotype_group"] == "J"
    assert df.loc[0, "vcf_path"] == str(vcf)


def test_load_sample_manifest_resolves_relative_file_against_vcf_dir(tmp_path):
    from biobank_agent.data.vcf_loader import load_sample_manifest

    vcf_dir = tmp_path / "vcfs"
    vcf_dir.mkdir()
    vcf = vcf_dir / "V1-55Y-M.genotyper.vcf.gz"
    vcf.touch()
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {
            "Donor": "V1",
            "Part": "Occipital",
            "sampleID": "V1-W-55Y-M",
            "Age": "55Y",
            "Sex": "Male",
            "Phenotype": "Vitiligo_White",
            "filePath": "V1-55Y-M.genotyper.vcf.gz",
        }
    ]).to_csv(manifest, index=False)

    df = load_sample_manifest(manifest, vcf_dirs=[vcf_dir])

    assert df.loc[0, "sample_id"] == "V1-55Y-M"
    assert df.loc[0, "vcf_path"] == str(vcf)
    assert df.loc[0, "phenotype_group"] == "V"


def test_discover_sample_manifest_uses_env_override(tmp_path, monkeypatch):
    from biobank_agent.data.vcf_loader import discover_sample_manifest

    manifest = tmp_path / "custom_manifest.tsv"
    manifest.write_text("sample_id\tfilePath\n", encoding="utf-8")
    monkeypatch.setenv("WGS_SAMPLE_INFO", str(manifest))

    assert discover_sample_manifest(tmp_path / "missing") == manifest


def test_wgs_environment_status_reports_degradation(monkeypatch):
    from biobank_agent.utils import wgs

    monkeypatch.setattr(wgs, "package_available", lambda name: name not in {"openpyxl", "gseapy"})
    monkeypatch.setattr(wgs, "find_executable", lambda *names: "/bin/tool" if names[0] in {"bcftools", "tabix"} else "")
    monkeypatch.setattr(wgs, "run_external", lambda *a, **kw: {"ok": False, "stdout": "", "stderr": "not executable"})

    status = wgs.wgs_environment_status()

    assert status["modules"]["qc_merge"] == "BLOCKED"
    assert status["modules"]["standard_gwas"] == "PARTIAL"
    assert status["modules"]["standard_annotation"] == "PARTIAL"
    assert status["executable_capabilities"]["bcftools_modern"] is False
    assert "plink2_or_plink" in status["missing_for_standard_workflow"]


def test_vcf_dir_discovery_includes_human_prompt_and_cluster_fallback(monkeypatch):
    from biobank_agent.skills.vcf_query import _get_vcf_dirs

    monkeypatch.delenv("VC_WGS_VCF_DIR", raising=False)
    monkeypatch.delenv("VC_VIRTUAL_VCF_DIR", raising=False)

    dirs = [str(p) for p in _get_vcf_dirs()]

    assert "data/vc_wgs_vcf" in dirs
    assert "input/Files/ResultData/VirtualCell_WGS_vcf" in dirs
    assert "/Files/ResultData/BW_WGS_vcf" in dirs


def test_vcf_sample_list_discovers_repo_local_data_dir(tmp_path, monkeypatch):
    from biobank_agent.skills.vcf_query import vcf_sample_list

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VC_WGS_VCF_DIR", raising=False)
    monkeypatch.delenv("VC_VIRTUAL_VCF_DIR", raising=False)
    vcf_dir = tmp_path / "data" / "vc_wgs_vcf"
    vcf_dir.mkdir(parents=True)
    vcf = vcf_dir / "S1-30Y-F.genotyper.vcf.gz"
    vcf.touch()
    (vcf_dir / f"{vcf.name}.tbi").touch()

    result = vcf_sample_list()

    assert result["n_samples"] == 1
    assert result["n_indexed"] == 1
    assert result["samples"][0]["filename"] == "S1-30Y-F"
