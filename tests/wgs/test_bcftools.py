"""Tests for biobank_agent/utils/bcftools.py — all subprocess calls mocked."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

import biobank_agent.utils.bcftools as bcf


# ---------------------------------------------------------------------------
# bcftools_path / _tabix_path discovery
# ---------------------------------------------------------------------------

class TestBcftoolsPath:
    def setup_method(self):
        # Reset cached path between tests
        bcf._BCFTOOLS = None

    def test_found_via_which(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/bcftools" if name == "bcftools" else None)
        assert bcf.bcftools_path() == "/usr/bin/bcftools"

    def test_found_via_sys_prefix(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: None)
        fake_bin = tmp_path / "bin" / "bcftools"
        fake_bin.parent.mkdir(parents=True)
        fake_bin.touch()
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        assert bcf.bcftools_path() == str(fake_bin)

    def test_not_found_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("sys.prefix", str(tmp_path / "nonexist"))
        with pytest.raises(FileNotFoundError, match="bcftools not found"):
            bcf.bcftools_path()


class TestTabixPath:
    def test_found_via_which(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/tabix" if name == "tabix" else None)
        assert bcf._tabix_path() == "/usr/bin/tabix"

    def test_found_via_sys_prefix(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: None)
        fake_bin = tmp_path / "bin" / "tabix"
        fake_bin.parent.mkdir(parents=True)
        fake_bin.touch()
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        assert bcf._tabix_path() == str(fake_bin)

    def test_not_found_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("sys.prefix", str(tmp_path / "nonexist"))
        with pytest.raises(FileNotFoundError, match="tabix not found"):
            bcf._tabix_path()


# ---------------------------------------------------------------------------
# run_bcftools
# ---------------------------------------------------------------------------

class TestRunBcftools:
    def test_success(self, monkeypatch):
        bcf._BCFTOOLS = "/usr/bin/bcftools"
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok\n", stderr=""
        )
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: result)
        out = bcf.run_bcftools(["stats", "f.vcf"])
        assert out.returncode == 0

    def test_failure_raises(self, monkeypatch):
        bcf._BCFTOOLS = "/usr/bin/bcftools"
        result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error msg"
        )
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: result)
        with pytest.raises(RuntimeError, match="bcftools failed"):
            bcf.run_bcftools(["merge", "f.vcf"])


# ---------------------------------------------------------------------------
# merge_vcfs
# ---------------------------------------------------------------------------

class TestMergeVcfs:
    def _mock_run_and_tabix(self, monkeypatch):
        calls = []

        def fake_run(args, **kw):
            calls.append(args)
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        bcf._BCFTOOLS = "/usr/bin/bcftools"
        monkeypatch.setattr(bcf, "_tabix_path", lambda: "/usr/bin/tabix")
        monkeypatch.setattr("subprocess.run", fake_run)
        return calls

    def test_single_vcf_uses_view(self, monkeypatch, tmp_path):
        calls = self._mock_run_and_tabix(monkeypatch)
        out = bcf.merge_vcfs(["/a.vcf.gz"], tmp_path / "out.vcf.gz")
        # First call is bcftools view, second is tabix
        assert "view" in calls[0]
        assert out == tmp_path / "out.vcf.gz"

    def test_multiple_vcfs_uses_merge(self, monkeypatch, tmp_path):
        calls = self._mock_run_and_tabix(monkeypatch)
        out = bcf.merge_vcfs(["/a.vcf.gz", "/b.vcf.gz"], tmp_path / "out.vcf.gz")
        assert "merge" in calls[0]

    def test_region_flag_added(self, monkeypatch, tmp_path):
        calls = self._mock_run_and_tabix(monkeypatch)
        bcf.merge_vcfs(["/a.vcf.gz", "/b.vcf.gz"], tmp_path / "out.vcf.gz", region="chr22")
        cmd = calls[0]
        assert "-r" in cmd
        assert "chr22" in cmd

    def test_optional_cache_reuses_existing_merge(self, monkeypatch, tmp_path):
        cache_dir = tmp_path / "cache"
        in1 = tmp_path / "a.vcf.gz"
        in2 = tmp_path / "b.vcf.gz"
        in1.write_text("a")
        in2.write_text("b")
        monkeypatch.setenv("WGS_MERGE_CACHE_DIR", str(cache_dir))

        calls = self._mock_run_and_tabix(monkeypatch)
        out1 = bcf.merge_vcfs([str(in1), str(in2)], tmp_path / "out1.vcf.gz", region="chr22")
        out1.write_text("merged")
        Path(str(out1) + ".tbi").write_text("index")
        cache_key = bcf._merge_cache_key([str(in1), str(in2)], "chr22", True)
        bcf._store_cached_vcf(out1, cache_dir / f"merged_{cache_key}.vcf.gz")

        calls.clear()
        out2 = bcf.merge_vcfs([str(in1), str(in2)], tmp_path / "out2.vcf.gz", region="chr22")
        assert calls == []
        assert out2.read_text() == "merged"
        assert Path(str(out2) + ".tbi").read_text() == "index"


# ---------------------------------------------------------------------------
# filter_vcf
# ---------------------------------------------------------------------------

class TestFilterVcf:
    def _mock_run(self, monkeypatch):
        calls = []

        def fake_run(args, **kw):
            calls.append(args)
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        bcf._BCFTOOLS = "/usr/bin/bcftools"
        monkeypatch.setattr(bcf, "_tabix_path", lambda: "/usr/bin/tabix")
        monkeypatch.setattr("subprocess.run", fake_run)
        return calls

    def test_include_expr(self, monkeypatch, tmp_path):
        calls = self._mock_run(monkeypatch)
        bcf.filter_vcf("/in.vcf.gz", tmp_path / "out.vcf.gz", include_expr="QUAL>=30")
        cmd = calls[0]
        assert "-i" in cmd
        assert "QUAL>=30" in cmd

    def test_exclude_expr(self, monkeypatch, tmp_path):
        calls = self._mock_run(monkeypatch)
        bcf.filter_vcf("/in.vcf.gz", tmp_path / "out.vcf.gz", exclude_expr="MAF<0.01")
        cmd = calls[0]
        assert "-e" in cmd

    def test_region_flag_added(self, monkeypatch, tmp_path):
        calls = self._mock_run(monkeypatch)
        bcf.filter_vcf("/in.vcf.gz", tmp_path / "out.vcf.gz", include_expr="QUAL>=30", region="chr22")
        cmd = calls[0]
        assert "-r" in cmd
        assert "chr22" in cmd


# ---------------------------------------------------------------------------
# stats_vcf
# ---------------------------------------------------------------------------

class TestStatsVcf:
    def test_parsing(self, monkeypatch):
        stdout = (
            "SN\t0\tnumber of records:\t1234\n"
            "SN\t0\tnumber of SNPs:\t1000\n"
            "SN\t0\tnumber of indels:\t200\n"
            "SN\t0\tnumber of MNPs:\t10\n"
            "SN\t0\tnumber of others:\t24\n"
            "SN\t0\tnumber of samples:\t28\n"
            "TSTV\t0\t800\t200\t4.0\n"
        )
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
        bcf._BCFTOOLS = "/usr/bin/bcftools"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: result)

        parsed = bcf.stats_vcf("/fake.vcf.gz")
        assert parsed["n_records"] == 1234
        assert parsed["n_snps"] == 1000
        assert parsed["n_indels"] == 200
        assert parsed["n_mnps"] == 10
        assert parsed["n_others"] == 24
        assert parsed["n_samples"] == 28
        assert parsed["ti_tv_ratio"] == 4.0


# ---------------------------------------------------------------------------
# get_tmp_dir
# ---------------------------------------------------------------------------

class TestGetTmpDir:
    def test_creates_dir(self, tmp_path):
        ctx = SimpleNamespace(report_dir=tmp_path / "reports")
        result = bcf.get_tmp_dir(ctx)
        assert result.exists()
        assert result == tmp_path / "reports" / "tmp"

    def test_no_report_dir_uses_default(self):
        ctx = SimpleNamespace()
        result = bcf.get_tmp_dir(ctx)
        # When report_dir is None: fallback = Path("./reports/tmp"), then /tmp appended
        assert result == Path("./reports/tmp/tmp")
