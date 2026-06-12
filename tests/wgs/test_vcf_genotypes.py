"""Tests for biobank_agent/utils/vcf_genotypes.py — cyvcf2 fully mocked."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.wgs.conftest import FakeVCF, FakeVariant, make_variants


# ---------------------------------------------------------------------------
# build_genotype_matrix
# ---------------------------------------------------------------------------

class TestBuildGenotypeMatrix:
    def _patch_cyvcf2(self, monkeypatch, fake_vcf):
        monkeypatch.setattr("cyvcf2.VCF", lambda path: fake_vcf)

    def test_basic_shape(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        samples = ["S1", "S2", "S3"]
        vs = [
            FakeVariant(ref="A", alt=["T"], gt_types=np.array([0, 1, 2], dtype=np.int8),
                        n_samples=3)
            for _ in range(20)
        ]
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=samples, variants=vs))
        G, vids, out_s = mod.build_genotype_matrix("/fake.vcf.gz", maf_min=0.0, maf_max=1.0)
        assert G.shape[0] == 3
        assert G.shape[1] == 20
        assert out_s == samples

    def test_maf_filter_low(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        samples = ["S1", "S2", "S3", "S4", "S5"]
        # All hom-ref → alt_freq=0 → filtered by maf_min=0.05
        vs = [FakeVariant(ref="A", alt=["T"], gt_types=np.array([0, 0, 0, 0, 0], dtype=np.int8))]
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=samples, variants=vs))
        G, vids, _ = mod.build_genotype_matrix("/f.vcf.gz", maf_min=0.05, maf_max=0.95)
        assert G.shape[1] == 0

    def test_maf_filter_high(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        samples = ["S1", "S2", "S3", "S4", "S5"]
        # All hom-alt → alt_freq=1.0 → filtered by maf_max=0.95
        vs = [FakeVariant(ref="A", alt=["T"], gt_types=np.array([2, 2, 2, 2, 2], dtype=np.int8))]
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=samples, variants=vs))
        G, vids, _ = mod.build_genotype_matrix("/f.vcf.gz", maf_min=0.0, maf_max=0.95)
        assert G.shape[1] == 0

    def test_snv_only_skips_indels(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        samples = ["S1", "S2", "S3"]
        indel = FakeVariant(ref="AT", alt=["A"], gt_types=np.array([0, 1, 2], dtype=np.int8))
        snp = FakeVariant(ref="A", alt=["T"], gt_types=np.array([0, 1, 2], dtype=np.int8))
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=samples, variants=[indel, snp]))
        G, vids, _ = mod.build_genotype_matrix("/f.vcf.gz", maf_min=0.0, maf_max=1.0, snv_only=True)
        assert G.shape[1] == 1

    def test_snv_only_false_includes_indels(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        samples = ["S1", "S2", "S3"]
        indel = FakeVariant(ref="AT", alt=["A"], gt_types=np.array([0, 1, 2], dtype=np.int8))
        snp = FakeVariant(ref="A", alt=["T"], gt_types=np.array([0, 1, 2], dtype=np.int8))
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=samples, variants=[indel, snp]))
        G, vids, _ = mod.build_genotype_matrix("/f.vcf.gz", maf_min=0.0, maf_max=1.0, snv_only=False)
        assert G.shape[1] == 2

    def test_max_variants_limit(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        samples = ["S1", "S2", "S3"]
        vs = [FakeVariant(ref="A", alt=["T"], gt_types=np.array([0, 1, 2], dtype=np.int8))
              for _ in range(50)]
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=samples, variants=vs))
        G, vids, _ = mod.build_genotype_matrix("/f.vcf.gz", maf_min=0.0, maf_max=1.0,
                                                max_variants=10)
        assert G.shape[1] == 10

    def test_missing_genotype_imputation(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        samples = ["S1", "S2", "S3", "S4"]
        # S1=0, S2=2, S3=missing, S4=2 → mean of non-missing = (0+2+2)/3 = 1.333
        v = FakeVariant(ref="A", alt=["T"], gt_types=np.array([0, 2, 3, 2], dtype=np.int8))
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=samples, variants=[v]))
        G, _, _ = mod.build_genotype_matrix("/f.vcf.gz", maf_min=0.0, maf_max=1.0)
        assert not np.isnan(G[2, 0])
        np.testing.assert_almost_equal(G[2, 0], (0 + 2 + 2) / 3, decimal=2)

    def test_empty_vcf(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        samples = ["S1", "S2"]
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=samples, variants=[]))
        G, vids, s = mod.build_genotype_matrix("/f.vcf.gz")
        assert G.shape == (2, 0)
        assert vids == []

    def test_region_param(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        called_with_region = []
        samples = ["S1", "S2", "S3"]

        class RegionVCF(FakeVCF):
            def __call__(self, region=None):
                called_with_region.append(region)
                return iter(self._variants)

        vs = [FakeVariant(ref="A", alt=["T"], gt_types=np.array([0, 1, 2], dtype=np.int8))]
        monkeypatch.setattr("cyvcf2.VCF",
                            lambda path: RegionVCF(samples=samples, variants=vs))
        mod.build_genotype_matrix("/f.vcf.gz", maf_min=0.0, maf_max=1.0, region="chr22:1-1000")
        assert called_with_region == ["chr22:1-1000"]

    def test_no_alt_skipped(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        samples = ["S1", "S2", "S3"]
        no_alt = FakeVariant(ref="A", alt=["."], gt_types=np.array([0, 0, 0], dtype=np.int8))
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=samples, variants=[no_alt]))
        G, _, _ = mod.build_genotype_matrix("/f.vcf.gz", maf_min=0.0, maf_max=1.0)
        assert G.shape[1] == 0

    def test_low_call_rate_skipped(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        samples = ["S1", "S2", "S3", "S4"]
        # 3 out of 4 missing → <50% valid → skipped
        v = FakeVariant(ref="A", alt=["T"], gt_types=np.array([1, 3, 3, 3], dtype=np.int8))
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=samples, variants=[v]))
        G, _, _ = mod.build_genotype_matrix("/f.vcf.gz", maf_min=0.0, maf_max=1.0)
        assert G.shape[1] == 0


# ---------------------------------------------------------------------------
# build_allele_counts
# ---------------------------------------------------------------------------

class TestBuildAlleleCounts:
    def _patch(self, monkeypatch, fake_vcf):
        monkeypatch.setattr("cyvcf2.VCF", lambda path: fake_vcf)

    def test_basic_counts(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        samples = ["C1", "C2", "T1", "T2"]
        # C1=het(1), C2=hom-alt(2), T1=hom-ref(0), T2=het(1)
        v = FakeVariant(ref="A", alt=["T"],
                        gt_types=np.array([1, 2, 0, 1], dtype=np.int8),
                        pos=1000)
        self._patch(monkeypatch, FakeVCF(samples=samples, variants=[v]))
        results = mod.build_allele_counts("/f.vcf.gz", ["C1", "C2"], ["T1", "T2"])
        assert len(results) == 1
        r = results[0]
        assert r["case_alt"] == 3  # 1 + 2
        assert r["case_ref"] == 1  # 2*2 - 3
        assert r["ctrl_alt"] == 1  # 0 + 1
        assert r["ctrl_ref"] == 3  # 2*2 - 1

    def test_case_not_in_vcf(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        samples = ["X1", "X2"]
        self._patch(monkeypatch, FakeVCF(samples=samples, variants=[]))
        results = mod.build_allele_counts("/f.vcf.gz", ["MISSING"], ["X1"])
        assert results == []

    def test_maf_filter(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        samples = ["C1", "C2", "T1", "T2"]
        # All hom-ref except one het → maf very low
        v = FakeVariant(ref="A", alt=["T"],
                        gt_types=np.array([0, 0, 0, 1], dtype=np.int8))
        self._patch(monkeypatch, FakeVCF(samples=samples, variants=[v]))
        results = mod.build_allele_counts("/f.vcf.gz", ["C1", "C2"], ["T1", "T2"],
                                          maf_min=0.2)
        assert len(results) == 0

    def test_indels_skipped(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        samples = ["C1", "T1"]
        indel = FakeVariant(ref="AT", alt=["A"],
                            gt_types=np.array([1, 0], dtype=np.int8))
        self._patch(monkeypatch, FakeVCF(samples=samples, variants=[indel]))
        results = mod.build_allele_counts("/f.vcf.gz", ["C1"], ["T1"])
        assert len(results) == 0

    def test_max_variants(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        samples = ["C1", "C2", "T1", "T2"]
        vs = [FakeVariant(ref="A", alt=["T"],
                          gt_types=np.array([0, 1, 2, 1], dtype=np.int8),
                          pos=i * 100)
              for i in range(50)]
        self._patch(monkeypatch, FakeVCF(samples=samples, variants=vs))
        results = mod.build_allele_counts("/f.vcf.gz", ["C1", "C2"], ["T1", "T2"],
                                          max_variants=5)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# get_sample_vcf_paths
# ---------------------------------------------------------------------------

class TestGetSampleVcfPaths:
    def test_from_query(self):
        import biobank_agent.utils.vcf_genotypes as mod
        df = pd.DataFrame({"sample_id": ["A", "B"], "vcf_path": ["/a.vcf", "/b.vcf"]})
        dm = SimpleNamespace(query=lambda sql: df)
        ctx = SimpleNamespace(dm=dm)
        paths = mod.get_sample_vcf_paths(ctx)
        assert paths == {"A": "/a.vcf", "B": "/b.vcf"}

    def test_fallback_on_query_failure(self, monkeypatch):
        import biobank_agent.utils.vcf_genotypes as mod
        dm = SimpleNamespace(query=lambda sql: (_ for _ in ()).throw(Exception("no table")))

        monkeypatch.setattr(
            "biobank_agent.data.vcf_loader.discover_vcf_files",
            lambda *a: [{"filename": "S1", "vcf_path": "/s1.vcf.gz"}],
        )
        monkeypatch.setattr(
            "biobank_agent.skills.vcf_query._get_vcf_dirs",
            lambda ctx=None: ["/fake/dir"],
        )
        ctx = SimpleNamespace(dm=dm)
        paths = mod.get_sample_vcf_paths(ctx)
        assert paths == {"S1": "/s1.vcf.gz"}
