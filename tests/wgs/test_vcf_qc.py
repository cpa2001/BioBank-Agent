"""Tests for biobank_agent/skills/vcf_qc.py — comprehensive branch coverage."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.wgs.conftest import (
    FakeVCF, FakeVariant, make_variants, ALL_SAMPLE_IDS, _J_IDS, _build_pheno_df,
)


# ---------------------------------------------------------------------------
# _compute_per_sample_qc  (internal function)
# ---------------------------------------------------------------------------

class TestComputePerSampleQc:
    """Tests for the internal _compute_per_sample_qc function."""

    def _patch_cyvcf2(self, monkeypatch, fake_vcf):
        monkeypatch.setattr("cyvcf2.VCF", lambda path: fake_vcf)

    def test_basic_qc_metrics(self, monkeypatch):
        from biobank_agent.skills.vcf_qc import _compute_per_sample_qc
        # 5 variants: 2 hom-ref, 1 het, 1 hom-alt, 1 missing
        vs = [
            FakeVariant(gt_types=np.array([0]), n_samples=1),  # hom-ref
            FakeVariant(gt_types=np.array([0]), n_samples=1),  # hom-ref
            FakeVariant(gt_types=np.array([1]), n_samples=1),  # het
            FakeVariant(gt_types=np.array([2]), n_samples=1),  # hom-alt
            FakeVariant(gt_types=np.array([3]), n_samples=1),  # missing
        ]
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=["S1"], variants=vs))
        result = _compute_per_sample_qc("/f.vcf.gz", "S1")
        assert result["n_total_variants"] == 5
        assert result["n_hom_ref"] == 2
        assert result["n_het"] == 1
        assert result["n_hom_alt"] == 1
        assert result["n_missing"] == 1
        assert result["call_rate"] == 0.8  # 4/5

    def test_het_ratio_includes_hom_ref(self, monkeypatch):
        """het_ratio denominator includes n_hom_ref."""
        from biobank_agent.skills.vcf_qc import _compute_per_sample_qc
        # 2 hom-ref, 1 het, 1 hom-alt → het_ratio = 1/(1+2+1) = 0.25
        vs = [
            FakeVariant(gt_types=np.array([0]), n_samples=1),
            FakeVariant(gt_types=np.array([0]), n_samples=1),
            FakeVariant(gt_types=np.array([1]), n_samples=1),
            FakeVariant(gt_types=np.array([2]), n_samples=1),
        ]
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=["S1"], variants=vs))
        result = _compute_per_sample_qc("/f.vcf.gz", "S1")
        assert result["het_ratio"] == 0.25

    def test_numpy_types_are_native_float(self, monkeypatch):
        """no numpy types in output."""
        from biobank_agent.skills.vcf_qc import _compute_per_sample_qc
        vs = make_variants(n=20, n_samples=1)
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=["S1"], variants=vs))
        result = _compute_per_sample_qc("/f.vcf.gz", "S1")
        for key, val in result.items():
            if val is not None and key != "sample_id" and key != "variant_only_vcf":
                assert type(val) in (int, float, bool), f"{key} is {type(val)}"

    def test_all_missing_genotypes(self, monkeypatch):
        from biobank_agent.skills.vcf_qc import _compute_per_sample_qc
        vs = [FakeVariant(gt_types=np.array([3]), n_samples=1) for _ in range(5)]
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=["S1"], variants=vs))
        result = _compute_per_sample_qc("/f.vcf.gz", "S1")
        assert result["call_rate"] == 0.0
        assert result["het_ratio"] == 0.0

    def test_no_dp_gq_fields(self, monkeypatch):
        from biobank_agent.skills.vcf_qc import _compute_per_sample_qc

        class NoDpGqVariant(FakeVariant):
            def format(self, tag):
                raise KeyError(tag)

        vs = [NoDpGqVariant(gt_types=np.array([1]), n_samples=1) for _ in range(3)]
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=["S1"], variants=vs))
        result = _compute_per_sample_qc("/f.vcf.gz", "S1")
        assert result["mean_dp"] is None
        assert result["mean_gq"] is None

    def test_ti_tv_computation(self, monkeypatch):
        from biobank_agent.skills.vcf_qc import _compute_per_sample_qc
        # 3 transitions (A→G), 1 transversion (A→T)
        vs = [
            FakeVariant(ref="A", alt=["G"], gt_types=np.array([1]), n_samples=1),
            FakeVariant(ref="A", alt=["G"], gt_types=np.array([1]), n_samples=1),
            FakeVariant(ref="A", alt=["G"], gt_types=np.array([1]), n_samples=1),
            FakeVariant(ref="A", alt=["T"], gt_types=np.array([1]), n_samples=1),
        ]
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=["S1"], variants=vs))
        result = _compute_per_sample_qc("/f.vcf.gz", "S1")
        assert result["ti_tv_ratio"] == 3.0

    def test_ti_tv_zero_transversions(self, monkeypatch):
        from biobank_agent.skills.vcf_qc import _compute_per_sample_qc
        # Only transitions → tv=0 → ti_tv_ratio=None (nan→None)
        vs = [FakeVariant(ref="A", alt=["G"], gt_types=np.array([1]), n_samples=1) for _ in range(3)]
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=["S1"], variants=vs))
        result = _compute_per_sample_qc("/f.vcf.gz", "S1")
        assert result["ti_tv_ratio"] is None

    def test_variant_only_vcf_detection(self, monkeypatch):
        from biobank_agent.skills.vcf_qc import _compute_per_sample_qc
        # No hom-ref → variant_only_vcf=True
        vs = [
            FakeVariant(gt_types=np.array([1]), n_samples=1),
            FakeVariant(gt_types=np.array([2]), n_samples=1),
        ]
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=["S1"], variants=vs))
        result = _compute_per_sample_qc("/f.vcf.gz", "S1")
        assert result["variant_only_vcf"] is True

    def test_region_param_forwards_to_vcf(self, monkeypatch):
        from biobank_agent.skills.vcf_qc import _compute_per_sample_qc
        called_regions = []

        class RegionVCF(FakeVCF):
            def __call__(self, region=None):
                called_regions.append(region)
                return iter(self._variants)

        monkeypatch.setattr("cyvcf2.VCF", lambda path: RegionVCF(samples=["S1"], variants=[]))
        _compute_per_sample_qc("/f.vcf.gz", "S1", region="chr22")
        assert called_regions == ["chr22"]

    def test_max_variants_limit(self, monkeypatch):
        from biobank_agent.skills.vcf_qc import _compute_per_sample_qc
        vs = [FakeVariant(gt_types=np.array([1]), n_samples=1) for _ in range(100)]
        self._patch_cyvcf2(monkeypatch, FakeVCF(samples=["S1"], variants=vs))
        result = _compute_per_sample_qc("/f.vcf.gz", "S1", max_variants=10)
        assert result["n_total_variants"] == 10


# ---------------------------------------------------------------------------
# vcf_qc (main skill function)
# ---------------------------------------------------------------------------

class TestVcfQcSkill:
    """Tests for the vcf_qc skill entry point."""

    def _setup_mocks(self, monkeypatch, mock_ctx, n_variants=50, extra_qc=None):
        """Common setup: patch _build_vcf_dm, _compute_per_sample_qc."""
        vcf_files = [{"filename": s, "vcf_path": f"/fake/{s}.vcf.gz"} for s in ALL_SAMPLE_IDS]
        vcf_dm = SimpleNamespace(list_vcf_files=lambda: vcf_files)
        monkeypatch.setattr(
            "biobank_agent.skills.vcf_query._build_vcf_dm",
            lambda ctx: vcf_dm,
        )

        def fake_compute(vcf_path, sample_id, region=None, max_variants=200000):
            if extra_qc and sample_id in extra_qc:
                return extra_qc[sample_id]
            return {
                "sample_id": sample_id,
                "n_total_variants": n_variants,
                "n_pass": n_variants - 2,
                "n_missing": 2,
                "n_het": 15,
                "n_hom_alt": 8,
                "n_hom_ref": 25,
                "call_rate": 0.96,
                "het_ratio": 0.3125,
                "ti_tv_ratio": 2.1,
                "mean_dp": 30.0,
                "median_dp": 28.0,
                "mean_gq": 40.0,
                "median_gq": 38.0,
                "variant_only_vcf": False,
            }

        monkeypatch.setattr("biobank_agent.skills.vcf_qc._compute_per_sample_qc", fake_compute)

    def test_no_vcf_files(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc
        vcf_dm = SimpleNamespace(list_vcf_files=lambda: [])
        monkeypatch.setattr("biobank_agent.skills.vcf_query._build_vcf_dm", lambda ctx: vcf_dm)
        result = vcf_qc(ctx=mock_ctx)
        assert "error" in result

    def test_all_samples_analyzed(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc
        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_qc(ctx=mock_ctx)
        assert result["n_samples_analyzed"] == 28

    def test_sample_ids_filter(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc
        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_qc(sample_ids="J001,J002", ctx=mock_ctx)
        assert result["n_samples_analyzed"] == 2

    def test_outlier_detection_3sd(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc
        # Inject one outlier with extremely low call_rate
        extra = {
            "J001": {
                "sample_id": "J001",
                "n_total_variants": 50, "n_pass": 48,
                "n_missing": 25, "n_het": 10, "n_hom_alt": 5, "n_hom_ref": 10,
                "call_rate": 0.5,  # way below normal 0.96
                "het_ratio": 0.3, "ti_tv_ratio": 2.0,
                "mean_dp": 30.0, "median_dp": 28.0,
                "mean_gq": 40.0, "median_gq": 38.0,
                "variant_only_vcf": False,
            }
        }
        self._setup_mocks(monkeypatch, mock_ctx, extra_qc=extra)
        result = vcf_qc(ctx=mock_ctx)
        assert "J001" in result["outlier_samples"]

    def test_no_outliers_when_uniform(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc
        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_qc(ctx=mock_ctx)
        assert result["outlier_samples"] == []

    def test_adaptive_call_rate_threshold(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc

        # Make all samples have low call_rate (0.7) so cohort mean < default 0.90
        def fake_compute(vcf_path, sample_id, region=None, max_variants=200000):
            return {
                "sample_id": sample_id,
                "n_total_variants": 50, "n_pass": 48,
                "n_missing": 5, "n_het": 15, "n_hom_alt": 8, "n_hom_ref": 22,
                "call_rate": 0.70,
                "het_ratio": 0.3, "ti_tv_ratio": 2.0,
                "mean_dp": 30.0, "median_dp": 28.0,
                "mean_gq": 40.0, "median_gq": 38.0,
                "variant_only_vcf": False,
            }

        vcf_files = [{"filename": s, "vcf_path": f"/fake/{s}.vcf.gz"} for s in ALL_SAMPLE_IDS]
        vcf_dm = SimpleNamespace(list_vcf_files=lambda: vcf_files)
        monkeypatch.setattr("biobank_agent.skills.vcf_query._build_vcf_dm", lambda ctx: vcf_dm)
        monkeypatch.setattr("biobank_agent.skills.vcf_qc._compute_per_sample_qc", fake_compute)

        result = vcf_qc(min_call_rate=0.90, ctx=mock_ctx)
        # With adaptive threshold, samples with 0.70 should pass since cohort mean is 0.70
        assert result["n_samples_pass"] > 0

    def test_qc_pass_fail_counts(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc
        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_qc(ctx=mock_ctx)
        assert result["n_samples_pass"] + result["n_samples_fail"] == result["n_samples_analyzed"]

    def test_hwe_test_skips_without_region(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc
        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_qc(region="", ctx=mock_ctx)
        assert result["hwe_fail_count"] is None

    def test_hwe_test_runs_with_region(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc
        self._setup_mocks(monkeypatch, mock_ctx)

        # Patch merge_vcfs for HWE
        monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs",
                            lambda paths, out, region=None: Path(out).touch() or Path(out))

        # Patch cyvcf2 for HWE iteration
        hwe_variants = []
        for i in range(100):
            gt = np.array([0]*10 + [1]*10 + [2]*8, dtype=np.int8)
            hwe_variants.append(FakeVariant(
                chrom="chr22", pos=16000000 + i, ref="A", alt=["T"],
                gt_types=gt, n_samples=28,
            ))

        monkeypatch.setattr("cyvcf2.VCF",
                            lambda path: FakeVCF(samples=ALL_SAMPLE_IDS, variants=hwe_variants))

        result = vcf_qc(region="chr22", ctx=mock_ctx)
        assert result["hwe_fail_count"] is not None
        assert isinstance(result["hwe_fail_count"], int)

    def test_hwe_test_skips_on_error(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc
        self._setup_mocks(monkeypatch, mock_ctx)

        # merge_vcfs raises for HWE
        def raise_merge(*a, **kw):
            raise RuntimeError("merge fail")

        monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs", raise_merge)

        result = vcf_qc(region="chr22", ctx=mock_ctx)
        assert result["hwe_fail_count"] is None

    def test_output_filtered(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc
        self._setup_mocks(monkeypatch, mock_ctx)

        def fake_filter(inp, out, include_expr=None):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).touch()
            return Path(out)

        monkeypatch.setattr("biobank_agent.utils.bcftools.filter_vcf", fake_filter)
        result = vcf_qc(output_filtered=True, ctx=mock_ctx)
        assert result["filtered_vcf_dir"] is not None

    def test_phenotype_group_coloring(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc
        self._setup_mocks(monkeypatch, mock_ctx)
        # The default pheno_df has phenotype_group column
        result = vcf_qc(ctx=mock_ctx)
        assert len(result["figures"]) > 0

    def test_no_phenotype_group(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc
        self._setup_mocks(monkeypatch, mock_ctx)
        # Remove phenotype_group from dm response
        df_no_grp = pd.DataFrame({"sample_id": ALL_SAMPLE_IDS, "age": [50] * 28})
        mock_ctx.dm.query = lambda sql, *a, **kw: df_no_grp
        result = vcf_qc(ctx=mock_ctx)
        assert len(result["figures"]) > 0

    def test_variant_only_vcf_note(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc
        extra = {
            "J001": {
                "sample_id": "J001",
                "n_total_variants": 50, "n_pass": 48,
                "n_missing": 0, "n_het": 30, "n_hom_alt": 20, "n_hom_ref": 0,
                "call_rate": 1.0, "het_ratio": 0.6,
                "ti_tv_ratio": 2.0, "mean_dp": 30.0, "median_dp": 28.0,
                "mean_gq": 40.0, "median_gq": 38.0,
                "variant_only_vcf": True,
            }
        }
        self._setup_mocks(monkeypatch, mock_ctx, extra_qc=extra)
        result = vcf_qc(ctx=mock_ctx)
        assert result["variant_only_vcf_note"] is not None
        assert "variant-only" in result["variant_only_vcf_note"].lower()

    def test_figures_stored_in_state(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc
        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_qc(ctx=mock_ctx)
        assert len(mock_ctx.state.figures) > 0

    def test_custom_data_stored(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc
        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_qc(ctx=mock_ctx)
        assert "vcf_qc_results" in mock_ctx.state.custom_data
        qc_data = mock_ctx.state.custom_data["vcf_qc_results"]
        assert "per_sample" in qc_data
        assert "pass_samples" in qc_data
        assert "fail_samples" in qc_data

    def test_qc_results_empty_when_all_fail(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_qc import vcf_qc

        def raise_for_all(vcf_path, sample_id, region=None, max_variants=200000):
            raise RuntimeError("VCF corrupt")

        vcf_files = [{"filename": s, "vcf_path": f"/fake/{s}.vcf.gz"} for s in ALL_SAMPLE_IDS]
        vcf_dm = SimpleNamespace(list_vcf_files=lambda: vcf_files)
        monkeypatch.setattr("biobank_agent.skills.vcf_query._build_vcf_dm", lambda ctx: vcf_dm)
        monkeypatch.setattr("biobank_agent.skills.vcf_qc._compute_per_sample_qc", raise_for_all)
        result = vcf_qc(ctx=mock_ctx)
        assert "error" in result
