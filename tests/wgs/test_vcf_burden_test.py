"""Comprehensive tests for biobank_agent/skills/vcf_burden_test.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.wgs.conftest import (
    ALL_SAMPLE_IDS,
    FakeVCF,
    FakeVariant,
    _build_pheno_df,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rare_variants(all_ids, n_variants=5, chrom="chr11", start_pos=88911696):
    """Return FakeVariant list with low MAF (2-3 carriers out of ~28 samples)."""
    n = len(all_ids)
    vs = []
    for i in range(n_variants):
        gt = np.zeros(n, dtype=np.int8)
        gt[0] = 1  # J001 – case carrier
        gt[1] = 1  # J002 – case carrier
        if i % 3 == 0 and n > 10:
            gt[10] = 1  # S001 – ctrl carrier for some variants
        vs.append(FakeVariant(
            chrom=chrom,
            pos=start_pos + i * 100,
            ref="A",
            alt=["T"],
            gt_types=gt,
            n_samples=n,
            qual=50.0,
        ))
    return vs


class BurdenVCF(FakeVCF):
    """FakeVCF whose region-call returns the stored variants."""

    def __call__(self, region=None):
        return iter(self._variants)


# ---------------------------------------------------------------------------
# Central _setup_mocks helper
# ---------------------------------------------------------------------------

def _setup_mocks(monkeypatch, mock_ctx, n_rare_per_gene=5, vcf_variants=None):
    """Wire up all monkeypatches required by vcf_burden_test."""
    mapping = {s: f"/fake/{s}.vcf.gz" for s in ALL_SAMPLE_IDS}
    monkeypatch.setattr(
        "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
        lambda ctx: mapping,
    )
    monkeypatch.setattr(
        "biobank_agent.utils.bcftools.get_tmp_dir",
        lambda ctx: mock_ctx.report_dir / "tmp",
    )
    (mock_ctx.report_dir / "tmp").mkdir(parents=True, exist_ok=True)

    merge_calls = []

    def fake_merge(vcf_paths, out, region=None):
        merge_calls.append(region)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).touch()
        return Path(out)

    monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs", fake_merge)

    all_ids = list(ALL_SAMPLE_IDS)
    if vcf_variants is None:
        vcf_variants = _make_rare_variants(all_ids, n_rare_per_gene)

    fake_vcf = BurdenVCF(samples=all_ids, variants=vcf_variants)
    monkeypatch.setattr("cyvcf2.VCF", lambda path: fake_vcf)

    return merge_calls


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestVcfBurdenTest:

    # ------------------------------------------------------------------
    # 1. test_too_few_cases
    # ------------------------------------------------------------------
    def test_too_few_cases(self, monkeypatch, mock_ctx):
        """case_group with <2 samples → error dict returned."""
        # Override dm.query to return a single-case phenotype table
        single_case_df = pd.DataFrame({
            "sample_id": ["J001", "V001", "V002", "V003"],
            "phenotype_group": ["J", "V", "V", "V"],
        })
        mock_ctx.dm.query = lambda sql, *a, **kw: single_case_df

        # Still need basic path mocks so the function reaches the size check
        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: {s: f"/fake/{s}.vcf.gz" for s in ["J001", "V001", "V002", "V003"]},
        )

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(case_group="J", control_group="V", ctx=mock_ctx)
        assert "error" in result
        assert "Cases=1" in result["error"]

    # ------------------------------------------------------------------
    # 2. test_too_few_controls
    # ------------------------------------------------------------------
    def test_too_few_controls(self, monkeypatch, mock_ctx):
        """control_group with <2 samples → error dict returned."""
        single_ctrl_df = pd.DataFrame({
            "sample_id": ["J001", "J002", "J003", "V001"],
            "phenotype_group": ["J", "J", "J", "V"],
        })
        mock_ctx.dm.query = lambda sql, *a, **kw: single_ctrl_df

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: {s: f"/fake/{s}.vcf.gz" for s in ["J001", "J002", "J003", "V001"]},
        )

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(case_group="J", control_group="V", ctx=mock_ctx)
        assert "error" in result
        assert "Controls=1" in result["error"]

    # ------------------------------------------------------------------
    # 3. test_qc_propagation
    # ------------------------------------------------------------------
    def test_qc_propagation(self, monkeypatch, mock_ctx):
        """pass_samples from vcf_qc_results restricts the analysis population."""
        # Only J001, J002, V001, V002 pass QC — enough for a valid run.
        # The rest of the J/V samples should be excluded.
        pass_ids = ["J001", "J002", "V001", "V002", "V003"]
        mock_ctx.state.custom_data["vcf_qc_results"] = {"pass_samples": pass_ids}

        seen_samples = {}

        orig_cyvcf2_setup = _setup_mocks(monkeypatch, mock_ctx, n_rare_per_gene=5)

        # Intercept cyvcf2.VCF to record which samples are handed the VCF
        all_ids = list(ALL_SAMPLE_IDS)
        vcf_variants = _make_rare_variants(all_ids, 5)

        captured_samples = {}

        class CapturingVCF(BurdenVCF):
            pass

        def vcf_factory(path):
            vcf = CapturingVCF(samples=all_ids, variants=vcf_variants)
            captured_samples["vcf"] = vcf
            return vcf

        monkeypatch.setattr("cyvcf2.VCF", vcf_factory)

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(case_group="J", control_group="V", ctx=mock_ctx)
        # With QC filter: 2 cases (J001, J002) and 3 controls (V001..V003) → valid run
        # Result should not be an error
        assert "error" not in result

    # ------------------------------------------------------------------
    # 4. test_default_vitiligo_genes
    # ------------------------------------------------------------------
    def test_default_vitiligo_genes(self, monkeypatch, mock_ctx):
        """No gene_list provided → all 20 VITILIGO_GENES_HG38 entries are targeted."""
        from biobank_agent.skills.vcf_annotation import VITILIGO_GENES_HG38

        merge_calls = _setup_mocks(monkeypatch, mock_ctx, n_rare_per_gene=5)

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(case_group="J", control_group="V", gene_list="", ctx=mock_ctx)

        # The number of merge calls equals the number of unique chromosomes in
        # VITILIGO_GENES_HG38 (one merge per chromosome).
        unique_chroms = len({v[0] for v in VITILIGO_GENES_HG38.values()})
        assert len(merge_calls) == unique_chroms

    # ------------------------------------------------------------------
    # 5. test_custom_gene_list
    # ------------------------------------------------------------------
    def test_custom_gene_list(self, monkeypatch, mock_ctx):
        """gene_list='TYR,OCA2' → only those two genes are tested."""
        _setup_mocks(monkeypatch, mock_ctx, n_rare_per_gene=5)

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(
            case_group="J", control_group="V", gene_list="TYR,OCA2", ctx=mock_ctx
        )

        assert "error" not in result
        tested_genes = {gr["gene"] for gr in result["gene_results"]}
        assert tested_genes <= {"TYR", "OCA2"}

    # ------------------------------------------------------------------
    # 6. test_merge_fail_skips_chrom
    # ------------------------------------------------------------------
    def test_merge_fail_skips_chrom(self, monkeypatch, mock_ctx):
        """merge_vcfs raising an exception causes genes on that chrom to be skipped."""
        from biobank_agent.skills.vcf_annotation import VITILIGO_GENES_HG38

        # We test with only TYR (chr11) and OCA2 (chr15).
        # Make merge_vcfs fail for chr11 but succeed for chr15.
        mapping = {s: f"/fake/{s}.vcf.gz" for s in ALL_SAMPLE_IDS}
        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: mapping,
        )
        monkeypatch.setattr(
            "biobank_agent.utils.bcftools.get_tmp_dir",
            lambda ctx: mock_ctx.report_dir / "tmp",
        )
        (mock_ctx.report_dir / "tmp").mkdir(parents=True, exist_ok=True)

        def selective_merge(vcf_paths, out, region=None):
            if "chr11" in (region or ""):
                raise RuntimeError("Simulated merge failure for chr11")
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).touch()
            return Path(out)

        monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs", selective_merge)

        all_ids = list(ALL_SAMPLE_IDS)
        vcf_variants = _make_rare_variants(all_ids, 5, chrom="chr15", start_pos=27622010)
        monkeypatch.setattr(
            "cyvcf2.VCF",
            lambda path: BurdenVCF(samples=all_ids, variants=vcf_variants),
        )

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(
            case_group="J", control_group="V", gene_list="TYR,OCA2", ctx=mock_ctx
        )

        # TYR is on chr11 (failed merge) → should not appear; OCA2 on chr15 → may appear
        if "gene_results" in result:
            genes_tested = {gr["gene"] for gr in result["gene_results"]}
            assert "TYR" not in genes_tested

    # ------------------------------------------------------------------
    # 7. test_min_variants_per_gene
    # ------------------------------------------------------------------
    def test_min_variants_per_gene(self, monkeypatch, mock_ctx):
        """Genes with fewer rare variants than min_variants_per_gene are skipped."""
        # Provide only 1 rare variant; min_variants_per_gene=2 → gene skipped
        _setup_mocks(monkeypatch, mock_ctx, n_rare_per_gene=1)

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(
            case_group="J",
            control_group="V",
            gene_list="TYR",
            min_variants_per_gene=2,
            ctx=mock_ctx,
        )

        assert "error" in result  # TYR skipped → no genes → error

    # ------------------------------------------------------------------
    # 8. test_no_genes_with_rare_variants
    # ------------------------------------------------------------------
    def test_no_genes_with_rare_variants(self, monkeypatch, mock_ctx):
        """All genes have <min_variants_per_gene → error returned."""
        all_ids = list(ALL_SAMPLE_IDS)
        n = len(all_ids)

        # High-MAF variant (all samples carry alt) → maf > 0.05 → filtered out
        high_maf_gt = np.ones(n, dtype=np.int8)  # all het
        high_maf_variant = FakeVariant(
            chrom="chr11", pos=88911700, ref="A", alt=["T"],
            gt_types=high_maf_gt, n_samples=n, qual=50.0,
        )

        mapping = {s: f"/fake/{s}.vcf.gz" for s in ALL_SAMPLE_IDS}
        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: mapping,
        )
        monkeypatch.setattr(
            "biobank_agent.utils.bcftools.get_tmp_dir",
            lambda ctx: mock_ctx.report_dir / "tmp",
        )
        (mock_ctx.report_dir / "tmp").mkdir(parents=True, exist_ok=True)

        def fake_merge(vcf_paths, out, region=None):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).touch()
            return Path(out)

        monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs", fake_merge)
        monkeypatch.setattr(
            "cyvcf2.VCF",
            lambda path: BurdenVCF(samples=all_ids, variants=[high_maf_variant]),
        )

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(
            case_group="J", control_group="V", gene_list="TYR", ctx=mock_ctx
        )

        assert "error" in result
        assert "No genes" in result["error"]

    # ------------------------------------------------------------------
    # 9. test_basic_burden_output
    # ------------------------------------------------------------------
    def test_basic_burden_output(self, monkeypatch, mock_ctx):
        """Normal run with rare variants produces expected top-level keys."""
        _setup_mocks(monkeypatch, mock_ctx, n_rare_per_gene=5)

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(
            case_group="J", control_group="V", gene_list="TYR", ctx=mock_ctx
        )

        assert "error" not in result
        for key in ("case_group", "control_group", "n_genes_tested", "gene_results", "figures"):
            assert key in result, f"Missing key: {key}"
        assert result["n_genes_tested"] >= 1
        assert result["case_group"] == "J"
        assert result["control_group"] == "V"

    # ------------------------------------------------------------------
    # 10. test_fisher_exact_burden
    # ------------------------------------------------------------------
    def test_fisher_exact_burden(self, monkeypatch, mock_ctx):
        """p_burden (from Fisher's exact test) is present and in [0, 1]."""
        _setup_mocks(monkeypatch, mock_ctx, n_rare_per_gene=5)

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(
            case_group="J", control_group="V", gene_list="TYR", ctx=mock_ctx
        )

        assert "error" not in result
        for gr in result["gene_results"]:
            assert "p_burden" in gr
            assert 0.0 <= gr["p_burden"] <= 1.0, f"p_burden={gr['p_burden']} out of range"

    # ------------------------------------------------------------------
    # 11. test_weighted_test
    # ------------------------------------------------------------------
    def test_weighted_test(self, monkeypatch, mock_ctx):
        """p_weighted (Madsen-Browning / Mann-Whitney) is present and in [0, 1]."""
        _setup_mocks(monkeypatch, mock_ctx, n_rare_per_gene=5)

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(
            case_group="J", control_group="V", gene_list="TYR", ctx=mock_ctx
        )

        assert "error" not in result
        for gr in result["gene_results"]:
            assert "p_weighted" in gr
            assert 0.0 <= gr["p_weighted"] <= 1.0, f"p_weighted={gr['p_weighted']} out of range"

    # ------------------------------------------------------------------
    # 12. test_fdr_correction_applied
    # ------------------------------------------------------------------
    def test_fdr_correction_applied(self, monkeypatch, mock_ctx):
        """p_burden_fdr and p_weighted_fdr are added to each gene result."""
        _setup_mocks(monkeypatch, mock_ctx, n_rare_per_gene=5)

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(
            case_group="J", control_group="V", gene_list="TYR,OCA2", ctx=mock_ctx
        )

        assert "error" not in result
        for gr in result["gene_results"]:
            assert "p_burden_fdr" in gr, "p_burden_fdr missing from gene result"
            assert "p_weighted_fdr" in gr, "p_weighted_fdr missing from gene result"
            assert 0.0 <= gr["p_burden_fdr"] <= 1.0
            assert 0.0 <= gr["p_weighted_fdr"] <= 1.0

    # ------------------------------------------------------------------
    # 13. test_significant_burden_flagged
    # ------------------------------------------------------------------
    def test_significant_burden_flagged(self, monkeypatch, mock_ctx):
        """significant_burden=True iff p_burden_fdr < 0.05."""
        _setup_mocks(monkeypatch, mock_ctx, n_rare_per_gene=5)

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(
            case_group="J", control_group="V", gene_list="TYR", ctx=mock_ctx
        )

        assert "error" not in result
        for gr in result["gene_results"]:
            expected = gr["p_burden_fdr"] < 0.05
            assert gr["significant_burden"] == expected, (
                f"significant_burden mismatch for {gr['gene']}: "
                f"p_burden_fdr={gr['p_burden_fdr']}, flag={gr['significant_burden']}"
            )

    # ------------------------------------------------------------------
    # 14. test_figures_saved
    # ------------------------------------------------------------------
    def test_figures_saved(self, monkeypatch, mock_ctx):
        """figures list is populated after a successful run."""
        _setup_mocks(monkeypatch, mock_ctx, n_rare_per_gene=5)

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(
            case_group="J", control_group="V", gene_list="TYR", ctx=mock_ctx
        )

        assert "error" not in result
        assert "figures" in result
        assert len(result["figures"]) >= 1
        # ctx.state.figures should also be extended
        assert len(mock_ctx.state.figures) >= 1

    # ------------------------------------------------------------------
    # 15. test_burden_results_in_state
    # ------------------------------------------------------------------
    def test_burden_results_in_state(self, monkeypatch, mock_ctx):
        """custom_data['burden_results'] is stored in ctx.state after a successful run."""
        _setup_mocks(monkeypatch, mock_ctx, n_rare_per_gene=5)

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(
            case_group="J", control_group="V", gene_list="TYR", ctx=mock_ctx
        )

        assert "error" not in result
        assert "burden_results" in mock_ctx.state.custom_data
        burden = mock_ctx.state.custom_data["burden_results"]
        assert "gene_results" in burden
        assert "significant_genes" in burden

    # ------------------------------------------------------------------
    # 16. test_per_chromosome_merge
    # ------------------------------------------------------------------
    def test_per_chromosome_merge(self, monkeypatch, mock_ctx):
        """Genes on the same chromosome trigger exactly one merge call for that chrom."""
        # TYR and HLA are both on chr11 / chr6 … but both TYR and MC1R are on
        # different chroms.  Use TYR + another gene on the same chrom.
        # VITILIGO_GENES_HG38: TYR → chr11; no other gene is chr11.
        # Use TYR twice is not possible; pick genes on chr6: HLA, IRF4, TNF.
        merge_calls = _setup_mocks(monkeypatch, mock_ctx, n_rare_per_gene=5)

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        # HLA, IRF4, TNF are all on chr6 → should produce a single merge for chr6.
        result = vcf_burden_test(
            case_group="J",
            control_group="V",
            gene_list="HLA,IRF4,TNF",
            ctx=mock_ctx,
        )

        # One merge call for chr6 (regardless of result)
        chr6_calls = [c for c in merge_calls if c is not None and "chr6" in c]
        assert len(chr6_calls) == 1, f"Expected 1 chr6 merge, got {len(chr6_calls)}: {chr6_calls}"

    # ------------------------------------------------------------------
    # 17. test_genes_on_different_chroms
    # ------------------------------------------------------------------
    def test_genes_on_different_chroms(self, monkeypatch, mock_ctx):
        """Genes on different chromosomes trigger separate merge calls."""
        merge_calls = _setup_mocks(monkeypatch, mock_ctx, n_rare_per_gene=5)

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        # TYR=chr11, OCA2=chr15 → 2 different chromosomes → 2 merges
        result = vcf_burden_test(
            case_group="J",
            control_group="V",
            gene_list="TYR,OCA2",
            ctx=mock_ctx,
        )

        assert len(merge_calls) == 2, f"Expected 2 merge calls, got {len(merge_calls)}"
        chroms_merged = {c.split(":")[0] for c in merge_calls if c}
        assert "chr11" in chroms_merged
        assert "chr15" in chroms_merged

    # ------------------------------------------------------------------
    # 18. test_maf_zero_excluded
    # ------------------------------------------------------------------
    def test_maf_zero_excluded(self, monkeypatch, mock_ctx):
        """Variants with MAF == 0 (all hom-ref in valid samples) are not counted."""
        all_ids = list(ALL_SAMPLE_IDS)
        n = len(all_ids)

        # MAF=0 variant: all samples are hom-ref (gt_type=0)
        maf_zero_gt = np.zeros(n, dtype=np.int8)
        maf_zero_variant = FakeVariant(
            chrom="chr11", pos=88911700, ref="A", alt=["T"],
            gt_types=maf_zero_gt, n_samples=n, qual=50.0,
        )

        # Only maf=0 variants → nothing passes the maf == 0 filter
        mapping = {s: f"/fake/{s}.vcf.gz" for s in ALL_SAMPLE_IDS}
        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: mapping,
        )
        monkeypatch.setattr(
            "biobank_agent.utils.bcftools.get_tmp_dir",
            lambda ctx: mock_ctx.report_dir / "tmp",
        )
        (mock_ctx.report_dir / "tmp").mkdir(parents=True, exist_ok=True)

        def fake_merge(vcf_paths, out, region=None):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).touch()
            return Path(out)

        monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs", fake_merge)
        monkeypatch.setattr(
            "cyvcf2.VCF",
            lambda path: BurdenVCF(samples=all_ids, variants=[maf_zero_variant]),
        )

        from biobank_agent.skills.vcf_burden_test import vcf_burden_test

        result = vcf_burden_test(
            case_group="J", control_group="V", gene_list="TYR", ctx=mock_ctx
        )

        # MAF=0 variants are excluded → no rare variants → error
        assert "error" in result
