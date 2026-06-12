"""Comprehensive tests for biobank_agent/skills/vcf_pca.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.wgs.conftest import ALL_SAMPLE_IDS, _build_pheno_df


# ---------------------------------------------------------------------------
# Shared mock-setup helper
# ---------------------------------------------------------------------------

def _setup_mocks(monkeypatch, mock_ctx, n_samples=28, n_variants=500):
    """Wire all four injectable dependencies with sensible defaults."""
    mapping = {s: f"/fake/{s}.vcf.gz" for s in ALL_SAMPLE_IDS[:n_samples]}
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

    rng = np.random.default_rng(42)

    def fake_geno(
        merged_vcf_path,
        maf_min=0.05,
        maf_max=0.95,
        max_variants=50000,
        snv_only=True,
        region=None,
    ):
        G = rng.choice([0, 1, 2], size=(n_samples, n_variants)).astype(np.float64)
        vids = [f"chr22:{16000000 + i * 100}:A:T" for i in range(n_variants)]
        return G, vids, list(mapping.keys())

    monkeypatch.setattr(
        "biobank_agent.utils.vcf_genotypes.build_genotype_matrix", fake_geno
    )
    return mapping


# ---------------------------------------------------------------------------
# Tests for _expand_chromosomes helper
# ---------------------------------------------------------------------------

class TestExpandChromosomes:
    """Unit tests for the _expand_chromosomes private helper."""

    def _fn(self):
        from biobank_agent.skills.vcf_pca import _expand_chromosomes
        return _expand_chromosomes

    def test_expand_chromosomes_default_empty_string(self):
        """Empty string should expand to all 22 autosomes."""
        fn = self._fn()
        result = fn("")
        assert len(result) == 22
        assert result[0] == "chr1"
        assert result[-1] == "chr22"

    def test_expand_chromosomes_all_keyword(self):
        """'all' should expand to chr1-chr22."""
        fn = self._fn()
        result = fn("all")
        assert result == [f"chr{i}" for i in range(1, 23)]

    def test_expand_chromosomes_chr1_22_keyword(self):
        """'chr1-22' should expand to all 22 autosomes."""
        fn = self._fn()
        result = fn("chr1-22")
        assert len(result) == 22

    def test_expand_chromosomes_range(self):
        """'chr5-7' should return exactly ['chr5', 'chr6', 'chr7']."""
        fn = self._fn()
        result = fn("chr5-7")
        assert result == ["chr5", "chr6", "chr7"]

    def test_expand_chromosomes_single(self):
        """'chr22' with no range should return a list with one element."""
        fn = self._fn()
        result = fn("chr22")
        assert result == ["chr22"]


# ---------------------------------------------------------------------------
# Tests for vcf_pca skill
# ---------------------------------------------------------------------------

class TestVcfPcaSkill:
    """Integration-level tests for the vcf_pca skill (all I/O mocked)."""

    # ------------------------------------------------------------------
    # Error-path tests
    # ------------------------------------------------------------------

    def test_no_sample_paths(self, monkeypatch, mock_ctx):
        """Empty sample mapping should return an error dict."""
        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: {},
        )
        from biobank_agent.skills.vcf_pca import vcf_pca

        result = vcf_pca(ctx=mock_ctx)
        assert "error" in result
        assert "No sample VCF" in result["error"]

    def test_fewer_than_3_samples(self, monkeypatch, mock_ctx):
        """Fewer than 3 samples should produce an error, not a PCA result."""
        two_samples = {s: f"/fake/{s}.vcf.gz" for s in ALL_SAMPLE_IDS[:2]}
        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: two_samples,
        )
        monkeypatch.setattr(
            "biobank_agent.utils.bcftools.get_tmp_dir",
            lambda ctx: mock_ctx.report_dir / "tmp",
        )
        (mock_ctx.report_dir / "tmp").mkdir(parents=True, exist_ok=True)
        from biobank_agent.skills.vcf_pca import vcf_pca

        result = vcf_pca(ctx=mock_ctx)
        assert "error" in result
        assert "3" in result["error"]

    def test_qc_propagation_filters_samples(self, monkeypatch, mock_ctx):
        """pass_samples in vcf_qc_results should restrict the sample set."""
        # Only keep the first 5 samples via QC pass list
        pass_samples = ALL_SAMPLE_IDS[:5]
        mock_ctx.state.custom_data["vcf_qc_results"] = {"pass_samples": pass_samples}

        full_mapping = {s: f"/fake/{s}.vcf.gz" for s in ALL_SAMPLE_IDS}
        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: full_mapping,
        )
        monkeypatch.setattr(
            "biobank_agent.utils.bcftools.get_tmp_dir",
            lambda ctx: mock_ctx.report_dir / "tmp",
        )
        (mock_ctx.report_dir / "tmp").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "biobank_agent.utils.bcftools.merge_vcfs",
            lambda vcf_paths, out, region=None: (
                Path(out).parent.mkdir(parents=True, exist_ok=True) or Path(out).touch() or Path(out)
            ),
        )

        n_samples = 5
        rng = np.random.default_rng(0)

        def fake_geno(merged_vcf_path, maf_min=0.05, maf_max=0.95,
                      max_variants=50000, snv_only=True, region=None):
            G = rng.choice([0, 1, 2], size=(n_samples, 100)).astype(np.float64)
            vids = [f"chr22:{16000000 + i * 100}:A:T" for i in range(100)]
            return G, vids, pass_samples

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.build_genotype_matrix", fake_geno
        )

        from biobank_agent.skills.vcf_pca import vcf_pca

        result = vcf_pca(chromosomes="chr22", ctx=mock_ctx)
        assert "error" not in result
        assert result["n_samples"] == 5

    def test_qc_propagation_fallback_no_qc(self, monkeypatch, mock_ctx):
        """Without vcf_qc_results in state, all 28 samples should be used."""
        _setup_mocks(monkeypatch, mock_ctx, n_samples=28, n_variants=100)
        from biobank_agent.skills.vcf_pca import vcf_pca

        result = vcf_pca(chromosomes="chr22", ctx=mock_ctx)
        assert "error" not in result
        assert result["n_samples"] == 28

    def test_merge_fail_continues(self, monkeypatch, mock_ctx):
        """A merge failure for one region should be skipped; others succeed."""
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

        call_count = {"n": 0}

        def flaky_merge(vcf_paths, out, region=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("bcftools crashed on purpose")
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).touch()
            return Path(out)

        monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs", flaky_merge)

        rng = np.random.default_rng(7)

        def fake_geno(merged_vcf_path, maf_min=0.05, maf_max=0.95,
                      max_variants=50000, snv_only=True, region=None):
            G = rng.choice([0, 1, 2], size=(28, 50)).astype(np.float64)
            vids = [f"chr22:{16000000 + i * 100}:A:T" for i in range(50)]
            return G, vids, list(mapping.keys())

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.build_genotype_matrix", fake_geno
        )

        from biobank_agent.skills.vcf_pca import vcf_pca

        # Use a two-chrom spec so the first fails and the second succeeds
        result = vcf_pca(chromosomes="chr21-22", ctx=mock_ctx)
        assert "error" not in result
        assert call_count["n"] >= 2
        assert result["skipped_regions"][0]["region"] == "chr21"
        assert result["skipped_regions"][0]["stage"] == "merge"
        assert "bcftools crashed" in result["skipped_regions"][0]["reason"]

    def test_no_common_variants(self, monkeypatch, mock_ctx):
        """build_genotype_matrix returning 0 variants everywhere → error."""
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
        monkeypatch.setattr(
            "biobank_agent.utils.bcftools.merge_vcfs",
            lambda vcf_paths, out, region=None: (
                Path(out).parent.mkdir(parents=True, exist_ok=True) or Path(out).touch() or Path(out)
            ),
        )

        def zero_geno(merged_vcf_path, maf_min=0.05, maf_max=0.95,
                      max_variants=50000, snv_only=True, region=None):
            # shape (28, 0) — no passing variants
            return np.empty((28, 0)), [], list(mapping.keys())

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.build_genotype_matrix", zero_geno
        )

        from biobank_agent.skills.vcf_pca import vcf_pca

        result = vcf_pca(chromosomes="chr22", ctx=mock_ctx)
        assert "error" in result
        assert "No common variants" in result["error"]

    # ------------------------------------------------------------------
    # Happy-path and output-content tests
    # ------------------------------------------------------------------

    def test_basic_pca_output(self, monkeypatch, mock_ctx):
        """Normal run with 28 samples and 500 variants returns expected keys."""
        _setup_mocks(monkeypatch, mock_ctx, n_samples=28, n_variants=500)
        from biobank_agent.skills.vcf_pca import vcf_pca

        result = vcf_pca(chromosomes="chr22", ctx=mock_ctx)

        assert "error" not in result
        assert result["n_samples"] == 28
        assert result["n_snps_used"] == 500
        assert "explained_variance_ratio" in result
        assert isinstance(result["pc_coordinates"], list)
        assert len(result["pc_coordinates"]) <= 10  # capped at first 10

    def test_max_variants_subsample(self, monkeypatch, mock_ctx):
        """When n_snps > max_variants, the result uses exactly max_variants SNPs."""
        _setup_mocks(monkeypatch, mock_ctx, n_samples=28, n_variants=500)
        from biobank_agent.skills.vcf_pca import vcf_pca

        result = vcf_pca(max_variants=200, chromosomes="chr22", ctx=mock_ctx)

        assert "error" not in result
        assert result["n_snps_used"] == 200

    def test_n_components_capped(self, monkeypatch, mock_ctx):
        """n_components larger than n_samples-1 should be silently capped."""
        _setup_mocks(monkeypatch, mock_ctx, n_samples=5, n_variants=100)
        from biobank_agent.skills.vcf_pca import vcf_pca

        # Requesting 50 PCs but only 5 samples → capped at min(50, 4, 100) = 4
        result = vcf_pca(n_components=50, chromosomes="chr22", ctx=mock_ctx)

        assert "error" not in result
        assert result["n_components"] <= 4

    def test_scree_plot_generated(self, monkeypatch, mock_ctx):
        """A figure path containing 'scree' must be present after a successful run."""
        _setup_mocks(monkeypatch, mock_ctx, n_samples=28, n_variants=200)
        from biobank_agent.skills.vcf_pca import vcf_pca

        result = vcf_pca(chromosomes="chr22", ctx=mock_ctx)

        assert "error" not in result
        scree_paths = [p for p in result["figures"] if "scree" in p]
        assert len(scree_paths) >= 1, "Expected at least one scree-plot figure"

    def test_pc_scatter_generated(self, monkeypatch, mock_ctx):
        """A figure path for PC1-vs-PC2 scatter must appear in the figures list."""
        _setup_mocks(monkeypatch, mock_ctx, n_samples=28, n_variants=200)
        from biobank_agent.skills.vcf_pca import vcf_pca

        result = vcf_pca(n_components=5, chromosomes="chr22", ctx=mock_ctx)

        assert "error" not in result
        pc_paths = [p for p in result["figures"] if "pc1" in p and "pc2" in p]
        assert len(pc_paths) >= 1, "Expected PC1-vs-PC2 scatter figure"

    def test_phenotype_group_colors(self, monkeypatch, mock_ctx):
        """phenotype_group should be attached to at least one pc_coordinates record."""
        _setup_mocks(monkeypatch, mock_ctx, n_samples=28, n_variants=200)
        from biobank_agent.skills.vcf_pca import vcf_pca

        result = vcf_pca(n_components=3, chromosomes="chr22", ctx=mock_ctx)

        assert "error" not in result
        records_with_group = [
            r for r in result["pc_coordinates"] if "phenotype_group" in r
        ]
        assert len(records_with_group) > 0, "Expected phenotype_group in pc_coordinates"

    def test_no_phenotype_group(self, monkeypatch, mock_ctx):
        """If the biomarkers query returns no phenotype_group column, PCA still runs."""
        _setup_mocks(monkeypatch, mock_ctx, n_samples=28, n_variants=200)
        # Override dm to return a DataFrame without phenotype_group
        pheno_df_no_group = pd.DataFrame(
            {"sample_id": ALL_SAMPLE_IDS, "age": range(28)}
        )
        mock_ctx.dm = SimpleNamespace(
            query=lambda sql, *a, **kw: pheno_df_no_group.copy(),
            subject_id_col="sample_id",
        )
        from biobank_agent.skills.vcf_pca import vcf_pca

        result = vcf_pca(n_components=3, chromosomes="chr22", ctx=mock_ctx)

        assert "error" not in result
        assert result["n_samples"] == 28
        # No record should carry phenotype_group since the column was absent
        for rec in result["pc_coordinates"]:
            assert "phenotype_group" not in rec

    def test_pca_stored_in_state(self, monkeypatch, mock_ctx):
        """After a successful run, pca_result must be stored in ctx.state.custom_data."""
        _setup_mocks(monkeypatch, mock_ctx, n_samples=28, n_variants=200)
        from biobank_agent.skills.vcf_pca import vcf_pca

        result = vcf_pca(n_components=5, chromosomes="chr22", ctx=mock_ctx)

        assert "error" not in result
        pca_result = mock_ctx.state.custom_data.get("pca_result")
        assert pca_result is not None, "pca_result not stored in state"
        assert "pcs" in pca_result
        assert "explained_variance" in pca_result
        assert "sample_order" in pca_result
        assert "n_snps" in pca_result
        assert isinstance(pca_result["pcs"], pd.DataFrame)

    def test_region_overrides_chromosomes(self, monkeypatch, mock_ctx):
        """When region= is supplied, the chromosomes parameter is ignored."""
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

        merge_regions: list[str | None] = []

        def recording_merge(vcf_paths, out, region=None):
            merge_regions.append(region)
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).touch()
            return Path(out)

        monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs", recording_merge)

        rng = np.random.default_rng(3)

        def fake_geno(merged_vcf_path, maf_min=0.05, maf_max=0.95,
                      max_variants=50000, snv_only=True, region=None):
            G = rng.choice([0, 1, 2], size=(28, 100)).astype(np.float64)
            vids = [f"chr22:{16000000 + i * 100}:A:T" for i in range(100)]
            return G, vids, list(mapping.keys())

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.build_genotype_matrix", fake_geno
        )

        from biobank_agent.skills.vcf_pca import vcf_pca

        result = vcf_pca(
            chromosomes="chr1-22",
            region="chr22:1-1000",
            ctx=mock_ctx,
        )

        assert "error" not in result
        # Only one merge call, and it must carry the exact region string
        assert merge_regions == ["chr22:1-1000"], (
            f"Expected single region 'chr22:1-1000', got {merge_regions}"
        )
