"""Comprehensive tests for biobank_agent/skills/vcf_association.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.wgs.conftest import ALL_SAMPLE_IDS, _build_pheno_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(tmp_path, pheno_df=None):
    """Build a minimal ctx object, optionally with a custom pheno_df."""
    if pheno_df is None:
        pheno_df = _build_pheno_df()

    dm = SimpleNamespace(
        query=lambda sql, *a, **kw: pheno_df.copy(),
        subject_id_col="sample_id",
    )
    state = SimpleNamespace(custom_data={}, figures=[], cohorts={})
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(dm=dm, state=state, report_dir=report_dir)


# ---------------------------------------------------------------------------
# TestGenomicInflation — unit tests for the internal helper
# ---------------------------------------------------------------------------

class TestGenomicInflation:
    """Direct tests for _genomic_inflation."""

    def _fn(self):
        from biobank_agent.skills.vcf_association import _genomic_inflation
        return _genomic_inflation

    def test_fewer_than_10_valid_returns_nan(self):
        fn = self._fn()
        p = np.array([0.1, 0.2, 0.3])
        assert np.isnan(fn(p))

    def test_out_of_range_values_filtered(self):
        fn = self._fn()
        # 0 and 1 are excluded; 11 valid interior values remain → >= 10 → finite result
        p = np.array([0.0, 1.0,
                      0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        result = fn(p)
        assert np.isfinite(result)

    def test_uniform_p_yields_lambda_near_one(self):
        fn = self._fn()
        rng = np.random.default_rng(0)
        p = rng.uniform(0.001, 0.999, size=10000)
        lam = fn(p)
        assert 0.8 < lam < 1.2

    def test_inflated_p_values_yield_lambda_above_one(self):
        fn = self._fn()
        # Very small p-values → chi2 > expected median → lambda > 1
        rng = np.random.default_rng(1)
        p = rng.uniform(1e-8, 0.001, size=500)
        lam = fn(p)
        assert lam > 1.2


# ---------------------------------------------------------------------------
# TestVcfAssociation — tests for the main skill
# ---------------------------------------------------------------------------

class TestVcfAssociation:

    def _setup_mocks(self, monkeypatch, mock_ctx, allele_counts=None):
        """Patch all external I/O so the skill runs without real files."""
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

        rng = np.random.default_rng(42)

        def fake_allele_counts(merged, case_ids, ctrl_ids,
                               maf_min=0.0, region=None, max_variants=0):
            if allele_counts is not None:
                return allele_counts
            results = []
            for i in range(50):
                results.append({
                    "chrom": "chr22",
                    "pos": 16000000 + i * 100,
                    "ref": "A",
                    "alt": "T",
                    "rsid": f"rs{100000 + i}",
                    "case_alt": int(rng.integers(0, 10)),
                    "case_ref": int(rng.integers(10, 30)),
                    "ctrl_alt": int(rng.integers(0, 10)),
                    "ctrl_ref": int(rng.integers(10, 30)),
                    "maf": round(float(rng.uniform(0.01, 0.4)), 4),
                })
            return results

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.build_allele_counts",
            fake_allele_counts,
        )

    # ------------------------------------------------------------------
    # 1. No phenotype data
    # ------------------------------------------------------------------
    def test_no_phenotype_data(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        mock_ctx.dm.query = lambda sql, *a, **kw: pd.DataFrame()
        result = vcf_association(case_group="J", control_group="V", ctx=mock_ctx)
        assert "error" in result
        assert "phenotype" in result["error"].lower() or "no" in result["error"].lower()

    # ------------------------------------------------------------------
    # 2. No phenotype_group column
    # ------------------------------------------------------------------
    def test_no_phenotype_group_column(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        df = pd.DataFrame({"sample_id": ALL_SAMPLE_IDS, "age": [40] * len(ALL_SAMPLE_IDS)})
        mock_ctx.dm.query = lambda sql, *a, **kw: df
        result = vcf_association(case_group="J", control_group="V", ctx=mock_ctx)
        assert "error" in result
        assert "phenotype_group" in result["error"]

    # ------------------------------------------------------------------
    # 3. Case group too small (non-existent group → 0 samples)
    # ------------------------------------------------------------------
    def test_case_group_too_small(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        result = vcf_association(case_group="Z", control_group="V", ctx=mock_ctx)
        assert "error" in result
        assert "Z" in result["error"]

    # ------------------------------------------------------------------
    # 4. Control group too small
    # ------------------------------------------------------------------
    def test_control_group_too_small(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        # Build a df where control_group "SINGLE" has only 1 sample
        pheno_df = _build_pheno_df()
        single_row = pd.DataFrame({
            "sample_id": ["X001"],
            "phenotype_group": ["SINGLE"],
            "vcf_path": ["/fake/X001.vcf.gz"],
        })
        pheno_df = pd.concat([pheno_df, single_row], ignore_index=True)
        mock_ctx.dm.query = lambda sql, *a, **kw: pheno_df.copy()

        result = vcf_association(case_group="J", control_group="SINGLE", ctx=mock_ctx)
        assert "error" in result
        assert "SINGLE" in result["error"] or "<2" in result["error"]

    # ------------------------------------------------------------------
    # 5. Basic association run — J vs V
    # ------------------------------------------------------------------
    def test_basic_association_output(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_association(
            case_group="J", control_group="V",
            region="chr22",
            ctx=mock_ctx,
        )
        assert "error" not in result
        assert result["case_group"] == "J"
        assert result["control_group"] == "V"
        assert result["n_cases"] == 10
        assert result["n_controls"] == 6
        assert result["n_variants_tested"] > 0
        assert result["analysis_mode"] == "exploratory_fisher_exact"
        assert "standard_gwas_available" in result
        assert "top_hits" in result
        assert "figures" in result

    # ------------------------------------------------------------------
    # 6. QC propagation — pass_samples filters case/ctrl
    # ------------------------------------------------------------------
    def test_qc_propagation(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        # Allow only 3 J samples and all V samples through QC
        pass_samples = ["J001", "J002", "J003"] + [f"V{i:03d}" for i in range(1, 7)]
        mock_ctx.state.custom_data["vcf_qc_results"] = {"pass_samples": pass_samples}

        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_association(
            case_group="J", control_group="V",
            region="chr22",
            ctx=mock_ctx,
        )
        assert "error" not in result
        # After QC filter only 3 J samples remain
        assert result["n_cases"] == 3
        assert result["n_controls"] == 6

    # ------------------------------------------------------------------
    # 7. No VCF files found for samples
    # ------------------------------------------------------------------
    def test_no_vcf_files(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: {},
        )
        result = vcf_association(case_group="J", control_group="V", ctx=mock_ctx)
        assert "error" in result
        assert "vcf" in result["error"].lower() or "no" in result["error"].lower()

    # ------------------------------------------------------------------
    # 8. Merge fail for a region — continues to next region
    # ------------------------------------------------------------------
    def test_merge_fail_continues(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

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

        def merge_sometimes_fails(vcf_paths, out, region=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("bcftools error")
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).touch()
            return Path(out)

        monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs", merge_sometimes_fails)

        rng = np.random.default_rng(7)

        def fake_allele_counts(merged, case_ids, ctrl_ids, **kw):
            return [
                {
                    "chrom": "chr22", "pos": 16000000 + i * 100,
                    "ref": "A", "alt": "T", "rsid": f"rs{i}",
                    "case_alt": int(rng.integers(0, 5)),
                    "case_ref": int(rng.integers(10, 20)),
                    "ctrl_alt": int(rng.integers(0, 5)),
                    "ctrl_ref": int(rng.integers(10, 20)),
                    "maf": 0.1,
                }
                for i in range(10)
            ]

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.build_allele_counts",
            fake_allele_counts,
        )

        # Use two regions so the second one can succeed after the first fails
        result = vcf_association(
            case_group="J", control_group="V",
            chromosomes="chr21-22",
            ctx=mock_ctx,
        )
        # The skill should continue past the merge failure
        assert "error" not in result
        assert result["n_variants_tested"] > 0

    # ------------------------------------------------------------------
    # 9. No variants tested → error
    # ------------------------------------------------------------------
    def test_no_variants_tested(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        self._setup_mocks(monkeypatch, mock_ctx, allele_counts=[])
        result = vcf_association(
            case_group="J", control_group="V",
            region="chr22",
            ctx=mock_ctx,
        )
        assert "error" in result
        assert "no variants" in result["error"].lower()

    # ------------------------------------------------------------------
    # 10. Fisher exact called → p_values populated
    # ------------------------------------------------------------------
    def test_fisher_exact_called(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_association(
            case_group="J", control_group="V",
            region="chr22",
            ctx=mock_ctx,
        )
        assert "error" not in result
        hits = result["top_hits"]
        assert len(hits) > 0
        for h in hits:
            assert "p_value" in h
            assert 0.0 <= h["p_value"] <= 1.0

    # ------------------------------------------------------------------
    # 11. Bonferroni correction present
    # ------------------------------------------------------------------
    def test_bonferroni_correction(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_association(
            case_group="J", control_group="V",
            region="chr22",
            ctx=mock_ctx,
        )
        assert "error" not in result
        assert "n_significant_bonferroni" in result
        assert isinstance(result["n_significant_bonferroni"], int)

        # Verify the key is also stored in gwas_results
        gwas = mock_ctx.state.custom_data.get("gwas_results", {})
        all_res = gwas.get("all_results", [])
        assert all_res, "all_results should be non-empty"
        for rec in all_res:
            assert "p_bonferroni" in rec
            assert 0.0 <= rec["p_bonferroni"] <= 1.0

    # ------------------------------------------------------------------
    # 12. FDR correction present
    # ------------------------------------------------------------------
    def test_fdr_correction(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_association(
            case_group="J", control_group="V",
            region="chr22",
            ctx=mock_ctx,
        )
        assert "error" not in result
        assert "n_significant_fdr" in result

        gwas = mock_ctx.state.custom_data.get("gwas_results", {})
        all_res = gwas.get("all_results", [])
        for rec in all_res:
            assert "p_fdr" in rec
            assert 0.0 <= rec["p_fdr"] <= 1.0

    # ------------------------------------------------------------------
    # 13. Lambda GC warning — substantial (> 1.2)
    # ------------------------------------------------------------------
    def test_lambda_gc_warning_substantial(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        # Build allele counts that will produce extremely small p-values
        # to drive lambda_gc well above 1.2
        rng = np.random.default_rng(99)
        skewed_counts = []
        for i in range(200):
            skewed_counts.append({
                "chrom": "chr22",
                "pos": 16000000 + i * 100,
                "ref": "A", "alt": "T", "rsid": f"rs{i}",
                # Large imbalance → tiny p-value
                "case_alt": 100,
                "case_ref": 0,
                "ctrl_alt": 0,
                "ctrl_ref": 100,
                "maf": 0.2,
            })

        self._setup_mocks(monkeypatch, mock_ctx, allele_counts=skewed_counts)
        result = vcf_association(
            case_group="J", control_group="V",
            region="chr22",
            ctx=mock_ctx,
        )
        assert "error" not in result
        if result["lambda_gc"] is not None and result["lambda_gc"] > 1.2:
            assert result["lambda_gc_warning"] is not None
            assert "substantial" in result["lambda_gc_warning"].lower()

    # ------------------------------------------------------------------
    # 14. Lambda GC warning — mild (1.1 < lambda <= 1.2)
    # ------------------------------------------------------------------
    def test_lambda_gc_warning_mild(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association
        from biobank_agent.skills.vcf_association import _genomic_inflation

        # Craft p-values that produce lambda in (1.1, 1.2]
        # Test _genomic_inflation directly, then verify warning branch in skill
        rng = np.random.default_rng(2024)
        # Mix: some small p-values mixed with uniform → moderate inflation
        p_uniform = rng.uniform(0.001, 0.999, size=400)
        p_small = rng.uniform(0.0001, 0.01, size=100)
        p_mixed = np.concatenate([p_uniform, p_small])
        lam = _genomic_inflation(p_mixed)

        # For the skill test: just confirm warning logic is exercised
        # by patching _genomic_inflation itself to return 1.15
        import biobank_agent.skills.vcf_association as assoc_mod
        original = assoc_mod._genomic_inflation

        def patched_inflation(p_vals):
            return 1.15

        monkeypatch.setattr(assoc_mod, "_genomic_inflation", patched_inflation)

        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_association(
            case_group="J", control_group="V",
            region="chr22",
            ctx=mock_ctx,
        )
        assert "error" not in result
        assert result["lambda_gc_warning"] is not None
        assert "mild" in result["lambda_gc_warning"].lower()

    # ------------------------------------------------------------------
    # 15. Lambda GC no warning (lambda < 1.1)
    # ------------------------------------------------------------------
    def test_lambda_gc_no_warning(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association
        import biobank_agent.skills.vcf_association as assoc_mod

        def patched_inflation(p_vals):
            return 0.98

        monkeypatch.setattr(assoc_mod, "_genomic_inflation", patched_inflation)

        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_association(
            case_group="J", control_group="V",
            region="chr22",
            ctx=mock_ctx,
        )
        assert "error" not in result
        assert result["lambda_gc_warning"] is None

    # ------------------------------------------------------------------
    # 16. Top hits sorted by p_value ascending
    # ------------------------------------------------------------------
    def test_top_hits_sorted(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_association(
            case_group="J", control_group="V",
            region="chr22",
            ctx=mock_ctx,
        )
        assert "error" not in result
        hits = result["top_hits"]
        p_vals = [h["p_value"] for h in hits]
        assert p_vals == sorted(p_vals), "top_hits should be sorted by p_value ascending"

    # ------------------------------------------------------------------
    # 17. Manhattan plot generated
    # ------------------------------------------------------------------
    def test_manhattan_generated(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_association(
            case_group="J", control_group="V",
            region="chr22",
            ctx=mock_ctx,
        )
        assert "error" not in result
        figures = result["figures"]
        assert any("manhattan" in f.lower() for f in figures), (
            f"No manhattan figure found in {figures}"
        )

    # ------------------------------------------------------------------
    # 18. QQ plot generated
    # ------------------------------------------------------------------
    def test_qq_generated(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_association(
            case_group="J", control_group="V",
            region="chr22",
            ctx=mock_ctx,
        )
        assert "error" not in result
        figures = result["figures"]
        assert any("qq" in f.lower() for f in figures), (
            f"No QQ figure found in {figures}"
        )

    # ------------------------------------------------------------------
    # 19. power_warning always present
    # ------------------------------------------------------------------
    def test_power_warning(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_association(
            case_group="J", control_group="V",
            region="chr22",
            ctx=mock_ctx,
        )
        assert "error" not in result
        assert "power_warning" in result
        assert result["power_warning"], "power_warning should be a non-empty string"
        assert "n_cases" in result["power_warning"] or "exploratory" in result["power_warning"].lower()

    # ------------------------------------------------------------------
    # 20. GWAS results stored in state.custom_data
    # ------------------------------------------------------------------
    def test_gwas_results_stored_in_state(self, monkeypatch, mock_ctx):
        from biobank_agent.skills.vcf_association import vcf_association

        self._setup_mocks(monkeypatch, mock_ctx)
        result = vcf_association(
            case_group="J", control_group="V",
            region="chr22",
            ctx=mock_ctx,
        )
        assert "error" not in result
        assert "gwas_results" in mock_ctx.state.custom_data

        gwas = mock_ctx.state.custom_data["gwas_results"]
        assert gwas["case_group"] == "J"
        assert gwas["control_group"] == "V"
        assert gwas["n_tested"] > 0
        assert "all_results" in gwas
        assert "lambda_gc" in gwas
